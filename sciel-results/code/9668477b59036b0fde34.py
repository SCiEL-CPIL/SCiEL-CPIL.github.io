"""Physics-based annealing for packing 200 equal circles in a unit square."""
import numpy as np
import time

N = 200
DEADLINE_FRACTION = 0.92  # safety margin before the 150 s wall-clock limit


def _score(positions):
    """Compute the maximum common radius for a given set of centres."""
    if positions.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(positions)):
        return 0.0
    # Distance from each centre to the nearest boundary
    boundary_dists = np.minimum(
        np.minimum(0.5 - positions[:, 0], positions[:, 0] + 0.5),
        np.minimum(0.5 - positions[:, 1], positions[:, 1] + 0.5)
    )
    min_boundary = np.min(boundary_dists)
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_boundary, min_pair)


def _repulsion_forces(positions, power=10):
    """Compute repulsive forces between all pairs of circles.
    
    Force magnitude ~ 1/d^power, directed away from each other.
    """
    n = len(positions)
    forces = np.zeros_like(positions)
    # Use vectorized computation in chunks to save memory
    chunk_size = 50
    for i_start in range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        chunk = positions[i_start:i_end]  # shape (c, 2)
        # diff[i, j] = chunk[i] - positions[j]
        diff = chunk[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (c, n, 2)
        dist_sq = np.sum(diff ** 2, axis=2)  # (c, n)
        # Avoid division by zero
        dist_sq = np.maximum(dist_sq, 1e-12)
        dist = np.sqrt(dist_sq)  # (c, n)
        # Force magnitude: k / dist^power, applied along direction
        # Only consider pairs where dist < some cutoff for efficiency
        inv_dist_pow = np.where(dist < 0.5, 1.0 / (dist ** power), 0.0)
        # Direction: diff / dist
        direction = diff / dist[:, :, np.newaxis]
        # Force on each particle in chunk from all others
        force_chunk = np.sum(inv_dist_pow[:, :, np.newaxis] * direction, axis=1)
        forces[i_start:i_end] += force_chunk
    return forces


def _boundary_forces(positions, strength=50.0):
    """Compute forces pushing circles away from boundaries."""
    forces = np.zeros_like(positions)
    for axis in range(2):
        # Left boundary: x = -0.5
        d_left = positions[:, axis] + 0.5
        # Right boundary: x = +0.5
        d_right = 0.5 - positions[:, axis]
        # If close to boundary, push inward
        threshold = 0.05
        # Left: if d_left < threshold, push in +axis direction
        mask_left = d_left < threshold
        forces[mask_left, axis] += strength * (threshold - d_left[mask_left])
        # Right: if d_right < threshold, push in -axis direction
        mask_right = d_right < threshold
        forces[mask_right, axis] -= strength * (threshold - d_right[mask_right])
    return forces


def _simulate(positions, n_steps, dt, power, boundary_strength):
    """Run a physics simulation with velocity Verlet-like integration."""
    positions = positions.copy()
    velocities = np.zeros_like(positions)
    
    for step in range(n_steps):
        # Compute forces
        f_rep = _repulsion_forces(positions, power=power)
        f_bound = _boundary_forces(positions, strength=boundary_strength)
        forces = f_rep + f_bound
        
        # Update velocities (simple Euler with damping)
        velocities = 0.95 * velocities + dt * forces
        positions = positions + dt * velocities
        
        # Project back inside the square
        positions[:, 0] = np.clip(positions[:, 0], -0.49, 0.49)
        positions[:, 1] = np.clip(positions[:, 1], -0.49, 0.49)
    
    return positions


def _gradient_ascent(positions, n_steps, lr_schedule='adaptive'):
    """Perform gradient ascent on a soft-min distance objective."""
    positions = positions.copy()
    best_pos = positions.copy()
    best_score = _score(positions)
    
    # Use a smooth approximation of the minimum distance
    # Objective: maximize min_ij d_ij / 2 and min_i boundary_dist
    # Soft min: -log(sum(exp(-d_i / t))) / t
    temperature = 0.001
    
    for step in range(n_steps):
        # Compute pairwise distances
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Soft min of pairwise distances
        # We want to maximize the minimum, so we use a soft-min that approximates min
        # For gradient: d/dx_j of softmin = sum_i exp(-d_ij/t) * (x_j - x_i) / (d_ij * t) / sum_k exp(-d_ik/t)
        
        # For efficiency, only consider nearby pairs
        # Use a simple approach: compute gradient of sum of inverse distances weighted by proximity to minimum
        
        # Find current minimum distance
        min_dist = np.min(dists)
        
        # Weight each pair by how close it is to the minimum
        weights = np.exp(-(dists - min_dist) / 0.005)
        weights[~np.isfinite(weights)] = 0
        np.fill_diagonal(weights, 0)
        
        # Gradient of sum of weights * d_ij
        # d/dx_j (sum_i w_ij * d_ij) = sum_i w_ij * (x_j - x_i) / d_ij
        diff_unit = diff / dists[:, :, np.newaxis]
        grad_pair = np.sum(weights[:, :, np.newaxis] * diff_unit, axis=1)
        
        # Boundary gradient
        grad_bound = np.zeros_like(positions)
        for axis in range(2):
            d_left = positions[:, axis] + 0.5
            d_right = 0.5 - positions[:, axis]
            min_bound = np.minimum(d_left, d_right)
            # Weight by proximity to minimum boundary distance
            weights_b = np.exp(-(min_bound - min_dist) / 0.005)
            weights_b[~np.isfinite(weights_b)] = 0
            grad_bound[:, axis] = weights_b * np.where(d_left < d_right, -1, 1)
        
        gradient = grad_pair + grad_bound * 0.5
        
        # Adaptive learning rate
        if lr_schedule == 'adaptive':
            base_lr = 0.001 * (1 - step / n_steps) ** 0.5
            # Scale by magnitude of gradient
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > 1e-10:
                lr = base_lr / grad_norm * 0.01
            else:
                lr = base_lr
        else:
            lr = 0.0001
        
        positions = positions + lr * gradient
        
        # Keep inside square
        positions[:, 0] = np.clip(positions[:, 0], -0.495, 0.495)
        positions[:, 1] = np.clip(positions[:, 1], -0.495, 0.495)
        
        # Check score periodically
        if step % 20 == 0:
            curr_score = _score(positions)
            if curr_score > best_score:
                best_score = curr_score
                best_pos = positions.copy()
    
    return best_pos, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 150 * DEADLINE_FRACTION
    
    np.random.seed(42)
    
    best_positions = None
    best_score = 0.0
    
    # Initial configurations
    def make_grid():
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N].copy()
    
    def make_random():
        return np.random.uniform(-0.49, 0.49, size=(N, 2))
    
    def make_hexagonal():
        """Hexagonal packing arrangement."""
        positions = []
        r = 0.036  # approximate radius
        dx = 2 * r
        dy = dx * np.sqrt(3) / 2
        row = 0
        while len(positions) < N:
            offset = 0 if row % 2 == 0 else dx / 2
            col = 0
            while len(positions) < N:
                x = offset + col * dx - 0.5 + dx / 2
                y = row * dy - 0.5 + dy / 2
                if -0.49 <= x <= 0.49 and -0.49 <= y <= 0.49:
                    positions.append([x, y])
                col += 1
            row += 1
        return np.array(positions[:N])
    
    initializers = [make_grid, make_random, make_hexagonal]
    
    for init_idx, init_func in enumerate(initializers):
        if time.time() > deadline:
            break
        
        positions = init_func()
        
        # Phase 1: Physics-based annealing
        if time.time() > deadline:
            break
        
        # Simulate with decreasing temperature
        n_sim_steps = 2000
        positions = _simulate(
            positions, 
            n_steps=n_sim_steps,
            dt=0.0001,
            power=8,
            boundary_strength=20.0
        )
        
        score = _score(positions)
        if score > best_score:
            best_score = score
            best_positions = positions.copy()
        
        # Phase 2: Gradient ascent refinement
        if time.time() > deadline:
            break
        
        positions, score = _gradient_ascent(positions, n_steps=500, lr_schedule='adaptive')
        
        if score > best_score:
            best_score = score
            best_positions = positions.copy()
        
        # Phase 3: More physics with smaller dt for fine-tuning
        if time.time() > deadline:
            break
        
        positions = _simulate(
            positions,
            n_steps=1000,
            dt=0.00005,
            power=10,
            boundary_strength=50.0
        )
        
        score = _score(positions)
        if score > best_score:
            best_score = score
            best_positions = positions.copy()
        
        # Phase 4: Final gradient ascent
        if time.time() > deadline:
            break
        
        positions, score = _gradient_ascent(positions, n_steps=300, lr_schedule='adaptive')
        
        if score > best_score:
            best_score = score
            best_positions = positions.copy()
    
    if best_positions is None:
        return make_grid()
    
    return best_positions
