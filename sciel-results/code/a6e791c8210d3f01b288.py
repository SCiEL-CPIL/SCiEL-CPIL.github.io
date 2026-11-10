"""Program for 200 equal circles in a unit square."""
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay

N = 200
TIME_LIMIT = 145.0  # seconds, leaving margin before 150s kill


def _compute_radius(centres):
    """Compute the packing radius for a set of centres."""
    if len(centres) != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check bounds: centres must be inside [-0.5, 0.5]^2
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return 0.0
    # Distance to walls
    wall_dists = np.min(np.stack([
        0.5 - centres[:, 0],
        0.5 + centres[:, 0],
        0.5 - centres[:, 1],
        0.5 + centres[:, 1]
    ], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    if min_wall <= 0:
        return 0.0
    # Minimum pairwise distance
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    return min(min_dist / 2.0, min_wall)


def _hexagonal_packing():
    """Create a hexagonal packing pattern, optimized for N points."""
    best_centres = None
    best_r = -1
    
    # Try various numbers of rows
    for n_rows in range(12, 20):
        # For hexagonal packing, alternating rows have offset
        for start_offset in [0, 1]:
            for n_cols in range(12, 20):
                # Calculate how many points fit in this configuration
                total_points = 0
                for row in range(n_rows):
                    if (row + start_offset) % 2 == 0:
                        total_points += n_cols
                    else:
                        total_points += n_cols - 1
                
                if total_points < N:
                    continue
                
                # Calculate spacing s
                s_horiz = 1.0 / n_cols
                if n_rows > 1:
                    s_vert = 2.0 / ((n_rows - 1) * np.sqrt(3))
                else:
                    s_vert = 1.0
                
                s = min(s_horiz, s_vert)
                r = s / 2.0
                
                if r <= best_r:
                    continue
                
                # Create the grid
                centres = []
                y_start = -0.5 + s * np.sqrt(3) / 2
                for row in range(n_rows):
                    y = y_start + row * s * np.sqrt(3) / 2
                    if (row + start_offset) % 2 == 0:
                        x_start = -0.5 + s / 2
                        for col in range(n_cols):
                            x = x_start + col * s
                            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                                centres.append((x, y))
                    else:
                        x_start = -0.5 + s
                        for col in range(n_cols - 1):
                            x = x_start + col * s
                            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                                centres.append((x, y))
                
                centres = np.array(centres)
                if len(centres) < N:
                    continue
                
                centres = centres[:N]
                actual_r = _compute_radius(centres)
                if actual_r > best_r:
                    best_r = actual_r
                    best_centres = centres
    
    if best_centres is not None:
        return best_centres
    
    # Fallback to grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _force_directed_optimize(centres, deadline, lr=0.001, momentum=0.9):
    """
    Vectorized force-directed relaxation.
    Computes forces from tight pairwise and wall constraints,
    and moves all points simultaneously using momentum.
    """
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    velocity = np.zeros_like(best_centres)
    
    # Precompute index pairs for efficiency
    i_idx, j_idx = np.triu_indices(N, k=1)
    n_pairs = len(i_idx)
    
    while time.time() < deadline:
        # Compute current radius
        r = _compute_radius(best_centres)
        if r <= 0:
            break
        
        # Compute pairwise distances
        dists = cdist(best_centres, best_centres)
        np.fill_diagonal(dists, np.inf)
        
        # Extract pairwise distances for all pairs
        pair_dists = dists[i_idx, j_idx]
        
        # Compute wall distances
        wall_dists = np.stack([
            0.5 - best_centres[:, 0],
            0.5 + best_centres[:, 0],
            0.5 - best_centres[:, 1],
            0.5 + best_centres[:, 1]
        ], axis=1)
        
        # Initialize forces
        forces = np.zeros_like(best_centres)
        
        # Pairwise repulsion (vectorized)
        # Only consider pairs that are somewhat close
        close_mask = pair_dists < 2.0 * r * 1.5
        if np.any(close_mask):
            close_i = i_idx[close_mask]
            close_j = j_idx[close_mask]
            close_d = pair_dists[close_mask]
            
            # Avoid division by zero
            valid = close_d > 1e-10
            if np.any(valid):
                close_i = close_i[valid]
                close_j = close_j[valid]
                close_d = close_d[valid]
                
                # Direction vectors
                dirs = (best_centres[close_i] - best_centres[close_j]) / close_d[:, np.newaxis]
                
                # Tightness: how close to the minimum constraint
                tightness = np.maximum(0, 2.0 * r - close_d) / (2.0 * r)
                f_mag = (tightness ** 2)[:, np.newaxis]
                
                # Accumulate forces
                np.add.at(forces, close_i, f_mag * dirs)
                np.add.at(forces, close_j, -f_mag * dirs)
        
        # Wall repulsion (vectorized)
        # Left wall: x = -0.5, push +x
        d_left = wall_dists[:, 0]
        tight_left = np.maximum(0, r - d_left) / r
        forces[:, 0] += (tight_left ** 2)
        
        # Right wall: x = 0.5, push -x
        d_right = wall_dists[:, 1]
        tight_right = np.maximum(0, r - d_right) / r
        forces[:, 0] -= (tight_right ** 2)
        
        # Bottom wall: y = -0.5, push +y
        d_bottom = wall_dists[:, 2]
        tight_bottom = np.maximum(0, r - d_bottom) / r
        forces[:, 1] += (tight_bottom ** 2)
        
        # Top wall: y = 0.5, push -y
        d_top = wall_dists[:, 3]
        tight_top = np.maximum(0, r - d_top) / r
        forces[:, 1] -= (tight_top ** 2)
        
        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1)
        max_force = np.max(force_norms) if np.max(force_norms) > 0 else 1.0
        if max_force > 0:
            forces = forces / max_force
        
        # Update velocity with momentum
        velocity = momentum * velocity + (1 - momentum) * forces
        
        # Adaptive learning rate: decay over time
        elapsed = time.time() - (deadline - 120)
        if elapsed > 0:
            lr_current = lr * (0.995 ** int(elapsed))
        else:
            lr_current = lr
        
        # Update positions
        new_centres = best_centres + lr_current * velocity
        
        # Project back into the square
        new_centres[:, 0] = np.clip(new_centres[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
        new_centres[:, 1] = np.clip(new_centres[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
        
        # Check if valid
        new_r = _compute_radius(new_centres)
        if new_r > 0 and new_r >= r - 1e-10:
            best_centres = new_centres
            if new_r > best_r:
                best_r = new_r
        else:
            # Revert and reduce velocity
            velocity *= 0.5
    
    return best_centres, best_r


def _gradient_ascent(centres, deadline, lr=0.0005):
    """
    Vectorized subgradient ascent on the radius.
    """
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    while time.time() < deadline:
        r = _compute_radius(best_centres)
        if r <= 0:
            break
        
        # Compute pairwise distances
        dists = cdist(best_centres, best_centres)
        np.fill_diagonal(dists, np.inf)
        
        # Compute wall distances
        wall_dists = np.stack([
            0.5 - best_centres[:, 0],
            0.5 + best_centres[:, 0],
            0.5 - best_centres[:, 1],
            0.5 + best_centres[:, 1]
        ], axis=1)
        
        # Compute subgradient (average of gradients of nearly-binding constraints)
        grad = np.zeros_like(best_centres)
        n_binding = 0
        
        # Pairwise constraints
        pair_dists = dists[i_idx, j_idx]
        binding_pair_mask = pair_dists < 2.0 * r * 1.01
        if np.any(binding_pair_mask):
            bi = i_idx[binding_pair_mask]
            bj = j_idx[binding_pair_mask]
            bd = pair_dists[binding_pair_mask]
            
            valid = bd > 1e-10
            if np.any(valid):
                bi = bi[valid]
                bj = bj[valid]
                bd = bd[valid]
                
                dirs = (best_centres[bi] - best_centres[bj]) / bd[:, np.newaxis]
                grad_contrib = dirs / 2.0
                
                np.add.at(grad, bi, grad_contrib)
                np.add.at(grad, bj, -grad_contrib)
                n_binding += len(bi)
        
        # Wall constraints
        # Left wall
        d_left = wall_dists[:, 0]
        binding_left = d_left < r * 1.01
        if np.any(binding_left):
            grad[binding_left, 0] += 1.0
            n_binding += np.sum(binding_left)
        
        # Right wall
        d_right = wall_dists[:, 1]
        binding_right = d_right < r * 1.01
        if np.any(binding_right):
            grad[binding_right, 0] -= 1.0
            n_binding += np.sum(binding_right)
        
        # Bottom wall
        d_bottom = wall_dists[:, 2]
        binding_bottom = d_bottom < r * 1.01
        if np.any(binding_bottom):
            grad[binding_bottom, 1] += 1.0
            n_binding += np.sum(binding_bottom)
        
        # Top wall
        d_top = wall_dists[:, 3]
        binding_top = d_top < r * 1.01
        if np.any(binding_top):
            grad[binding_top, 1] -= 1.0
            n_binding += np.sum(binding_top)
        
        if n_binding > 0:
            grad /= n_binding
        
        # Update
        new_centres = best_centres + lr * grad
        
        # Project into square
        new_centres[:, 0] = np.clip(new_centres[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
        new_centres[:, 1] = np.clip(new_centres[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
        
        new_r = _compute_radius(new_centres)
        if new_r > 0 and new_r >= r - 1e-10:
            best_centres = new_centres
            if new_r > best_r:
                best_r = new_r
        else:
            # Reduce learning rate
            lr *= 0.85
    
    return best_centres, best_r


def _local_optimization(centres, max_iters=300, initial_step=0.01):
    """Local optimization using force-directed method with constraint handling."""
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 5)
    
    # Use force-directed optimization
    best_centres, best_r = _force_directed_optimize(best_centres, deadline, lr=0.002, momentum=0.95)
    
    # Then gradient ascent for fine-tuning
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        best_centres, best_r = _gradient_ascent(best_centres, deadline, lr=0.001)
    
    return best_centres, best_r


def _simulated_annealing(centres, max_iters=1000, t_start=0.01, t_end=1e-6):
    """Simulated annealing optimization - kept as fallback."""
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    current_centres = centres.copy()
    current_r = best_r
    
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 5)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Temperature schedule
        t = t_start * (t_end / t_start) ** (it / max_iters)
        
        # Pick a random point to move
        idx = np.random.randint(N)
        
        # Random perturbation
        angle = np.random.uniform(0, 2 * np.pi)
        magnitude = np.random.uniform(0, t)
        delta = magnitude * np.array([np.cos(angle), np.sin(angle)])
        
        new_centres = current_centres.copy()
        new_centres[idx] += delta
        
        # Check bounds
        if np.any(new_centres < -0.5) or np.any(new_centres > 0.5):
            continue
        
        new_r = _compute_radius(new_centres)
        
        # Acceptance criterion
        if new_r > current_r:
            current_centres = new_centres
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centres = current_centres.copy()
        else:
            # Accept with probability exp((new_r - current_r) / t)
            delta_r = new_r - current_r
            if np.random.random() < np.exp(delta_r / max(t, 1e-12)):
                current_centres = new_centres
                current_r = new_r
    
    return best_centres, best_r


def construct():
    start_time = time.time()
    
    # Start with hexagonal packing as initial configuration
    centres = _hexagonal_packing()
    current_r = _compute_radius(centres)
    
    # Force-directed optimization first (new structural approach)
    if time.time() - start_time < TIME_LIMIT - 20:
        centres, current_r = _local_optimization(centres)
    
    # Simulated annealing for further exploration
    if time.time() - start_time < TIME_LIMIT - 5:
        centres, current_r = _simulated_annealing(centres)
    
    # Final force-directed refinement
    if time.time() - start_time < TIME_LIMIT - 3:
        centres, current_r = _local_optimization(centres, max_iters=50, initial_step=0.005)
    
    # Final check
    final_r = _compute_radius(centres)
    if final_r <= 0:
        # Fallback to a valid grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return centres
