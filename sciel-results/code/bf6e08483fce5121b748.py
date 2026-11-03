import numpy as np
import time

N = 50
DEADLINE_SECONDS = 115  # Safety margin before 120s limit

def _score(points):
    """Compute the maximum valid radius for the given sphere centers."""
    if points.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    if np.any(points > 0.5) or np.any(points < -0.5):
        return 0.0
        
    # Distance between all pairs
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Distance to walls: cube is [-0.5, 0.5]^3
    wall_dists = 0.5 - np.abs(points)
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall)

def _potential_energy(points, power=4, wall_power=2):
    """Compute repulsion potential energy. Lower is better (more spread out)."""
    # Pair repulsion
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, 1.0) # Avoid division by zero, though diagonal is 0
    
    # We want to minimize sum(1/d^power)
    # Using dists_sq for speed: 1/d^power = 1/(dists_sq^(power/2))
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, 1e10) # Large distance for diagonal -> 0 contribution
    pair_energy = np.sum(1.0 / np.power(dists, power))
    
    # Wall repulsion
    wall_dists = 0.5 - np.abs(points)
    # Clip to avoid infinity if a point is exactly on wall (shouldn't happen with clipping)
    wall_dists_clipped = np.clip(wall_dists, 1e-6, None)
    wall_energy = np.sum(1.0 / np.power(wall_dists_clipped, wall_power))
    
    return pair_energy + wall_energy

def _potential_grad(points, power=4, wall_power=2):
    """Gradient of the repulsion potential energy."""
    grad = np.zeros_like(points)
    
    # Pair gradient
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, 1e10)
    
    # d/dx_i (1/d_ij^power) = -power * (x_i - x_j) / d_ij^(power+2)
    # Note: d_ij = |x_i - x_j|
    # We sum over j != i
    # Using broadcasting:
    # factor_ij = -power / dists_ij^(power+2)
    # grad_i += sum_j factor_ij * (x_i - x_j)
    
    # To avoid O(N^2) loops in Python, we can vectorize.
    # dists is (N, N)
    # We need (x_i - x_j) which is diff (N, N, 3)
    # factor is (N, N)
    
    factor = -power * np.power(dists, -(power + 2))
    np.fill_diagonal(factor, 0.0)
    
    # grad[i] = sum_j factor[i, j] * diff[i, j]
    # diff[i, j] = points[i] - points[j]
    grad += np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    # Wall gradient
    wall_dists = 0.5 - np.abs(points)
    wall_dists_clipped = np.clip(wall_dists, 1e-6, None)
    
    # d/dx_ik (1/wall_dist_ik^wall_power) = -wall_power * sign(x_ik) / wall_dist_ik^(wall_power+1)
    signs = np.sign(points)
    # Handle zeros
    signs[signs == 0] = 1.0
    
    wall_factor = -wall_power * np.power(wall_dists_clipped, -(wall_power + 1))
    grad += wall_factor * signs
    
    return grad

def _optimize_repulsion(points, max_time, power=4, wall_power=2):
    """
    Gradient descent on repulsion potential to spread spheres out.
    """
    start_time = time.time()
    current_points = points.copy()
    current_energy = _potential_energy(current_points, power, wall_power)
    best_points = current_points.copy()
    best_score = _score(current_points)
    
    lr = 0.001
    momentum = 0.9
    velocity = np.zeros_like(current_points)
    
    while time.time() - start_time < max_time:
        grad = _potential_grad(current_points, power, wall_power)
        
        # Gradient Descent with Momentum
        velocity = momentum * velocity - lr * grad
        new_points = current_points + velocity
        new_points = np.clip(new_points, -0.5, 0.5)
        
        new_energy = _potential_energy(new_points, power, wall_power)
        
        # Accept if energy decreases, or with small probability if it increases (simple SA-like)
        # For now, just accept if better, else reject. Momentum helps escape shallow minima.
        if new_energy < current_energy:
            current_points = new_points
            current_energy = new_energy
            # Check score occasionally
            if np.random.random() < 0.05:
                s = _score(current_points)
                if s > best_score:
                    best_score = s
                    best_points = current_points.copy()
        else:
            # Reject, but keep velocity? Or reset?
            # Standard momentum keeps velocity.
            pass
            
        # Decay learning rate slowly
        lr *= 0.999
        
        if lr < 1e-6:
            break
            
    return best_points, best_score

def _local_search(points, max_time):
    """
    Local search to maximize the minimum distance.
    Iteratively find the bottleneck (closest pair or wall) and move to alleviate it.
    """
    start_time = time.time()
    current_points = points.copy()
    current_score = _score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    lr = 0.005
    decay = 0.995
    
    while time.time() - start_time < max_time:
        # Calculate all pair distances and wall distances
        diff = current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        wall_dists = 0.5 - np.abs(current_points)
        
        min_pair_val = np.min(dists) / 2.0
        min_wall_val = np.min(wall_dists)
        
        if min_pair_val < min_wall_val:
            # Find the closest pair
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = idx
            # Move them apart
            direction = current_points[i] - current_points[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
                current_points[i] += lr * direction
                current_points[j] -= lr * direction
        else:
            # Find the closest wall constraint
            # Flatten wall_dists to find global min
            flat_idx = np.argmin(wall_dists)
            i = flat_idx // 3
            k = flat_idx % 3
            
            # Move point i away from wall k
            sign = np.sign(current_points[i, k])
            if sign == 0:
                sign = 1.0
            current_points[i, k] += lr * sign
        
        current_points = np.clip(current_points, -0.5, 0.5)
        
        new_score = _score(current_points)
        if new_score > best_score:
            best_score = new_score
            best_points = current_points.copy()
            
        lr *= decay
        if lr < 1e-7:
            break
            
    return best_points, best_score

def _generate_grid_init(n):
    """
    Generate initial configuration from a 4x4x4 grid.
    """
    # 4x4x4 = 64 points
    coords = np.array(np.meshgrid(np.linspace(-0.45, 0.45, 4), 
                                  np.linspace(-0.45, 0.45, 4), 
                                  np.linspace(-0.45, 0.45, 4))).T.reshape(-1, 3)
    
    # Randomly select 50 points
    indices = np.random.choice(len(coords), n, replace=False)
    points = coords[indices]
    
    # Add small noise to break symmetry
    points += np.random.randn(n, 3) * 0.001
    
    return np.clip(points, -0.5, 0.5)

def _generate_bcc_init(n):
    """
    Generate an initial configuration based on a Body-Centered Cubic (BCC) lattice.
    """
    points = []
    a = 1.0 / 2.0  # lattice parameter
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Corner
                points.append([(i - 1) * a, (j - 1) * a, (k - 1) * a])
                # Body center
                points.append([(i - 0.5) * a, (j - 0.5) * a, (k - 0.5) * a])
    
    points = np.array(points)
    
    # We have 54 points, need 50. Remove 4 corner points from the outer shell
    remove_idx = []
    for idx, p in enumerate(points):
        if all(abs(abs(c) - 0.5) < 1e-6 for c in p):
            remove_idx.append(idx)
            if len(remove_idx) == 4:
                break
    
    if len(remove_idx) < 4:
        remaining = [i for i in range(len(points)) if i not in remove_idx]
        extra = np.random.choice(remaining, size=4 - len(remove_idx), replace=False)
        remove_idx.extend(extra)
    
    keep_idx = [i for i in range(len(points)) if i not in remove_idx]
    points = points[keep_idx[:n]]
    
    points -= np.mean(points, axis=0)
    max_val = np.max(np.abs(points))
    if max_val > 0:
        points *= (0.48 / max_val)
    
    points += np.random.randn(n, 3) * 0.003
    
    return np.clip(points, -0.5, 0.5)

def construct():
    """Construct the sphere centers."""
    start_time = time.time()
    best_points = None
    best_score = 0.0
    
    # Strategies: Grid Init + Repulsion + Local Search
    # BCC Init + Repulsion + Local Search
    
    while time.time() - start_time < DEADLINE_SECONDS:
        remaining = DEADLINE_SECONDS - (time.time() - start_time)
        if remaining < 5.0:
            break
        
        # Alternate between initializations
        if np.random.random() < 0.5:
            points = _generate_grid_init(N)
        else:
            points = _generate_bcc_init(N)
        
        # Phase 1: Repulsion Optimization (fast, spreads out)
        # Budget: 60% of remaining time, min 1s
        t_rep = max(1.0, remaining * 0.6)
        opt_points, opt_score = _optimize_repulsion(points, t_rep)
        
        # Phase 2: Local Search (slow, tightens)
        # Budget: 40% of remaining time, min 1s
        t_local = max(1.0, remaining * 0.4)
        opt_points, opt_score = _local_search(opt_points, t_local)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points.copy()
            
    if best_points is None:
        best_points = _generate_grid_init(N)
        
    return best_points

if __name__ == "__main__":
    centers = construct()
    print(f"Score: {_score(centers)}")
    print(f"Shape: {centers.shape}")
