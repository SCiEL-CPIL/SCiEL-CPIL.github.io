import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scipy.spatial import Delaunay

N = 50

def compute_radius(positions):
    """
    Compute the maximum common radius for given circle centers.
    positions: array of shape (N, 2)
    """
    # Boundary constraint: distance to nearest wall
    # Square is [-0.5, 0.5] x [-0.5, 0.5]
    # Distance from x_i to wall is 0.5 - |x_i|
    min_bound = 0.5 - np.max(np.abs(positions))
    if min_bound <= 0:
        return 0.0
    
    # Pairwise distance constraint
    # Using cdist for speed
    dists = cdist(positions, positions)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    return min(min_bound, min_dist / 2.0)

def fast_radius_approx(positions):
    """
    Fast approximation of radius using Delaunay triangulation.
    The minimum edge length in the Delaunay triangulation is a lower bound
    on the minimum pairwise distance (and often the exact minimum).
    """
    min_bound = 0.5 - np.max(np.abs(positions))
    if min_bound <= 0:
        return 0.0
    
    try:
        tri = Delaunay(positions)
        edges = tri.simplices
        # Compute lengths of all edges in the triangulation
        edge_lengths = []
        for simplex in edges:
            for i in range(3):
                for j in range(i+1, 3):
                    d = np.linalg.norm(positions[simplex[i]] - positions[simplex[j]])
                    edge_lengths.append(d)
        min_edge = np.min(edge_lengths)
        return min(min_bound, min_edge / 2.0)
    except Exception:
        # Fallback to exact if Delaunay fails (e.g., collinear points)
        return compute_radius(positions)

def soft_min_radius(x, temperature=0.01):
    """
    Smooth approximation of the min radius using log-sum-exp.
    Maximizing the soft-min is equivalent to maximizing the worst-case constraint.
    """
    pos = x.reshape(N, 2)
    
    # Boundary terms: 0.5 - |x_i| and 0.5 - |y_i|
    bound_terms = np.concatenate([
        0.5 - np.abs(pos[:, 0]),
        0.5 - np.abs(pos[:, 1])
    ])
    
    # Pairwise distance terms / 2
    # Only upper triangle to avoid double counting and save memory/time
    dists = cdist(pos, pos)
    np.fill_diagonal(dists, np.inf)
    iu = np.triu_indices(N, k=1)
    pair_terms = dists[iu] / 2.0
    
    # Combine all constraints
    # We want to maximize the minimum of these values.
    # Soft min: -T * log(sum(exp(-val/T)))
    all_terms = np.concatenate([bound_terms, pair_terms])
    
    # Stability check
    if np.any(all_terms < 0):
        # If any term is negative, the radius is effectively 0 (invalid)
        # We return a large negative number to penalize this heavily
        return -10.0

    neg_terms = -all_terms / temperature
    # logsumexp handles stability
    soft_min = -temperature * logsumexp(neg_terms)
    
    return soft_min

def nelder_mead_optimize(positions, time_limit=5.0, seed=0):
    """
    Optimize positions using Nelder-Mead on the negative radius.
    This is a derivative-free method that can handle the non-smooth
    min-function better than gradient-based methods in some cases.
    """
    start_time = time.time()
    best_pos = positions.copy()
    best_r = compute_radius(positions)
    
    # Create a custom objective that uses fast radius approximation
    # but verifies with exact radius occasionally
    call_count = [0]
    
    def objective(x):
        pos = x.reshape(N, 2)
        # Check bounds
        if np.any(np.abs(pos) > 0.5 + 1e-6):
            return 10.0  # Large penalty
        
        # Use fast approximation for speed
        r = fast_radius_approx(pos)
        
        # Every 10th call, verify with exact radius
        call_count[0] += 1
        if call_count[0] % 10 == 0:
            r = compute_radius(pos)
        
        return -r
    
    # Nelder-Mead parameters
    # initial_simplex_size controls the size of the initial simplex
    # We want it small enough to not jump too far, but large enough to explore
    initial_simplex_size = 0.01
    
    try:
        result = minimize(
            objective, 
            best_pos.flatten(), 
            method='Nelder-Mead',
            options={
                'xatol': 1e-7,
                'fatol': 1e-7,
                'maxiter': 5000,
                'maxfev': 10000,
                'initial_simplex_size': initial_simplex_size,
                'adaptive': True
            }
        )
        
        new_pos = result.x.reshape(N, 2)
        if np.all(np.abs(new_pos) <= 0.5):
            new_r = compute_radius(new_pos)
            if new_r > best_r:
                best_r = new_r
                best_pos = new_pos.copy()
                
    except Exception as e:
        pass
    
    return best_pos, best_r

def repulsion_init(n_iter=100, seed=0):
    """
    Generate an initial configuration using a simple repulsion force model.
    This helps escape the degenerate local minima of pure random starts.
    """
    np.random.seed(seed)
    pos = np.random.uniform(-0.4, 0.4, (N, 2))
    
    # Simple iterative repulsion
    # Force: sum of (pos_j - pos_i) / ||pos_j - pos_i||^2 * strength
    # Boundary Force: push away from walls
    strength = 0.01
    
    for _ in range(n_iter):
        forces = np.zeros_like(pos)
        
        # Pairwise repulsion
        # Vectorized pairwise differences
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :] # (N, N, 2)
        dist_sq = np.sum(diff**2, axis=2) + 1e-9 # (N, N)
        
        # 1/d^2 term (inverse square repulsion)
        # We want force direction (pos_i - pos_j) * strength / d^2
        # Note: diff is pos_i - pos_j? No, pos[i] - pos[j].
        # Force on i due to j should be along (pos_i - pos_j).
        inv_dist_sq = np.zeros_like(dist_sq)
        mask = dist_sq > 1e-6
        inv_dist_sq[mask] = 1.0 / dist_sq[mask]
        
        # Multiply diff by inv_dist_sq
        # diff is (N, N, 2). inv_dist_sq is (N, N, 1)
        weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
        
        # Sum over neighbors (axis 1)
        repulsion = np.sum(weighted_diff, axis=1)
        
        # Boundary attraction/repulsion
        # Push away from walls if close
        # Wall at -0.5: force is +1 if x < -0.5 + margin
        # Simple spring towards center? No, that clusters.
        # We want them to fill the space.
        # Repulsion from walls:
        # If x < -0.4, force += (1.0)
        # If x > 0.4, force -= (1.0)
        
        margin = 0.45
        wall_force = np.zeros_like(pos)
        
        # Left wall
        wall_force[pos[:, 0] < -margin, 0] += 1.0
        # Right wall
        wall_force[pos[:, 0] > margin, 0] -= 1.0
        # Bottom wall
        wall_force[pos[:, 1] < -margin, 1] += 1.0
        # Top wall
        wall_force[pos[:, 1] > margin, 1] -= 1.0
        
        forces = repulsion * strength + wall_force * 0.01
        
        pos += forces
        pos = np.clip(pos, -0.49, 0.49)
        
    return pos

def polish(positions, time_limit=5.0):
    """
    Local optimization using scipy minimize with annealing temperature.
    """
    start_time = time.time()
    positions = positions.copy()
    current_r = compute_radius(positions)
    
    best_pos = positions.copy()
    best_r = current_r
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Annealing schedule for temperature
    # Start with T=0.02 (smooth), end with T=0.001 (sharp)
    temps = [0.02, 0.01, 0.005, 0.002, 0.001]
    
    for T in temps:
        if time.time() - start_time > time_limit:
            break
            
        def objective(x):
            if time.time() - start_time > time_limit:
                return 0
            pos = x.reshape(N, 2)
            if np.any(np.abs(pos) > 0.5):
                return 0
            return -soft_min_radius(x, temperature=T)
        
        try:
            result = minimize(objective, best_pos.flatten(), method='L-BFGS-B', 
                              bounds=bounds, options={'maxiter': 100, 'maxls': 30, 'ftol': 1e-8})
            new_pos = result.x.reshape(N, 2)
            if np.all(np.abs(new_pos) <= 0.5):
                new_r = compute_radius(new_pos)
                if new_r > best_r:
                    best_r = new_r
                    best_pos = new_pos.copy()
        except Exception:
            pass
            
    # Final precise polish with very low temperature
    remaining = time_limit - (time.time() - start_time)
    if remaining > 1.0:
        def final_objective(x):
            if time.time() - start_time > time_limit:
                return 0
            pos = x.reshape(N, 2)
            if np.any(np.abs(pos) > 0.5):
                return 0
            return -soft_min_radius(x, temperature=0.0005)
        
        try:
            result = minimize(final_objective, best_pos.flatten(), method='L-BFGS-B', 
                              bounds=bounds, options={'maxiter': 200, 'maxls': 50, 'ftol': 1e-10})
            new_pos = result.x.reshape(N, 2)
            if np.all(np.abs(new_pos) <= 0.5):
                new_r = compute_radius(new_pos)
                if new_r > best_r:
                    best_r = new_r
                    best_pos = new_pos.copy()
        except Exception:
            pass

    return best_pos

def hybrid_polish(positions, time_limit=10.0, seed=0):
    """
    Hybrid polishing: alternate between L-BFGS-B (gradient) and Nelder-Mead (derivative-free).
    This combines the strengths of both: gradient for fast convergence, NM for escaping
    local minima in the non-smooth landscape.
    """
    start_time = time.time()
    best_pos = positions.copy()
    best_r = compute_radius(positions)
    
    # Phase 1: Gradient-based polish
    t1 = min(4.0, time_limit * 0.4)
    best_pos = polish(best_pos, time_limit=t1)
    best_r = compute_radius(best_pos)
    
    # Phase 2: Nelder-Mead exploration
    remaining = time_limit - (time.time() - start_time)
    if remaining > 2.0:
        nm_pos, nm_r = nelder_mead_optimize(best_pos, time_limit=remaining * 0.6, seed=seed)
        if nm_r > best_r:
            best_r = nm_r
            best_pos = nm_pos.copy()
    
    # Phase 3: Final gradient polish
    remaining = time_limit - (time.time() - start_time)
    if remaining > 1.0:
        best_pos = polish(best_pos, time_limit=remaining)
        best_r = compute_radius(best_pos)
    
    return best_pos

def generate_grid_init():
    """Generate a hexagonal grid initialization as a strong baseline."""
    positions = []
    sqrt3 = np.sqrt(3.0)
    
    best_pos = None
    best_r = 0
    
    # Try a few grid variations
    for n_rows in range(6, 9):
        n_cols = int(np.ceil(N / n_rows))
        
        s_x = 1.0 / (n_cols + 0.5)
        s_y = s_x * sqrt3 / 2.0
        
        total_height = (n_rows - 1) * s_y + s_x
        if total_height > 1.0:
            scale = 1.0 / total_height
            s_x *= scale
            s_y *= scale
        
        pos_list = []
        for row in range(n_rows):
            y = -0.5 + s_x/2 + row * s_y
            if y > 0.5 - s_x/2:
                continue
            n_in_row = n_cols if row % 2 == 0 else n_cols - 1
            if row % 2 == 1:
                x_start = -0.5 + s_x
            else:
                x_start = -0.5 + s_x/2
            
            for col in range(n_in_row):
                x = x_start + col * s_x
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    pos_list.append([x, y])
                if len(pos_list) >= N:
                    break
            if len(pos_list) >= N:
                break
        
        if len(pos_list) >= N:
            pos_arr = np.array(pos_list[:N])
            r = compute_radius(pos_arr)
            if r > best_r:
                best_r = r
                best_pos = pos_arr.copy()
    
    if best_pos is not None:
        return best_pos
    return repulsion_init(50, seed=1)

def push_apart_repair(pos, target_r, max_iters=100, tol=1e-10):
    """
    Iteratively push overlapping circles apart until no overlaps remain
    or target radius is achieved. This is a local repair step.
    """
    pos = pos.copy()
    n = len(pos)
    
    for _ in range(max_iters):
        # Find all pairs that are too close
        dists = cdist(pos, pos)
        np.fill_diagonal(dists, np.inf)
        
        # Find pairs with distance < 2 * target_r
        iu = np.triu_indices(n, k=1)
        close_pairs = np.where(dists[iu] < 2 * target_r)[0]
        
        if len(close_pairs) == 0:
            break
        
        # Compute forces to push apart
        forces = np.zeros_like(pos)
        for idx in close_pairs:
            i, j = iu[0][idx], iu[1][idx]
            d = dists[i, j]
            if d < 1e-10:
                # Random direction
                forces[i] += np.random.randn(2) * 0.001
                forces[j] -= np.random.randn(2) * 0.001
            else:
                direction = (pos[i] - pos[j]) / d
                # Push apart by half the overlap
                overlap = 2 * target_r - d
                forces[i] += direction * overlap * 0.5
                forces[j] -= direction * overlap * 0.5
        
        # Apply forces with adaptive step size
        step_size = min(0.01, np.max(np.abs(forces)) * 0.5 + 1e-6)
        pos += forces * step_size
        
        # Keep within bounds
        pos = np.clip(pos, -0.49, 0.49)
    
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_pos = None
    best_r = 0.0
    
    # 1. Start with Grid - use hybrid polish
    pos = generate_grid_init()
    pos = hybrid_polish(pos, time_limit=min(25.0, deadline - time.time()), seed=0)
    r = compute_radius(pos)
    if r > best_r:
        best_r = r
        best_pos = pos.copy()
    
    # 2. Try several Repulsion-Initialized Random Starts
    # Each start gets a specific time budget
    n_starts = 4
    for i in range(n_starts):
        if time.time() > deadline - 10.0: # Keep 10s for final polish
            break
            
        seed = 100 + i
        pos_init = repulsion_init(150, seed=seed)
        
        # Use hybrid polish for each start
        current_time = time.time()
        remaining_starts = n_starts - i
        alloc_time = min(20.0, (deadline - 10.0 - current_time) / remaining_starts)
        
        if alloc_time < 3.0:
            break
            
        pos_polished = hybrid_polish(pos_init, time_limit=alloc_time, seed=seed)
        r = compute_radius(pos_polished)
        
        if r > best_r:
            best_r = r
            best_pos = pos_polished.copy()
            
    # 3. Final Polish on the Best Result with repair
    if time.time() < deadline - 2.0:
        remaining = deadline - 2.0 - time.time()
        if remaining > 5.0:
            # Do a final high-precision polish using hybrid
            best_pos = hybrid_polish(best_pos, time_limit=remaining * 0.7, seed=42)
            best_r = compute_radius(best_pos)
            
            # Apply push_apart_repair to squeeze out more radius
            if time.time() < deadline - 1.0:
                target_r = best_r * 1.001
                repaired = push_apart_repair(best_pos, target_r, max_iters=50)
                r_repaired = compute_radius(repaired)
                if r_repaired > best_r:
                    best_r = r_repaired
                    best_pos = repaired.copy()
            
            # Final gradient polish
            if time.time() < deadline - 0.5:
                best_pos = polish(best_pos, time_limit=min(3.0, deadline - 0.5 - time.time()))
                best_r = compute_radius(best_pos)
                
    # Ensure valid output
    if best_pos is None:
        best_pos = generate_grid_init()
        
    best_pos = np.clip(best_pos, -0.5, 0.5)
    
    return best_pos

if __name__ == "__main__":
    # Quick self-test
    t0 = time.time()
    res = construct()
    t1 = time.time()
    r = compute_radius(res)
    print(f"Time: {t1-t0:.2f}s, Radius: {r:.10f}")
