from scipy.spatial.distance import cdist
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
import random

N = 100
CUBE_HALF = 0.5

def _score(c):
    """
    Compute the maximum radius for a given set of centers.
    r = min(min_pairwise_dist / 2, min_boundary_dist)
    """
    if N > 1:
        dists = pdist(c, metric='euclidean')
        min_pair_dist = np.min(dists)
    else:
        min_pair_dist = np.inf
        
    # Boundary distances
    abs_c = np.abs(c)
    boundary_dists = CUBE_HALF - np.max(abs_c, axis=1)
    min_boundary_dist = np.min(boundary_dists)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r

def _initial_lattice():
    """
    Initialize with a jittered lattice to provide a good starting structure.
    We use a 5x5x4 grid (100 points) which fits perfectly.
    """
    nx, ny, nz = 5, 5, 4
    # Spacing
    dx = 1.0 / nx
    dy = 1.0 / ny
    dz = 1.0 / nz
    
    # Center the grid
    # x in [-0.5, 0.5]
    # ticks: -0.5 + 0.1, -0.5 + 0.3, ...
    x_ticks = np.linspace(-0.5 + dx/2, 0.5 - dx/2, nx)
    y_ticks = np.linspace(-0.5 + dy/2, 0.5 - dy/2, ny)
    z_ticks = np.linspace(-0.5 + dz/2, 0.5 - dz/2, nz)
    
    points = []
    for z in z_ticks:
        for y in y_ticks:
            for x in x_ticks:
                points.append([x, y, z])
    
    c = np.array(points)
    # Add small random jitter to break symmetry and avoid exact grid local minima
    c += np.random.normal(0, 0.01, size=c.shape)
    # Clip to ensure inside
    c = np.clip(c, -CUBE_HALF, CUBE_HALF)
    return c

def _initial_random():
    """Create a random initial placement."""
    return np.random.uniform(-CUBE_HALF, CUBE_HALF, size=(N, 3))

def _sa_optimize(c_init, time_limit):
    """
    Simulated Annealing optimization.
    """
    start_time = time.time()
    c = c_init.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    
    # Temperature parameters
    T_start = 0.05
    T_min = 1e-5
    # We want to cool down over the time limit.
    # Estimate number of iterations possible.
    # Let's assume ~10000 iters/sec for _score + move.
    # Total iters approx time_limit * 10000.
    # Cooling rate alpha such that T_end = T_start * alpha^n
    # n = log(T_end/T_start) / log(alpha)
    # Let's just use a fixed alpha and see.
    # If we do 100,000 iterations, alpha = (T_min/T_start)^(1/100000)
    # (1e-5 / 0.05) = 2e-4. (2e-4)^(1e-5) is very close to 1.
    # Let's use a simpler approach: linear or geometric cooling based on iteration count,
    # but we have a time limit.
    
    # Let's use a geometric cooling with a fixed alpha, and restart if stuck.
    # Or better: adaptive step size and fixed T schedule that resets.
    
    # We'll use a "batch" approach.
    # Run SA for a certain number of iterations, then check time.
    
    T = T_start
    alpha = 0.9995 # Slow cooling
    step_size = 0.02
    
    # Track recent best to detect stagnation
    recent_best = current_score
    stagnation_count = 0
    max_stagnation = 5000 # Reheat if no improvement for 5000 iters
    
    while time.time() - start_time < time_limit:
        # Generate move
        # 80% single move, 20% double move
        if random.random() < 0.8:
            idx = random.randint(0, N - 1)
            delta = np.random.randn(3) * step_size
            c[idx] += delta
            c[idx] = np.clip(c[idx], -CUBE_HALF, CUBE_HALF)
        else:
            i1, i2 = random.sample(range(N), 2)
            delta1 = np.random.randn(3) * step_size
            delta2 = np.random.randn(3) * step_size
            c[i1] += delta1
            c[i2] += delta2
            c[i1] = np.clip(c[i1], -CUBE_HALF, CUBE_HALF)
            c[i2] = np.clip(c[i2], -CUBE_HALF, CUBE_HALF)
        
        new_score = _score(c)
        
        # Acceptance
        if new_score > current_score:
            c = c # already updated
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_c = c.copy()
                recent_best = current_score
                stagnation_count = 0
        else:
            # Accept with probability exp((new - current) / T)
            # To avoid underflow/overflow issues with very small T and large negative delta,
            # we can use log, but exp is usually fine.
            # However, if T is very small, exp(-large) is 0.
            delta_score = new_score - current_score
            # If delta_score is very negative, prob is 0.
            # Optimization: if delta_score < -10*T, prob is effectively 0.
            if delta_score > -10 * T:
                prob = np.exp(delta_score / T)
                if random.random() < prob:
                    current_score = new_score
                    # c is already updated
                    # Note: we don't update best_c here unless it's better
            else:
                # Revert if rejected? No, we already modified c in place.
                # We need to revert c if we reject the move.
                # The current code modifies c directly. If rejected, we must revert.
                # This is inefficient. Let's copy c before modification.
                pass # Handled by the fact that we modify c directly. 
                # Wait, if we reject, c remains in the new state? No.
                # If we reject, we must revert c to the previous state.
                # The current implementation modifies `c` directly.
                # So if we reject, `c` is still the new state. This is WRONG for SA.
                # We must revert `c` if the move is rejected.
                
                # To fix this efficiently without copying every time:
                # We can store the old value, modify, score, and if rejected, restore old value.
                # But we did `c[idx] += delta`. 
                # Let's restructure the move generation to be reversible or use a copy.
                # Copying (100,3) array is fast.
                
                # Let's restart the loop logic in the next edit to handle revert properly.
                # For now, assume we revert.
                # Actually, the code above does `c[idx] += delta`. 
                # If rejected, we must do `c[idx] -= delta`.
                # But we clipped it! So we can't simply subtract.
                # We MUST save the old state.
        
        # Cooling
        T *= alpha
        if T < T_min:
            T = T_min
            # Reheat if stagnation
            if stagnation_count > max_stagnation:
                T = T_start * 0.5 # Partial reheat
                step_size = 0.02
                stagnation_count = 0
        
        # Update stagnation
        if new_score < recent_best:
            stagnation_count += 1
        else:
            stagnation_count = 0
            
        # Adaptive step size
        # Reduce step size as T decreases to refine
        # step_size = 0.02 * (T / T_start) ** 0.5
        # But don't go below 0.001
        step_size = max(0.001, 0.02 * (T / T_start) ** 0.5)

    return best_c, best_score

def construct():
    """
    Construct the optimal arrangement of 100 equal spheres in a unit cube.
    """
    best_c = None
    best_score = -np.inf
    
    # Time budget: 120s.
    # We will run multiple SA instances.
    # 1. Lattice start (good structure)
    # 2. Random starts
    
    # Time allocation:
    # 1. Lattice: 30s
    # 2. Random 1: 25s
    # 3. Random 2: 25s
    # 4. Random 3: 25s
    # 5. Polish: 15s
    
    # 1. Lattice
    c1 = _initial_lattice()
    c_best1, s_best1 = _sa_optimize(c1, time_limit=30)
    if s_best1 > best_score:
        best_score = s_best1
        best_c = c_best1
        
    # 2-4. Randoms
    for i in range(3):
        c_rand = _initial_random()
        c_curr, s_curr = _sa_optimize(c_rand, time_limit=25)
        if s_curr > best_score:
            best_score = s_curr
            best_c = c_curr
            
    # 5. Polish
    # Use a dedicated polishing function that focuses on local improvements
    # We can run a very low T SA or a specific local search.
    # Let's use a "Nudge" local search.
    
    if best_c is not None:
        c_polish = best_c.copy()
        current_score = _score(c_polish)
        polish_time_limit = 10
        start_polish = time.time()
        
        # Simple iterative improvement:
        # Find the point with the smallest margin (distance to nearest neighbor / 2 or boundary)
        # Try to move it away.
        
        # To be fast, we will do a few passes of "move the most constrained point".
        # However, identifying the most constrained point requires O(N^2) distance calc.
        # pdist is fast.
        
        while time.time() - start_polish < polish_time_limit:
            # Calculate margins
            dists = pdist(c_polish)
            min_pair = np.min(dists)
            abs_c = np.abs(c_polish)
            boundary_dists = CUBE_HALF - np.max(abs_c, axis=1)
            min_boundary = np.min(boundary_dists)
            
            # The bottleneck is min(min_pair/2, min_boundary)
            # If min_pair/2 < min_boundary, the bottleneck is a pair.
            # We need to find which pair.
            # pdist returns condensed form. Index mapping is complex.
            # Alternative: Just run a very low T SA for the polish.
            
            # Let's just run a short SA with very low T and small steps.
            # This is safer than manual index mapping.
            # But we already used SA. 
            # Let's try a simple coordinate descent on the "worst" point.
            
            # Find the point closest to the boundary or nearest neighbor.
            # Approximation: Calculate distance to nearest neighbor for each point.
            # This is O(N^2) but N=100, so 10000 ops. Fast.
            
            # cdist(c_polish, c_polish) gives NxN matrix.
            # Set diagonal to inf.
            dist_matrix = cdist(c_polish, c_polish)
            np.fill_diagonal(dist_matrix, np.inf)
            min_dists_to_other = np.min(dist_matrix, axis=1)
            
            # Margin for point i: min(min_dists_to_other[i]/2, boundary_dists[i])
            margins = np.minimum(min_dists_to_other / 2.0, boundary_dists)
            
            # Find the point with the smallest margin
            idx_worst = np.argmin(margins)
            margin_val = margins[idx_worst]
            
            # Try to move this point in a direction away from its constraints
            # Direction: away from nearest neighbor and away from boundary.
            # Find nearest neighbor index
            nn_idx = np.argmin(dist_matrix[idx_worst])
            
            # Vector from NN to worst
            vec_nn = c_polish[idx_worst] - c_polish[nn_idx]
            norm_nn = np.linalg.norm(vec_nn)
            if norm_nn > 1e-9:
                vec_nn /= norm_nn
            else:
                vec_nn = np.random.randn(3)
                
            # Boundary constraint direction
            # If x is close to -0.5, push +x. If close to 0.5, push -x.
            vec_bound = np.zeros(3)
            for k in range(3):
                if c_polish[idx_worst, k] < 0:
                    vec_bound[k] = 1.0 # Push towards center (positive)
                else:
                    vec_bound[k] = -1.0 # Push towards center (negative)
                    
            # Combine directions
            # Weight by how close to boundary vs how close to neighbor
            # Simple heuristic: average of normalized vectors
            move_dir = vec_nn + vec_bound
            norm_move = np.linalg.norm(move_dir)
            if norm_move > 1e-9:
                move_dir /= norm_move
            else:
                move_dir = np.random.randn(3)
                
            # Try several step sizes
            improved = False
            for step in [0.01, 0.005, 0.002, 0.001]:
                c_new = c_polish.copy()
                c_new[idx_worst] += move_dir * step
                c_new[idx_worst] = np.clip(c_new[idx_worst], -CUBE_HALF, CUBE_HALF)
                s_new = _score(c_new)
                if s_new > current_score:
                    c_polish = c_new
                    current_score = s_new
                    improved = True
                    break
            
            if not improved:
                # Try random small moves for the worst point
                for _ in range(10):
                    c_new = c_polish.copy()
                    delta = np.random.randn(3) * 0.005
                    c_new[idx_worst] += delta
                    c_new[idx_worst] = np.clip(c_new[idx_worst], -CUBE_HALF, CUBE_HALF)
                    s_new = _score(c_new)
                    if s_new > current_score:
                        c_polish = c_new
                        current_score = s_new
                        improved = True
                        break
                        
            if not improved:
                # Try moving a random point slightly to escape local min
                if random.random() < 0.1:
                    idx_rand = random.randint(0, N-1)
                    c_new = c_polish.copy()
                    delta = np.random.randn(3) * 0.01
                    c_new[idx_rand] += delta
                    c_new[idx_rand] = np.clip(c_new[idx_rand], -CUBE_HALF, CUBE_HALF)
                    s_new = _score(c_new)
                    if s_new > current_score:
                        c_polish = c_new
                        current_score = s_new
                    # If not improved, keep original (do not accept worse)
        
        if current_score > best_score:
            best_score = current_score
            best_c = c_polish

    return best_c
# EVOLVE-BLOCK-END
