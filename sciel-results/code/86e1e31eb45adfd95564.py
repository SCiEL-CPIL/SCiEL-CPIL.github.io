import numpy as np
import time
from scipy.optimize import minimize

N = 200

# EVOLVE-BLOCK-START
def construct():
    """
    Uses Simulated Annealing with local refinement to maximize the minimum
    distance between circles and between circles and the square boundary.
    """
    
    def evaluate(points):
        """
        Returns the minimum radius r for the given configuration.
        r = min(min_pair_dist / 2, min_wall_dist)
        """
        # 1. Distance to walls
        wall_dists = np.minimum.reduce([
            0.5 - points[:, 0],
            0.5 + points[:, 0],
            0.5 - points[:, 1],
            0.5 + points[:, 1]
        ])
        min_wall = np.min(wall_dists)
        
        # 2. Distance between pairs
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        r = min(min_wall, min_pair / 2.0)
        return r

    def get_constrained_points(points, n_constraints=5):
        """
        Find up to n_constraints most violated constraints.
        Returns list of (type, indices, value) tuples.
        """
        constraints = []
        
        # Wall constraints
        wall_dists = np.minimum.reduce([
            0.5 - points[:, 0],
            0.5 + points[:, 0],
            0.5 - points[:, 1],
            0.5 + points[:, 1]
        ])
        wall_order = np.argsort(wall_dists)
        for idx in wall_order[:n_constraints]:
            constraints.append(('wall', idx, wall_dists[idx]))
        
        # Pair constraints
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_order = np.argsort(dists, axis=None)
        seen_pairs = set()
        count = 0
        for flat_idx in pair_order:
            i, j = np.unravel_index(flat_idx, dists.shape)
            if i == j:
                continue
            pair_key = (min(i, j), max(i, j))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            constraints.append(('pair', (i, j), dists[i, j] / 2.0))
            count += 1
            if count >= n_constraints:
                break
        
        return constraints

    def improve(points, temp, n_moves=3):
        """
        Perform a single step of optimization by moving multiple constrained points.
        """
        p = points.copy()
        constraints = get_constrained_points(p, n_constraints=n_moves)
        
        # Adaptive step size
        step_size = 0.01 * temp / 0.05 + 0.003
        
        for ctype, indices, cval in constraints:
            if ctype == 'wall':
                idx = indices
                x, y = p[idx]
                dx_pos = 0.5 - x
                dx_neg = 0.5 + x
                dy_pos = 0.5 - y
                dy_neg = 0.5 + y
                
                min_dist = min(dx_pos, dx_neg, dy_pos, dy_neg)
                
                if min_dist == dx_pos:
                    p[idx, 0] -= step_size
                elif min_dist == dx_neg:
                    p[idx, 0] += step_size
                elif min_dist == dy_pos:
                    p[idx, 1] -= step_size
                else:
                    p[idx, 1] += step_size
                
                p[idx] += np.random.normal(0, temp * 0.3, size=2)
                
            else:
                i, j = indices
                vec = p[j] - p[i]
                norm = np.linalg.norm(vec)
                if norm > 1e-9:
                    vec = vec / norm
                else:
                    vec = np.array([1.0, 0.0])
                
                p[i] -= vec * step_size
                p[j] += vec * step_size
                
                p[i] += np.random.normal(0, temp * 0.3, size=2)
                p[j] += np.random.normal(0, temp * 0.3, size=2)
        
        # Small global perturbation
        p += np.random.normal(0, temp * 0.05, size=p.shape)
        p = np.clip(p, -0.5, 0.5)
        
        return p

    def local_refine(points):
        """
        Use scipy L-BFGS-B to locally optimize the packing.
        Maximize r = min(min_pair_dist/2, min_wall_dist)
        """
        def neg_radius(flat):
            pts = flat.reshape(N, 2)
            r = evaluate(pts)
            return -r
        
        # Try a few random initial perturbations
        best_result = None
        best_val = -1
        
        for attempt in range(3):
            if attempt == 0:
                x0 = points.flatten().copy()
            else:
                x0 = points.flatten() + np.random.normal(0, 0.005, size=N*2)
                x0 = np.clip(x0.reshape(N, 2), -0.49, 0.49).flatten()
            
            try:
                res = minimize(neg_radius, x0, method='L-BFGS-B',
                              bounds=[(-0.5, 0.5)] * (N * 2),
                              options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_result = res.x.reshape(N, 2)
            except:
                continue
        
        if best_result is not None:
            return best_result
        return points

    # Multiple restarts
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    best_points = None
    best_score = -1
    
    n_restarts = 5
    
    for restart in range(n_restarts):
        # Check time
        if time.time() - start_time > time_limit - 30:
            break
        
        # Initialization: Grid with jitter
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = grid[:N].copy()
        points += np.random.normal(0, 0.01 + 0.01 * restart, size=points.shape)
        points = np.clip(points, -0.49, 0.49)
        
        current_points = points.copy()
        current_score = evaluate(current_points)
        
        # SA parameters
        initial_temp = 0.04
        final_temp = 0.002
        n_sa_steps = 3000
        cooling_rate = (final_temp / initial_temp) ** (1.0 / n_sa_steps)
        
        temp = initial_temp
        
        for step in range(n_sa_steps):
            if time.time() - start_time > time_limit - 20:
                break
            
            new_points = improve(current_points, temp, n_moves=3)
            new_score = evaluate(new_points)
            
            if new_score >= current_score:
                current_points = new_points
                current_score = new_score
            else:
                prob = np.exp((new_score - current_score) / (temp + 1e-12))
                if np.random.rand() < prob:
                    current_points = new_points
                    current_score = new_score
            
            temp *= cooling_rate
        
        # Local refinement
        if time.time() - start_time < time_limit - 10:
            refined = local_refine(current_points)
            refined_score = evaluate(refined)
            if refined_score > current_score:
                current_points = refined
                current_score = refined_score
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
    
    return best_points
# EVOLVE-BLOCK-END
