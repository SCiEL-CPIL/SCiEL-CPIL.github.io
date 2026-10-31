import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    def score_points(pts):
        """
        Calculate the squared ratio of max to min pairwise distance.
        """
        d = pdist(pts, 'euclidean')
        min_d = np.min(d)
        if min_d < 1e-8:
            return 1e10
        max_d = np.max(d)
        return (max_d / min_d) ** 2

    def prepare_params_from_pair(pts, i_min, j_min):
        """
        Reduces dimensionality by fixing points i_min and j_min to (0,0) and (1,0).
        Returns (params, permutation) or (None, None) if invalid.
        """
        p0 = pts[i_min]
        p1 = pts[j_min]
        
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-8:
            return None, None

        # Rotation matrix to align v with x-axis
        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])

        # Transform all points
        transformed = (pts - p0) @ R_mat.T / norm_v
        
        # Reorder so i_min is 0 and j_min is 1
        perm = list(range(N))
        perm[0] = i_min
        perm[1] = j_min
        remaining = [i for i in range(N) if i != i_min and i != j_min]
        perm[2:] = remaining
        
        transformed = transformed[perm]
        
        # transformed[0] should be [0,0]
        # transformed[1] should be [1,0]
        params = transformed[2:].flatten()
        return params, perm

    def objective(params):
        """
        Objective function for optimization.
        Reconstructs points from params (p0=0, p1=1, rest=params) and computes score.
        """
        if time.time() > deadline:
            return 1e10
            
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        return score_points(pts)

    def soft_objective(params, beta=50.0):
        """
        Smooth surrogate for the objective function to allow gradient-based optimization
        of the min/max terms.
        """
        if time.time() > deadline:
            return 1e10
            
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        # Vectorized distance calculation
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        d_sq = np.maximum(d_sq, 1e-12)
        
        # Soft max using log-sum-exp for numerical stability
        max_d_sq = np.max(d_sq)
        soft_max_d_sq = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
        
        # Soft min using log-sum-exp for numerical stability
        min_d_sq = np.min(d_sq)
        soft_min_d_sq = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))
        
        if soft_min_d_sq < 1e-8:
            return 1e10
            
        return soft_max_d_sq / soft_min_d_sq

    def generate_hex_seed():
        """
        Generate a hexagonal lattice subset of 16 points.
        """
        R = 5
        hex_pts = []
        for q in range(-R, R + 1):
            for r in range(-R, R + 1):
                x = q + 0.5 * r
                y = (np.sqrt(3) / 2) * r
                hex_pts.append((x, y))
        hex_pts = np.array(hex_pts)
        
        # Select 16 points closest to the origin to form a compact cluster
        dists = np.sum(hex_pts**2, axis=1)
        idx = np.argsort(dists)[:16]
        return hex_pts[idx]

    def generate_circular_seed():
        """
        Generate points on concentric circles.
        """
        # Inner ring: 6 points
        angles_inner = np.linspace(0, 2*np.pi, 6, endpoint=False)
        r_inner = 0.5
        inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
        
        # Outer ring: 10 points
        angles_outer = np.linspace(0, 2*np.pi, 10, endpoint=False) + np.pi/10
        r_outer = 1.5
        outer_pts = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
        
        return np.vstack([inner_pts, outer_pts])

    def generate_optimized_hex():
        """
        Generate an optimized hexagonal arrangement.
        """
        # Start with hex seed and do a quick optimization
        hex_seed = generate_hex_seed()
        
        # Find closest pair
        d = pdist(hex_seed, 'euclidean')
        iu = np.triu_indices(N, 1)
        min_idx = np.argmin(d)
        i_min, j_min = iu[0][min_idx], iu[1][min_idx]
        
        params, perm = prepare_params_from_pair(hex_seed, i_min, j_min)
        if params is None:
            return hex_seed
        
        # Quick Powell optimization
        try:
            res = minimize(objective, params, method='Powell', 
                           options={'maxiter': 100, 'maxfev': 1000, 'disp': False})
            if res.fun < objective(params):
                pts = np.zeros((N, 2))
                pts[0] = [0.0, 0.0]
                pts[1] = [1.0, 0.0]
                pts[2:] = res.x.reshape(-1, 2)
                inv_perm = np.argsort(perm)
                return pts[inv_perm]
        except Exception:
            pass
        
        return hex_seed

    def generate_other_seeds():
        """Generate alternative initial configurations."""
        seeds = []
        
        # Square Grid 4x4
        grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        seeds.append(grid)
        
        # Two concentric rings
        angles_inner = np.linspace(0, 2*np.pi, 8, endpoint=False)
        angles_outer = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
        r_inner = 0.5
        r_outer = 1.5
        inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
        outer_pts = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
        ring_pts = np.vstack([inner_pts, outer_pts])
        seeds.append(ring_pts)
        
        # Random Uniform
        rng = np.random.default_rng(42)
        rand_pts = rng.uniform(-1, 1, size=(16, 2))
        seeds.append(rand_pts)
        
        # Circular arrangement
        seeds.append(generate_circular_seed())
        
        # Optimized hex
        seeds.append(generate_optimized_hex())
        
        return seeds

    best_score = 1e10
    best_params = None
    best_perm = None

    # 1. Primary Seed: Hexagonal Lattice
    hex_seed = generate_hex_seed()
    
    # 2. Secondary Seeds
    other_seeds = generate_other_seeds()
    
    all_seeds = [hex_seed] + other_seeds
    
    # Add perturbed versions of the hex seed to escape potential local minima
    rng = np.random.default_rng(123)
    for _ in range(5):
        perturbed = hex_seed + rng.normal(0, 0.05, size=(16, 2))
        all_seeds.append(perturbed)

    # We will try optimizing from multiple "anchor" pairs for each seed.
    # Instead of just the global min pair, we try the top 5 closest pairs.
    
    for seed_pts in all_seeds:
        if time.time() > deadline - 20:
            break
            
        # Find the 5 closest pairs in the seed to use as anchors
        d = pdist(seed_pts, 'euclidean')
        iu = np.triu_indices(N, 1)
        i_idx, j_idx = iu[0], iu[1]
        sorted_pairs = np.argsort(d)[:5]
        
        for pair_idx in sorted_pairs:
            if time.time() > deadline - 15:
                break
                
            i_min = i_idx[pair_idx]
            j_min = j_idx[pair_idx]
            
            params, perm = prepare_params_from_pair(seed_pts, i_min, j_min)
            if params is None:
                continue
                
            # Initial score
            curr_score = objective(params)
            if curr_score < best_score:
                best_score = curr_score
                best_params = params.copy()
                best_perm = perm.copy()
                
            # Optimization Phase 1: Powell (Derivative-free, good for non-smooth start)
            try:
                res = minimize(objective, params, method='Powell', 
                               options={'maxiter': 300, 'maxfev': 3000, 'disp': False})
                
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x.copy()
                    best_perm = perm.copy()
                    
            except Exception:
                pass

            # Optimization Phase 2: L-BFGS-B on Soft Objective (Gradient-based polish)
            # Use Beta Annealing: Start with lower beta, then increase
            if time.time() < deadline - 10:
                try:
                    # Pass 1: Lower beta for smoother landscape
                    res2 = minimize(soft_objective, res.x, args=(15.0,), method='L-BFGS-B',
                                    options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
                    
                    # Pass 2: Higher beta for sharper approximation
                    res3 = minimize(soft_objective, res2.x, args=(50.0,), method='L-BFGS-B',
                                    options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False})
                    
                    # Pass 3: Even higher beta for near-exact gradient
                    if time.time() < deadline - 8:
                        res4 = minimize(soft_objective, res3.x, args=(150.0,), method='L-BFGS-B',
                                        options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12, 'disp': False})
                        res3 = res4
                    
                    # Check exact score of the result
                    exact_score = objective(res3.x)
                    if exact_score < best_score:
                        best_score = exact_score
                        best_params = res3.x.copy()
                        best_perm = perm.copy()
                        
                except Exception:
                    pass

    # Final Polish: If we have time, do one more very tight L-BFGS-B pass on the best found
    if best_params is not None and time.time() < deadline - 5:
        try:
            # Use a very high beta for near-exact gradient
            def final_soft_objective(params, beta=300.0):
                return soft_objective(params, beta)
                
            res5 = minimize(final_soft_objective, best_params, method='L-BFGS-B',
                            options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-14, 'disp': False})
            
            exact_score = objective(res5.x)
            if exact_score < best_score:
                best_score = exact_score
                best_params = res5.x.copy()
                best_perm = best_perm.copy()
        except Exception:
            pass

    # Reconstruct final points from best_params and best_perm
    if best_params is not None:
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = best_params.reshape(-1, 2)
        
        # Reverse permutation
        inv_perm = np.argsort(best_perm)
        final_pts = pts[inv_perm]
        
        # Center the points for numerical stability (optional, but good practice)
        final_pts -= final_pts.mean(axis=0)
        
        return final_pts
    else:
        # Fallback: return hex seed centered
        pts = generate_hex_seed()
        pts -= pts.mean(axis=0)
        return pts
