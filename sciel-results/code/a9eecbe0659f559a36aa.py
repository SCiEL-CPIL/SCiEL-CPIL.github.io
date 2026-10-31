import numpy as np
from scipy.optimize import minimize, basinhopping
import time
from itertools import combinations

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute upper triangle indices
    iu = np.triu_indices(N, 1)
    
    def get_sq_dists(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        return sq_dists[iu]

    def exact_score(pts):
        if len(pts) < 2:
            return 1e10
        sq_dists = get_sq_dists(pts)
        min_sq = np.min(sq_dists)
        max_sq = np.max(sq_dists)
        if min_sq < 1e-12:
            return 1e10
        return max_sq / min_sq

    def soft_score_sq(pts, beta=100.0):
        sq_dists = get_sq_dists(pts)
        sq_dists = np.maximum(sq_dists, 1e-12)
        
        max_sq = np.max(sq_dists)
        exp_terms_max = np.exp(beta * (sq_dists - max_sq))
        sum_exp_max = np.sum(exp_terms_max)
        soft_max_sq = max_sq + (1.0 / beta) * np.log(sum_exp_max)
        
        min_sq = np.min(sq_dists)
        exp_terms_min = np.exp(-beta * (sq_dists - min_sq))
        sum_exp_min = np.sum(exp_terms_min)
        soft_min_sq = min_sq - (1.0 / beta) * np.log(sum_exp_min)
            
        if soft_min_sq < 1e-12:
            return 1e10
            
        return soft_max_sq / soft_min_sq

    def normalize(pts):
        pts = pts - np.mean(pts, axis=0)
        rms = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
        if rms > 1e-9:
            pts = pts / rms
        return pts

    # 1. Generate the 19-point hexagonal lattice
    # Center
    hex_pts = [(0.0, 0.0)]
    # Ring 1
    for i in range(6):
        angle = i * np.pi / 3
        hex_pts.append((np.cos(angle), np.sin(angle) * np.sqrt(3))) # Scaled to match lattice geometry? 
        # Standard hex lattice vectors: (1,0), (0.5, sqrt(3)/2)
        # Let's use proper lattice coords
    # Let's regenerate properly
    lattice_pts = []
    # Basis vectors
    u = np.array([1.0, 0.0])
    v = np.array([0.5, np.sqrt(3)/2.0])
    
    # Generate points in a hexagonal shape of radius 2
    # i, j from -2 to 2 such that |i|+|j|+|i-j| <= 4? No, just filter by distance
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * u + j * v
            dist_sq = np.dot(p, p)
            if dist_sq <= 4.5: # Radius 2 squared is 4. Add margin
                candidates.append(p)
    
    # Sort by distance to origin to get the compact sets
    candidates.sort(key=lambda p: np.dot(p, p))
    # Take the first 19 (should be the 1+6+12 set)
    # The 19 points are those with lattice distance <= 2
    # Lattice distance squared for (i,j) is i^2 + j^2 + i*j
    # For i=2,j=0: 4. i=1,j=1: 3. i=2,j=-1: 3.
    # So all points with i^2+j^2+ij <= 4 are included.
    # Count:
    # 0: (0,0) -> 1
    # 1: (1,0),(0,1),(-1,0),(0,-1),(1,-1),(-1,1) -> 6
    # 3: (2,0),(1,1),(0,2),(-2,0),(-1,-1),(0,-2),(2,-1),(1,-2),(-2,1),(-1,2),(2,1),(-2,-1) -> 12
    # Total 19.
    
    hex_set = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            d_sq = i*i + j*j + i*j
            if d_sq <= 4:
                p = i * u + j * v
                hex_set.append(p)
                
    hex_set = np.array(hex_set)
    # Ensure we have 19 points
    if len(hex_set) != 19:
        # Fallback to manual construction if logic fails
        # But the logic above is standard.
        pass
        
    # 2. Evaluate all subsets of size 16 from these 19 points
    best_lattice_score = 1e10
    best_lattice_pts = None
    
    indices = range(19)
    # We remove 3 points from 19
    for removed in combinations(indices, 3):
        removed_set = set(removed)
        subset_indices = [i for i in indices if i not in removed_set]
        subset_pts = hex_set[subset_indices]
        
        # Quick score
        score = exact_score(subset_pts)
        if score < best_lattice_score:
            best_lattice_score = score
            best_lattice_pts = subset_pts.copy()
            
    # Normalize the best lattice points
    best_lattice_pts = normalize(best_lattice_pts)
    
    best_score = best_lattice_score
    best_params = best_lattice_pts.copy()
    
    # 3. Local Optimization from the best lattice seed
    # Use Nelder-Mead for fine-tuning
    if time.time() < deadline - 20:
        try:
            def exact_obj(params):
                pts = params.reshape(N, 2)
                return exact_score(pts)
            
            # Nelder-Mead is good for low-dimensional (32 vars) non-smooth problems
            res = minimize(
                exact_obj, 
                best_params.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.reshape(N, 2)
        except Exception:
            pass

    # 4. Basin Hopping for global escape
    # Use soft objective for basin hopping as it's smoother
    if time.time() < deadline - 30:
        try:
            def soft_obj(params):
                pts = params.reshape(N, 2)
                return soft_score_sq(pts, beta=50.0)
            
            def step(params):
                # Random step
                rng = np.random.default_rng()
                return params + rng.normal(0, 0.1, size=params.shape)
                
            bh = basinhopping(
                soft_obj, 
                x0=best_params.flatten(), 
                stepfunc=step,
                minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 100}},
                niter=50,
                T=1000.0,
                stepsize=0.5,
                interval=10,
                disp=False
            )
            
            # Check the exact score of the best found by basin hopping
            bh_pts = bh.x.reshape(N, 2)
            bh_score = exact_score(bh_pts)
            
            if bh_score < best_score:
                best_score = bh_score
                best_params = bh_pts.copy()
                
                # Polish the basin hopping result
                if time.time() < deadline - 10:
                    res = minimize(
                        exact_obj, 
                        best_params.flatten(), 
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                    )
                    if res.fun < best_score:
                        best_score = res.fun
                        best_params = res.x.reshape(N, 2)
        except Exception:
            pass

    # 5. Final check and normalization
    final_pts = best_params.reshape(N, 2)
    
    # Ensure distinct points
    dists_sq = get_sq_dists(final_pts)
    if np.min(dists_sq) < 1e-10:
        final_pts += np.random.uniform(-1e-3, 1e-3, final_pts.shape)
        final_pts = normalize(final_pts)
        
    return final_pts
