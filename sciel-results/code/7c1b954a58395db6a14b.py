import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580

    def score(pts):
        """Compute (d_min / d_max)^2 for the point configuration."""
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (n, d) or not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def smooth_objective(x, beta=10.0):
        """
        Smooth approximation of - (d_min / d_max)^2 using log-sum-exp.
        d_min ~ -1/beta * log(sum(exp(-beta * d)))
        d_max ~  1/beta * log(sum(exp( beta * d)))
        """
        pts = x.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        
        dists = pdist(pts)
        if len(dists) == 0 or np.any(~np.isfinite(dists)):
            return 1e6

        # Numerical stability for log-sum-exp
        min_term = -beta * dists
        min_max_val = np.max(min_term)
        if min_max_val > 700:
             min_log_sum_exp = min_max_val + np.log(np.sum(np.exp(min_term - min_max_val)))
        else:
             min_log_sum_exp = np.log(np.sum(np.exp(min_term)))
        d_min_approx = -min_log_sum_exp / beta
        
        max_term = beta * dists
        max_max_val = np.max(max_term)
        if max_max_val > 700:
             max_log_sum_exp = max_max_val + np.log(np.sum(np.exp(max_term - max_max_val)))
        else:
             max_log_sum_exp = np.log(np.sum(np.exp(max_term)))
        d_max_approx = max_log_sum_exp / beta
        
        if d_max_approx < 1e-12:
            return 1e6
            
        ratio = d_min_approx / d_max_approx
        return -ratio ** 2

    def exact_objective(x):
        """Exact objective for local polishing."""
        pts = x.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        dists = pdist(pts)
        if len(dists) == 0:
            return 1e6
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 1e6
        return -(d_min / d_max) ** 2

    # Generate the tetrahedral structure: 4 vertices + 6 edge midpoints
    # The 4th point (index 13) is not fixed; it will be optimized.
    # Actually, let's just use this as a good seed. The prompt suggests 
    # "14 points... 4 vertices + 6 midpoints + 4 face centers"? No, 4+6=10. 
    # 14-10=4. Maybe 4 face centers? That makes a 14-point structure (the 1-simplex 
    # barycentric subdivision has 2^3 vertices + ... wait.
    # A tetrahedron has 4 vertices, 6 edges, 4 faces.
    # Subdividing each face into 4 gives 4*4=16 face centers? No.
    # Let's just use the seed: 4 vertices of a regular tetrahedron.
    # Then 6 points on the edges.
    # And then 4 more points? Maybe face centers?
    # Let's construct a specific candidate for seed.
    
    def get_seed_tetra():
        # Regular tetrahedron vertices (scaled to unit circumradius for stability, then scale doesn't matter)
        # Vertices of a regular tetrahedron centered at origin:
        # (1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)
        V = np.array([
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1]
        ], dtype=float)
        V = V / np.linalg.norm(V[0]) # normalize to radius 1
        
        # Midpoints of edges
        edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        M = np.array([(V[i] + V[j]) / 2.0 for i, j in edges])
        
        # Face centers (centroids of triangles)
        faces = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
        F = np.array([(V[i] + V[j] + V[k]) / 3.0 for i, j, k in faces])
        
        # Total points: 4 (V) + 6 (M) + 4 (F) = 14
        pts = np.vstack([V, M, F])
        pts = pts - np.mean(pts, axis=0)
        return pts

    # Initial candidate from this highly symmetric structure
    seed_pts = get_seed_tetra()
    current_best_pts = seed_pts.copy()
    current_best_score = score(current_best_pts)

    best_score = current_best_score
    best_points = current_best_pts

    # 1. Local Optimization from the symmetric seed
    # We expect this structure to be near-optimal or at least in a good basin.
    # Use Powell for the non-smooth objective directly on the symmetric structure.
    # Since the structure has symmetry, breaking it might help.
    
    if time.time() < deadline - 100:
        # Multi-scale Powell optimization
        x0 = current_best_pts.flatten()
        
        # First, try to refine the symmetric structure directly
        try:
            res = minimize(
                exact_objective,
                x0,
                method='Powell',
                options={'maxiter': 500, 'ftol': 1e-14, 'xtol': 1e-14}
            )
            cand = res.x.reshape((n, d))
            cand = cand - np.mean(cand, axis=0)
            s = score(cand)
            if s > best_score:
                best_score = s
                best_points = cand.copy()
        except Exception:
            pass
        
        # Now, perturb and refine to break symmetry if beneficial
        rng = np.random.RandomState(123)
        for perturb_scale in [0.1, 0.5, 1.0, 2.0]:
            if time.time() > deadline - 100:
                break
            x0_pert = best_points.flatten() + rng.randn(n * d) * perturb_scale
            try:
                res = minimize(
                    exact_objective,
                    x0_pert,
                    method='Powell',
                    options={'maxiter': 1000, 'ftol': 1e-14, 'xtol': 1e-14}
                )
                cand = res.x.reshape((n, d))
                cand = cand - np.mean(cand, axis=0)
                s = score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
            except Exception:
                pass

    # 2. Differential Evolution for global search (in case symmetric structure is not optimal)
    # Use the symmetric structure as part of the population? DE doesn't directly support this,
    # but we can run DE and compare.
    # Let's run a shorter DE to see if we can beat the local optimum from the seed.
    
    if time.time() < deadline - 200:
        bounds = [(-5, 5)] * (n * d)
        try:
            # Use smooth objective for DE to make it faster and more stable
            de_result = differential_evolution(
                smooth_objective,
                bounds,
                args=(50.0,), # High beta for accuracy
                maxiter=300,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=True,
                updating='immediate',
                init='sobol'
            )
            cand = de_result.x.reshape((n, d))
            cand = cand - np.mean(cand, axis=0)
            s = score(cand)
            if s > best_score:
                best_score = s
                best_points = cand.copy()
        except Exception:
            pass

    # 3. Final high-precision polishing
    if time.time() < deadline - 30:
        x0 = best_points.flatten()
        
        # Nelder-Mead with tight tolerance
        try:
            res = minimize(
                exact_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'fatol': 1e-15, 'xatol': 1e-15, 'adaptive': True}
            )
            cand = res.x.reshape((n, d))
            cand = cand - np.mean(cand, axis=0)
            s = score(cand)
            if s > best_score:
                best_score = s
                best_points = cand.copy()
        except Exception:
            pass
        
        # One more Powell pass from a tiny perturbation
        try:
            if time.time() < deadline - 20:
                x0_pert = best_points.flatten() + np.random.RandomState(1234).randn(n * d) * 1e-6
                res = minimize(
                    exact_objective,
                    x0_pert,
                    method='Powell',
                    options={'maxiter': 3000, 'ftol': 1e-15, 'xtol': 1e-15}
                )
                cand = res.x.reshape((n, d))
                cand = cand - np.mean(cand, axis=0)
                s = score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
        except Exception:
            pass

    # Ensure output is valid
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = seed_pts

    return best_points

BET: {mechanism: "Seeding optimization with a highly symmetric tetrahedral structure (4 vertices + 6 edge midpoints + 4 face centers) provides a geometrically informed starting point that is likely closer to the global optimum than random or Fibonacci sphere seeds, enabling local optimizers to converge to a higher score by avoiding distant local minima.", q: 0.7, delta: [0.5, 5.0]}
