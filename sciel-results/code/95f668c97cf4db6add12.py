import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull

def get_score(points):
    """
    Calculate the score: minimum triangle area divided by convex hull area.
    """
    N = len(points)
    if N < 3:
        return 0.0
    
    # Check for degeneracy
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # Vectorized triangle area calculation
    # We need the minimum area over all triples (i, j, k)
    # 13 points -> 286 triangles. 
    # Instead of loops, we can compute all pairwise cross products?
    # Actually, for N=13, a simple loop is fast enough (< 1ms).
    # But let's try to be efficient.
    
    min_area = float('inf')
    # Precompute indices for speed? No, just simple loops are fine for N=13
    for i in range(N):
        pi = points[i]
        for j in range(i + 1, N):
            pj = points[j]
            # Vector from i to j
            v1 = pj - pi
            for k in range(j + 1, N):
                pk = points[k]
                v2 = pk - pi
                # Cross product magnitude in 2D is |x1*y2 - y1*x2|
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    
    return min_area / hull_area

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We should stop at 145s to return safely.
    deadline = start_time + 145.0
    
    N = 13
    
    # Define bounds for the search space.
    # We can normalize the hull to unit area later, but for optimization,
    # it's better to constrain the points to a fixed box, e.g., [0, 1] x [0, 1].
    # The score is scale-invariant (area/area), so the absolute size doesn't matter,
    # only the relative positions.
    bounds = [(0, 1) for _ in range(N * 2)]
    
    def objective(x_flat):
        pts = x_flat.reshape(N, 2)
        # Add a small penalty if points are too close to each other to avoid numerical issues
        # in hull calculation, though ConvexHull usually handles it.
        # Also, if the hull area is very small, the score is ill-defined or 0.
        try:
            # Quick check for duplicate points
            # If any two points are identical, area is 0.
            # Differential Evolution handles this naturally by minimizing the score.
            return -get_score(pts)
        except Exception:
            return 1e6 # Large penalty for invalid states
    
    # Initial population strategy:
    # Differential Evolution creates a random initial population within bounds.
    # We can provide a custom initial population to bias towards "good" structures
    # like a circle, but DE is global. Let's stick to standard DE but with a 
    # decent maxiter and popsize.
    
    # Parameters for Differential Evolution
    # popsize: 15 * N * 2 ? 15*26 = 390 individuals.
    # Each evaluation of an individual takes ~1-2ms.
    # 390 * 2000 iterations = 780,000 evaluations.
    # 780,000 * 1ms = 780s. Too slow.
    
    # We need to be much faster or use fewer iterations.
    # Let's estimate get_score cost more accurately.
    # 286 triangles. Pure Python loops are slow.
    # Let's optimize get_score using NumPy vectorization for the area calculation.
    
    def get_score_fast(points):
        N = len(points)
        if N < 3:
            return 0.0
        
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 0.0
        except Exception:
            return 0.0

        # Vectorized area calculation
        # points shape (N, 2)
        # We want area of triangle (i, j, k)
        # Area = 0.5 * | (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i) |
        
        # Create indices
        i_idx, j_idx, k_idx = np.indices((N, N, N)).reshape(3, -1)
        # Filter valid triangles i < j < k
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        k_idx = k_idx[mask]
        
        pi = points[i_idx]
        pj = points[j_idx]
        pk = points[k_idx]
        
        v1 = pj - pi
        v2 = pk - pi
        
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        return min_area / hull_area

    # Re-define objective with fast score
    def objective_fast(x_flat):
        pts = x_flat.reshape(N, 2)
        try:
            return -get_score_fast(pts)
        except Exception:
            return 1e6

    # Adjust DE parameters for speed
    # popsize=10 (10*26=260 individuals)
    # maxiter=500
    # Total evals ~ 130,000.
    # If get_score_fast takes 0.5ms, total time ~ 65s. This fits.
    
    # We can run multiple DE runs with different random seeds/initializations
    # or just one long run.
    # Given the 145s budget, we can afford ~200,000-300,000 evaluations if fast enough.
    
    best_score = 0.0
    best_points = None
    
    # Initial guess: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    # Shift to [0,1] box
    init_pts = (init_pts - init_pts.min(axis=0)) / (init_pts.max(axis=0) - init_pts.min(axis=0))
    best_points = init_pts.copy()
    best_score = get_score_fast(best_points)
    
    # Run Differential Evolution
    # We will run it in chunks to check time
    # DE is stochastic. We can restart it if it stalls, but DE itself is robust.
    
    try:
        # Use a fixed seed for reproducibility in this instance, 
        # but in a real scenario we might randomize. 
        # The task allows standard library, numpy, scipy.
        
        # To maximize diversity, we can use 'best1bin' or 'current-to-best1bin'
        # Let's use standard 'best1bin'
        
        # We'll run DE for a fixed number of iterations, then check time.
        # If time remains, we can refine the best point found using a local optimizer
        # like Nelder-Mead or L-BFGS-B (with gradient approximation if needed, 
        # but Nelder-Mead is derivative-free and good for refinement).
        
        res = differential_evolution(
            objective_fast, 
            bounds, 
            seed=42,
            maxiter=1000,
            popsize=15,
            tol=1e-6,
            init='latinhypercube', # Better coverage of the space
            polish=False # We will polish manually to save overhead inside DE
        )
        
        if res.fun < -best_score: # res.fun is negative score
            best_score = -res.fun
            best_points = res.x.reshape(N, 2)
            
    except Exception as e:
        pass
    
    # Check time
    if time.time() < deadline - 20:
        # Refine the best point found using Nelder-Mead
        # This helps escape the coarse grid of DE
        
        from scipy.optimize import minimize
        
        def neg_score_flat(x_flat):
            pts = x_flat.reshape(N, 2)
            return -get_score_fast(pts)
            
        x_start = best_points.flatten()
        
        try:
            res_local = minimize(
                neg_score_flat, 
                x_start, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if -res_local.fun > best_score:
                best_score = -res_local.fun
                best_points = res_local.x.reshape(N, 2)
                
        except Exception:
            pass
        
        # If still time, try another DE run with a different seed or perturbation
        if time.time() < deadline - 10:
            try:
                # Perturb best points
                x_start2 = best_points.flatten() + np.random.normal(0, 0.05, N*2)
                # Clip to bounds
                x_start2 = np.clip(x_start2, 0, 1)
                
                res2 = differential_evolution(
                    objective_fast, 
                    bounds, 
                    seed=123,
                    maxiter=500,
                    popsize=10,
                    init='latinhypercube',
                    polish=False
                )
                
                if res2.fun < -best_score:
                    best_score = -res2.fun
                    best_points = res2.x.reshape(N, 2)
                    
                # Final polish
                if time.time() < deadline - 2:
                    res_local2 = minimize(
                        neg_score_fast, 
                        best_points.flatten(), 
                        method='Nelder-Mead', 
                        options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                    )
                    if -res_local2.fun > best_score:
                        best_score = -res_local2.fun
                        best_points = res_local2.x.reshape(N, 2)
                        
            except Exception:
                pass

    return best_points

if __name__ == "__main__":
    # Quick test
    pts = construct()
    print(f"Score: {get_score(pts)}")
    print(f"Points:\n{pts}")
