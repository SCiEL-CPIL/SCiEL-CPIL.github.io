import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools

N = 14

def compute_score(points):
    """
    Compute min triangle area / convex hull area.
    Points: (N, 2) array.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
        
    # Indices for all combinations of 3 points
    # N=14, 364 triangles.
    # Pre-computed indices for speed
    i_idx, j_idx, k_idx = np.indices((n, n, n)).reshape(3, -1)
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
        
    return min_area / hull_area

def neg_score_full(points):
    return -compute_score(points)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    # Pre-compute triangle indices for speed
    tri_indices = np.array(list(itertools.combinations(range(N), 3)))
    i_idx = tri_indices[:, 0]
    j_idx = tri_indices[:, 1]
    k_idx = tri_indices[:, 2]

    def get_triangles(points):
        p_i = points[i_idx]
        p_j = points[j_idx]
        p_k = points[k_idx]
        v1 = p_j - p_i
        v2 = p_k - p_i
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return 0.5 * np.abs(cross)

    def get_hull_area(points):
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return 1e-12

    def objective(x):
        # x contains coords for P2 to P13
        p2_13 = x.reshape((N-2, 2))
        p0 = np.array([0.0, 0.0])
        p1 = np.array([1.0, 0.0])
        points = np.vstack((p0, p1, p2_13))
        
        # Check for degenerate points (very close together)
        # If any two points are too close, penalty
        # This is O(N^2)
        # But 14 points is small.
        
        try:
            hull_area = get_hull_area(points)
            if hull_area < 1e-10:
                return 1.0 # High cost for invalid
            areas = get_triangles(points)
            min_area = np.min(areas)
            if min_area < 1e-10:
                return 1.0
            return -min_area / hull_area
        except Exception:
            return 1.0

    def objective_soft(x, beta=50.0):
        # Soft-min objective for smoother gradient
        p2_13 = x.reshape((N-2, 2))
        p0 = np.array([0.0, 0.0])
        p1 = np.array([1.0, 0.0])
        points = np.vstack((p0, p1, p2_13))
        
        try:
            hull_area = get_hull_area(points)
            if hull_area < 1e-10:
                return 1.0
            areas = get_triangles(points)
            # Log-sum-exp for soft min
            # We want to maximize min(areas), so minimize -min(areas)
            # soft_min(a) = -1/beta * log(sum(exp(-beta * a)))
            # We want to minimize this.
            # But areas can be large, so scale?
            # Let's just use np.min for now in the hard objective,
            # and use this for gradient-based refinement.
            
            # Actually, for L-BFGS, we need a differentiable objective.
            # Let's use a smooth approximation of the minimum.
            # min(a) approx -1/beta * log(sum(exp(-beta * a)))
            # To avoid overflow, shift by max
            a_shifted = -areas + np.max(areas)
            soft_min_val = -np.max(areas) - (1.0/beta) * np.log(np.sum(np.exp(-beta * a_shifted)))
            
            return -soft_min_val / hull_area
        except Exception:
            return 1.0

    # Strategy:
    # 1. Global search with Differential Evolution to find a good basin.
    # 2. Local refinement with L-BFGS-B on a smooth objective to escape flat minima.
    # 3. Final polish with hard objective if needed.

    bounds = [(-10.0, 10.0)] * (2 * (N - 2))
    
    best_x = None
    best_score = -float('inf')

    # Phase 1: Differential Evolution
    # Use a smaller population and fewer iterations to leave time for refinement
    try:
        # We want to finish DE in ~60s
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=100,
            popsize=10,
            tol=1e-8,
            mutation=(0.5, 1.5),
            recombination=0.9,
            polish=False, # We will do our own polish
            updating='deferred'
        )
        
        if res.fun < best_score:
            best_score = res.fun
            best_x = res.x
            
    except Exception:
        pass

    if best_x is None:
        # Fallback: Regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        p_reg = np.column_stack((np.cos(angles), np.sin(angles)))
        return p_reg

    # Phase 2: Local Refinement with L-BFGS-B on smooth objective
    # This exploits differentiability to escape local minima that trap gradient-free methods
    if time.time() - start_time < deadline - 20:
        try:
            # Use a moderate beta for smoothness
            res_lbfgs = minimize(
                objective_soft, 
                best_x, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res_lbfgs.fun < best_score:
                best_score = res_lbfgs.fun
                best_x = res_lbfgs.x
        except Exception:
            pass

    # Phase 3: Final Polish with Hard Objective (Nelder-Mead or Powell)
    # Nelder-Mead doesn't use bounds, but we can project or just hope it stays in bounds
    # Powell is safer with bounds? No, Powell doesn't support bounds in older scipy.
    # Let's use L-BFGS-B again with the hard objective? No, it's not differentiable.
    # Let's use Nelder-Mead with a smaller step size, or just accept the L-BFGS result.
    # Actually, let's try one more L-BFGS-B with a higher beta (sharper min)
    if time.time() - start_time < deadline - 10:
        try:
            res_lbfgs2 = minimize(
                lambda x: objective_soft(x, beta=200.0), 
                best_x, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res_lbfgs2.fun < best_score:
                best_score = res_lbfgs2.fun
                best_x = res_lbfgs2.x
        except Exception:
            pass

    # Reconstruct points
    p2_13 = best_x.reshape((N-2, 2))
    points = np.vstack((np.array([0.0, 0.0]), np.array([1.0, 0.0]), p2_13))
    
    return points
