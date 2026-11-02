"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

# Vertices of the unit equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
VERTS = np.array([V0, V1, V2])


def bary_to_cart(bary):
    """Convert barycentric coordinates (n,3) to Cartesian (n,2)."""
    return bary[:, 0:1] * V0 + bary[:, 1:2] * V1 + bary[:, 2:3] * V2


def cart_to_bary(pts):
    """Convert Cartesian (n,2) to barycentric (n,3)."""
    # Solve: p = a*V0 + b*V1 + c*V2, a+b+c=1
    # Using linear algebra: p = V1*b + V2*c (since V0=0, a=1-b-c)
    # p = b*(V1-V0) + c*(V2-V0)
    A = np.array([[V1[0] - V0[0], V2[0] - V0[0]],
                  [V1[1] - V0[1], V2[1] - V0[1]]])
    # A @ [b, c]^T = p - V0
    diff = pts - V0
    bc = np.linalg.solve(A.T, diff.T).T  # (n, 2)
    bary = np.column_stack([1.0 - bc[:, 0] - bc[:, 1], bc[:, 0], bc[:, 1]])
    return bary


def min_triangle_area_bary(bary):
    """Compute the minimum area of any triangle formed by 3 of the N points (in barycentric coords)."""
    n = bary.shape[0]
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area in barycentric: the 2x2 determinant
                # det = bary[i,1]*bary[j,2] + bary[j,1]*bary[k,2] + bary[k,1]*bary[i,2]
                #     - bary[i,2]*bary[j,1] - bary[j,2]*bary[k,1] - bary[k,2]*bary[i,1]
                det = (bary[i, 1] * bary[j, 2] + bary[j, 1] * bary[k, 2] + bary[k, 1] * bary[i, 2]
                       - bary[i, 2] * bary[j, 1] - bary[j, 2] * bary[k, 1] - bary[k, 2] * bary[i, 1])
                area = 0.5 * abs(det)
                if area < min_area:
                    min_area = area
    return min_area


def objective_neg_min_area(x, t):
    """Annealed soft-min objective. x is flat barycentric coords (n*3)."""
    bary = x.reshape(-1, 3)
    # Normalize barycentric to sum to 1 (soft constraint)
    bary_norm = bary / bary.sum(axis=1, keepdims=True)
    
    n = bary_norm.shape[0]
    # Compute all triangle areas
    areas = np.empty((n * (n - 1) * (n - 2)) // 6)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                det = (bary_norm[i, 1] * bary_norm[j, 2] + bary_norm[j, 1] * bary_norm[k, 2] + bary_norm[k, 1] * bary_norm[i, 2]
                       - bary_norm[i, 2] * bary_norm[j, 1] - bary_norm[j, 2] * bary_norm[k, 1] - bary_norm[k, 2] * bary_norm[i, 1])
                areas[idx] = 0.5 * abs(det)
                idx += 1
    
    # Soft min: -log(sum(exp(-areas/t))) / n_triangles approximates min for small t
    # Actually, for maximization of min, we want to minimize -softmin(areas)
    # softmin(areas) = -t * log(sum(exp(-areas/t)))
    # We want to maximize softmin, so minimize -softmin = t * log(sum(exp(-areas/t)))
    # For numerical stability:
    a_min = areas.min()
    shifted = -areas / t
    shifted -= shifted.max()
    log_sum = np.log(np.sum(np.exp(shifted)))
    softmin_val = a_min - t * log_sum  # This is an approximation; let's use the proper formula
    
    # Proper soft min: min_i a_i ≈ -t * log(sum_i exp(-a_i / t))
    # To maximize min, minimize t * log(sum exp(-a_i/t))
    val = t * log_sum + a_min  # Wait, let me redo this
    
    # Actually: -t * log(sum exp(-a_i/t)) is a smooth approximation to min(a_i)
    # So to maximize min(a_i), we maximize -t*log(sum(exp(-a_i/t)))
    # Equivalently, minimize t*log(sum(exp(-a_i/t)))
    # For stability: t * (log_sum + max_shift) where max_shift = max(-a_i/t)
    # Let's just compute: t * log(sum(exp(-a_i/t)))
    # = t * log(sum(exp((-a_i - a_min)/t) * exp(a_min/t)))
    # = t * (a_min/t + log(sum(exp((-a_i - a_min)/t))))
    # = a_min + t * log(sum(exp(-(a_i - a_min)/t)))
    
    result = a_min + t * log_sum
    return -result  # Negate because we're minimizing


def objective_direct(x):
    """Direct objective: negative of minimum triangle area."""
    bary = x.reshape(-1, 3)
    bary_norm = bary / bary.sum(axis=1, keepdims=True)
    min_area = min_triangle_area_bary(bary_norm)
    return -min_area


def make_initial_bary(seed):
    """Create initial barycentric coordinates with some structure."""
    rng = np.random.RandomState(seed)
    bary = np.zeros((N, 3))
    
    # Start with points spread around
    # Include vertices and midpoints as a heuristic starting point
    # Then add interior points
    
    # Use a grid-like approach in barycentric space
    # For 11 points, try a pattern
    # 3 vertices, 3 edge midpoints, 1 center, 4 interior
    bary[0] = [1.0, 0.0, 0.0]  # V0
    bary[1] = [0.0, 1.0, 0.0]  # V1
    bary[2] = [0.0, 0.0, 1.0]  # V2
    bary[3] = [0.5, 0.5, 0.0]  # Mid V0-V1
    bary[4] = [0.5, 0.0, 0.5]  # Mid V0-V2
    bary[5] = [0.0, 0.5, 0.5]  # Mid V1-V2
    bary[6] = [1/3, 1/3, 1/3]  # Center
    
    # 4 more interior points - place them at positions that maximize spacing
    # Use a simple heuristic: place at 1/4 positions
    bary[7] = [0.25, 0.25, 0.5]
    bary[8] = [0.25, 0.5, 0.25]
    bary[9] = [0.5, 0.25, 0.25]
    bary[10] = [0.1, 0.4, 0.5]
    
    # Add small random perturbation
    bary += rng.uniform(-0.01, 0.01, size=bary.shape)
    bary = np.clip(bary, 0.001, None)
    bary = bary / bary.sum(axis=1, keepdims=True)
    
    return bary


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    best_bary = None
    best_score = -np.inf
    
    # Multiple restarts with different seeds and annealing schedules
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    
    for seed_idx, seed in enumerate(seeds):
        if time.time() > deadline - 10.0:
            break
        
        bary = make_initial_bary(seed)
        x0 = bary.flatten()
        
        # Annealed optimization
        t_values = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
        
        for t in t_values:
            if time.time() > deadline - 5.0:
                break
            
            def obj(x):
                return objective_neg_min_area(x, t)
            
            try:
                res = minimize(obj, x0, method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.success or res.fun < best_score:
                    # Check if this is better
                    bary_candidate = res.x.reshape(-1, 3)
                    bary_candidate = np.clip(bary_candidate, 0.0, None)
                    bary_candidate = bary_candidate / bary_candidate.sum(axis=1, keepdims=True)
                    score = min_triangle_area_bary(bary_candidate)
                    if score > best_score:
                        best_score = score
                        best_bary = bary_candidate.copy()
                    x0 = res.x
            except Exception:
                pass
        
        # After annealing, do a final direct optimization
        if time.time() > deadline - 5.0:
            break
        
        try:
            res = minimize(objective_direct, x0, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            bary_candidate = res.x.reshape(-1, 3)
            bary_candidate = np.clip(bary_candidate, 0.0, None)
            bary_candidate = bary_candidate / bary_candidate.sum(axis=1, keepdims=True)
            score = min_triangle_area_bary(bary_candidate)
            if score > best_score:
                best_score = score
                best_bary = bary_candidate.copy()
        except Exception:
            pass
    
    # If no valid result, use the initial heuristic
    if best_bary is None:
        best_bary = make_initial_bary(42)
    
    # Convert to Cartesian
    pts = bary_to_cart(best_bary)
    
    # Ensure points are inside the triangle (clip to boundary)
    # Verify barycentric coords are valid
    bary_final = cart_to_bary(pts)
    bary_final = np.clip(bary_final, 0.0, None)
    bary_final = bary_final / bary_final.sum(axis=1, keepdims=True)
    pts = bary_to_cart(bary_final)
    
    return pts
# EVOLVE-BLOCK-END
