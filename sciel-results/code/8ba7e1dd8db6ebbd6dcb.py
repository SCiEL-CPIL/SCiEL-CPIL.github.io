import numpy as np
import time
from scipy.optimize import minimize

N = 11

# Pre-compute indices for combinations of 3 from 11
# 11 choose 3 = 165
# Using array for vectorized operations
_tri_indices = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def _all_areas(pts):
    """Vectorized computation of all 165 triangle areas."""
    pi = pts[_tri_indices[:, 0]]
    pj = pts[_tri_indices[:, 1]]
    pk = pts[_tri_indices[:, 2]]
    
    # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    
    return 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)

def _min_area(pts):
    """Exact minimum area."""
    return float(np.min(_all_areas(pts)))

def _soft_min_area(pts, beta=50.0):
    """
    Smooth approximation of min area for gradient-based optimization.
    As beta -> infinity, this approaches the true min area.
    """
    areas = _all_areas(pts)
    m = np.min(areas)
    # Shift to avoid overflow/underflow in exp
    # We want to approximate min(a_i)
    # log-sum-exp trick: min(a) = -1/beta * log(sum(exp(-beta * a)))
    # To improve stability, subtract min first: min(a) = m - 1/beta * log(sum(exp(-beta * (a - m))))
    shifted = -beta * (areas - m)
    shifted = np.clip(shifted, -500, 500) # Numerical stability
    log_sum = np.log(np.sum(np.exp(shifted)))
    return m - log_sum / beta

def _clamp_to_triangle(p):
    """
    Project a point onto the interior of the equilateral triangle using barycentric coordinates.
    """
    x, y = p
    # Barycentric coordinates relative to V0, V1, V2
    # p = a*V0 + b*V1 + c*V2
    # y = c * H  => c = 2y / sqrt(3)
    # x = b*1 + c*0.5 => b = x - 0.5c
    # a = 1 - b - c
    
    c = 2 * y / np.sqrt(3)
    b = x - 0.5 * c
    a = 1 - b - c
    
    eps = 1e-9
    # Clamp to [eps, 1-2eps] to ensure strict interior and valid barycentric sum
    a = max(eps, min(1 - 2*eps, a))
    b = max(eps, min(1 - 2*eps, b))
    c = max(eps, min(1 - 2*eps, c))
    
    # Normalize just in case clamping broke the sum (it shouldn't significantly, but good practice)
    s = a + b + c
    a /= s; b /= s; c /= s
    
    return a * V0 + b * V1 + c * V2

def _clamp_all(pts):
    """Clamp all points to be inside the triangle."""
    return np.array([_clamp_to_triangle(p) for p in pts])

def _objective_smooth(flat_pts, beta=50.0):
    """Objective to minimize: negative soft-min area."""
    pts = flat_pts.reshape(-1, 2)
    pts = _clamp_all(pts)
    return -_soft_min_area(pts, beta)

def _objective_hard(flat_pts):
    """Objective to minimize: negative exact min area."""
    pts = flat_pts.reshape(-1, 2)
    pts = _clamp_all(pts)
    return -_min_area(pts)

def _init_structured():
    """
    Structured initialization leveraging symmetry and key geometric points.
    Combines vertices, midpoints, centroid, and a ring.
    """
    pts = []
    # 1. Vertices (pulled in slightly to avoid degenerate areas if other points are close? 
    #    Actually, for max-min area, vertices are usually optimal. 
    #    But we must ensure they are strictly inside for the optimizer? 
    #    The clamping handles boundary. Let's start with vertices.
    pts.append(V0.copy())
    pts.append(V1.copy())
    pts.append(V2.copy())
    
    # 2. Midpoints of edges
    pts.append((V0 + V1) / 2)
    pts.append((V1 + V2) / 2)
    pts.append((V2 + V0) / 2)
    
    # 3. Centroid
    pts.append((V0 + V1 + V2) / 3)
    
    # 4. Ring around centroid (4 points)
    cx, cy = 1/3, H/3
    r = 0.12
    # Angles to avoid alignment with existing points
    for angle in [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]:
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        pts.append([px, py])
        
    return np.array(pts)

def _init_grid():
    """Triangular grid initialization."""
    pts = []
    # 4 rows
    y0 = 0.05
    for i in range(4):
        x = 0.05 + i * (0.9 / 3)
        pts.append([x, y0])
    y1 = 0.30
    for i in range(4):
        x = 0.125 + i * (0.75 / 3)
        pts.append([x, y1])
    y2 = 0.55
    for i in range(3):
        x = 0.20 + i * (0.6 / 2)
        pts.append([x, y2])
    return np.array(pts)

def _init_random(seed):
    """Random initialization inside triangle."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(N):
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        p = r1 * V1 + r2 * V2
        pts.append(p)
    return np.array(pts)

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We aim to finish by 145s.
    deadline = start_time + 145.0
    
    def time_left():
        return deadline - time.time()
    
    best_pts = None
    best_score = -1.0
    
    # Phase 1: Multi-start Smooth Optimization (L-BFGS-B)
    # We use a range of beta values to go from "very smooth" (easy to optimize, bad approximation)
    # to "sharper" (better approximation, harder to optimize).
    
    init_candidates = [
        _init_structured(),
        _init_grid()
    ]
    # Add a few random starts
    for seed in [1, 42, 123]:
        init_candidates.append(_init_random(seed))
    
    # Beta schedule: Start smooth, end sharp
    betas = [20.0, 50.0, 100.0, 200.0]
    
    for init in init_candidates:
        if time_left() < 10.0:
            break
        
        current_pts = init.copy()
        
        for beta in betas:
            if time_left() < 5.0:
                break
            
            try:
                # Flatten points for scipy
                x0 = current_pts.flatten()
                
                # L-BFGS-B is good for this box-constrained (via clamping) problem
                # Note: We don't use bounds directly because clamping is smoother for the gradient
                # than hard bounds in some implementations, but L-BFGS-B handles bounds well too.
                # However, since we clamp inside the objective, the gradient is effectively zero
                # at the boundary (or rather, the projection handles it). 
                # To be safe, we can provide loose bounds.
                bounds = [(0, 1), (0, H)] * N
                
                res = minimize(
                    _objective_smooth, 
                    x0, 
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                opt_pts = _clamp_all(res.x.reshape(-1, 2))
                score = _min_area(opt_pts)
                
                if score > best_score:
                    best_score = score
                    best_pts = opt_pts.copy()
                
                # Update current points for next beta (annealing)
                current_pts = opt_pts
                
            except Exception as e:
                pass # Skip if optimization fails
    
    if best_pts is None:
        # Fallback
        best_pts = _init_structured()
        best_score = _min_area(best_pts)
    
    # Phase 2: Nelder-Mead Polish on Hard Objective
    # Nelder-Mead is derivative-free and can handle the non-differentiability of min() better
    # for final refinement steps.
    
    if time_left() > 15.0:
        # Run NM on the best solution found so far
        try:
            res = minimize(
                _objective_hard,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            nm_pts = _clamp_all(res.x.reshape(-1, 2))
            nm_score = _min_area(nm_pts)
            
            if nm_score > best_score:
                best_score = nm_score
                best_pts = nm_pts.copy()
        except Exception:
            pass
        
        # Try NM on a few perturbed versions of the best solution to escape local minima
        rng = np.random.default_rng(999)
        for _ in range(5):
            if time_left() < 10.0:
                break
            
            # Small perturbation
            perturbed = best_pts + rng.normal(0, 0.01, size=(N, 2))
            perturbed = _clamp_all(perturbed)
            
            try:
                res = minimize(
                    _objective_hard,
                    perturbed.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                nm_pts = _clamp_all(res.x.reshape(-1, 2))
                nm_score = _min_area(nm_pts)
                
                if nm_score > best_score:
                    best_score = nm_score
                    best_pts = nm_pts.copy()
            except Exception:
                pass

    return best_pts

# Test/Verification block (not executed in evaluation)
if __name__ == "__main__":
    pts = construct()
    print(f"Best score: {_min_area(pts)}")
    print(f"Points:\n{pts}")
