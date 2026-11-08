"""Heilbronn triangles: place 11 points to maximize min triangle area."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize, minimize_scalar

N = 11
_H = np.sqrt(3) / 2.0

# Precompute all C(11,3) = 165 triangle index triples
_combos = list(combinations(range(N), 3))
_idx_i = np.array([c[0] for c in _combos])
_idx_j = np.array([c[1] for c in _combos])
_idx_k = np.array([c[2] for c in _combos])


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the N points.
    pts has shape (N, 2)."""
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _project_to_triangle(x):
    """Project point x (shape (2,)) onto the equilateral triangle with vertices
    (0,0), (1,0), (0.5, sqrt(3)/2)."""
    x = np.asarray(x, dtype=float)
    # Barycentric coordinates: x = a*(0,0) + b*(1,0) + c*(0.5, sqrt(3)/2)
    # b + 0.5*c = x[0]
    # 0.866*c = x[1]  => c = x[1]/0.866
    # a = 1 - b - c
    c = x[1] / _H
    b = x[0] - 0.5 * c
    a = 1.0 - b - c
    if a < 0:
        # Project onto edge b-c (the edge from (1,0) to (0.5, sqrt(3)/2))
        # This edge: points (t, sqrt(3)/2 * (1-t)) for t in [0.5, 1]
        # Actually the edge from (1,0) to (0.5, H)
        # Parametrize: (1 - 0.5*s, H*s) for s in [0,1]
        # Minimize |x - (1-0.5s, H*s)|^2
        # d/ds: -0.5*(x[0] - 1 + 0.5s) + H*(x[1] - H*s) = 0
        # -0.5*x[0] + 0.5 + 0.25s + H*x[1] - H^2*s = 0
        # s*(0.25 - H^2) = 0.5*x[0] - 0.5 - H*x[1]
        # H^2 = 3/4, so 0.25 - 3/4 = -1/2
        # s*(-1/2) = 0.5*x[0] - 0.5 - H*x[1]
        # s = -1*(0.5*x[0] - 0.5 - H*x[1]) = 0.5 + H*x[1] - 0.5*x[0]
        s = 0.5 + _H * x[1] - 0.5 * x[0]
        s = np.clip(s, 0, 1)
        return np.array([1.0 - 0.5 * s, _H * s])
    elif b < 0:
        # Project onto edge a-c (the edge from (0,0) to (0.5, sqrt(3)/2))
        # Parametrize: (0.5*s, H*s) for s in [0,1]
        # d/ds: 0.5*(x[0] - 0.5s) + H*(x[1] - H*s) = 0
        # 0.5*x[0] - 0.25s + H*x[1] - H^2*s = 0
        # s*(0.25 + H^2) = 0.5*x[0] + H*x[1]
        # H^2 = 3/4, so 0.25 + 3/4 = 1
        # s = 0.5*x[0] + H*x[1]
        s = 0.5 * x[0] + _H * x[1]
        s = np.clip(s, 0, 1)
        return np.array([0.5 * s, _H * s])
    elif c < 0:
        # Project onto edge a-b (the edge from (0,0) to (1,0), i.e., y=0)
        return np.array([x[0], 0.0])
    else:
        return x


def _objective_soft_min(x_flat, temperature=0.05):
    """Soft-min surrogate for the min triangle area objective.
    Returns negative of the soft-min (to minimize)."""
    pts = x_flat.reshape(N, 2)
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    # Soft min: -temperature * log(sum(exp(-area/temperature)))
    # Use log-sum-exp trick for numerical stability
    min_val = np.min(areas)
    exps = np.exp(-(areas - min_val) / temperature)
    soft_min = min_val - temperature * np.log(np.sum(exps))
    return -soft_min


def _objective_soft_min_grad(x_flat, temperature=0.05):
    """Gradient of the soft-min objective."""
    pts = x_flat.reshape(N, 2)
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    min_val = np.min(areas)
    exps = np.exp(-(areas - min_val) / temperature)
    weights = exps / np.sum(exps)
    
    # Gradient of -soft_min with respect to areas
    # d(-soft_min)/d(area_k) = weights[k]
    # But we need gradient w.r.t. x_flat
    
    # For each triangle, the area is 0.5 * |cross|
    # d(area)/d(cross) = 0.5 * sign(cross)
    # cross = (pj[0]-pi[0])*(pk[1]-pi[1]) - (pj[1]-pi[1])*(pk[0]-pi[0])
    # This is complex to differentiate directly. Let's use finite differences or a simpler approach.
    
    # Actually, let's just use the gradient of the true min for the active set
    # and blend with the soft weights
    # For simplicity, compute gradient numerically for the soft part
    # Or better: use the fact that the gradient of the soft min is a weighted average of gradients of individual areas
    
    # Let's compute gradient of each area w.r.t. each point coordinate
    # For triangle (i,j,k): area = 0.5 * |cross|
    # cross = (x_j - x_i) x (x_k - x_i) where x is 2D cross product
    # = (x_j0 - x_i0)(x_k1 - x_i1) - (x_j1 - x_i1)(x_k0 - x_i0)
    
    # d(cross)/d(x_i0) = -(x_k1 - x_i1) + (x_j1 - x_i1) = x_j1 - x_k1
    # d(cross)/d(x_i1) = (x_k0 - x_i0) - (x_j0 - x_i0) = x_j0 - x_k0
    # d(cross)/d(x_j0) = (x_k1 - x_i1)
    # d(cross)/d(x_j1) = -(x_k0 - x_i0)
    # d(cross)/d(x_k0) = -(x_j1 - x_i1)
    # d(cross)/d(x_k1) = (x_j0 - x_i0)
    
    sign_cross = np.sign(cross)
    # Handle zero cross
    sign_cross[np.abs(cross) < 1e-12] = 1.0
    
    grad = np.zeros_like(x_flat)
    for t in range(len(_combos)):
        w = weights[t] * 0.5 * sign_cross[t]
        if abs(w) < 1e-15:
            continue
        i, j, k = _combos[t]
        xi = pts[i]
        xj = pts[j]
        xk = pts[k]
        # d(area)/d(x_i)
        grad[2*i] += w * (xj[1] - xk[1])
        grad[2*i+1] += w * (xj[0] - xk[0])
        # d(area)/d(x_j)
        grad[2*j] += w * (xk[1] - xi[1])
        grad[2*j+1] += w * (-(xk[0] - xi[0]))
        # d(area)/d(x_k)
        grad[2*k] += w * (-(xj[1] - xi[1]))
        grad[2*k+1] += w * (xj[0] - xi[0])
    
    # We're minimizing -soft_min, so gradient is negative of the above
    return -grad


def _symmetric_initial_config():
    """Create an initial configuration with some symmetry."""
    pts = np.zeros((N, 2))
    # Place points in a grid-like pattern within the triangle
    # Use barycentric coordinates
    # Row 0: 3 points along bottom
    # Row 1: 3 points
    # Row 2: 3 points
    # Row 3: 2 points
    # Total: 11
    
    # Let's use a more systematic approach
    # Place points at fractional positions
    configs = [
        # Bottom row
        [0.1, 0.01], [0.5, 0.01], [0.9, 0.01],
        # Second row
        [0.25, 0.15], [0.5, 0.15], [0.75, 0.15],
        # Third row
        [0.35, 0.35], [0.5, 0.35], [0.65, 0.35],
        # Top
        [0.45, 0.55], [0.55, 0.55],
    ]
    for i, (x, y) in enumerate(configs):
        pts[i] = _project_to_triangle([x, y])
    return pts


def _nelder_mead_polish(pts, max_iter=500):
    """Polish using Nelder-Mead on the true min-area objective."""
    def obj(x_flat):
        p = x_flat.reshape(N, 2)
        # Project each point to ensure feasibility
        p = np.array([_project_to_triangle(p[i]) for i in range(N)])
        return -_min_triangle_area(p)
    
    x0 = pts.flatten()
    result = minimize(obj, x0, method='Nelder-Mead', 
                      options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
    best_pts = np.array([_project_to_triangle(result.x.reshape(N, 2)[i]) for i in range(N)])
    return best_pts


def construct():
    deadline = time.time() + 130  # Safety margin
    
    # Start with symmetric initial config
    best_pts = _symmetric_initial_config()
    best_score = _min_triangle_area(best_pts)
    
    # Phase 1: L-BFGS-B with soft-min surrogate
    temperature = 0.05
    x0 = best_pts.flatten()
    
    def obj_lbfgs(x_flat):
        return _objective_soft_min(x_flat, temperature)
    
    def grad_lbfgs(x_flat):
        return _objective_soft_min_grad(x_flat, temperature)
    
    try:
        result = minimize(obj_lbfgs, x0, jac=grad_lbfgs, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-8})
        if time.time() < deadline:
            cand_pts = np.array([_project_to_triangle(result.x.reshape(N, 2)[i]) for i in range(N)])
            cand_score = _min_triangle_area(cand_pts)
            if cand_score > best_score:
                best_pts = cand_pts
                best_score = cand_score
    except Exception:
        pass
    
    # Phase 2: Try different temperatures and restarts
    for temp in [0.02, 0.08, 0.15]:
        if time.time() > deadline - 30:
            break
        x0 = best_pts.flatten()
        
        def obj_t(x_flat):
            return _objective_soft_min(x_flat, temp)
        
        def grad_t(x_flat):
            return _objective_soft_min_grad(x_flat, temp)
        
        try:
            result = minimize(obj_t, x0, jac=grad_t, method='L-BFGS-B',
                              options={'maxiter': 100, 'ftol': 1e-8})
            if time.time() < deadline:
                cand_pts = np.array([_project_to_triangle(result.x.reshape(N, 2)[i]) for i in range(N)])
                cand_score = _min_triangle_area(cand_pts)
                if cand_score > best_score:
                    best_pts = cand_pts
                    best_score = cand_score
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead polish on the true objective
    if time.time() < deadline - 10:
        polished = _nelder_mead_polish(best_pts, max_iter=300)
        polished_score = _min_triangle_area(polished)
        if polished_score > best_score:
            best_pts = polished
            best_score = polished_score
    
    # Phase 4: Local search - perturb each point and keep if better
    if time.time() < deadline - 5:
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                for _ in range(5):
                    new_pts = best_pts.copy()
                    new_pts[i] = best_pts[i] + np.random.normal(0, 0.01, 2)
                    new_pts[i] = _project_to_triangle(new_pts[i])
                    new_score = _min_triangle_area(new_pts)
                    if new_score > best_score:
                        best_pts = new_pts
                        best_score = new_score
                        improved = True
                        break
            if not improved:
                break
    
    return best_pts
