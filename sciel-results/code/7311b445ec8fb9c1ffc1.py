import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

def _validate(x, y, r, tol=1e-7):
    """Check if the packing is valid (no overlaps, within bounds, positive radii)."""
    if np.any(r < -tol):
        return False
    # Boundary
    if np.any(r > x + tol) or np.any(r > 1 - x + tol) or np.any(r > y + tol) or np.any(r > 1 - y + tol):
        return False
    if np.any(x < -tol) or np.any(x > 1 + tol) or np.any(y < -tol) or np.any(y > 1 + tol):
        return False
    # Overlaps
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if d < r[i] + r[j] - tol:
                return False
    return True

def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0)."""
    return np.logaddexp(0, z / eps) * eps

def _get_contact_edges(circles):
    """Get contact edges from Delaunay triangulation of centers."""
    centers = circles[:, :2]
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for k in range(3):
                a, b = simplex[k], simplex[(k + 1) % 3]
                edges.add((min(a, b), max(a, b)))
        return list(edges)
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit (evaluator kills at 150s, leave 10s buffer)
    
    # 1. Generate initial layouts
    def get_initial_layout_hex():
        # Hexagonal grid
        r0 = 0.12
        centers = []
        for row in range(10):
            for col in range(10):
                x = col * 2 * r0
                y = row * np.sqrt(3) * r0
                if row % 2 == 1:
                    x += r0
                centers.append((x, y))
        
        centers.sort(key=lambda p: (p[0]-0.5)**2 + (p[1]-0.5)**2)
        centers = centers[:N]
        
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y
        
        scale = min(0.8 / width, 0.8 / height)
        
        scaled_centers = []
        for x, y in centers:
            sx = (x - min_x) * scale + 0.1
            sy = (y - min_y) * scale + 0.1
            scaled_centers.append((sx, sy))
            
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((scaled_centers[i][0]-scaled_centers[j][0])**2 + (scaled_centers[i][1]-scaled_centers[j][1])**2)
                if d < min_dist:
                    min_dist = d
        min_boundary = np.inf
        for x, y in scaled_centers:
            min_boundary = min(min_boundary, x, 1-x, y, 1-y)
            
        r_init = min(min_dist / 2, min_boundary) * 0.95
        
        v = []
        for x, y in scaled_centers:
            v.extend([x, y, r_init])
        return np.array(v)

    def get_initial_layout_grid():
        side = int(np.ceil(np.sqrt(N)))
        positions = []
        r = 0.5 / side * 0.9
        for i in range(side):
            for j in range(side):
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                positions.append([x, y, r])
                if len(positions) >= N:
                    break
            if len(positions) >= N:
                break
        while len(positions) < N:
            positions.append([0.5, 0.5, r * 0.5])
        return np.array(positions[:N]).flatten()

    def get_initial_layout_corner():
        rng = np.random.default_rng(42)
        positions = []
        corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92],
                   [0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5]]
        for cx, cy in corners:
            positions.append([cx, cy, 0.08])
        for _ in range(N - len(positions)):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            positions.append([x, y, rng.uniform(0.06, 0.10)])
        return np.array(positions[:N]).flatten()

    # Objective function with adaptive penalty and optional contact graph
    def objective(params, penalty_weight, smooth_eps, contact_edges=None):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]

        penalty = 0.0

        # Boundary violations (smooth)
        lx = r - x
        ux = x + r - 1.0
        ly = r - y
        uy = y + r - 1.0

        penalty += np.sum(_softplus(lx, smooth_eps) ** 2)
        penalty += np.sum(_softplus(ux, smooth_eps) ** 2)
        penalty += np.sum(_softplus(ly, smooth_eps) ** 2)
        penalty += np.sum(_softplus(uy, smooth_eps) ** 2)

        # Negative radius
        penalty += np.sum(_softplus(-r, smooth_eps) ** 2)

        # Overlaps
        if contact_edges is not None and len(contact_edges) > 0:
            for (i, j) in contact_edges:
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                overlap = r[i] + r[j] - dist
                penalty += _softplus(overlap, smooth_eps) ** 2
        else:
            # Full pairwise
            n = len(r)
            idx_i, idx_j = np.triu_indices(n, k=1)
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist_sq = dx * dx + dy * dy
            dist = np.sqrt(np.maximum(dist_sq, 1e-12))
            overlap = r[idx_i] + r[idx_j] - dist
            penalty += np.sum(_softplus(overlap, smooth_eps) ** 2)

        return -np.sum(r) + penalty_weight * penalty

    # Gradient for objective (approximate for L-BFGS-B, or use finite diff if needed, but L-BFGS-B can do finite diff)
    # To speed up, we can provide gradient. But softplus makes it complex. 
    # Let's stick to finite difference for robustness with softplus, or implement a simple gradient.
    # Given the budget, L-BFGS-B with finite diff on 78 variables might be okay if iterations are low.
    # However, the current program uses a custom gradient. Let's try to adapt the gradient for the new objective.
    
    def gradient(params, penalty_weight, smooth_eps, contact_edges=None):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        grad = np.zeros_like(params)
        
        # Gradient of -sum(r)
        grad[2::3] -= 1.0
        
        # Gradient of penalties
        # Helper for softplus derivative: d/dz logaddexp(0, z/eps)*eps = eps * sigmoid(z/eps) = eps * exp(-z/eps) / (1 + exp(-z/eps))
        # Or simply: if z > 0, approx 1; if z < 0, approx 0. 
        # Let's use the sigmoid form for smoothness.
        
        def softplus_deriv(z, eps):
            return eps / (1 + np.exp(-z / eps))
        
        # Boundary
        # lx = r - x
        # ux = x + r - 1
        # ly = r - y
        # uy = y + r - 1
        
        # dP/dx_i:
        # From lx: 2 * softplus(lx) * d(softplus(lx))/dx_i * (-1)
        # From ux: 2 * softplus(ux) * d(softplus(ux))/dx_i * (1)
        
        for i in range(N):
            # x
            lx = r[i] - x[i]
            ux = x[i] + r[i] - 1.0
            d_lx = softplus_deriv(lx, smooth_eps)
            d_ux = softplus_deriv(ux, smooth_eps)
            
            grad[i*3] += -2 * penalty_weight * _softplus(lx, smooth_eps) * d_lx * (-1)
            grad[i*3] += 2 * penalty_weight * _softplus(ux, smooth_eps) * d_ux * (1)
            
            # y
            ly = r[i] - y[i]
            uy = y[i] + r[i] - 1.0
            d_ly = softplus_deriv(ly, smooth_eps)
            d_uy = softplus_deriv(uy, smooth_eps)
            
            grad[i*3+1] += -2 * penalty_weight * _softplus(ly, smooth_eps) * d_ly * (-1)
            grad[i*3+1] += 2 * penalty_weight * _softplus(uy, smooth_eps) * d_uy * (1)
            
            # r
            grad[i*3+2] += 2 * penalty_weight * _softplus(lx, smooth_eps) * d_lx * (1)
            grad[i*3+2] += 2 * penalty_weight * _softplus(ux, smooth_eps) * d_ux * (1)
            grad[i*3+2] += 2 * penalty_weight * _softplus(ly, smooth_eps) * d_ly * (1)
            grad[i*3+2] += 2 * penalty_weight * _softplus(uy, smooth_eps) * d_uy * (1)
            
            # Negative radius: softplus(-r)
            nr = -r[i]
            d_nr = softplus_deriv(nr, smooth_eps)
            grad[i*3+2] += 2 * penalty_weight * _softplus(nr, smooth_eps) * d_nr * (-1)
        
        # Overlaps
        if contact_edges is not None and len(contact_edges) > 0:
            edges = contact_edges
        else:
            edges = [(i, j) for i in range(N) for j in range(i+1, N)]
            
        for i, j in edges:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(np.maximum(dist_sq, 1e-12))
            overlap = r[i] + r[j] - dist
            
            if overlap > -smooth_eps * 10: # Only compute gradient if close to constraint
                d_overlap = softplus_deriv(overlap, smooth_eps)
                
                # d(overlap)/dx_i = -dx/dist
                # d(overlap)/dx_j = dx/dist
                # d(overlap)/dy_i = -dy/dist
                # d(overlap)/dy_j = dy/dist
                # d(overlap)/dr_i = 1
                # d(overlap)/dr_j = 1
                
                factor = 2 * penalty_weight * _softplus(overlap, smooth_eps) * d_overlap
                
                grad[i*3] += factor * (-dx/dist)
                grad[j*3] += factor * (dx/dist)
                grad[i*3+1] += factor * (-dy/dist)
                grad[j*3+1] += factor * (dy/dist)
                grad[i*3+2] += factor * 1
                grad[j*3+2] += factor * 1
        
        return grad

    bounds = [(0, 1), (0, 1), (0, 0.5)] * N
    
    best_score = -1
    best_v = None
    
    # Initial layouts to try
    initial_layouts = [
        get_initial_layout_hex(),
        get_initial_layout_grid(),
        get_initial_layout_corner()
    ]
    
    # Multi-start with adaptive penalty
    for init_idx, v_init in enumerate(initial_layouts):
        if time.time() > deadline:
            break
            
        v_current = v_init.copy()
        
        # Phase 1: Low penalty, focus on layout
        penalty = 5.0
        smooth_eps = 1e-3
        contact_edges = None # Use full pairwise for robustness initially
        
        try:
            res = minimize(objective, v_current, jac=lambda p: gradient(p, penalty, smooth_eps, contact_edges), 
                          method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < objective(v_current, penalty, smooth_eps, contact_edges):
                v_current = res.x
        except Exception:
            pass
        
        if time.time() > deadline:
            break
            
        # Phase 2: Medium penalty, use contact graph
        penalty = 50.0
        smooth_eps = 1e-4
        circles = v_current.reshape(N, 3)
        contact_edges = _get_contact_edges(circles)
        
        try:
            res = minimize(objective, v_current, jac=lambda p: gradient(p, penalty, smooth_eps, contact_edges), 
                          method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < objective(v_current, penalty, smooth_eps, contact_edges):
                v_current = res.x
        except Exception:
            pass
        
        if time.time() > deadline:
            break
            
        # Phase 3: High penalty, refine
        penalty = 500.0
        smooth_eps = 1e-5
        circles = v_current.reshape(N, 3)
        contact_edges = _get_contact_edges(circles)
        
        try:
            res = minimize(objective, v_current, jac=lambda p: gradient(p, penalty, smooth_eps, contact_edges), 
                          method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < objective(v_current, penalty, smooth_eps, contact_edges):
                v_current = res.x
        except Exception:
            pass
        
        # Check validity and update best
        x = v_current[0::3]
        y = v_current[1::3]
        r = v_current[2::3]
        
        if _validate(x, y, r):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_v = v_current.copy()
        
        # If time remains, do a few random perturbations from the best found so far
        if best_v is not None:
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                v_perturbed = best_v + np.random.normal(0, 0.02, size=3*N)
                v_perturbed[0::3] = np.clip(v_perturbed[0::3], 0, 1)
                v_perturbed[1::3] = np.clip(v_perturbed[1::3], 0, 1)
                v_perturbed[2::3] = np.clip(v_perturbed[2::3], 0, 0.5)
                
                penalty = 500.0
                smooth_eps = 1e-5
                circles = v_perturbed.reshape(N, 3)
                contact_edges = _get_contact_edges(circles)
                
                try:
                    res = minimize(objective, v_perturbed, jac=lambda p: gradient(p, penalty, smooth_eps, contact_edges), 
                                  method='L-BFGS-B', bounds=bounds, 
                                  options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
                    if res.fun < objective(v_perturbed, penalty, smooth_eps, contact_edges):
                        v_perturbed = res.x
                except Exception:
                    pass
                
                x = v_perturbed[0::3]
                y = v_perturbed[1::3]
                r = v_perturbed[2::3]
                
                if _validate(x, y, r):
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_v = v_perturbed.copy()

    if best_v is None:
        # Fallback
        return get_initial_layout_hex().reshape(N, 3)
    
    return best_v.reshape(N, 3)
