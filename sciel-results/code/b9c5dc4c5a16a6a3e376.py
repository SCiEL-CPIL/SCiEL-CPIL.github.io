"""Circle packing in perimeter-4 rectangle: pebble growth + SLSQP polish."""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 21
START = time.time()
DEADLINE = START + 140.0


def _time_ok():
    return time.time() < DEADLINE - 5.0


def _pebble_grow(cx, cy, W, H, max_iter=500):
    """Grow circles from initial positions, maintaining non-overlap.
    
    At each step, compute the maximum radius each circle can have given
    current positions and other radii, then grow all radii proportionally.
    Also allow small position adjustments to relieve pressure.
    """
    n = len(cx)
    cx = cx.copy().astype(float)
    cy = cy.copy().astype(float)
    r = np.full(n, 1e-6)
    
    for it in range(max_iter):
        if not _time_ok():
            break
        
        # Compute max allowed radius for each circle
        max_r = np.full(n, np.inf)
        # Wall constraints
        for i in range(n):
            max_r[i] = min(cx[i], W - cx[i], cy[i], H - cy[i])
        
        # Pairwise constraints: r_i + r_j <= dist_ij
        # We want to grow r_i as much as possible given current r_j
        # r_i <= dist_ij - r_j for all j
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                max_r[i] = min(max_r[i], d - r[j])
        
        max_r = np.maximum(max_r, 1e-8)
        
        # Grow: set r_i to a fraction of max_r_i, but also keep relative growth
        # Use a relaxation: r_new = max(r_old, 0.95 * max_r)
        # Actually, let's just set r_i = max_r_i * 0.999 to stay safe
        # But we want all circles to grow together. Let's use:
        # r_i_new = min(max_r_i, r_i * (1 + growth)) where growth adapts
        
        # Better approach: find the bottleneck
        # r_i can be at most max_r_i
        # We want to maximize sum, so set r_i = max_r_i for all i
        # But this might cause issues if two circles are close
        # Let's just set r_i = max_r_i * 0.999
        r_new = max_r * 0.999
        r_new = np.maximum(r_new, r * 0.999)  # don't shrink
        r_new = np.minimum(r_new, max_r)
        
        # Check if we can improve positions
        # Compute forces to push circles apart
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.hypot(dx, dy)
                if d < 1e-12:
                    continue
                min_d = r_new[i] + r_new[j]
                if d < min_d * 1.01:
                    # Slight overlap or close
                    overlap = min_d - d
                    if overlap > 0:
                        nx = dx / d
                        ny = dy / d
                        fx[i] += nx * overlap
                        fy[i] += ny * overlap
                        fx[j] -= nx * overlap
                        fy[j] -= ny * overlap
        
        # Wall forces
        for i in range(n):
            if cx[i] < r_new[i] * 1.05:
                fx[i] += (r_new[i] * 1.05 - cx[i]) * 2.0
            if cx[i] > W - r_new[i] * 1.05:
                fx[i] -= (cx[i] - (W - r_new[i] * 1.05)) * 2.0
            if cy[i] < r_new[i] * 1.05:
                fy[i] += (r_new[i] * 1.05 - cy[i]) * 2.0
            if cy[i] > H - r_new[i] * 1.05:
                fy[i] -= (cy[i] - (H - r_new[i] * 1.05)) * 2.0
        
        # Update positions with small step
        lr = 0.1
        cx_new = cx + lr * fx
        cy_new = cy + lr * fy
        
        # Project to valid range
        for i in range(n):
            lo_x = r_new[i]
            hi_x = W - r_new[i]
            lo_y = r_new[i]
            hi_y = H - r_new[i]
            cx_new[i] = np.clip(cx_new[i], lo_x, hi_x)
            cy_new[i] = np.clip(cy_new[i], lo_y, hi_y)
        
        # Check validity of new configuration
        valid = True
        for i in range(n):
            if cx_new[i] < r_new[i] - 1e-9 or cx_new[i] > W - r_new[i] + 1e-9:
                valid = False
                break
            if cy_new[i] < r_new[i] - 1e-9 or cy_new[i] > H - r_new[i] + 1e-9:
                valid = False
                break
        if valid:
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.hypot(cx_new[i] - cx_new[j], cy_new[i] - cy_new[j])
                    if d + 1e-9 < r_new[i] + r_new[j]:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            cx = cx_new
            cy = cy_new
            r = r_new
        else:
            # Try without position update, just radius update
            r = np.minimum(r_new, max_r * 0.99)
            # Check if this is valid
            valid2 = True
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                    if d + 1e-9 < r[i] + r[j]:
                        valid2 = False
                        break
                if not valid2:
                    break
            if valid2:
                pass
            else:
                # Reduce growth
                r = r * 1.005
                valid3 = True
                for i in range(n):
                    for j in range(i + 1, n):
                        d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                        if d + 1e-9 < r[i] + r[j]:
                            valid3 = False
                            break
                    if not valid3:
                        break
                if not valid3:
                    # Stuck, break
                    break
                r = r * 1.005  # apply the small growth
    
    return cx, cy, r


def _evaluate(cx, cy, r, W, H):
    """Check validity and return sum of radii, or None if invalid."""
    n = len(cx)
    cx = np.clip(cx, 0.0, W)
    cy = np.clip(cy, 0.0, H)
    r = np.maximum(r, 1e-8)
    # Check wall constraints
    for i in range(n):
        if cx[i] < r[i] - 1e-7 or cx[i] > W - r[i] + 1e-7:
            return None
        if cy[i] < r[i] - 1e-7 or cy[i] > H - r[i] + 1e-7:
            return None
    # Check pairwise
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d + 1e-7 < r[i] + r[j]:
                return None
    return float(np.sum(r))


def _slsqp_polish(cx, cy, r, W, H, max_iter=200):
    """Use SLSQP to polish the solution."""
    n = len(cx)
    x0 = np.column_stack([cx, cy, r]).flatten()
    
    def objective(x):
        cx_ = x[:n]
        cy_ = x[n:2*n]
        r_ = x[2*n:]
        # Penalty for violations
        penalty = 0.0
        for i in range(n):
            if cx_[i] < r_[i]:
                penalty += (r_[i] - cx_[i]) ** 2
            if cx_[i] > W - r_[i]:
                penalty += (cx_[i] - (W - r_[i])) ** 2
            if cy_[i] < r_[i]:
                penalty += (r_[i] - cy_[i]) ** 2
            if cy_[i] > H - r_[i]:
                penalty += (cy_[i] - (H - r_[i])) ** 2
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(cx_[i] - cx_[j], cy_[i] - cy_[j])
                if d < r_[i] + r_[j]:
                    penalty += (r_[i] + r_[j] - d) ** 2
        return -np.sum(r_) + 100.0 * penalty
    
    def gradient(x):
        cx_ = x[:n]
        cy_ = x[n:2*n]
        r_ = x[2*n:]
        grad = np.zeros(3 * n)
        grad[2*n:] = -1.0  # d(-sum r)/dr = -1
        
        # Wall penalties
        for i in range(n):
            if cx_[i] < r_[i]:
                grad[i] += 2.0 * 100.0 * (r_[i] - cx_[i]) * (-1.0)
                grad[2*n+i] += 2.0 * 100.0 * (r_[i] - cx_[i]) * 1.0
            if cx_[i] > W - r_[i]:
                grad[i] += 2.0 * 100.0 * (cx_[i] - (W - r_[i])) * 1.0
                grad[2*n+i] += 2.0 * 100.0 * (cx_[i] - (W - r_[i])) * (-1.0)
            if cy_[i] < r_[i]:
                grad[n+i] += 2.0 * 100.0 * (r_[i] - cy_[i]) * (-1.0)
                grad[2*n+i] += 2.0 * 100.0 * (r_[i] - cy_[i]) * 1.0
            if cy_[i] > H - r_[i]:
                grad[n+i] += 2.0 * 100.0 * (cy_[i] - (H - r_[i])) * 1.0
                grad[2*n+i] += 2.0 * 100.0 * (cy_[i] - (H - r_[i])) * (-1.0)
        
        # Pairwise penalties
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx_[i] - cx_[j]
                dy = cy_[i] - cy_[j]
                d = np.hypot(dx, dy)
                req = r_[i] + r_[j]
                if d < req and d > 1e-12:
                    viol = req - d
                    dv = 2.0 * 100.0 * viol
                    grad[i] += dv * (-dx / d)
                    grad[n+i] += dv * (-dy / d)
                    grad[j] += dv * (dx / d)
                    grad[n+j] += dv * (dy / d)
                    grad[2*n+i] += dv * 1.0
                    grad[2*n+j] += dv * 1.0
        
        return grad
    
    try:
        res = minimize(objective, x0, jac=gradient, method='SLSQP',
                       options={'maxiter': max_iter, 'ftol': 1e-12})
        if res.x is not None:
            cx_opt = res.x[:n]
            cy_opt = res.x[n:2*n]
            r_opt = res.x[2*n:]
            s = _evaluate(cx_opt, cy_opt, r_opt, W, H)
            if s is not None:
                return cx_opt, cy_opt, r_opt, s
    except Exception:
        pass
    
    return cx, cy, r, _evaluate(cx, cy, r, W, H)


def _generate_seed(W, H, method='grid'):
    """Generate initial circle positions."""
    if method == 'grid':
        # Try different grid configurations
        best = None
        for kx in range(3, 7):
            for ky in range(3, 7):
                if kx * ky < N:
                    continue
                cx = np.linspace(0, W, kx + 2)[1:-1]
                cy = np.linspace(0, H, ky + 2)[1:-1]
                pts = []
                for yy in cy:
                    for xx in cx:
                        pts.append([xx, yy])
                if len(pts) >= N:
                    # Take N points, centered
                    pts = pts[:N]
                    cx_arr = np.array([p[0] for p in pts])
                    cy_arr = np.array([p[1] for p in pts])
                    # Check if all can fit
                    min_r = np.min(np.minimum(np.minimum(cx_arr, W - cx_arr), 
                                              np.minimum(cy_arr, H - cy_arr)))
                    if min_r > 0.05:
                        if best is None or min_r > best[2]:
                            best = (cx_arr, cy_arr, min_r)
        if best is not None:
            return best[0], best[1]
    
    # Fallback: random
    cx = np.random.uniform(0.1 * W, 0.9 * W, N)
    cy = np.random.uniform(0.1 * H, 0.9 * H, N)
    return cx, cy


def construct():
    best_sum = 0.0
    best_cx = None
    best_cy = None
    best_r = None
    best_W = 1.0
    best_H = 1.0
    
    random.seed(42)
    np.random.seed(42)
    
    # Try different aspect ratios
    Ws = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
    
    for W in Ws:
        if not _time_ok():
            break
        H = 2.0 - W
        
        # Try multiple seed methods
        for method in ['grid', 'random']:
            if not _time_ok():
                break
            
            for trial in range(3 if method == 'random' else 1):
                if not _time_ok():
                    break
                
                cx, cy = _generate_seed(W, H, method)
                
                # Pebble growth
                cx_g, cy_g, r_g = _pebble_grow(cx, cy, W, H, max_iter=300)
                s = _evaluate(cx_g, cy_g, r_g, W, H)
                if s is None:
                    continue
                
                # SLSQP polish
                cx_p, cy_p, r_p, s_p = _slsqp_polish(cx_g, cy_g, r_g, W, H, max_iter=150)
                if s_p is not None and s_p > best_sum:
                    best_sum = s_p
                    best_cx = cx_p
                    best_cy = cy_p
                    best_r = r_p
                    best_W = W
                    best_H = H
    
    # Additional polish on the best solution
    if best_cx is not None and _time_ok():
        cx_f, cy_f, r_f, s_f = _slsqp_polish(best_cx, best_cy, best_r, best_W, best_H, max_iter=200)
        if s_f is not None and s_f > best_sum:
            best_sum = s_f
            best_cx = cx_f
            best_cy = cy_f
            best_r = r_f
    
    # Final validation and cleanup
    if best_cx is None:
        # Fallback: simple grid
        W, H = 1.0, 1.0
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4])
        best_cx = np.array([c[0] for c in cells[:N]])
        best_cy = np.array([c[1] for c in cells[:N]])
        best_r = np.array([c[2] for c in cells[:N]])
        best_W, best_H = W, H
    
    # Clip to valid range
    cx_out = np.clip(best_cx, 0.0, best_W)
    cy_out = np.clip(best_cy, 0.0, best_H)
    r_out = np.maximum(best_r, 1e-6)
    
    # Enforce wall constraints
    lo = np.minimum(np.minimum(cx_out, cy_out), np.minimum(best_W - cx_out, best_H - cy_out))
    r_out = np.minimum(r_out, lo)
    r_out = np.maximum(r_out, 1e-6)
    
    out = np.column_stack([cx_out, cy_out, r_out])
    
    # Final check
    s_final = _evaluate(cx_out, cy_out, r_out, best_W, best_H)
    if s_final is None:
        # Return a simple valid configuration
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return out
