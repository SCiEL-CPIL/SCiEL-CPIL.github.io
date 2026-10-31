"""Circle packing in perimeter-4 rectangle: hexagonal init + SLSQP with analytic gradients + multi-start."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 135.0  # safety margin before 150s kill


def _hex_init(W, H, n_rows=None):
    """Create a hexagonal packing initialization."""
    if n_rows is None:
        n_rows = max(1, int(np.ceil(np.sqrt(N * H / W))))
    
    # Hexagonal spacing
    # For a hex grid with row spacing dy and column spacing dx:
    # dy = sqrt(3)/2 * dx for optimal hex packing
    # We want to fit as many as possible in W x H
    
    # Start with a reasonable guess for circle radius
    # Area per circle in hex packing: r^2 * 2*pi/sqrt(3)
    # Total area: W*H
    # N * r^2 * 2*pi/sqrt(3) <= W*H
    # r <= sqrt(W*H*sqrt(3)/(N*2*pi))
    r_guess = np.sqrt(W * H * np.sqrt(3) / (N * 2 * np.pi)) * 0.95
    
    # Hex spacing
    dx = 2 * r_guess * 1.01  # slight overlap avoidance
    dy = dx * np.sqrt(3) / 2
    
    # How many columns fit?
    n_cols = max(1, int(np.floor(W / dx)) + 1)
    
    # Generate hex positions
    pts = []
    row = 0
    while len(pts) < N and row < n_rows + 10:
        offset = (dx / 2) if (row % 2 == 1) else 0
        col = 0
        while col < n_cols + 10 and len(pts) < N:
            cx = offset + col * dx
            cy = row * dy
            if 0 <= cx <= W and 0 <= cy <= H:
                r = min(cx, cy, W - cx, H - cy) * 0.99
                if r > 1e-4:
                    pts.append([cx, cy, r])
            col += 1
        row += 1
    
    # If we don't have enough, add some random ones
    while len(pts) < N:
        cx = random.uniform(0.05 * W, 0.95 * W)
        cy = random.uniform(0.05 * H, 0.95 * H)
        r = min(cx, cy, W - cx, H - cy) * 0.3
        r = max(r, 1e-4)
        pts.append([cx, cy, r])
    
    return np.array(pts[:N])


def _square_grid_init(W, H):
    """Create a square grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    pts = []
    for i in range(side):
        for j in range(side):
            if len(pts) >= N:
                break
            cx = (i + 0.5) * W / side
            cy = (j + 0.5) * H / side
            r = min(cx, cy, W - cx, H - cy) * 0.99
            pts.append([cx, cy, max(r, 1e-4)])
    return np.array(pts[:N])


def _random_init(W, H):
    """Create a random initialization."""
    pts = []
    for _ in range(N):
        cx = random.uniform(0.05 * W, 0.95 * W)
        cy = random.uniform(0.05 * H, 0.95 * H)
        r = min(cx, cy, W - cx, H - cy) * 0.3
        r = max(r, 1e-4)
        pts.append([cx, cy, r])
    return np.array(pts)


def _objective(x, W, H, mu=1e3):
    """Return (value, grad) where value = -sum(r) + mu * penalty."""
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = len(cx)
    
    # Clamp to reasonable range
    cx = np.clip(cx, -1.0, W + 1.0)
    cy = np.clip(cy, -1.0, H + 1.0)
    r = np.clip(r, -1.0, 2.0)
    
    val = -np.sum(r)
    grad = np.zeros_like(x)
    grad[:, 2] = -1.0
    
    # Boundary constraint: r <= min(cx, cy, W-cx, H-cy)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    viol_b = np.maximum(r - lo, 0.0)
    val += mu * np.sum(viol_b ** 2)
    
    for i in range(n):
        if viol_b[i] <= 0:
            continue
        d_left = cx[i]
        d_bottom = cy[i]
        d_right = W - cx[i]
        d_top = H - cy[i]
        min_d = lo[i]
        eps = 1e-9
        g_cx = 0.0
        g_cy = 0.0
        if d_left <= min_d + eps:
            g_cx += 1.0
        if d_right <= min_d + eps:
            g_cx += -1.0
        if d_bottom <= min_d + eps:
            g_cy += 1.0
        if d_top <= min_d + eps:
            g_cy += -1.0
        cnt = (d_left <= min_d + eps) + (d_right <= min_d + eps) + \
              (d_bottom <= min_d + eps) + (d_top <= min_d + eps)
        if cnt > 0:
            g_cx /= cnt
            g_cy /= cnt
        dv = 2.0 * mu * viol_b[i]
        grad[i, 0] += dv * (-g_cx)
        grad[i, 1] += dv * (-g_cy)
        grad[i, 2] += dv * 1.0
    
    # r > 0
    viol_r = np.maximum(-r, 0.0)
    val += mu * np.sum(viol_r ** 2)
    for i in range(n):
        if viol_r[i] > 0:
            dv = 2.0 * mu * viol_r[i]
            grad[i, 2] += dv * (-1.0)
    
    # Non-overlap
    mu_o = mu
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.hypot(dx, dy)
            req = r[i] + r[j]
            viol = req - d
            if viol <= 0:
                continue
            val += mu_o * viol ** 2
            if d < 1e-12:
                grad[i, 0] += 2.0 * mu_o * viol * 1.0
                grad[j, 0] += 2.0 * mu_o * viol * (-1.0)
                grad[i, 2] += 2.0 * mu_o * viol * 1.0
                grad[j, 2] += 2.0 * mu_o * viol * 1.0
                continue
            dv = 2.0 * mu_o * viol
            grad[i, 0] += dv * (-dx / d)
            grad[i, 1] += dv * (-dy / d)
            grad[j, 0] += dv * (dx / d)
            grad[j, 1] += dv * (dy / d)
            grad[i, 2] += dv * 1.0
            grad[j, 2] += dv * 1.0
    
    return val, grad


def _evaluate(x, W, H):
    """Return sum of radii if x is valid, else None."""
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = len(cx)
    cx = np.clip(cx, 0.0, W)
    cy = np.clip(cy, 0.0, H)
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d + 1e-8 < r[i] + r[j]:
                return None
    return float(np.sum(r))


def _optimize_from(x0, W, H, max_iter=300):
    """Run SLSQP from x0."""
    best_x = x0.copy()
    best_sum = _evaluate(x0, W, H)
    if best_sum is None:
        best_sum = 0.0
    
    def fun(x):
        v, _ = _objective(x, W, H)
        return v
    
    def grad(x):
        _, g = _objective(x, W, H)
        return g
    
    try:
        res = minimize(fun, x0, jac=grad, method='SLSQP',
                       options={'maxiter': max_iter, 'ftol': 1e-10, 'disp': 0})
        if res.x is not None:
            s = _evaluate(res.x, W, H)
            if s is not None and s > best_sum:
                best_sum = s
                best_x = res.x
    except Exception:
        pass
    
    if time.time() < DEADLINE - 10:
        try:
            res2 = minimize(fun, best_x, jac=grad, method='SLSQP',
                            options={'maxiter': 200, 'ftol': 1e-12, 'disp': 0})
            if res2.x is not None:
                s = _evaluate(res2.x, W, H)
                if s is not None and s > best_sum:
                    best_sum = s
                    best_x = res2.x
        except Exception:
            pass
    
    return best_x, best_sum


def construct():
    best_sum = 0.0
    best_arr = None
    best_W = 1.0
    best_H = 1.0
    
    def time_ok():
        return time.time() < DEADLINE
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.67, 1.4, 0.71]
    
    for W_ratio in aspect_ratios:
        if not time_ok():
            break
        
        W = W_ratio
        H = 2.0 - W
        if W <= 0 or H <= 0:
            continue
        
        # Try multiple initializations
        inits = []
        
        # Hexagonal
        for n_rows in [3, 4, 5]:
            inits.append(_hex_init(W, H, n_rows))
        
        # Square grid
        inits.append(_square_grid_init(W, H))
        
        # Random
        for _ in range(3):
            inits.append(_random_init(W, H))
        
        for x0 in inits:
            if not time_ok():
                break
            
            # Perturb slightly
            x0_perturbed = x0 + np.random.normal(0, 0.01, x0.shape)
            
            res_x, res_sum = _optimize_from(x0_perturbed, W, H)
            
            if res_sum > best_sum:
                best_sum = res_sum
                best_arr = res_x.copy()
                best_W = W
                best_H = H
    
    if best_arr is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Final validation and clamping
    W, H = best_W, best_H
    cx = np.clip(best_arr[:, 0], 0.0, W)
    cy = np.clip(best_arr[:, 1], 0.0, H)
    r = np.maximum(best_arr[:, 2], 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    
    result = np.column_stack([cx, cy, r])
    return result
