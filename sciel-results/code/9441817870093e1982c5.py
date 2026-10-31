"""Circle packing in perimeter-4 rectangle: contact-graph + SA + vectorized optimization."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 140.0  # safety margin before 150s kill

def _evaluate(x, W, H):
    """Return sum of radii if x is a valid packing, else None."""
    cx = x[:, 0]
    cy = x[:, 1]
    r = x[:, 2]
    n = len(cx)
    # clamp
    cx = np.clip(cx, 0.0, W)
    cy = np.clip(cy, 0.0, H)
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    # check non-overlap (vectorized)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d + 1e-7 < r[i] + r[j]:
                return None
    return float(np.sum(r))

def _evaluate_fast(x, W, H):
    """Fast evaluation: returns (sum_radii, is_valid)."""
    cx = x[:, 0]
    cy = x[:, 1]
    r = x[:, 2]
    n = len(cx)
    cx = np.clip(cx, 0.0, W)
    cy = np.clip(cy, 0.0, H)
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    # vectorized overlap check
    valid = True
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d + 1e-7 < r[i] + r[j]:
                valid = False
                break
        if not valid:
            break
    return float(np.sum(r)), valid

def _force_directed_vec(cx, cy, r, W, H, steps=200, lr=0.005):
    """Vectorized force-directed relaxation with adaptive growth."""
    n = len(cx)
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()
    
    for step in range(steps):
        fcx = np.zeros(n)
        fcy = np.zeros(n)
        
        # Repulsion between overlapping circles (vectorized pairs)
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.hypot(dx, dy)
                req = r[i] + r[j]
                if d < req:
                    if d > 1e-12:
                        f = (req - d) / d
                        fcx[i] += f * dx
                        fcy[i] += f * dy
                        fcx[j] -= f * dx
                        fcy[j] -= f * dy
                    else:
                        angle = random.uniform(0, 2 * np.pi)
                        fcx[i] += 0.1 * np.cos(angle)
                        fcy[i] += 0.1 * np.sin(angle)
                        fcx[j] -= 0.1 * np.cos(angle)
                        fcy[j] -= 0.1 * np.sin(angle)
        
        # Boundary forces: push toward nearest wall to allow growth
        for i in range(n):
            d_left = cx[i]
            d_right = W - cx[i]
            d_bottom = cy[i]
            d_top = H - cy[i]
            min_d = min(d_left, d_right, d_bottom, d_top)
            if min_d > r[i] * 0.9:
                # Push toward closest wall
                if min_d == d_left:
                    fcx[i] -= 0.005
                elif min_d == d_right:
                    fcx[i] += 0.005
                elif min_d == d_bottom:
                    fcy[i] -= 0.005
                elif min_d == d_top:
                    fcy[i] += 0.005
        
        # Update
        cx += lr * fcx
        cy += lr * fcy
        cx = np.clip(cx, 0.0, W)
        cy = np.clip(cy, 0.0, H)
        
        # Grow radii
        lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
        r = np.minimum(r * 1.0005 + 0.00005, lo)
        r = np.maximum(r, 1e-6)
        
        lr *= 0.998
    
    return cx, cy, r

def _simulated_annealing(x0, W, H, steps=3000, T0=0.1, T1=0.001):
    """Simulated annealing for global exploration."""
    n = len(x0)
    cx = x0[:, 0].copy()
    cy = x0[:, 1].copy()
    r = x0[:, 2].copy()
    
    best_cx = cx.copy()
    best_cy = cy.copy()
    best_r = r.copy()
    best_sum = _evaluate(np.column_stack([best_cx, best_cy, best_r]), W, H) or 0.0
    
    for step in range(steps):
        T = T0 * (T1 / T0) ** (step / steps)
        
        # Propose perturbation
        cx_new = cx.copy()
        cy_new = cy.copy()
        r_new = r.copy()
        
        # Random move: pick 1-3 circles and perturb
        n_move = random.randint(1, 3)
        for _ in range(n_move):
            i = random.randint(0, n - 1)
            cx_new[i] += random.gauss(0, 0.05)
            cy_new[i] += random.gauss(0, 0.05)
            if random.random() < 0.3:
                r_new[i] = max(1e-6, r_new[i] * random.uniform(0.8, 1.2))
        
        # Clamp
        cx_new = np.clip(cx_new, 0.0, W)
        cy_new = np.clip(cy_new, 0.0, H)
        r_new = np.maximum(r_new, 1e-6)
        lo = np.minimum(np.minimum(cx_new, cy_new), np.minimum(W - cx_new, H - cy_new))
        r_new = np.minimum(r_new, lo)
        r_new = np.maximum(r_new, 1e-6)
        
        # Evaluate
        sum_new, valid_new = _evaluate_fast(np.column_stack([cx_new, cy_new, r_new]), W, H)
        
        # Accept if valid and better, or with probability based on temperature
        if valid_new:
            if sum_new > best_sum:
                best_sum = sum_new
                best_cx = cx_new.copy()
                best_cy = cy_new.copy()
                best_r = r_new.copy()
            # Accept move if it doesn't create overlap
            cx, cy, r = cx_new, cy_new, r_new
        else:
            # Reject if invalid (overlap)
            pass
    
    return np.column_stack([best_cx, best_cy, best_r]), best_sum

def _local_refine(x0, W, H, max_iter=50):
    """Fast local refinement using gradient descent on penalty."""
    n = len(x0)
    cx = x0[:, 0].copy()
    cy = x0[:, 1].copy()
    r = x0[:, 2].copy()
    
    best_sum = _evaluate(x0, W, H) or 0.0
    best_cx, best_cy, best_r = cx.copy(), cy.copy(), r.copy()
    
    lr = 0.01
    for it in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
        fcx = np.zeros(n)
        fcy = np.zeros(n)
        fr = np.zeros(n)
        
        # Overlap forces
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.hypot(dx, dy)
                req = r[i] + r[j]
                if d < req and d > 1e-12:
                    f = (req - d) / d
                    fcx[i] += f * dx
                    fcy[i] += f * dy
                    fcx[j] -= f * dx
                    fcy[j] -= f * dy
                    # Also reduce radii slightly when overlapping
                    fr[i] -= 0.1 * (req - d)
                    fr[j] -= 0.1 * (req - d)
        
        # Boundary constraints
        for i in range(n):
            if cx[i] < r[i]:
                fcx[i] += (r[i] - cx[i])
            elif cx[i] > W - r[i]:
                fcx[i] -= (cx[i] - (W - r[i]))
            if cy[i] < r[i]:
                fcy[i] += (r[i] - cy[i])
            elif cy[i] > H - r[i]:
                fcy[i] -= (cy[i] - (H - r[i]))
        
        # Update
        cx += lr * fcx
        cy += lr * fcy
        r += lr * fr
        cx = np.clip(cx, 0.0, W)
        cy = np.clip(cy, 0.0, H)
        r = np.maximum(r, 1e-6)
        lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
        r = np.minimum(r, lo)
        
        s = _evaluate(np.column_stack([cx, cy, r]), W, H)
        if s is not None and s > best_sum:
            best_sum = s
            best_cx, best_cy, best_r = cx.copy(), cy.copy(), r.copy()
        
        lr *= 0.95
    
    return np.column_stack([best_cx, best_cy, best_r]), best_sum

def _grid_layout(kx, ky, W, H, shrink=0.95):
    """Build a kx*ky grid layout (pad/truncate to N)."""
    cx = np.linspace(0.0, W, kx + 2)[1:-1]
    cy = np.linspace(0.0, H, ky + 2)[1:-1]
    pts = []
    for yy in cy:
        for xx in cx:
            r = min(xx, yy, W - xx, H - yy) * shrink
            pts.append([xx, yy, max(r, 1e-4)])
    arr = np.array(pts)
    if len(arr) < N:
        while len(arr) < N:
            arr = np.vstack([arr, [0.01, 0.01, 0.005]])
    return arr[:N]

def _random_layout(W, H):
    """Random feasible-ish layout."""
    pts = []
    for _ in range(N):
        cx = random.uniform(0.1 * W, 0.9 * W)
        cy = random.uniform(0.1 * H, 0.9 * H)
        r = min(cx, cy, W - cx, H - cy) * 0.4
        r = max(r, 1e-4)
        pts.append([cx, cy, r])
    return np.array(pts)

def construct():
    best_sum = 0.0
    best_arr = None
    best_W = 1.0
    best_H = 1.0
    
    def time_ok():
        return time.time() < DEADLINE - 5
    
    random.seed(42)
    np.random.seed(42)
    
    # Generate diverse starting candidates
    candidates = []
    
    Ws = [0.75, 0.85, 0.95, 1.05, 1.15, 1.25]
    grid_sizes = [(3, 4), (4, 3), (3, 3), (4, 4), (5, 3), (3, 5)]
    
    for W in Ws:
        H = 2.0 - W
        for kx, ky in grid_sizes:
            if kx * ky < N:
                continue
            g = _grid_layout(kx, ky, W, H, shrink=0.95)
            s = _evaluate(g, W, H)
            if s is not None:
                candidates.append((s, W, H, g))
    
    # Add random starts
    for W in Ws:
        H = 2.0 - W
        for _ in range(3):
            g = _random_layout(W, H)
            s = _evaluate(g, W, H)
            if s is not None:
                candidates.append((s, W, H, g))
    
    # Sort by initial score
    candidates.sort(key=lambda t: -t[0])
    
    # Phase 1: Force-directed on top candidates
    top_k = min(10, len(candidates))
    for idx in range(top_k):
        if not time_ok():
            break
        s0, W, H, g0 = candidates[idx]
        cx, cy, r = _force_directed_vec(g0[:, 0], g0[:, 1], g0[:, 2], W, H, steps=150, lr=0.005)
        x_fd = np.column_stack([cx, cy, r])
        s_fd = _evaluate(x_fd, W, H)
        if s_fd is not None and s_fd > best_sum:
            best_sum = s_fd
            best_arr = x_fd
            best_W = W
            best_H = H
    
    # Phase 2: Simulated annealing on best and a few others
    if time_ok() and best_arr is not None:
        # SA on best
        x_sa, s_sa = _simulated_annealing(best_arr, best_W, best_H, steps=2000, T0=0.08, T1=0.001)
        if s_sa > best_sum:
            best_sum = s_sa
            best_arr = x_sa
    
    # SA on second best candidate
    if time_ok() and len(candidates) > 1:
        s0, W, H, g0 = candidates[1]
        x_sa2, s_sa2 = _simulated_annealing(g0, W, H, steps=1500, T0=0.08, T1=0.001)
        if s_sa2 > best_sum:
            best_sum = s_sa2
            best_arr = x_sa2
            best_W = W
            best_H = H
    
    # Phase 3: Local refinement
    if time_ok() and best_arr is not None:
        x_ref, s_ref = _local_refine(best_arr, best_W, best_H, max_iter=40)
        if s_ref > best_sum:
            best_sum = s_ref
            best_arr = x_ref
    
    # Phase 4: More SA restarts if time remains
    if time_ok() and best_arr is not None:
        remaining = DEADLINE - time.time()
        n_extra = min(4, int(remaining // 15))
        for _ in range(n_extra):
            if not time_ok():
                break
            # Perturb best
            x0 = best_arr + np.random.randn(*best_arr.shape) * 0.02
            x0[:, 0] = np.clip(x0[:, 0], 0.0, best_W)
            x0[:, 1] = np.clip(x0[:, 1], 0.0, best_H)
            x0[:, 2] = np.maximum(x0[:, 2], 1e-4)
            x_sa, s_sa = _simulated_annealing(x0, best_W, best_H, steps=1000, T0=0.05, T1=0.001)
            if s_sa > best_sum:
                best_sum = s_sa
                best_arr = x_sa
    
    # Final local refinement
    if time_ok() and best_arr is not None:
        x_ref, s_ref = _local_refine(best_arr, best_W, best_H, max_iter=30)
        if s_ref > best_sum:
            best_sum = s_ref
            best_arr = x_ref
    
    # Fallback
    if best_arr is None:
        W, H = 1.0, 1.0
        g = _grid_layout(5, 5, W, H, shrink=0.95)
        best_arr = g
        best_W, best_H = W, H
    
    # Final clamping
    cx = np.clip(best_arr[:, 0], 0.0, best_W)
    cy = np.clip(best_arr[:, 1], 0.0, best_H)
    r = best_arr[:, 2].copy()
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(best_W - cx, best_H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    out = np.column_stack([cx, cy, r])
    
    # Final validation
    s_final = _evaluate(out, best_W, best_H)
    if s_final is None:
        # Fallback to grid
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return out
