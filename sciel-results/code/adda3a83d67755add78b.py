"""Circle packing in perimeter-4 rectangle: unified SLSQP + force-directed polish."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 145.0  # safety margin before 150s kill

def _evaluate(x, W, H):
    """Return sum of radii if x is a valid packing, else None."""
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
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d + 1e-7 < r[i] + r[j]:
                return None
    return float(np.sum(r))

def _force_relax(cx, cy, r, W, H, max_iter=300, dt=0.003):
    """Force-directed relaxation to resolve overlaps and wall violations."""
    n = len(cx)
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()
    
    for _ in range(max_iter):
        if time.time() > DEADLINE - 3:
            break
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Wall repulsion
        for i in range(n):
            if cx[i] - r[i] < 0:
                fx[i] += (cx[i] - r[i]) * 20
            if cx[i] + r[i] > W:
                fx[i] -= (cx[i] + r[i] - W) * 20
            if cy[i] - r[i] < 0:
                fy[i] += (cy[i] - r[i]) * 20
            if cy[i] + r[i] > H:
                fy[i] -= (cy[i] + r[i] - H) * 20
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 10
                    fy[i] += (dy / d) * overlap * 10
                    fx[j] -= (dx / d) * overlap * 10
                    fy[j] -= (dy / d) * overlap * 10
        
        # Update positions
        cx += dt * fx
        cy += dt * fy
        
        # Clamp positions
        for i in range(n):
            cx[i] = np.clip(cx[i], r[i], W - r[i])
            cy[i] = np.clip(cy[i], r[i], H - r[i])
            r[i] = np.maximum(r[i], 1e-6)
            lo = np.minimum(np.minimum(cx[i], cy[i]), np.minimum(W - cx[i], H - cy[i]))
            r[i] = np.minimum(r[i], lo)
            r[i] = np.maximum(r[i], 1e-6)
    
    return cx, cy, r

def _unified_slsqp(z0, maxiter=200):
    """
    SLSQP optimization on unified parameter vector [x1,y1,r1,...,xN,yN,rN,w].
    h = 2.0 - w
    """
    n = N
    penalty_weight = 500.0
    
    def objective(z):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w
        
        if w < 0.3 or h < 0.3:
            return 1e6
        
        obj = -np.sum(r)
        
        # Overlap penalty
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                dist_req = r[i] + r[j]
                if d < dist_req:
                    obj += penalty_weight * (dist_req - d) ** 2
        
        # Boundary penalty
        for i in range(n):
            if x[i] < r[i]:
                obj += penalty_weight * (r[i] - x[i]) ** 2
            elif x[i] > w - r[i]:
                obj += penalty_weight * (x[i] - (w - r[i])) ** 2
            if y[i] < r[i]:
                obj += penalty_weight * (r[i] - y[i]) ** 2
            elif y[i] > h - r[i]:
                obj += penalty_weight * (y[i] - (h - r[i])) ** 2
        
        # Keep aspect ratio in reasonable range
        if w < 0.5:
            obj += penalty_weight * (0.5 - w) ** 2
        elif w > 1.5:
            obj += penalty_weight * (w - 1.5) ** 2
        
        return obj
    
    bounds = [(0, 2)] * (3 * n) + [(0.3, 1.7)]
    
    try:
        res = minimize(
            objective,
            z0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-10, 'maxls': 50},
        )
    except Exception:
        return None, 0.0
    
    z_opt = res.x
    x = z_opt[0::3][:n]
    y = z_opt[1::3][:n]
    r = z_opt[2::3][:n]
    w = z_opt[-1]
    h = 2.0 - w
    
    # Clamp and validate
    x = np.clip(x, 0, w)
    y = np.clip(y, 0, h)
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(x, y), np.minimum(w - x, h - y))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    
    out = np.column_stack([x, y, r])
    s = _evaluate(out, w, h)
    if s is not None:
        return out, s
    return None, 0.0

def _polish_result(cx, cy, r, W, H, time_budget=10.0):
    """Polish a valid packing with force-directed radius growth."""
    n = len(cx)
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()
    
    dt = 0.001
    max_iter = 500
    start = time.time()
    
    best_sum = _evaluate(np.column_stack([cx, cy, r]), W, H) or 0.0
    best_cx = cx.copy()
    best_cy = cy.copy()
    best_r = r.copy()
    
    for it in range(max_iter):
        if time.time() - start > time_budget:
            break
        
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Wall repulsion
        for i in range(n):
            if cx[i] - r[i] < 0:
                fx[i] += (cx[i] - r[i]) * 20
            if cx[i] + r[i] > W:
                fx[i] -= (cx[i] + r[i] - W) * 20
            if cy[i] - r[i] < 0:
                fy[i] += (cy[i] - r[i]) * 20
            if cy[i] + r[i] > H:
                fy[i] -= (cy[i] + r[i] - H) * 20
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 10
                    fy[i] += (dy / d) * overlap * 10
                    fx[j] -= (dx / d) * overlap * 10
                    fy[j] -= (dy / d) * overlap * 10
        
        # Update positions
        cx += dt * fx
        cy += dt * fy
        
        # Try to grow radii slightly
        for i in range(n):
            lo = np.minimum(np.minimum(cx[i], cy[i]), np.minimum(W - cx[i], H - cy[i]))
            # Check if growing would cause overlap
            can_grow = True
            new_r = r[i] * 1.002
            if new_r <= lo:
                for j in range(n):
                    if i != j:
                        d = np.sqrt((cx[i] - cx[j])**2 + (cy[i] - cy[j])**2)
                        if d < new_r + r[j]:
                            can_grow = False
                            break
                if can_grow:
                    r[i] = new_r
        
        # Clamp
        for i in range(n):
            cx[i] = np.clip(cx[i], 0, W)
            cy[i] = np.clip(cy[i], 0, H)
            r[i] = np.maximum(r[i], 1e-6)
            lo = np.minimum(np.minimum(cx[i], cy[i]), np.minimum(W - cx[i], H - cy[i]))
            r[i] = np.minimum(r[i], lo)
            r[i] = np.maximum(r[i], 1e-6)
        
        s_new = _evaluate(np.column_stack([cx, cy, r]), W, H)
        if s_new is not None and s_new > best_sum:
            best_sum = s_new
            best_cx = cx.copy()
            best_cy = cy.copy()
            best_r = r.copy()
    
    return best_cx, best_cy, best_r, best_sum

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

def _honeycomb_layout(W, H):
    """Hexagonal/honeycomb-like layout for better packing density."""
    pts = []
    base_r = min(W, H) / 8.0
    rows = int(H / (base_r * 1.732)) + 2
    cols = int(W / (base_r * 1.5)) + 2
    
    for row in range(rows):
        y = row * base_r * 1.732
        if y > H:
            break
        offset = (base_r * 1.5) if (row % 2 == 1) else 0
        for col in range(cols):
            x = col * base_r * 1.5 + offset
            if x > W:
                break
            r = min(x, y, W - x, H - y) * 0.9
            r = max(r, 1e-4)
            pts.append([x, y, r])
    
    arr = np.array(pts)
    if len(arr) < N:
        while len(arr) < N:
            arr = np.vstack([arr, [0.01, 0.01, 0.005]])
    return arr[:N]

def construct():
    best_sum = 0.0
    best_arr = None
    best_W = 1.0
    best_H = 1.0
    
    def time_ok():
        return time.time() < DEADLINE - 3
    
    random.seed(42)
    np.random.seed(42)
    
    # Generate diverse initial guesses with varying aspect ratios
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5]
    side = int(np.ceil(np.sqrt(N)))
    
    candidates = []
    
    for w0 in aspect_ratios:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        
        # Grid-based start
        cx = np.zeros(N)
        cy = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, side)
            cx[i] = (col + 0.5) / side * w0
            cy[i] = (row + 0.5) / side * h0
            r[i] = min(w0 / side, h0 / side) * 0.45
            r[i] = max(r[i], 1e-4)
        
        # Force relax to resolve overlaps
        cx_rel, cy_rel, r_rel = _force_relax(cx, cy, r, w0, h0, max_iter=200)
        g = np.column_stack([cx_rel, cy_rel, r_rel])
        s = _evaluate(g, w0, h0)
        if s is not None:
            candidates.append((s, w0, h0, g))
        
        # Honeycomb start
        h = _honeycomb_layout(w0, h0)
        cx_h, cy_h, r_h = h[:, 0], h[:, 1], h[:, 2]
        cx_hr, cy_hr, r_hr = _force_relax(cx_h, cy_h, r_h, w0, h0, max_iter=200)
        g_h = np.column_stack([cx_hr, cy_hr, r_hr])
        s_h = _evaluate(g_h, w0, h0)
        if s_h is not None:
            candidates.append((s_h, w0, h0, g_h))
        
        # Random start
        g_r = _random_layout(w0, h0)
        cx_r, cy_r, r_r = g_r[:, 0], g_r[:, 1], g_r[:, 2]
        cx_rr, cy_rr, r_rr = _force_relax(cx_r, cy_r, r_r, w0, h0, max_iter=200)
        g_rr = np.column_stack([cx_rr, cy_rr, r_rr])
        s_rr = _evaluate(g_rr, w0, h0)
        if s_rr is not None:
            candidates.append((s_rr, w0, h0, g_rr))
    
    # Add pure random starts
    for _ in range(6):
        w0 = random.uniform(0.6, 1.4)
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        g_r = _random_layout(w0, h0)
        cx_r, cy_r, r_r = g_r[:, 0], g_r[:, 1], g_r[:, 2]
        cx_rr, cy_rr, r_rr = _force_relax(cx_r, cy_r, r_r, w0, h0, max_iter=200)
        g_rr = np.column_stack([cx_rr, cy_rr, r_rr])
        s_rr = _evaluate(g_rr, w0, h0)
        if s_rr is not None:
            candidates.append((s_rr, w0, h0, g_rr))
    
    candidates.sort(key=lambda t: -t[0])
    
    # Phase 1: Unified SLSQP on top candidates
    top_k = min(10, len(candidates))
    for idx in range(top_k):
        if not time_ok():
            break
        s0, W, H, g0 = candidates[idx]
        
        # Convert to unified parameter vector
        z0 = np.zeros(3 * N + 1)
        z0[0::3][:N] = g0[:, 0]
        z0[1::3][:N] = g0[:, 1]
        z0[2::3][:N] = g0[:, 2]
        z0[-1] = W
        
        remaining_time = DEADLINE - time.time()
        maxiter = int(min(200, remaining_time * 20))
        
        x_opt, s_opt = _unified_slsqp(z0, maxiter=maxiter)
        if x_opt is not None and s_opt > best_sum:
            best_sum = s_opt
            best_arr = x_opt
            best_W = W
            best_H = H
    
    # Phase 2: Polish the best result
    if time_ok() and best_arr is not None:
        cx_p, cy_p, r_p, s_p = _polish_result(
            best_arr[:, 0], best_arr[:, 1], best_arr[:, 2],
            best_W, best_H,
            time_budget=min(15, DEADLINE - time.time() - 5)
        )
        if s_p > best_sum:
            best_sum = s_p
            best_arr = np.column_stack([cx_p, cy_p, r_p])
    
    # Phase 3: More SLSQP on next candidates if time remains
    if time_ok() and len(candidates) > top_k:
        for idx in range(top_k, min(top_k + 5, len(candidates))):
            if not time_ok():
                break
            s0, W, H, g0 = candidates[idx]
            
            z0 = np.zeros(3 * N + 1)
            z0[0::3][:N] = g0[:, 0]
            z0[1::3][:N] = g0[:, 1]
            z0[2::3][:N] = g0[:, 2]
            z0[-1] = W
            
            remaining_time = DEADLINE - time.time()
            maxiter = int(min(150, remaining_time * 15))
            
            x_opt, s_opt = _unified_slsqp(z0, maxiter=maxiter)
            if x_opt is not None and s_opt > best_sum:
                best_sum = s_opt
                best_arr = x_opt
                best_W = W
                best_H = H
    
    # Phase 4: Final polish
    if time_ok() and best_arr is not None:
        cx_p, cy_p, r_p, s_p = _polish_result(
            best_arr[:, 0], best_arr[:, 1], best_arr[:, 2],
            best_W, best_H,
            time_budget=min(10, DEADLINE - time.time() - 3)
        )
        if s_p > best_sum:
            best_sum = s_p
            best_arr = np.column_stack([cx_p, cy_p, r_p])
    
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
    
    s_final = _evaluate(out, best_W, best_H)
    if s_final is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return out
