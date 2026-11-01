"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time

def _pack_rect(w, h):
    """Pack N circles in a w x h rectangle, maximize sum of radii."""
    n = N
    # Initial guess: grid layout
    cols = int(np.ceil(np.sqrt(n * w / h))) if h > 0 else int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    cw, ch = w / cols, h / rows
    x0 = []
    for i in range(cols):
        for j in range(rows):
            if len(x0) >= n:
                break
            x0.append(cw / 2 + i * cw)
            x0.append(ch / 2 + j * ch)
            x0.append(min(cw, ch) / 2 * 0.9)
    x0 = np.array(x0[:n * 3])
    
    # Bounds: x in [r, w-r], y in [r, h-r], r in [0, min(w,h)/2]
    rmax = min(w, h) / 2
    bounds = []
    for i in range(n):
        bounds.append((0, w))  # x
        bounds.append((0, h))  # y
        bounds.append((0, rmax))  # r
    
    def objective(v):
        r = v[2::3]
        return -np.sum(r)
    
    # Constraints: all pairs non-overlapping, and inside rectangle
    # We'll use a penalty-free approach with explicit constraints via SLSQP
    # But SLSQP can handle many constraints. Let's define them.
    
    def constraints_func(v):
        xs = v[0::3]
        ys = v[1::3]
        rs = v[2::3]
        cons = []
        # Inside rectangle: x >= r, x <= w - r, y >= r, y <= h - r
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2], 'jac': None})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: (w - v[3*i]) - v[3*i+2], 'jac': None})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2], 'jac': None})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: (h - v[3*i+1]) - v[3*i+2], 'jac': None})
        # Non-overlap: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
        for i in range(n):
            for j in range(i+1, n):
                cons.append({'type': 'ineq', 'fun': lambda v, i=i, j=j: (v[3*i]-v[3*j])**2 + (v[3*i+1]-v[3*j+1])**2 - (v[3*i+2]+v[3*j+2])**2, 'jac': None})
        return cons
    
    # SLSQP with many constraints might be slow. Let's try a different approach.
    # Use a penalty method or trust-constr.
    
    # Let's use a simpler heuristic + local refinement.
    # First, try a grid-based initialization with varied aspect ratios.
    
    best_val = 0
    best_v = None
    
    for trial in range(5):
        np.random.seed(42 + trial)
        # Perturb initial positions
        x0_trial = x0.copy()
        x0_trial[:n] += np.random.uniform(-0.1, 0.1, n) * w
        x0_trial[n:2*n] += np.random.uniform(-0.1, 0.1, n) * h
        x0_trial[2*n:] *= np.random.uniform(0.8, 1.2, n)
        x0_trial = np.clip(x0_trial, 0, [w, h, rmax] * n)
        
        # Simple local search: iteratively push overlapping circles apart
        v = x0_trial.copy()
        for iteration in range(200):
            improved = False
            xs = v[0::3].copy()
            ys = v[1::3].copy()
            rs = v[2::3].copy()
            
            # Check overlaps and push apart
            for i in range(n):
                for j in range(i+1, n):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = rs[i] + rs[j]
                    if dist < min_dist - 1e-8:
                        if dist < 1e-10:
                            dx = 0.01
                            dy = 0.01
                            dist = np.sqrt(dx*dx + dy*dy)
                        push = (min_dist - dist) / 2
                        dx_norm = dx / dist
                        dy_norm = dy / dist
                        xs[i] += push * dx_norm
                        ys[i] += push * dy_norm
                        xs[j] -= push * dx_norm
                        ys[j] -= push * dy_norm
                        improved = True
            
            # Clamp to bounds
            for i in range(n):
                xs[i] = np.clip(xs[i], rs[i], w - rs[i])
                ys[i] = np.clip(ys[i], rs[i], h - rs[i])
            
            v[0::3] = xs
            v[1::3] = ys
            
            if not improved:
                break
        
        # Verify no overlaps
        valid = True
        for i in range(n):
            if v[3*i] < v[3*i+2] - 1e-6 or v[3*i] > w - v[3*i+2] + 1e-6:
                valid = False
                break
            if v[3*i+1] < v[3*i+2] - 1e-6 or v[3*i+1] > h - v[3*i+2] + 1e-6:
                valid = False
                break
        if valid:
            for i in range(n):
                for j in range(i+1, n):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < v[3*i+2] + v[3*j+2] - 1e-6:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            val = np.sum(v[2::3])
            if val > best_val:
                best_val = val
                best_v = v.copy()
    
    if best_v is None:
        # Fallback: small circles in grid
        r = 0.01
        v = np.zeros(n*3)
        cols = int(np.ceil(np.sqrt(n)))
        for i in range(n):
            cx = i % cols
            cy = i // cols
            v[3*i] = (cx + 0.5) * w / cols
            v[3*i+1] = (cy + 0.5) * h / rows
            v[3*i+2] = r
        return v, np.sum(v[2::3])
    
    return best_v, best_val

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    best_sum = 0
    best_result = None
    
    # Try different aspect ratios
    # Perimeter = 2(w+h) = 4 => w+h = 2
    aspect_ratios = [1.0, 1.5, 0.667, 2.0, 0.5, 1.2, 0.833, 1.333, 0.75, 1.25]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        v, val = _pack_rect(w, h)
        if val > best_sum:
            best_sum = val
            best_result = v.copy()
            best_w, best_h = w, h
    
    if best_result is None:
        # Fallback
        w, h = 1, 1
        v, val = _pack_rect(w, h)
        best_result = v
        best_w, best_h = w, h
    
    # Scale to perimeter 4
    # Current perimeter is 2*(best_w + best_h) = 4 already
    # So just return as is
    
    return best_result
# EVOLVE-BLOCK-END
