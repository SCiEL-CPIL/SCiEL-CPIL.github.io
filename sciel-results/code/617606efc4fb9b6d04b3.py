"""Circle packing in perimeter-4 rectangle: Hard-Constraint SLSQP + Aggressive Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _get_max_radius(cx, cy, r, W, H, i):
    """Calculate the maximum possible radius for circle i given current positions of others."""
    r_wall = min(cx[i], W - cx[i], cy[i], H - cy[i])
    r_max = r_wall
    for j in range(N):
        if i == j:
            continue
        dx = cx[i] - cx[j]
        dy = cy[i] - cy[j]
        d = np.sqrt(dx * dx + dy * dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
    return max(0.0, r_max)


def _polish_aggressive(z, time_budget):
    """
    Aggressive polish:
    1. Iteratively expand each circle to its max possible radius.
    2. If stuck, perform small random jiggles on positions to escape local minima, then re-expand.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    best_val = np.sum(r)
    best_state = (x.copy(), y.copy(), r.copy())
    
    rng = np.random.default_rng(123)

    # Phase 1: Pure expansion
    for _ in range(100):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            r_max = _get_max_radius(x, y, r, w, h, i)
            if r_max > r[i] + 1e-9:
                r[i] = r_max
                improved = True
        if not improved:
            break
        
        if np.sum(r) > best_val:
            best_val = np.sum(r)
            best_state = (x.copy(), y.copy(), r.copy())

    x, y, r = best_state

    # Phase 2: Jiggle and Re-expand
    # Try to escape local minima by slightly perturbing positions
    for attempt in range(20):
        if time.time() > time_budget:
            break
            
        # Save current state
        x_orig = x.copy()
        y_orig = y.copy()
        r_orig = r.copy()
        
        # Apply small random jitter to positions
        # Scale jitter based on average radius to be effective
        avg_r = np.mean(r)
        jitter_scale = avg_r * 0.05
        x_jit = x + rng.normal(0, jitter_scale, n)
        y_jit = y + rng.normal(0, jitter_scale, n)
        
        # Ensure jittered positions are within bounds
        x_jit = np.clip(x_jit, r, w - r)
        y_jit = np.clip(y_jit, r, h - r)
        
        # Try to expand radii from jittered positions
        # We need to check validity after each expansion step
        x_new = x_jit.copy()
        y_new = y_jit.copy()
        r_new = r_orig.copy() # Start with original radii, try to grow
        
        valid = True
        # First, check if jittered positions with original radii are valid
        if not _is_valid(x_new, y_new, r_new, w, h):
            # If not valid, try to shrink radii slightly to fit, or skip this jitter
            # Simple approach: skip this jitter if invalid
            continue
            
        # Expand
        for _ in range(50):
            improved = False
            for i in range(n):
                r_max = _get_max_radius(x_new, y_new, r_new, w, h, i)
                if r_max > r_new[i] + 1e-9:
                    r_new[i] = r_max
                    improved = True
            if not improved:
                break
            
            if np.sum(r_new) > best_val:
                best_val = np.sum(r_new)
                best_state = (x_new.copy(), y_new.copy(), r_new.copy())
                
        # If we found a better state, adopt it as the new starting point for next jitter
        if np.sum(r_new) > np.sum(r_orig):
            x, y, r = x_new, y_new, r_new
        else:
            # Restore original if jitter didn't help
            x, y, r = x_orig, y_orig, r_orig

    return np.array([x, y, r, [w]]).T.flatten()


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    def objective(z):
        """Minimize negative sum of radii."""
        return -np.sum(z[2::3][:N])

    # Generate diverse initial guesses
    def get_initial_guesses():
        guesses = []
        
        # 1. Grid-based with varied aspect ratios
        aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            # Approximate grid
            cols = int(np.ceil(np.sqrt(N * w0 / h0)))
            rows = int(np.ceil(N / cols))
            for i in range(N):
                row = i // cols
                col = i % cols
                cx = (col + 0.5) / cols * w0
                cy = (row + 0.5) / rows * h0
                r0 = min(w0 / cols, h0 / rows) * 0.48
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        # 2. Hexagonal lattice seeds
        for w0 in [1.0, 0.9, 1.1, 0.8, 1.2]:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            # Estimate density
            # Hex packing density ~0.9069
            # Area per circle ~ pi * r^2 / 0.9069
            # Total area = w0 * h0
            # N * pi * r^2 / 0.9069 <= w0 * h0
            # r <= sqrt(w0 * h0 * 0.9069 / (N * pi))
            r_est = np.sqrt(w0 * h0 * 0.9069 / (N * np.pi)) * 0.95
            r_est = min(r_est, w0/4, h0/4) # Safety
            
            cols = int(np.ceil(w0 / (r_est * 1.732)))
            rows = int(np.ceil(h0 / (r_est * 1.5)))
            
            idx = 0
            for row in range(rows):
                if idx >= N:
                    break
                y_pos = (row + 0.5) * (h0 / rows)
                offset = (w0 / cols) / 2.0 if (row % 2 == 1) else 0.0
                for col in range(cols):
                    if idx >= N:
                        break
                    x_pos = (col + 0.5) * (w0 / cols) + offset
                    if x_pos < r_est or x_pos > w0 - r_est:
                        continue
                    z[3 * idx] = x_pos
                    z[3 * idx + 1] = y_pos
                    z[3 * idx + 2] = r_est
                    idx += 1
            
            # If we didn't fill N, just use the first N from a simple grid as fallback for this seed
            if idx < N:
                # Revert to simple grid for this aspect ratio
                cols = int(np.ceil(np.sqrt(N * w0 / h0)))
                rows = int(np.ceil(N / cols))
                for i in range(N):
                    row = i // cols
                    col = i % cols
                    cx = (col + 0.5) / cols * w0
                    cy = (row + 0.5) / rows * h0
                    r0 = min(w0 / cols, h0 / rows) * 0.48
                    z[3 * i] = cx
                    z[3 * i + 1] = cy
                    z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    # Constraints for SLSQP
    # Variables: x0, y0, r0, x1, y1, r1, ..., xN-1, yN-1, rN-1, w
    # Indices: 3*i, 3*i+1, 3*i+2 for circle i. Last index is w.
    
    def make_constraints():
        cons = []
        
        # Boundary constraints
        for i in range(N):
            # x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*i] - z[3*i+2]})
            # w - x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[-1] - z[3*i] - z[3*i+2]})
            # y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*i+1] - z[3*i+2]})
            # h - y_i - r_i >= 0  (h = 2 - w)
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: (2.0 - z[-1]) - z[3*i+1] - z[3*i+2]})
            
        # Pairwise constraints
        for i in range(N):
            for j in range(i + 1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda z, i=i, j=j: (z[3*i] - z[3*j])**2 + (z[3*i+1] - z[3*j+1])**2 - (z[3*i+2] + z[3*j+2])**2
                })
                
        return cons

    constraints = make_constraints()
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7)) # w

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 30:
            break

        # Ensure initial guess is roughly valid or close
        # SLSQP can handle slight violations if constraints are active, but better to start valid
        # We'll rely on SLSQP to find the feasible region.
        
        try:
            res = minimize(objective, z0, method='SLSQP',
                           bounds=bounds,
                           constraints=constraints,
                           options={'maxiter': 500, 'ftol': 1e-12, 'maxls': 100})
            
            z_opt = res.x
            x = z_opt[0::3][:N]
            y = z_opt[1::3][:N]
            r = z_opt[2::3][:N]
            w = z_opt[-1]
            h = 2.0 - w
            
            # Check validity
            if _is_valid(x, y, r, w, h):
                val = np.sum(r)
                if val > best_value:
                    best_value = val
                    best_result = z_opt.copy()
        except Exception:
            continue

    # Polish the best result
    if best_result is not None and time.time() < deadline - 10:
        polished = _polish_aggressive(best_result, deadline - 5)
        if polished is not None:
            x = polished[0::3][:N]
            y = polished[1::3][:N]
            r = polished[2::3][:N]
            w = polished[-1]
            h = 2.0 - w
            if _is_valid(x, y, r, w, h):
                val = np.sum(r)
                if val > best_value:
                    best_value = val
                    best_result = polished

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Format output
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    
    return np.column_stack((x, y, r))

