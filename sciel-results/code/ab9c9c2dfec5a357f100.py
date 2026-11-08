"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
TIME_LIMIT = 150.0  # seconds
SAFETY_MARGIN = 5.0  # seconds to return before limit


def _objective(v, penalty=1000.0):
    """
    Objective: -sum(r) + penalty * sum(violations^2)
    """
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    obj = -np.sum(r)
    pen = 0.0
    
    # Boundary constraints
    v_x_low = np.maximum(0.0, r - x)
    v_x_high = np.maximum(0.0, x + r - 1.0)
    v_y_low = np.maximum(0.0, r - y)
    v_y_high = np.maximum(0.0, y + r - 1.0)
    
    pen += np.sum(v_x_low**2) + np.sum(v_x_high**2) + np.sum(v_y_low**2) + np.sum(v_y_high**2)
    
    # Circle-circle constraints
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy + 1e-12)
            viol = r[i] + r[j] - d
            if viol > 0:
                pen += viol * viol
                
    return obj + penalty * pen


def _objective_grad(v, penalty=1000.0):
    """
    Gradient of the penalized objective.
    """
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    grad = np.zeros(3*N)
    
    # Gradient of -sum(r)
    grad[2*N:] = -1.0
    
    # Gradient of penalty terms
    v_x_low = r - x
    v_x_high = x + r - 1.0
    v_y_low = r - y
    v_y_high = y + r - 1.0
    
    mask_x_low = v_x_low > 0
    mask_x_high = v_x_high > 0
    mask_y_low = v_y_low > 0
    mask_y_high = v_y_high > 0
    
    # Grads for x
    grad[:N][mask_x_low] += penalty * 2 * v_x_low[mask_x_low] * (-1)
    grad[:N][mask_x_high] += penalty * 2 * v_x_high[mask_x_high] * (1)
    
    # Grads for y
    grad[N:2*N][mask_y_low] += penalty * 2 * v_y_low[mask_y_low] * (-1)
    grad[N:2*N][mask_y_high] += penalty * 2 * v_y_high[mask_y_high] * (1)
    
    # Grads for r
    grad[2*N:][mask_x_low] += penalty * 2 * v_x_low[mask_x_low]
    grad[2*N:][mask_x_high] += penalty * 2 * v_x_high[mask_x_high]
    grad[2*N:][mask_y_low] += penalty * 2 * v_y_low[mask_y_low]
    grad[2*N:][mask_y_high] += penalty * 2 * v_y_high[mask_y_high]
    
    # Circle-circle gradients
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy + 1e-12)
            viol = r[i] + r[j] - d
            if viol > 0:
                factor = 2.0 * penalty * viol
                gx = dx / d
                gy = dy / d
                
                grad[i] += factor * (-gx)
                grad[j] += factor * (gx)
                grad[N+i] += factor * (-gy)
                grad[N+j] += factor * (gy)
                grad[2*N+i] += factor * 1.0
                grad[2*N+j] += factor * 1.0
                
    return grad


def _make_grid_initial(seed=0):
    """Create a 6x6 grid initial configuration using 26 cells."""
    rng = np.random.RandomState(seed)
    side = 6
    positions = []
    radii = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                x += rng.uniform(-0.01, 0.01)
                y += rng.uniform(-0.01, 0.01)
                r = 0.45 / side
                positions.append([x, y])
                radii.append(r)
    
    x = np.array([p[0] for p in positions])
    y = np.array([p[1] for p in positions])
    r = np.array(radii)
    return np.concatenate([x, y, r])


def _make_hex_initial(seed=0):
    """Create a hexagonal packing initial guess."""
    rng = np.random.RandomState(seed)
    pos = []
    rad = []
    rows = [5, 5, 5, 5, 6]  # total = 26
    row_y = []
    for k in range(len(rows)):
        row_y.append((k + 0.5) / len(rows))
    
    idx = 0
    for k, count in enumerate(rows):
        y = row_y[k]
        if k % 2 == 0:
            xs = [(i + 0.5) / count for i in range(count)]
        else:
            xs = [(i + 0.5) / count for i in range(count)]
        for x in xs:
            if idx < N:
                x += rng.uniform(-0.01, 0.01)
                y += rng.uniform(-0.01, 0.01)
                pos.append([x, y])
                rad.append(0.08)
                idx += 1
    
    x = np.array([p[0] for p in pos])
    y = np.array([p[1] for p in pos])
    r = np.array(rad)
    return np.concatenate([x, y, r])


def _make_random_initial(seed=0):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    r = np.full(N, 0.05)
    return np.concatenate([x, y, r])


def _make_mixed_initial(seed=0):
    """Create a mixed initial configuration with two large circles."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    rad = np.zeros(N)
    pos[0] = [0.25, 0.5]
    pos[1] = [0.75, 0.5]
    rad[0] = 0.2
    rad[1] = 0.2
    for i in range(2, N):
        angle = 2 * np.pi * i / (N - 2)
        r = 0.35
        pos[i] = [0.5 + r * np.cos(angle), 0.5 + r * np.sin(angle)]
        rad[i] = 0.06
    
    x = pos[:, 0] + rng.uniform(-0.01, 0.01, N)
    y = pos[:, 1] + rng.uniform(-0.01, 0.01, N)
    return np.concatenate([x, y, rad])


def _repair(v):
    """
    Repair a solution to be strictly feasible by reducing radii.
    Returns the repaired parameter vector and a boolean indicating success.
    """
    x = v[:N].copy()
    y = v[N:2*N].copy()
    r = v[2*N:].copy()
    
    # Clamp positions to [0,1]
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)
    
    # Ensure boundary feasibility
    for i in range(N):
        r[i] = min(r[i], x[i], 1 - x[i], y[i], 1 - y[i])
        r[i] = max(r[i], 0)
    
    # Enforce non-overlap by reducing radii iteratively
    for _ in range(50):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-12:
                    # Circles are at the same position; need to separate or shrink
                    # For now, just shrink both to 0
                    r[i] = 0
                    r[j] = 0
                    changed = True
                    continue
                max_allowed = (dist - 1e-7) / 2.0
                if max_allowed < 0:
                    max_allowed = 0
                current_sum = r[i] + r[j]
                if current_sum > dist - 1e-7:
                    # Scale down both radii proportionally
                    if current_sum > 1e-12:
                        scale = max(0, (dist - 1e-7)) / current_sum
                        r[i] *= scale
                        r[j] *= scale
                    else:
                        r[i] = 0
                        r[j] = 0
                    changed = True
        
        # Re-check boundary
        for i in range(N):
            r[i] = min(r[i], x[i], 1 - x[i], y[i], 1 - y[i])
            r[i] = max(r[i], 0)
        
        if not changed:
            break
    
    # Final validation
    for i in range(N):
        if x[i] < r[i] - 1e-6 or x[i] > 1.0 - r[i] + 1e-6:
            return None, False
        if y[i] < r[i] - 1e-6 or y[i] > 1.0 - r[i] + 1e-6:
            return None, False
        for j in range(i+1, N):
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if d < r[i] + r[j] - 1e-6:
                return None, False
    
    return np.concatenate([x, y, r]), True


def construct():
    """Pack N circles in unit square maximizing sum of radii."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    best_value = -np.inf
    best_v = None
    
    # Generate multiple initial guesses
    initial_guesses = []
    for seed in range(5):
        initial_guesses.append(_make_grid_initial(seed))
    for seed in range(3):
        initial_guesses.append(_make_hex_initial(seed))
    for seed in range(3):
        initial_guesses.append(_make_random_initial(seed))
    for seed in range(2):
        initial_guesses.append(_make_mixed_initial(seed))
    
    # Optimize from each initial guess
    for v0 in initial_guesses:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                _objective,
                v0,
                method='L-BFGS-B',
                jac=_objective_grad,
                bounds=[(0.0, 1.0)] * (2 * N) + [(0.001, 0.5)] * N,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            
            v = result.x
            repaired_v, feasible = _repair(v)
            
            if feasible:
                val = np.sum(repaired_v[2*N:])
                if val > best_value:
                    best_value = val
                    best_v = repaired_v
        except Exception:
            pass
    
    # If we have a feasible solution, try to polish it with Nelder-Mead
    if best_v is not None and time.time() < deadline:
        try:
            result_nm = minimize(
                _objective,
                best_v,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9}
            )
            
            v = result_nm.x
            repaired_v, feasible = _repair(v)
            
            if feasible:
                val = np.sum(repaired_v[2*N:])
                if val > best_value:
                    best_value = val
                    best_v = repaired_v
        except Exception:
            pass
    
    # If still no feasible solution, fall back to a simple grid
    if best_v is None:
        v0 = _make_grid_initial(0)
        repaired_v, feasible = _repair(v0)
        if feasible:
            best_v = repaired_v
            best_value = np.sum(repaired_v[2*N:])
        else:
            # Last resort: very small circles
            x = np.full(N, 0.5)
            y = np.full(N, 0.5)
            r = np.full(N, 0.01)
            # Spread them out a bit
            for i in range(N):
                angle = 2 * np.pi * i / N
                x[i] = 0.5 + 0.3 * np.cos(angle)
                y[i] = 0.5 + 0.3 * np.sin(angle)
            best_v = np.concatenate([x, y, r])
            best_value = np.sum(r)
    
    x = best_v[:N]
    y = best_v[N:2*N]
    r = best_v[2*N:]
    
    result = np.column_stack([x, y, r])
    return result


# Keep the original function signature
def construct():
    return _construct_impl()


def _construct_impl():
    """Actual implementation."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    best_value = -np.inf
    best_v = None
    
    # Generate multiple initial guesses
    initial_guesses = []
    for seed in range(5):
        initial_guesses.append(_make_grid_initial(seed))
    for seed in range(3):
        initial_guesses.append(_make_hex_initial(seed))
    for seed in range(3):
        initial_guesses.append(_make_random_initial(seed))
    for seed in range(2):
        initial_guesses.append(_make_mixed_initial(seed))
    
    # Optimize from each initial guess
    for v0 in initial_guesses:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                _objective,
                v0,
                method='L-BFGS-B',
                jac=_objective_grad,
                bounds=[(0.0, 1.0)] * (2 * N) + [(0.001, 0.5)] * N,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            
            v = result.x
            repaired_v, feasible = _repair(v)
            
            if feasible:
                val = np.sum(repaired_v[2*N:])
                if val > best_value:
                    best_value = val
                    best_v = repaired_v
        except Exception:
            pass
    
    # If we have a feasible solution, try to polish it with Nelder-Mead
    if best_v is not None and time.time() < deadline:
        try:
            result_nm = minimize(
                _objective,
                best_v,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9}
            )
            
            v = result_nm.x
            repaired_v, feasible = _repair(v)
            
            if feasible:
                val = np.sum(repaired_v[2*N:])
                if val > best_value:
                    best_value = val
                    best_v = repaired_v
        except Exception:
            pass
    
    # If still no feasible solution, fall back to a simple grid
    if best_v is None:
        v0 = _make_grid_initial(0)
        repaired_v, feasible = _repair(v0)
        if feasible:
            best_v = repaired_v
            best_value = np.sum(repaired_v[2*N:])
        else:
            # Last resort: very small circles
            x = np.full(N, 0.5)
            y = np.full(N, 0.5)
            r = np.full(N, 0.01)
            for i in range(N):
                angle = 2 * np.pi * i / N
                x[i] = 0.5 + 0.3 * np.cos(angle)
                y[i] = 0.5 + 0.3 * np.sin(angle)
            best_v = np.concatenate([x, y, r])
            best_value = np.sum(r)
    
    x = best_v[:N]
    y = best_v[N:2*N]
    r = best_v[2*N:]
    
    result = np.column_stack([x, y, r])
    return result


# The actual entry point
def construct():
    return _construct_impl()
