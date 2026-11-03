"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_penalties(state, w, h, beta):
    """Compute total penalty for given state.
    state: (3N,) array [x1,y1,r1, x2,y2,r2, ...]
    Returns: (penalty_value, gradient)
    """
    N = len(state) // 3
    xs = state[0::3]
    ys = state[1::3]
    rs = state[2::3]
    
    penalty = 0.0
    grad = np.zeros_like(state)
    
    # Overlap penalties
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = rs[i] + rs[j]
            if dist < min_dist:
                # Violation: dist < min_dist
                viol = min_dist - dist
                # Gradient of penalty = beta * 2 * viol * d(viol)/d(state)
                # d(viol)/d(xs[i]) = -dx/dist (when dist > 0)
                if dist > 1e-12:
                    g = beta * 2.0 * viol / dist
                    grad[3*i] += g * dx
                    grad[3*i+1] += g * dy
                    grad[3*i+2] += beta * 2.0 * viol  # d(viol)/d(rs[i]) = 1
                    grad[3*j] -= g * dx
                    grad[3*j+1] -= g * dy
                    grad[3*j+2] += beta * 2.0 * viol
                else:
                    # Circles coincident, push apart
                    grad[3*i+2] += beta * 2.0 * viol
                    grad[3*j+2] += beta * 2.0 * viol
                penalty += beta * viol * viol
    
    # Boundary penalties
    for i in range(N):
        # Left wall: xs[i] - rs[i] >= 0
        v = rs[i] - xs[i]
        if v > 0:
            penalty += beta * v * v
            grad[3*i] += beta * 2.0 * v      # d/d(xs[i]) of -v = -1, but we want to increase xs
            # Actually: v = rs[i] - xs[i], penalty = beta*v^2
            # d(penalty)/d(xs[i]) = beta * 2*v * d(v)/d(xs[i]) = beta*2*v*(-1) = -beta*2*v
            # d(penalty)/d(rs[i]) = beta * 2*v * d(v)/d(rs[i]) = beta*2*v*(1) = beta*2*v
            grad[3*i] = -beta * 2.0 * v
            grad[3*i+2] = beta * 2.0 * v
        
        # Right wall: (w - xs[i]) - rs[i] >= 0
        v = rs[i] + xs[i] - w
        if v > 0:
            penalty += beta * v * v
            grad[3*i] = beta * 2.0 * v       # d(v)/d(xs[i]) = 1
            grad[3*i+2] = beta * 2.0 * v     # d(v)/d(rs[i]) = 1
        
        # Bottom wall: ys[i] - rs[i] >= 0
        v = rs[i] - ys[i]
        if v > 0:
            penalty += beta * v * v
            grad[3*i+1] = -beta * 2.0 * v
            grad[3*i+2] = beta * 2.0 * v
        
        # Top wall: (h - ys[i]) - rs[i] >= 0
        v = rs[i] + ys[i] - h
        if v > 0:
            penalty += beta * v * v
            grad[3*i+1] = beta * 2.0 * v
            grad[3*i+2] = beta * 2.0 * v
    
    return penalty, grad


def _objective_and_grad(state, w, h, beta):
    """Objective: -sum(r) + penalty. Minimize this."""
    N = len(state) // 3
    rs = state[2::3]
    obj = -np.sum(rs)
    penalty, pgrad = _compute_penalties(state, w, h, beta)
    obj += penalty
    grad = -np.zeros_like(state)  # d(-sum(r))/d(state)
    grad[2::3] = -1.0             # d(-sum(r))/d(r_i) = -1
    grad += pgrad
    return obj, grad


def _optimize_for_aspect(w, h, timeout=30.0):
    """Optimize circle packing for given rectangle dimensions w x h (w+h=2)."""
    start_time = time.time()
    
    # Initialize in a grid pattern
    side = int(np.ceil(np.sqrt(N)))
    # Place circles in a grid within [0,w] x [0,h]
    cols = side
    rows = side
    init_state = np.zeros(3*N)
    for idx in range(N):
        i = idx % cols
        j = idx // cols
        x = (i + 0.5) * w / cols
        y = (j + 0.5) * h / rows
        r = min(w/cols, h/rows) * 0.45
        init_state[3*idx] = x
        init_state[3*idx+1] = y
        init_state[3*idx+2] = r
    
    # Multi-phase optimization with decreasing beta
    betas = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05]
    
    best_state = init_state.copy()
    best_val = -np.inf
    
    for beta in betas:
        if time.time() - start_time > timeout - 5:
            break
        
        def obj_grad(s):
            if time.time() - start_time > timeout - 2:
                # Return current value and stop
                N_local = len(s) // 3
                rs_local = s[2::3]
                return -np.sum(rs_local) + 1e10, np.zeros_like(s)
            return _objective_and_grad(s, w, h, beta)
        
        result = minimize(
            obj_grad,
            init_state,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'gtol': 1e-6, 'maxls': 50}
        )
        
        if result.success or result.fun < best_val:
            init_state = result.x
            # Compute actual sum of radii
            rs = init_state[2::3]
            val = np.sum(rs)
            if val > best_val:
                best_val = val
                best_state = init_state.copy()
    
    return best_state, best_val


def _validate(state, w, h, tol=1e-6):
    """Check if packing is valid: no overlaps, all inside rectangle."""
    N = len(state) // 3
    xs = state[0::3]
    ys = state[1::3]
    rs = state[2::3]
    
    # Check radii positive
    if np.any(rs <= 0):
        return False
    
    # Check boundary
    if np.any(xs < rs - tol) or np.any(xs > w - rs + tol):
        return False
    if np.any(ys < rs - tol) or np.any(ys > h - rs + tol):
        return False
    
    # Check overlaps
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    
    return True


def _fix_state(state, w, h):
    """Fix small violations by shrinking radii slightly."""
    N = len(state) // 3
    xs = state[0::3].copy()
    ys = state[1::3].copy()
    rs = state[2::3].copy()
    
    # Clamp to boundaries
    for i in range(N):
        rs[i] = min(rs[i], xs[i], w - xs[i], ys[i], h - ys[i])
    
    # Iteratively fix overlaps by reducing radii
    for _ in range(50):
        fixed = True
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rs[i] + rs[j]
                if dist < min_dist:
                    # Reduce both radii
                    excess = min_dist - dist
                    # Reduce each by half the excess, but not below 0
                    red_i = min(excess/2, rs[i]*0.5)
                    red_j = min(excess/2, rs[j]*0.5)
                    rs[i] -= red_i
                    rs[j] -= red_j
                    fixed = False
        
        # Also fix boundary again
        for i in range(N):
            rs[i] = min(rs[i], xs[i], w - xs[i], ys[i], h - ys[i])
        
        if fixed:
            break
    
    state_fixed = np.zeros(3*N)
    state_fixed[0::3] = xs
    state_fixed[1::3] = ys
    state_fixed[2::3] = rs
    return state_fixed


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_state = None
    best_val = 0.0
    best_w = 1.0
    
    # Try various aspect ratios
    # w + h = 2, so w in (0, 2)
    # Try w from 0.6 to 1.4 (h from 1.4 to 0.6)
    aspect_ratios = np.linspace(0.6, 1.4, 9)
    
    for w in aspect_ratios:
        if time.time() > deadline - 30:
            break
        h = 2.0 - w
        try:
            state, val = _optimize_for_aspect(w, h, timeout=25.0)
            if val > best_val:
                best_val = val
                best_state = state
                best_w = w
        except Exception:
            continue
    
    if best_state is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Validate and fix
    h = 2.0 - best_w
    if not _validate(best_state, best_w, h):
        best_state = _fix_state(best_state, best_w, h)
        if not _validate(best_state, best_w, h):
            # More aggressive fix
            best_state = _fix_state(best_state, best_w, h)
    
    # Final validation
    if not _validate(best_state, best_w, h, tol=1e-5):
        # Fallback to safe grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_state
# EVOLVE-BLOCK-END

