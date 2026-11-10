import numpy as np
import time
from scipy.optimize import minimize

def _union_length_1d(X, c, N=8):
    """
    Compute the 1D union length of N intervals at height c.
    Interval i: [X[i] + c*i/N, X[i] + c*i/N + 1/N]
    Uses sweep-line: sort endpoints, track coverage count.
    """
    side = 1.0 / N
    starts = X + c * np.arange(N) / N
    ends = starts + side
    
    # Create events: (position, type) where type=+1 for start, -1 for end
    # To handle ties correctly: when start and end are at same position,
    # we process end first (so the interval ending there doesn't extend coverage)
    # Actually for union length, it doesn't matter for measure-zero points.
    # Let's just sort all 2N points with type info.
    
    events_x = np.empty(2 * N)
    events_x[:N] = starts
    events_x[N:] = ends
    
    # Sort by position; for ties, we need to be careful.
    # For union measure, ties have measure zero so it doesn't affect the result.
    order = np.argsort(events_x, kind='stable')
    sorted_pos = events_x[order]
    # Track which are starts (0..N-1 in events_x) vs ends (N..2N-1)
    is_start = np.zeros(2 * N, dtype=bool)
    is_start[:N] = True
    sorted_is_start = is_start[order]
    
    # Sweep
    total_len = 0.0
    count = 0
    prev_pos = sorted_pos[0]
    
    for idx in range(2 * N):
        if sorted_is_start[idx]:
            if count == 0:
                total_len += sorted_pos[idx] - prev_pos
            count += 1
        else:
            count -= 1
            if count == 0:
                total_len += sorted_pos[idx] - prev_pos
        prev_pos = sorted_pos[idx]
    
    return total_len

def _compute_volume(X, Y, N=8):
    """
    Compute V = ∫₀¹ Lₓ(c) · Lᵧ(c) dc using adaptive quadrature.
    """
    from scipy.integrate import quad
    
    def integrand(c):
        lx = _union_length_1d(X, c, N)
        ly = _union_length_1d(Y, c, N)
        return lx * ly
    
    result, _ = quad(integrand, 0.0, 1.0, limit=200, epsabs=1e-10, epsrel=1e-8)
    return result

def _objective(params, N=8):
    """
    Objective: volume of the union of 64 sheared tubes.
    params: 16 values = [X0..X7, Y0..Y7]
    """
    X = params[:N]
    Y = params[N:]
    return _compute_volume(X, Y, N)

def _objective_fast(params, N=8):
    """
    Faster objective using fixed quadrature with fewer points.
    The union length is piecewise linear in c (breakpoints where endpoints cross).
    Product Lx*Ly is piecewise quadratic. We can find breakpoints exactly.
    """
    X = params[:N]
    Y = params[N:]
    
    # Find all breakpoints in (0,1) where any endpoint ordering changes.
    # For 1D: breakpoints when X[i]+c*i/N = X[j]+c*j/N  or  X[i]+c*i/N+1/N = X[j]+c*j/N
    # i.e., c = N*(X[j]-X[i])/(i-j)  or  c = N*(X[j]+1/N-X[i])/(i-j)  etc.
    
    bp_x = set([0.0, 1.0])
    bp_y = set([0.0, 1.0])
    
    for i in range(N):
        for j in range(i + 1, N):
            denom = i - j
            # starts equal: X[i]+c*i/N = X[j]+c*j/N => c*(i-j)/N = X[j]-X[i]
            c1 = N * (X[j] - X[i]) / denom
            if 0.0 < c1 < 1.0:
                bp_x.add(round(c1, 12))
            # start of i = end of j: X[i]+c*i/N = X[j]+c*j/N+1/N
            c2 = N * (X[j] + 1.0/N - X[i]) / denom
            if 0.0 < c2 < 1.0:
                bp_x.add(round(c2, 12))
            # end of i = start of j
            c3 = N * (X[j] - X[i] - 1.0/N) / denom
            if 0.0 < c3 < 1.0:
                bp_x.add(round(c3, 12))
            # ends equal
            c4 = N * (X[j] + 1.0/N - X[i] - 1.0/N) / denom
            # same as c1, skip
            # Actually: X[i]+c*i/N+1/N = X[j]+c*j/N+1/N => same as c1
    
    for i in range(N):
        for j in range(i + 1, N):
            denom = i - j
            c1 = N * (Y[j] - Y[i]) / denom
            if 0.0 < c1 < 1.0:
                bp_y.add(round(c1, 12))
            c2 = N * (Y[j] + 1.0/N - Y[i]) / denom
            if 0.0 < c2 < 1.0:
                bp_y.add(round(c2, 12))
            c3 = N * (Y[j] - Y[i] - 1.0/N) / denom
            if 0.0 < c3 < 1.0:
                bp_y.add(round(c3, 12))
    
    # Merge breakpoints
    all_bp = sorted(bp_x | bp_y)
    
    # Evaluate Lx(c) * Ly(c) at midpoints of each interval, use trapezoidal
    # Actually, within each interval (no breakpoint), Lx is linear and Ly is linear,
    # so Lx*Ly is quadratic. We can integrate exactly using 3-point Gauss or
    # just use the values at endpoints + midpoint.
    
    if len(all_bp) < 2:
        return 1e10
    
    total = 0.0
    for k in range(len(all_bp) - 1):
        a = all_bp[k]
        b = all_bp[k + 1]
        if b - a < 1e-14:
            continue
        # Sample at a, (a+b)/2, b
        c1 = a
        c2 = 0.5 * (a + b)
        c3 = b
        
        f1 = _union_length_1d(X, c1, N) * _union_length_1d(Y, c1, N)
        f2 = _union_length_1d(X, c2, N) * _union_length_1d(Y, c2, N)
        f3 = _union_length_1d(X, c3, N) * _union_length_1d(Y, c3, N)
        
        # Simpson's rule for quadratic
        total += (b - a) / 6.0 * (f1 + 4*f2 + f3)
    
    return total

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    
    N = 8
    
    # Initial guess: spread positions to maximize overlap potential
    # Key insight: we want tubes to overlap as much as possible.
    # At c=0, tube (i,j) is at [X[i], X[i]+1/N] x [Y[j], Y[j]+1/N].
    # At c=1, it's at [X[i]+i/N, X[i]+i/N+1/N] x [Y[j]+j/N, Y[j]+j/N+1/N].
    # To maximize overlap, we want the X positions to be such that
    # at all heights, the intervals cover similar ranges.
    
    # Strategy: make X[i] decrease with i so that the shearing brings
    # later tubes back toward earlier ones.
    # X[i] = -i/N * alpha for some alpha > 0
    
    # Try several initializations
    initializations = []
    
    # 1. Linear shear to counteract
    alpha = 0.5
    X_lin = -np.arange(N) * alpha / N
    Y_lin = -np.arange(N) * alpha / N
    initializations.append(np.concatenate([X_lin, Y_lin]))
    
    # 2. Stronger shear
    alpha = 1.0
    X_lin = -np.arange(N) * alpha / N
    Y_lin = -np.arange(N) * alpha / N
    initializations.append(np.concatenate([X_lin, Y_lin]))
    
    # 3. Uniform spread (baseline)
    X_uni = np.linspace(-0.5, 0.5, N)
    Y_uni = np.linspace(-0.5, 0.5, N)
    initializations.append(np.concatenate([X_uni, Y_uni]))
    
    # 4. Random perturbations around linear
    rng_init = np.random.default_rng(42)
    for _ in range(3):
        X_r = -np.arange(N) * 0.5 / N + rng_init.normal(0, 0.1, N)
        Y_r = -np.arange(N) * 0.5 / N + rng_init.normal(0, 0.1, N)
        initializations.append(np.concatenate([X_r, Y_r]))
    
    # Evaluate all initializations with fast objective
    best_params = None
    best_val = float('inf')
    
    for p in initializations:
        try:
            val = _objective_fast(p, N)
            if val < best_val and np.all(np.isfinite(p)):
                best_val = val
                best_params = p.copy()
        except Exception:
            pass
    
    if best_params is None:
        # Fallback
        best_params = initializations[0]
        best_val = _objective_fast(best_params, N)
    
    # Now optimize with multiple methods and restarts
    methods = ['Nelder-Mead', 'Powell', 'COBYLA']
    all_results = [(best_params, best_val)]
    
    for method in methods:
        if time.time() > deadline - 30:
            break
        
        try:
            if method == 'Nelder-Mead':
                res = minimize(_objective_fast, best_params, args=(N,), method=method,
                              options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
            elif method == 'Powell':
                res = minimize(_objective_fast, best_params, args=(N,), method=method,
                              options={'maxiter': 1000, 'maxfev': 5000})
            elif method == 'COBYLA':
                res = minimize(_objective_fast, best_params, args=(N,), method=method,
                              options={'maxiter': 5000, 'rhobeg': 1e-3})
            
            if res.fun < best_val and np.all(np.isfinite(res.x)):
                best_val = res.fun
                best_params = res.x.copy()
            all_results.append((res.x, res.fun))
        except Exception:
            pass
    
    # Local restarts from best
    rng_restart = np.random.default_rng(123)
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        perturb_scale = 0.05 * (1 - r / num_restarts) + 0.01
        p_pert = best_params + rng_restart.normal(0, perturb_scale, 2 * N)
        
        # Quick evaluate
        try:
            val_pert = _objective_fast(p_pert, N)
            if val_pert < best_val:
                best_val = val_pert
                best_params = p_pert.copy()
        except Exception:
            pass
        
        # Optimize from perturbed point
        if time.time() > deadline - 15:
            break
        
        try:
            res = minimize(_objective_fast, p_pert, args=(N,), method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_val and np.all(np.isfinite(res.x)):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final refinement with high-precision objective
    if time.time() < deadline - 10:
        try:
            val_precise = _compute_volume(best_params[:N], best_params[N:], N)
            # Use precise value as reference but keep fast for optimization
        except Exception:
            pass
    
    # Construct output
    X_opt = best_params[:N]
    Y_opt = best_params[N:]
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = X_opt[i_flat]
    base_y = Y_opt[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    # Validate: finite and reasonable
    if not np.all(np.isfinite(positions)):
        # Fallback to linear shear
        X_fallback = -np.arange(N) * 0.5 / N
        Y_fallback = -np.arange(N) * 0.5 / N
        base_x = X_fallback[i_flat]
        base_y = Y_fallback[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
    
    return positions
