"""Combined program: Zadoff-Chu chirp family + smooth soft-max L-BFGS polish + discrete bit-flip refinement."""
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 400

# EVOLVE-BLOCK-START

def _autoconv_lin(x):
    """Linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    X = np.fft.rfft(x, fsize)
    c = np.fft.irfft(X * X, fsize)[:size]
    return c

def _metric_offdiag(x):
    """Compute max |c_k| for k != 0 divided by (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    if len(c) > 1:
        max_val = np.max(np.abs(c[1:]))
    else:
        max_val = 0.0
    return max_val / (s * s)

def _objective_smooth(x, temp=1.0):
    """Smooth approximation of the off-diagonal metric for optimization."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    c = _autoconv_lin(x)
    if len(c) > 1:
        abs_c = np.abs(c[1:])
    else:
        return 0.0
    max_c = np.max(abs_c)
    soft_max = max_c + temp * np.log(np.sum(np.exp((abs_c - max_c) / temp)))
    return soft_max / (s * s)

def _zadoff_chu_real(n, u, N_local):
    """Real part of Zadoff-Chu-like sequence: cos(pi * u * n * (n+1) / N)."""
    return np.cos(np.pi * u * n * (n + 1) / N_local)

def _chirp_family_score(u, n, N_local):
    """Score for a given chirp parameter u."""
    x = _zadoff_chu_real(n, u, N_local)
    return _metric_offdiag(x)

def _optimize_chirp_param(n, N_local, u_range=(-5.0, 5.0)):
    """Find best chirp parameter u via Brent's method on a grid + local refine."""
    # First, coarse grid search
    u_grid = np.linspace(u_range[0], u_range[1], 50)
    best_u = u_grid[0]
    best_score = float('inf')
    for u in u_grid:
        s = _chirp_family_score(u, n, N_local)
        if s < best_score:
            best_score = s
            best_u = u
    
    # Local refinement around best grid point
    delta = (u_range[1] - u_range[0]) / 50
    try:
        res = minimize_scalar(
            lambda u: _chirp_family_score(u, n, N_local),
            bounds=(best_u - delta, best_u + delta),
            method='bounded',
            options={'xatol': 1e-6}
        )
        if res.fun < best_score:
            best_u = res.x
            best_score = res.fun
    except Exception:
        pass
    
    return best_u, best_score

def _local_bit_flip(x0, max_passes=30, deadline=None):
    """Local optimization by flipping signs of individual elements."""
    x = x0.copy().astype(float)
    current_score = _metric_offdiag(x)
    if not np.isfinite(current_score):
        return x, current_score
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        idx = np.random.permutation(N)
        for j in idx:
            if deadline is not None and time.time() > deadline:
                break
            old_val = x[j]
            x[j] = -x[j]
            new_score = _metric_offdiag(x)
            if new_score < current_score - 1e-12:
                current_score = new_score
                improved = True
            else:
                x[j] = old_val
        if not improved:
            break
    return x, current_score

def _smooth_optimize(x0, deadline):
    """Two-stage L-BFGS-B with annealed temperature."""
    x = x0.copy().astype(float)
    s = np.sum(x)
    if abs(s) < 1e-6:
        return x0, _metric_offdiag(x0)
    x = x / s
    
    best_x = x.copy()
    best_val = _metric_offdiag(best_x)
    
    try:
        res1 = minimize(
            lambda x: _objective_smooth(x, temp=1.0),
            x,
            method='L-BFGS-B',
            options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        x_mid = res1.x
        s_mid = np.sum(x_mid)
        if abs(s_mid) > 1e-6:
            x_mid = x_mid / s_mid
        val_mid = _metric_offdiag(x_mid)
        if val_mid < best_val:
            best_val = val_mid
            best_x = x_mid.copy()
    except Exception:
        x_mid = x
    
    if time.time() < deadline:
        try:
            res2 = minimize(
                lambda x: _objective_smooth(x, temp=0.1),
                x_mid,
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            x_opt = res2.x
            s_opt = np.sum(x_opt)
            if abs(s_opt) > 1e-6:
                x_opt = x_opt / s_opt
            val_opt = _metric_offdiag(x_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_x = x_opt.copy()
        except Exception:
            pass
    
    return best_x, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = np.random.default_rng(42)
    n = np.arange(N)
    
    best_x = np.ones(N)
    best_val = _metric_offdiag(best_x)
    
    # Phase 1: Search the Zadoff-Chu chirp family
    candidates = []
    
    # Optimize chirp parameter
    u_best, s_chirp = _optimize_chirp_param(n, N, u_range=(-5.0, 5.0))
    x_chirp = _zadoff_chu_real(n, u_best, N)
    s_chirp = _metric_offdiag(x_chirp)
    if s_chirp < best_val:
        best_val = s_chirp
        best_x = x_chirp.copy()
    candidates.append(x_chirp.copy())
    
    # Try a few more chirp parameters with different u ranges
    for u_range in [(-1.0, 1.0), (-2.0, 2.0), (0.5, 3.0)]:
        if time.time() > deadline - 40:
            break
        u_b, _ = _optimize_chirp_param(n, N, u_range=u_range)
        x_c = _zadoff_chu_real(n, u_b, N)
        candidates.append(x_c.copy())
    
    # Phase 2: Add diverse initial candidates
    # Random binary with non-zero mean
    for _ in range(8):
        if time.time() > deadline - 30:
            break
        x = rng.choice([-1, 1], N)
        if abs(np.sum(x)) > 10:
            candidates.append(x.copy())
    
    # Multi-sine patterns
    for freq in [1, 2, 3, 5]:
        if time.time() > deadline - 30:
            break
        x = np.sin(2 * np.pi * freq * n / N)
        candidates.append(x.copy())
    
    # Block patterns
    for block_size in [10, 20, 50]:
        if time.time() > deadline - 30:
            break
        x = np.zeros(N)
        for i in range(N):
            if (i // block_size) % 2 == 0:
                x[i] = 1.0
            else:
                x[i] = -1.0
        x[0] += 0.1  # ensure non-zero sum
        candidates.append(x.copy())
    
    # Phase 3: Smooth optimize each candidate, then bit-flip polish
    for x0 in candidates:
        if time.time() > deadline - 20:
            break
        
        s0 = np.sum(x0)
        if abs(s0) < 1e-6:
            continue
        
        x_smooth, val_smooth = _smooth_optimize(x0, deadline - 10)
        
        # Bit-flip polish on the smoothed result
        x_final, val_final = _local_bit_flip(x_smooth, max_passes=20, deadline=deadline - 5)
        
        if val_final < best_val:
            best_val = val_final
            best_x = x_final.copy()
    
    return best_x

# EVOLVE-BLOCK-END
