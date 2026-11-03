import numpy as np
from scipy.optimize import minimize
import time
import warnings

warnings.filterwarnings('ignore')

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

# Precompute the continuous autoconvolution kernel
# Step function on [-1/4, 1/4] with N equal intervals
# x_i = -1/4 + i/N * (1/2) for i=0..N
# dw = 1/(2*N) = 1/100... wait, interval is [-1/4, 1/4] so width = 1/2
# dw = (1/2) / N = 1/(2N) = 1/100

DW = 1.0 / (2 * N)  # width of each step
X = np.linspace(-0.25, 0.25, N + 1)  # N+1 edges

# For autoconvolution of step functions, we need to evaluate at many t points
# The autoconvolution is piecewise linear with breakpoints at X[i]+X[j] for all i,j
# So we sample t on a fine grid

# Number of t samples - need enough to capture the sup
N_T = 4 * N  # 200 samples over [-0.5, 0.5]
T_GRID = np.linspace(-0.5, 0.5, N_T + 1)
T_MID = (T_GRID[:-1] + T_GRID[1:]) / 2.0  # midpoints for integration... 
# Actually for sup, we just need to evaluate at points. The sup of a piecewise linear 
# function occurs at breakpoints or endpoints. But for optimization, a dense grid works.

# Precompute K: K[i, j, k] = overlap length between interval i and (interval j shifted by t_k)
# Interval i: [X[i], X[i+1]]
# Interval j shifted by t: [X[j]+t, X[j+1]+t]
# Overlap = max(0, min(X[i+1], X[j+1]+t) - max(X[i], X[j]+t))

# Vectorized computation
i_idx = np.arange(N)[:, None, None]  # (N,1,1)
j_idx = np.arange(N)[None, :, None]  # (1,N,1)
k_idx = np.arange(N_T)[None, None, :]  # (1,1,N_T)

X_i = X[i_idx]       # (N,1,1)
X_i1 = X[i_idx + 1]  # (N,1,1)
X_j = X[j_idx]       # (1,N,1)
X_j1 = X[j_idx + 1]  # (1,N,1)
T_k = T_MID[k_idx]   # (1,1,N_T)

lo = np.maximum(X_i, T_k - X_j1)  # (N,N,N_T)
hi = np.minimum(X_i1, T_k - X_j)  # (N,N,N_T)

K = np.maximum(0.0, hi - lo)  # (N,N,N_T)

# Flatten for matrix multiplication: autoconv[k] = sum_{i,j} h_i*h_j*K[i,j,k]
K_flat = K.reshape(N * N, N_T)  # (N^2, N_T)


def compute_score(h):
    """Compute L2^2 / (L1 * sup|autoconv|) for nonnegative step heights h."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    
    L1 = np.sum(h)
    if L1 < 1e-15:
        return 0.0
    
    L2_sq = np.dot(h, h)
    
    # Autoconvolution: autoconv[k] = sum_{i,j} h_i*h_j*K[i,j,k]
    h_outer = np.outer(h, h).flatten()  # (N^2,)
    autoconv = h_outer @ K_flat  # (N_T,)
    
    sup_conv = np.max(np.abs(autoconv))
    if sup_conv < 1e-15:
        return 0.0
    
    # The ratio L2^2/(L1*sup) is scale invariant, so DW cancels out
    # L2^2_actual = L2_sq * DW, L1_actual = L1 * DW, sup_actual = sup_conv * DW
    # ratio = (L2_sq*DW) / (L1*DW * sup_conv*DW) = L2_sq / (L1 * sup_conv * DW)
    # Wait, let me re-derive:
    # L1 = integral h dx = sum h_i * DW
    # L2^2 = integral h^2 dx = sum h_i^2 * DW
    # autoconv(t) = integral h(x)h(t-x) dx, which for step functions = sum_{i,j} h_i*h_j * overlap(i,j,t)
    # overlap has units of length, so autoconv has units of h^2 * length
    # sup|autoconv| has units of h^2 * length
    # ratio = (sum h_i^2 * DW) / ((sum h_i * DW) * sup|autoconv|)
    #       = (sum h_i^2) / (sum h_i * sup|autoconv| / DW)
    # Since K already gives the overlap length, sup_conv = sup|autoconv| has units of length
    # So ratio = L2_sq * DW / (L1 * DW * sup_conv) = L2_sq / (L1 * sup_conv)
    # The DW cancels! So we can just use L2_sq / (L1 * sup_conv)
    
    return L2_sq / (L1 * sup_conv)


def score_neg(h):
    return -compute_score(h)


def score_neg_smooth(h, beta=50.0):
    """Smooth surrogate using log-sum-exp for the sup norm."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    
    L1 = np.sum(h)
    if L1 < 1e-15:
        return 0.0
    
    L2_sq = np.dot(h, h)
    
    h_outer = np.outer(h, h).flatten()
    autoconv = h_outer @ K_flat
    
    abs_ac = np.abs(autoconv)
    max_val = np.max(abs_ac)
    if max_val < 1e-15:
        return 0.0
    
    # Log-sum-exp for smooth max
    shifted = beta * (abs_ac - max_val)
    # Clip for numerical stability
    shifted = np.clip(shifted, -500, 500)
    log_sum = max_val + np.log(np.sum(np.exp(shifted))) / beta
    
    if log_sum < 1e-15:
        return 0.0
    
    return -(L2_sq / (L1 * log_sum))


def coordinate_descent(h, max_rounds=20, deadline_check=None):
    """Coordinate descent polish."""
    h = np.asarray(h, dtype=float).copy()
    h = np.maximum(h, 0.0)
    best_val = compute_score(h)
    
    trial_values = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    
    for _round in range(max_rounds):
        if deadline_check and deadline_check():
            break
        improved = False
        for i in range(N):
            old = h[i]
            best_i_val = old
            best_i_score = best_val
            
            # Try a range of values
            candidates = set()
            candidates.add(0.0)
            candidates.add(old)
            for tv in trial_values:
                candidates.add(tv)
            # Relative perturbations
            if old > 0:
                for factor in [0.1, 0.3, 0.5, 0.7, 1.3, 1.5, 2.0, 3.0]:
                    candidates.add(old * factor)
            # Scale-aware: if L1 ~ 1, typical h ~ 1/N
            typical = 1.0 / N
            for tv in [typical, 2*typical, 5*typical, 10*typical]:
                candidates.add(tv)
            
            for cv in candidates:
                cv = max(0.0, cv)
                h[i] = cv
                v = compute_score(h)
                if v > best_i_score + 1e-12:
                    best_i_score = v
                    best_i_val = cv
            
            h[i] = best_i_val
            if best_i_val != old:
                improved = True
            if best_i_score > best_val:
                best_val = best_i_score
        
        if not improved:
            break
    
    return h, best_val


def construct():
    t0 = time.time()
    
    def elapsed():
        return time.time() - t0
    
    def time_left():
        return DEADLINE - elapsed()
    
    # Start with a good initial guess: two adjacent non-zero bins
    # For two adjacent bins with heights a, b:
    # autoconv is a trapezoid, sup at t=0
    # Let's start with a simple profile
    
    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_h[1] = 1.0
    best_val = compute_score(best_h)
    
    # Also try the uniform
    h_unif = np.ones(N)
    v_unif = compute_score(h_unif)
    if v_unif > best_val:
        best_val = v_unif
        best_h = h_unif
    
    # Try sparse: single bin
    h_single = np.zeros(N)
    h_single[0] = 1.0
    v_single = compute_score(h_single)
    if v_single > best_val:
        best_val = v_single
        best_h = h_single
    
    # Phase 1: Smooth surrogate optimization from multiple diverse starts
    starts = [
        best_h.copy(),
        np.array([1.0, 1.0] + [0.0]*(N-2)),
        np.array([1.0, 0.5, 0.5] + [0.0]*(N-3)),
        np.array([2.0, 1.0] + [0.0]*(N-2)),
        np.array([1.0, 2.0] + [0.0]*(N-2)),
        np.array([1.0, 1.0, 1.0] + [0.0]*(N-3)),
        np.ones(N) * 0.5,
        np.array([1.0] + [0.0]*(N-1)),
        # Concentrated in center
        np.array([0.0]*(N//2-2) + [1.0, 1.0, 1.0, 1.0] + [0.0]*(N - (N//2+2))),
        # Triangular
        np.array([min(1.0, abs(i - N/2 + 0.5)/(N/4)) for i in range(N)]),
    ]
    
    for s in starts:
        if time_left() < 30:
            break
        s = np.maximum(s, 0.0)
        # Normalize for numerical stability
        s_sum = np.sum(s)
        if s_sum > 0:
            s = s / s_sum * N  # scale so L1 ~ N, typical h ~ 1
        
        try:
            res = minimize(score_neg_smooth, s, method='L-BFGS-B',
                          bounds=[(0, 100)] * N,
                          args=(30.0,),
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            h_trial = np.maximum(res.x, 0.0)
            v_trial = compute_score(h_trial)
            if v_trial > best_val:
                best_val = v_trial
                best_h = h_trial.copy()
        except Exception:
            pass
    
    # Phase 2: Increase smoothness (higher beta)
    for beta in [80.0, 150.0, 300.0]:
        if time_left() < 20:
            break
        try:
            res = minimize(score_neg_smooth, best_h, method='L-BFGS-B',
                          bounds=[(0, 100)] * N,
                          args=(beta,),
                          options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-10})
            h_trial = np.maximum(res.x, 0.0)
            v_trial = compute_score(h_trial)
            if v_trial > best_val:
                best_val = v_trial
                best_h = h_trial.copy()
        except Exception:
            pass
    
    # Phase 3: Direct optimization with exact objective
    if time_left() > 15:
        try:
            res = minimize(score_neg, best_h, method='L-BFGS-B',
                          bounds=[(0, 100)] * N,
                          options={'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-10})
            h_trial = np.maximum(res.x, 0.0)
            v_trial = compute_score(h_trial)
            if v_trial > best_val:
                best_val = v_trial
                best_h = h_trial.copy()
        except Exception:
            pass
    
    # Phase 4: Coordinate descent polish
    if time_left() > 5:
        h_polished, v_polished = coordinate_descent(
            best_h, max_rounds=15,
            deadline_check=lambda: elapsed() > DEADLINE - 3
        )
        if v_polished > best_val:
            best_val = v_polished
            best_h = h_polished
    
    # Ensure non-negative
    best_h = np.maximum(best_h, 0.0)
    
    return list(best_h)
