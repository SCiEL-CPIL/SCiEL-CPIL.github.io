"""
Structural change: formulate as an SDP via the Fejér–Riesz spectral representation
and solve the dual directly with scipy.optimize.linprog (HiGHS LP).
This avoids non-convex local minima of the smoothed max-overlap gradient approach.
"""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START

def _score(h):
    s = h.sum()
    if abs(s - N / 2.0) > 1e-8:
        return float('inf')
    if h.min() < -1e-10 or h.max() > 1.0 + 1e-10:
        return float('inf')
    inv_h = 1.0 - h
    H = np.fft.fft(h)
    IH = np.fft.fft(inv_h)
    corr = np.real(np.fft.ifft(H * np.conj(IH)))
    return float(np.max(corr))

def _normalize(h):
    h = h.copy()
    target = N / 2.0
    diff = target - h.sum()
    for _ in range(200):
        if abs(diff) < 1e-15:
            break
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
        diff = target - h.sum()
    diff = target - h.sum()
    if abs(diff) > 1e-15:
        if diff > 0:
            idxs = np.where(h < 1.0)[0]
            room = 1.0 - h[idxs]
            order = np.argsort(-room)
            for i in order:
                add = min(diff, room[i])
                h[idxs[i]] += add
                diff -= add
                if abs(diff) < 1e-15:
                    break
        else:
            idxs = np.where(h > 0.0)[0]
            order = np.argsort(h[idxs])
            for i in order:
                sub = min(-diff, h[idxs[i]])
                h[idxs[i]] -= sub
                diff += sub
                if abs(diff) < 1e-15:
                    break
    return h

def _solve_sdp():
    """
    Solve the SDP relaxation via the dual LP.
    Dual variables: t (scalar), u_k for k=1..N-1
    Minimize t subject to:
      1 + sum_k u_k cos(2 pi k n / N) >= 0  for all n
      sum_k u_k = 0
    """
    from scipy.optimize import linprog
    
    # Variables: [t, u_1, u_2, ..., u_{N-1}]
    n_vars = N  # 1 + (N-1)
    
    # Objective: minimize t
    c = np.zeros(n_vars)
    c[0] = 1.0
    
    # Inequality constraints: A_ub @ x <= b_ub
    # 1 + sum_k u_k cos(2 pi k n / N) >= 0  =>  -1 - sum_k u_k cos(...) <= 0
    A_ub = np.zeros((N, n_vars))
    b_ub = np.zeros(N)
    for n in range(N):
        A_ub[n, 0] = -1.0
        for k in range(1, N):
            A_ub[n, k] = -np.cos(2 * np.pi * k * n / N)
        b_ub[n] = 0.0
    
    # Equality constraint: sum_k u_k = 0  =>  [0, 1, 1, ..., 1] @ x = 0
    A_eq = np.zeros((1, n_vars))
    A_eq[0, 1:] = 1.0
    b_eq = np.zeros(1)
    
    bounds = [(None, None)] * n_vars
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs',
                  options={'presolve': True, 'time_limit': 100})
    
    if not res.success:
        return None
    
    t = res.x[0]
    u = res.x[1:]
    
    # Construct h from spectral mask:
    # h_hat[k] = (1 + sum_l u_l cos(2 pi l k / N)) / 2  (symmetric real)
    # But we need h such that the overlap spectral mask is exactly 1 + sum u_k cos(...)
    # The overlap at shift s has DFT: |H_k|^2 (1 - 2 Re(H_k conj(IH_k))) ... 
    # Actually, the overlap function o[s] = sum_i h[i] (1-h[(i+s)%N])
    # Its DFT: O_k = H_k * IH_k^* (with appropriate normalization)
    # We want max_s o[s] = t, which means the spectral mask 1 - O_k/N >= 0
    
    # From the dual, we get u_k such that the spectral mask is non-negative.
    # The optimal h corresponds to h_hat[k] that achieves this spectral mask.
    # For the Fejér-Riesz construction, we can set:
    # h_hat[k] = 0.5 * (1 + sum_l u_l cos(2 pi l k / N))
    # But this gives h in [0,1] only if the spectral mask is in [0,1]
    
    h_hat = np.zeros(N, dtype=complex)
    for k in range(N):
        val = 0.5
        for l in range(1, N):
            val += 0.5 * u[l-1] * np.cos(2 * np.pi * l * k / N)
        h_hat[k] = val
    
    h = np.real(np.fft.ifft(h_hat))
    h = np.clip(h, 0.0, 1.0)
    h = _normalize(h)
    
    return h, t

def _hill_climb(h, budget_s=5.0):
    """Local search: random pair perturbations preserving sum."""
    rng = np.random.default_rng(123)
    best = h.copy()
    best_sc = _score(best)
    deadline = time.time() + budget_s
    no_improve = 0
    while time.time() < deadline and no_improve < 300:
        i = rng.integers(0, N)
        j = rng.integers(0, N)
        if i == j:
            no_improve += 1
            continue
        eps = rng.uniform(0.001, 0.02)
        if best[i] + eps > 1.0 or best[j] - eps < 0.0:
            no_improve += 1
            continue
        h_new = best.copy()
        h_new[i] += eps
        h_new[j] -= eps
        sc = _score(h_new)
        if sc < best_sc:
            best = h_new
            best_sc = sc
            no_improve = 0
        else:
            no_improve += 1
    return best, best_sc

def construct():
    start = time.time()
    deadline = start + 140.0
    
    best_h = np.full(N, 0.5)
    best_sc = _score(best_h)
    
    # Try SDP relaxation
    try:
        sdp_res = _solve_sdp()
        if sdp_res is not None:
            h_sdp, t_sdp = sdp_res
            sc_sdp = _score(h_sdp)
            if sc_sdp < best_sc:
                best_h = h_sdp
                best_sc = sc_sdp
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        return _normalize(best_h)
    
    # Local search from SDP result
    if time.time() < deadline - 5:
        h_lc, sc_lc = _hill_climb(best_h, budget_s=min(20.0, deadline - time.time() - 5))
        if sc_lc < best_sc:
            best_h = h_lc
            best_sc = sc_lc
    
    # Try a few other starts
    rng = np.random.default_rng(7)
    if time.time() < deadline - 15:
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            h_rand = _normalize(rng.random(N))
            sc_r = _score(h_rand)
            if sc_r < best_sc:
                best_h = h_rand
                best_sc = sc_r
                h_lc, sc_lc = _hill_climb(best_h, budget_s=min(10.0, deadline - time.time() - 5))
                if sc_lc < best_sc:
                    best_h = h_lc
                    best_sc = sc_lc
    
    return _normalize(best_h)

# EVOLVE-BLOCK-END
