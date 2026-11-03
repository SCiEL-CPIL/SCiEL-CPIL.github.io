import time
import numpy as np
from scipy.optimize import minimize

N = 100
_T_START = time.time()
_DEADLINE = _T_START + 145  # Safety margin

def _compute_cs(h):
    """Compute C_k for all k using FFT-based linear autocorrelation."""
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs

def _score(h):
    h = np.asarray(h, dtype=np.float64)
    cs = _compute_cs(h)
    return float(np.max(cs))

def _project(h, target_sum):
    """Project h onto {x : 0 <= x_i <= 1, sum(x) = target_sum}."""
    h = np.asarray(h, dtype=np.float64).copy()
    low = -np.max(h) - 1.0
    high = 1.0 - np.min(h) + 1.0
    
    def sum_of_clipped(offset):
        return np.sum(np.clip(h + offset, 0.0, 1.0))
        
    for _ in range(50):
        mid = (low + high) / 2.0
        s_mid = sum_of_clipped(mid)
        if s_mid > target_sum:
            high = mid
        else:
            low = mid
            
    offset = (low + high) / 2.0
    h_proj = np.clip(h + offset, 0.0, 1.0)
    
    diff = target_sum - np.sum(h_proj)
    if abs(diff) > 1e-12:
        if diff > 0:
            idx = np.argmin(h_proj)
            h_proj[idx] += diff
        else:
            idx = np.argmax(h_proj)
            h_proj[idx] += diff
        h_proj = np.clip(h_proj, 0.0, 1.0)
    return h_proj

def _epigraph_obj_grad(z, beta=1.0):
    """
    Objective and gradient for epigraph formulation.
    z = [t, h_0, ..., h_{N-1}]
    Objective: t
    Constraints handled separately or via penalty? 
    SLSQP handles constraints. We provide obj and jac for the objective.
    """
    t = z[0]
    h = z[1:]
    return t, np.concatenate(([1.0], np.zeros(N)))

def _epigraph_constraints(z):
    """
    Returns list of constraint dicts.
    """
    t = z[0]
    h = z[1:]
    
    # Equality constraint: sum(h) = N/2
    cons_eq = {
        'type': 'eq',
        'fun': lambda x: np.sum(x[1:]) - N / 2.0,
        'jac': lambda x: np.concatenate(([0.0], np.ones(N)))
    }
    
    # Inequality constraints: t - C_k(h) >= 0
    # We compute cs first
    cs = _compute_cs(h)
    
    def ineq_fun(x):
        t_x = x[0]
        h_x = x[1:]
        cs_x = _compute_cs(h_x)
        return t_x - cs_x
        
    def ineq_jac(x):
        t_x = x[0]
        h_x = x[1:]
        # Compute gradients of C_k w.r.t h
        # C_k = (2/N)(N/2 - r_k)
        # dC_k/dh_j = -(2/N) dr_k/dh_j
        # dr_k/dh_j = h_{j-k} (if j>=k) + h_{j+k} (if j+k<N)
        
        jac_h = np.zeros((N-1, N)) # k=1 to N-1, j=0 to N-1
        # We can vectorize this
        # For each k, the gradient is a shifted h.
        # Let's just loop for clarity and speed if N is small
        for k in range(1, N):
            # d r_k / d h_j
            # term 1: h_{j-k} for j >= k
            # term 2: h_{j+k} for j < N-k
            grad = np.zeros(N)
            if k < N:
                grad[k:] = h_x[:N-k] # h_{j-k} where j=k..N-1 -> h_0..h_{N-k-1}
                grad[:N-k] += h_x[k:] # h_{j+k} where j=0..N-k-1 -> h_k..h_{N-1}
            # dC_k/dh_j = -(2/N) * grad
            jac_h[k-1, :] = -(2.0 / N) * grad
            
        # Full jac: [1, jac_h]
        # Shape: (N-1, N+1)
        jac_full = np.zeros((N-1, N+1))
        jac_full[:, 0] = 1.0
        jac_full[:, 1:] = jac_h
        return jac_full

    cons_ineq = {
        'type': 'ineq',
        'fun': ineq_fun,
        'jac': ineq_jac
    }
    
    return [cons_eq, cons_ineq]

def _run_epigraph_slsqp(h_init, maxiter=500):
    """
    Run SLSQP on the epigraph formulation.
    """
    h_init = _project(h_init, N/2.0)
    t_init = _score(h_init) + 0.1 # Start t slightly above current max
    
    z0 = np.concatenate(([t_init], h_init))
    
    # Bounds: t in [0, 2], h in [0, 1]
    bounds = [(0.0, 2.0)] + [(0.0, 1.0)] * N
    
    cons = _epigraph_constraints(z0)
    
    try:
        res = minimize(_epigraph_obj_grad, z0, jac=True, 
                       method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': 1e-9, 'disp': False})
        
        if res.success or res.fun < _score(h_init):
            h_opt = _project(res.x[1:], N/2.0)
            return h_opt
    except Exception as e:
        pass
        
    return h_init

def construct():
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _project(h, N / 2.0)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    i = np.arange(N)
    
    # Initial candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.full(N, 0.5))
    
    # 2. Alternating
    h_alt = np.zeros(N)
    h_alt[0::2] = 1.0
    h_alt[1::2] = 0.0
    candidates.append(h_alt)
    
    # 3. Cosine
    candidates.append(0.5 + 0.5 * np.cos(2.0 * np.pi * i / N))
    
    # 4. Sine
    candidates.append(0.5 + 0.5 * np.sin(2.0 * np.pi * i / N))
    
    # 5. Random
    for _ in range(10):
        candidates.append(np.random.rand(N))
        
    # 6. Spectral peaks (random frequencies)
    for _ in range(5):
        f = np.random.randint(1, N//2)
        candidates.append(0.5 + 0.5 * np.cos(2.0 * np.pi * f * i / N))

    for h0 in candidates:
        _update_best(h0)
        
    # Run Epigraph SLSQP from best candidates
    # We will run from the top 3 distinct initial scores
    scored_candidates = []
    seen_hashes = set()
    for h0 in candidates:
        h_proj = _project(h0, N/2.0)
        s = _score(h_proj)
        # Simple hash to avoid duplicates
        h_tuple = tuple(np.round(h_proj, 2))
        if h_tuple not in seen_hashes:
            seen_hashes.add(h_tuple)
            scored_candidates.append((s, h_proj))
            
    scored_candidates.sort(key=lambda x: x[0])
    
    # Run SLSQP on top 5
    for idx, (s, h0) in enumerate(scored_candidates[:5]):
        if time.time() > _DEADLINE - 10:
            break
        h_opt = _run_epigraph_slsqp(h0, maxiter=200)
        _update_best(h_opt)
        
    # If time remains, do a second pass with higher precision or more iterations on the best
    if best_h is not None and time.time() < _DEADLINE - 5:
        h_opt2 = _run_epigraph_slsqp(best_h, maxiter=300)
        _update_best(h_opt2)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _project(best_h, N / 2.0)
    return best_h
