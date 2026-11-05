"""Rewritten program for the third autocorrelation inequality task using band-limited signal optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 145.0  # wall-clock limit with safety margin
_start_time = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _start_time)


def _evaluate(h):
    """Compute the task value: abs(2 * N * max(convolve(h,h)) / sum(h)^2)."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _build_h(params):
    """
    Build h from parameters: h[n] = offset + sum_k amp[k] * cos(2*pi*freq[k]*n/N + phase[k])
    params layout: [offset, freq1, phase1, amp1, freq2, phase2, amp2, ...]
    Number of sinusoids K is inferred from len(params).
    """
    K = (len(params) - 1) // 3
    if K <= 0:
        return np.ones(N)
    
    offset = params[0]
    h = np.full(N, offset, dtype=np.float64)
    
    n = np.arange(N, dtype=np.float64)
    for k in range(K):
        idx = 1 + 3 * k
        freq = params[idx]
        phase = params[idx + 1]
        amp = params[idx + 2]
        h += amp * np.cos(2.0 * np.pi * freq * n / N + phase)
    
    return h


def _smooth_max(x, tau=0.05):
    """Smooth approximation of max(x) using logsumexp."""
    # Scale to avoid overflow
    m = np.max(x)
    if not np.isfinite(m):
        return m
    scaled = (x - m) / tau
    # Clip to avoid overflow in exp
    scaled = np.clip(scaled, -50, 50)
    return m + tau * np.log(np.sum(np.exp(scaled)))


def _objective(params, tau=0.05):
    """Objective: smooth approximation of the task value, plus regularization."""
    h = _build_h(params)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h)
    # Use smooth max as surrogate
    sm = _smooth_max(conv, tau)
    val = abs(2.0 * N * sm / (s * s))
    # Add small regularization to keep h well-behaved
    reg = 1e-8 * np.sum(h * h)
    return val + reg


def _objective_true(params):
    """True objective for final evaluation."""
    h = _build_h(params)
    return _evaluate(h)


def _optimize_one(rng, tau_start=0.1, tau_end=0.01, max_iter=500):
    """Run one optimization attempt with adaptive tau."""
    K = rng.integers(2, 6)  # 2 to 5 sinusoids
    dim = 1 + 3 * K
    
    # Random initialization
    offset = rng.uniform(-1, 1)
    params = np.empty(dim)
    params[0] = offset
    for k in range(K):
        params[1 + 3*k] = rng.uniform(0.5, 5.0)      # freq
        params[1 + 3*k + 1] = rng.uniform(0, 2*np.pi)  # phase
        params[1 + 3*k + 2] = rng.uniform(-1, 1)       # amp
    
    best_params = params.copy()
    best_val = _objective_true(params)
    
    if _time_left() < 5:
        return best_params, best_val
    
    # Adaptive tau: start large, decrease over iterations
    n_restarts = 3
    for restart in range(n_restarts):
        if _time_left() < 10:
            break
        
        # For later restarts, perturb around best
        if restart == 0:
            x0 = params.copy()
        else:
            x0 = best_params + rng.normal(0, 0.1, size=dim)
        
        # Run L-BFGS-B with adaptive tau
        current_tau = tau_start * (tau_end / tau_start) ** (restart / max(n_restarts - 1, 1))
        
        def obj_with_tau(p):
            return _objective(p, tau=current_tau)
        
        def grad_with_tau(p):
            # Numerical gradient
            eps = 1e-7
            g = np.zeros_like(p)
            for i in range(dim):
                p_plus = p.copy()
                p_plus[i] += eps
                p_minus = p.copy()
                p_minus[i] -= eps
                g[i] = (obj_with_tau(p_plus) - obj_with_tau(p_minus)) / (2 * eps)
            return g
        
        result = minimize(
            obj_with_tau, x0, method='L-BFGS-B',
            jac=grad_with_tau,
            options={'maxiter': max_iter // n_restarts, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        # Evaluate true objective
        true_val = _objective_true(result.x)
        if true_val < best_val:
            best_val = true_val
            best_params = result.x.copy()
    
    return best_params, best_val


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Seed: constant vector
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    rng = np.random.default_rng(12345)
    
    # Multiple optimization attempts with different numbers of sinusoids
    n_attempts = 15
    for attempt in range(n_attempts):
        if time.time() > deadline - 15:
            break
        
        params, val = _optimize_one(rng)
        if val < best_val:
            best_val = val
            best_h = _build_h(params)
        
        # Also try the raw optimized h without rebuilding (in case of numerical issues)
        h_candidate = _build_h(params)
        v_candidate = _evaluate(h_candidate)
        if v_candidate < best_val:
            best_val = v_candidate
            best_h = h_candidate
    
    # Final local refinement: coordinate descent on h directly
    if time.time() < deadline - 5:
        h = best_h.copy()
        for _ in range(200):
            if time.time() > deadline - 2:
                break
            improved = False
            for i in range(N):
                old = h[i]
                # Try small perturbations
                for delta in [-0.01, -0.005, 0.005, 0.01, -0.05, 0.05]:
                    h[i] = old + delta
                    v = _evaluate(h)
                    if v < best_val:
                        best_val = v
                        improved = True
                    else:
                        h[i] = old
                if improved:
                    best_h = h.copy()
                    improved = False
    
    return best_h
# EVOLVE-BLOCK-END
