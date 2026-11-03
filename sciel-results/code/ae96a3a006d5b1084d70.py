"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _autoconv(x):
    """Compute autoconvolution of x (linear, full)."""
    # Use FFT for efficiency
    n = len(x)
    size = 2 * n - 1
    f = np.fft.rfft(x, size=size)
    ac = np.fft.irfft(f * np.conj(f), n=size)
    return ac

def _max_abs_autoconv(x):
    """Maximum absolute value of the autoconvolution."""
    ac = _autoconv(x)
    return np.max(np.abs(ac))

def _squared_integral(x):
    """(sum(x))^2"""
    return np.sum(x) ** 2

def _c3_prime(x):
    """C3' = max(|autoconv|) / (sum(x))^2, handling zero integral."""
    si = np.sum(x)
    if abs(si) < 1e-10:
        return np.inf
    return _max_abs_autoconv(x) / (si * si)

def _smooth_max_abs(x, beta=10.0):
    """Smooth surrogate for max(|autoconv|) using log-sum-exp."""
    ac = _autoconv(x)
    abs_ac = np.abs(ac)
    # Soft max via log-sum-exp
    m = np.max(abs_ac)
    if m < 1e-10:
        return 0.0
    # Use scaled log-sum-exp for stability
    scaled = abs_ac / m
    log_sum_exp = np.log(np.sum(np.exp(beta * (scaled - 1.0)))) / beta + 1.0
    return m * log_sum_exp

def _objective(x, beta=10.0):
    """Objective: C3' using smooth surrogate for max."""
    si = np.sum(x)
    if abs(si) < 1e-10:
        return 1e10
    sm = _smooth_max_abs(x, beta)
    return sm / (si * si)

def _gradient(x, beta=10.0):
    """Analytical gradient of the smooth objective."""
    si = np.sum(x)
    if abs(si) < 1e-10:
        return np.zeros_like(x)
    
    ac = _autoconv(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-10:
        return np.zeros_like(x)
    
    # Gradient of soft-max: softmax-like weights
    scaled = abs_ac / m
    exp_vals = np.exp(beta * (scaled - 1.0))
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # weights sum to 1
    
    # d(smooth_max)/d(ac_k) = weights[k] * sign(ac_k) * (1/m) * ... 
    # Actually: smooth_max = m * log(sum(exp(beta*(ac/m - 1)))) 
    # Let's use a simpler approach: gradient of sum(exp(beta*abs_ac))
    # d/dx of sum_k exp(beta * |ac_k|)
    
    # ac_k = sum_i x_i x_{k-i} (with appropriate indexing)
    # d(ac_k)/d(x_j) = x_{k-j} + x_{j-k} (for valid indices, accounting for symmetry)
    
    # Let's compute gradient numerically for robustness within the smooth framework
    # But since we have analytical insight, let's try:
    # grad_obj = (1/si^2) * d(sm)/dx - 2*sm/si^3 * d(si)/dx
    # d(si)/dx = 1 for all j
    
    # For d(sm)/dx_j where sm = sum_k exp(beta * |ac_k|) / sum_k exp(beta * |ac_k|) * m ... 
    # This is getting complex. Let's use a finite difference check or a simpler smooth objective.
    
    # Alternative: use the epigraph formulation with auxiliary t
    # Minimize t + lambda * sum(exp(x_i)) s.t. |ac_k| <= t for all k
    # This is not directly differentiable at kinks.
    
    # Let's use a simpler smooth objective: sum of squares of (|ac_k| - t)^2 for |ac_k| > t
    # Or just use the log-sum-exp properly.
    
    # Let me just return a numerical gradient for reliability in this context
    eps = 1e-5
    grad = np.zeros_like(x)
    f0 = _objective(x, beta)
    for j in range(len(x)):
        xp = x.copy()
        xp[j] += eps
        grad[j] = (_objective(xp, beta) - f0) / eps
    
    return grad

def construct():
    """Construct signed step sequence minimizing C3'."""
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Initial guess: Zadoff-Chu-like or random signs
    best_x = None
    best_score = np.inf
    
    # Try a few initializations
    initializations = []
    
    # 1. Random signs
    for seed in [42, 123, 456]:
        np.random.seed(seed)
        x = np.random.choice([-1.0, 1.0], size=N)
        initializations.append(x)
    
    # 2. Zadoff-Chu sequence (real part of complex ZC)
    # ZC sequence: x[n] = exp(-j * pi * u * n^2 / N) for N odd, or similar
    # For even N, use: x[n] = exp(-j * pi * u * n^2 / (N+1))
    for u in [1, 3, 5, 7]:
        n = np.arange(N)
        # Use N+1 to avoid periodicity issues for even N
        phase = -np.pi * u * n**2 / (N + 1)
        x_zc = np.real(np.exp(1j * phase))
        # Quantize to signs
        x_signs = np.sign(x_zc)
        x_signs[x_signs == 0] = 1.0
        initializations.append(x_signs)
    
    # 3. Alternating signs
    x_alt = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    initializations.append(x_alt)
    
    # 4. Block signs
    for block_size in [4, 8, 16]:
        x_block = np.zeros(N)
        for i in range(N):
            x_block[i] = 1.0 if (i // block_size) % 2 == 0 else -1.0
        initializations.append(x_block)
    
    # Evaluate all initializations
    for x0 in initializations:
        if time.time() > deadline:
            break
        score = _c3_prime(x0)
        if score < best_score:
            best_score = score
            best_x = x0.copy()
    
    # Local optimization from best initialization
    if best_x is not None and time.time() < deadline:
        # Use L-BFGS-B with bounds [-1, 1]
        # Project to signs after optimization
        try:
            result = minimize(
                _objective,
                best_x,
                method='L-BFGS-B',
                bounds=[(-1.0, 1.0)] * N,
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if time.time() < deadline:
                x_opt = result.x
                # Project to signs
                x_signs = np.sign(x_opt)
                x_signs[x_signs == 0] = 1.0
                score = _c3_prime(x_signs)
                if score < best_score:
                    best_score = score
                    best_x = x_signs.copy()
        except Exception:
            pass
    
    # Local search on sign flips
    if best_x is not None and time.time() < deadline:
        x = best_x.copy()
        current_score = _c3_prime(x)
        improved = True
        max_iterations = 200
        
        for it in range(max_iterations):
            if time.time() > deadline:
                break
            if not improved:
                break
            improved = False
            # Try flipping each position
            for j in range(N):
                if time.time() > deadline:
                    break
                x[j] = -x[j]
                new_score = _c3_prime(x)
                if new_score < current_score - 1e-10:
                    current_score = new_score
                    improved = True
                    best_x = x.copy()
                    best_score = new_score
                else:
                    x[j] = -x[j]  # revert
            if improved:
                x = best_x.copy()
    
    # Final polish: try more iterations if time permits
    if best_x is not None and time.time() < deadline - 10:
        x = best_x.copy()
        current_score = _c3_prime(x)
        # More aggressive local search
        for it in range(100):
            if time.time() > deadline - 5:
                break
            improved = False
            for j in range(N):
                if time.time() > deadline - 5:
                    break
                x[j] = -x[j]
                new_score = _c3_prime(x)
                if new_score < current_score - 1e-10:
                    current_score = new_score
                    improved = True
                    best_x = x.copy()
                    best_score = new_score
                else:
                    x[j] = -x[j]
            if not improved:
                break
    
    # Return the best signed step sequence
    if best_x is None:
        return np.ones(N)
    
    # Ensure output is a sequence of signed step heights (±1)
    result = np.sign(best_x)
    result[result == 0] = 1.0
    return result

# EVOLVE-BLOCK-END
