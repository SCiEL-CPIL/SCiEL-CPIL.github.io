"""Rewritten program for signed step functions minimizing max |autoconv| / (integral)^2."""
import numpy as np
import time
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

N = 400

# EVOLVE-BLOCK-START

def _autoconv(x):
    """Compute full autoconvolution of x with itself (linear, not circular)."""
    # For a step function of length N, autoconvolution has length 2N-1
    # Use FFT for efficiency
    n_fft = 2 * N
    X = np.fft.rfft(x, n=n_fft)
    ac = np.fft.irfft(X * X, n=n_fft)
    # The linear autoconvolution is the first 2N-1 samples (shifted)
    # Actually, for linear convolution of x with x, the result length is 2N-1
    # rfft with n_fft=2N gives circular convolution, which for n_fft >= 2N-1 equals linear
    return ac[:2 * N - 1]

def _score(x):
    """Compute the C3' score: max |autoconv| / (sum(x))^2"""
    integral = np.sum(x)
    if abs(integral) < 1e-10:
        return 1e10
    ac = _autoconv(x)
    max_abs = np.max(np.abs(ac))
    return max_abs / (integral * integral)

def _soft_max_abs_ac(x, temperature=1.0):
    """Soft approximation of max |autoconv| using log-sum-exp."""
    ac = _autoconv(x)
    abs_ac = np.abs(ac)
    # Scale for numerical stability
    max_val = np.max(abs_ac)
    if max_val < 1e-10:
        return 0.0
    scaled = abs_ac / temperature
    # Log-sum-exp trick
    m = np.max(scaled)
    lse = m + np.log(np.sum(np.exp(scaled - m)))
    return lse * temperature

def _objective(params, temperature=1.0):
    """Objective function: soft max |autoconv| / (sum)^2 with penalty for non-step values."""
    x = np.tanh(params)  # Map to (-1, 1)
    # Penalize values not close to ±1 (encourage step function)
    step_penalty = np.mean((1 - x**2)**2) * 0.1
    integral = np.sum(x)
    if abs(integral) < 1e-10:
        return 1e10
    soft_max = _soft_max_abs_ac(x, temperature)
    base_obj = soft_max / (integral * integral)
    return base_obj + step_penalty

def _construct_local_search(initial_x, max_iter=500, time_limit=100.0):
    """Local search by flipping signs of individual elements."""
    start_time = time.time()
    x = initial_x.copy().astype(float)
    current_score = _score(x)
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        improved = False
        # Try flipping each element
        for i in range(N):
            x[i] = -x[i]
            new_score = _score(x)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                x[i] = -x[i]  # revert
        if not improved:
            break
    return x, current_score

def _construct_gradient_based(initial_params, max_iter=200, time_limit=80.0):
    """Gradient-based optimization of phase/sign parameters."""
    start_time = time.time()
    
    def obj_and_grad(params):
        if time.time() - start_time > time_limit:
            return 1e10, np.zeros_like(params)
        val = _objective(params)
        # Numerical gradient
        eps = 1e-6
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            params_minus = params.copy()
            params_minus[i] -= eps
            grad[i] = (_objective(params_plus) - _objective(params_minus)) / (2 * eps)
        return val, grad
    
    best_x = np.tanh(initial_params).copy()
    best_score = _score(best_x)
    
    for temp in [1.0, 0.5, 0.1]:
        if time.time() - start_time > time_limit:
            break
        res = minimize(lambda p: _objective(p, temp), initial_params, 
                      method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-8})
        if time.time() - start_time > time_limit:
            break
        x_new = np.tanh(res.x)
        # Snap to ±1
        x_new = np.sign(x_new)
        x_new[x_new == 0] = 1
        score_new = _score(x_new)
        if score_new < best_score:
            best_score = score_new
            best_x = x_new.copy()
            initial_params = res.x.copy()
    
    return best_x, best_score

def construct():
    """Main entry point: construct signed step function minimizing C3'."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Start with random ±1 initialization
    np.random.seed(42)
    
    best_x = None
    best_score = np.inf
    
    # Multiple restarts with different strategies
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Random initial sign pattern
        if restart == 0:
            # Start with all ones (baseline)
            x0 = np.ones(N)
        elif restart == 1:
            # Alternating pattern
            x0 = np.sign(np.cos(np.arange(N) * np.pi / 2))
            x0[x0 == 0] = 1
        elif restart == 2:
            # Random
            x0 = np.random.choice([-1, 1], size=N)
        elif restart == 3:
            # Block pattern: blocks of 10
            block_len = 10
            n_blocks = N // block_len
            blocks = np.random.choice([-1, 1], size=n_blocks)
            x0 = np.repeat(blocks, block_len)
        else:
            # Random with some structure
            x0 = np.random.choice([-1, 1], size=N)
            # Apply some smoothing
            kernel = np.ones(5) / 5
            x_smooth = np.convolve(x0, kernel, mode='same')
            x0 = np.sign(x_smooth)
            x0[x0 == 0] = 1
        
        # Local search first (fast)
        if time.time() > deadline - 20:
            x_local, score_local = x0, _score(x0)
        else:
            x_local, score_local = _construct_local_search(x0, max_iter=200, 
                                                            time_limit=min(15, deadline - time.time() - 10))
        
        if score_local < best_score:
            best_score = score_local
            best_x = x_local.copy()
        
        # Gradient-based refinement if time allows
        if time.time() > deadline - 15:
            continue
        
        # Convert to parameter space
        params0 = np.arctanh(np.clip(x0, -0.99, 0.99))
        x_grad, score_grad = _construct_gradient_based(params0, max_iter=100,
                                                        time_limit=min(20, deadline - time.time() - 10))
        
        if score_grad < best_score:
            best_score = score_grad
            best_x = x_grad.copy()
    
    # Final polish: more iterations of local search on best solution
    if best_x is not None and time.time() < deadline - 5:
        x_polished, score_polished = _construct_local_search(best_x, max_iter=500,
                                                              time_limit=min(10, deadline - time.time() - 2))
        if score_polished < best_score:
            best_score = score_polished
            best_x = x_polished
    
    if best_x is None:
        best_x = np.ones(N)
    
    # Ensure output is a sequence of signed step heights
    return best_x.astype(float)

# EVOLVE-BLOCK-END
