import numpy as np
import time
from scipy.optimize import minimize

# Global state for time tracking
START_TIME = time.time()
TIME_LIMIT = 115  # Leave 5 seconds safety margin for 120s limit
N = 400

def _check_deadline():
    return (time.time() - START_TIME) > TIME_LIMIT

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    # Pad to length 2*N - 1 to avoid circular convolution effects
    # Next power of 2 for FFT efficiency
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective_exact(x):
    """Compute the exact objective: max(|autoconv|) / (sum x)^2"""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def _softmax_obj_grad(x, beta):
    """
    Compute the soft-max surrogate objective and its gradient.
    Objective: softmax_beta(|x*x|) / (sum x)^2
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(x)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    # Softmax of abs_ac
    max_val = np.max(abs_ac)
    if max_val < 1e-15:
        return 1e10, np.zeros_like(x)
    
    # Numerical stability: subtract max
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    
    val = soft_max / (s * s)
    
    # Gradient calculation
    # d(val)/dx = (1/s^2) * d(soft_max)/dx - (2*soft_max/s^3) * 1
    
    # d(soft_max)/dx_j = sum_k weights_k * sign(ac_k) * d(ac_k)/dx_j
    # d(ac_k)/dx_j = x[j+k] + x[j-k] (correlation terms)
    
    # Let g_k = weights_k * sign(ac_k)
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac)
    
    # We need sum_k g_k * (x[j+k] + x[j-k])
    # This is the cross-correlation of g and x.
    # term1_j = sum_k g_k x[j+k]
    # term2_j = sum_k g_k x[j-k]
    
    L_fft = 1
    while L_fft < 3*N: # Need enough length for correlation
        L_fft *= 2
        
    G = np.fft.rfft(g, n=L_fft)
    Xf = np.fft.rfft(x, n=L_fft)
    
    # Cross-correlation: ifft(conj(G) * Xf) gives sum g_k x_{k+j}
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
    
    # sum g_k x_{j-k} is ifft(G * Xf) reversed or similar
    # Actually, sum_k g_k x_{j-k} is the correlation of g and x.
    # Let's use the property: corr[g, x][j] = sum_k g_k x_{j-k}
    # In FFT: ifft(conj(G) * Xf) gives sum_k g_k x_{k+j} (which is term1)
    # ifft(G * conj(Xf)) ... no.
    # Let's just compute term2 directly via FFT as well.
    # sum_k g_k x_{j-k} = sum_k g_k x_{-(k-j)} = sum_k g_k x_{j-k}
    # This is the cross-correlation.
    # ifft(conj(G) * Xf) is sum_k g_k x_{k+j}.
    # ifft(G * Xf) is sum_k g_k x_{k-j} ? No, ifft(G * Xf) is convolution.
    
    # Let's stick to the convolution definition:
    # (g * x)[j] = sum_k g_k x_{j-k}
    # This is exactly term2.
    term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
    
    d_softmax_dx = term1 + term2
    
    grad = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max / (s ** 3)) * np.ones(N)
    
    return val, grad

def construct():
    best_x = np.ones(N)
    best_val = _objective_exact(best_x)
    
    if _check_deadline():
        return best_x

    # Phase 1: Diverse Initializations
    initializations = [
        np.ones(N),
        np.linspace(2.0, 1.0, N),
        np.linspace(1.0, 2.0, N),
        1.0 + 0.5 * np.cos(np.pi * np.linspace(0, 1, N)),
        1.0 + np.cos(np.pi * np.linspace(0, 1, N)),
        np.exp(-2.0 * np.linspace(0, 1, N)),
        np.exp(2.0 * np.linspace(0, 1, N) - 2.0),
    ]
    
    # Add random smooth signals
    rng = np.random.default_rng(42)
    for _ in range(10):
        freqs = rng.uniform(0.1, 3.0, size=5)
        phases = rng.uniform(0, 2*np.pi, size=5)
        amps = rng.uniform(0.1, 0.5, size=5)
        t = np.linspace(0, 1, N)
        sig = 1.0 + sum(a * np.cos(2*np.pi*f*t + p) for a, f, p in zip(amps, freqs, phases))
        initializations.append(sig)
        
    # Add signed initializations (alternating with noise)
    for _ in range(5):
        base = np.sign(np.sin(np.linspace(0, 5*np.pi, N)))
        noise = rng.normal(0, 0.5, N)
        sig = base + noise
        # Ensure non-zero sum
        if abs(np.sum(sig)) < 1e-6:
            sig[0] += 1.0
        initializations.append(sig)

    for x0 in initializations:
        if _check_deadline():
            break
        val = _objective_exact(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()

    if _check_deadline():
        return best_x

    # Phase 2: Optimization with increasing beta
    # Start with lower beta for smooth landscape, increase for sharpness
    betas = [10.0, 20.0, 50.0, 100.0]
    
    current_x = best_x.copy()
    
    for beta in betas:
        if _check_deadline():
            break
        
        def _obj(x):
            val, grad = _softmax_obj_grad(x, beta)
            return val, grad
            
        try:
            # Use L-BFGS-B
            res = minimize(_obj, current_x, method='L-BFGS-B', jac=True,
                           options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-12})
            if res.fun < best_val:
                # Verify with exact objective
                exact_val = _objective_exact(res.x)
                if exact_val < best_val:
                    best_val = exact_val
                    best_x = res.x.copy()
                current_x = res.x.copy()
            else:
                # If softmax objective didn't improve, keep current_x but check if exact did
                # (unlikely if softmax approximates well, but possible)
                pass
        except Exception:
            pass

    if _check_deadline():
        return best_x

    # Phase 3: Local Refinement with Exact Objective
    # Use a simple coordinate descent or Nelder-Mead on the exact objective
    # Since L-BFGS on softmax might get stuck in local minima of the surrogate,
    # a local search on the true objective can polish the result.
    
    # We'll use a simple random perturbation search (Simulated Annealing style)
    # or just Nelder-Mead if the dimensionality allows (400 is high, so NM is slow).
    # Let's try a few steps of coordinate descent with line search.
    
    def _coord_descent(x, max_iters=1000):
        x = x.copy()
        current_val = _objective_exact(x)
        step = 0.05
        for i in range(max_iters):
            if _check_deadline():
                break
            improved = False
            for j in range(N):
                # Try positive and negative step
                for sign in [-1, 1]:
                    trial = x.copy()
                    trial[j] += sign * step
                    val = _objective_exact(trial)
                    if val < current_val:
                        x = trial
                        current_val = val
                        improved = True
                        break
            if not improved:
                step *= 0.8
                if step < 1e-4:
                    break
        return x, current_val

    if not _check_deadline():
        best_x, best_val = _coord_descent(best_x, max_iters=500)

    return best_x
