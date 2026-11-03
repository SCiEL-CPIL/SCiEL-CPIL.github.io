"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.fftpack import fft, ifft

def _autoconv(x):
    """Compute autoconvolution of x using FFT."""
    n = len(x)
    # Use a size that is a power of 2 for speed, at least 2*n
    nfft = 1 << (2 * n - 1).bit_length()
    F = fft(x, n=nfft)
    ac = ifft(F * F.conj())
    return ac[:n]

def _score(x):
    """Compute the ratio: max|autoconv| / (sum(x)^2).
    We want to minimize this."""
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return 1e10
    ac = _autoconv(x)
    max_abs = np.max(np.abs(ac))
    return max_abs / (integral * integral)

def _grad_score(x):
    """Approximate gradient of the score via finite differences or analytical if possible.
    For L-BFGS-B, we can use numerical gradient or provide analytic.
    Let's use numerical gradient for simplicity, but it might be slow.
    Actually, let's try to compute analytic gradient for better performance."""
    # Score S = max_i |ac_i| / (sum x)^2
    # This is complex due to max. Let's use subgradient or just numerical.
    # Given the time constraints, numerical gradient with central differences might be too slow for N=400.
    # Let's try a simpler approach: coordinate descent + random restarts, or use the fact that
    # we can minimize max|ac| subject to sum(x)=1 (normalization).
    # Actually, the score is homogeneous of degree 0? No.
    # If we scale x by c, ac scales by c^2, integral scales by c, so score scales by c^2/c^2 = 1.
    # So we can fix sum(x) = 1 (or any non-zero constant) and minimize max|ac|.
    # This simplifies the problem significantly.
    
    # Let's reformulate: fix sum(x) = 1. Minimize max_i |ac_i|.
    # This is a convex problem if we relax max to a quadratic or use SOCP, but we can use L-BFGS-B on a smooth approximation.
    # Or, we can minimize the sum of squares of ac? No, that's different.
    # Let's stick to minimizing max|ac| with constraint sum(x)=1.
    # We can use L-BFGS-B with a penalty for the constraint or use equality constraint.
    # L-BFGS-B supports bounds but not equality constraints directly.
    # We can fix one variable to satisfy the sum constraint, or use a penalty.
    # Let's fix x[0] = 1 - sum(x[1:]) and optimize x[1:].
    
    return None  # We'll use a different strategy below.

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    best_x = None
    best_score = np.inf
    
    # Strategy:
    # 1. Normalize so that sum(x) = 1.
    # 2. Minimize max|ac| using L-BFGS-B on a smooth approximation (e.g., log-sum-exp of |ac|) or just use numerical gradient on the max function (subgradient).
    # 3. Try multiple random initializations.
    
    # Smooth approximation of max|ac|: log(sum(exp(beta * ac)) + sum(exp(-beta * ac))) / beta
    # This is smooth and differentiable.
    
    def smooth_max_abs_ac(x):
        n = len(x)
        nfft = 1 << (2 * n - 1).bit_length()
        F = fft(x, n=nfft)
        ac = ifft(F * F.conj())[:n]
        # Smooth max of |ac|
        beta = 100.0  # High beta approximates max
        pos = np.exp(beta * ac)
        neg = np.exp(-beta * ac)
        # Avoid overflow
        max_pos = np.max(pos)
        max_neg = np.max(neg)
        log_sum_pos = np.log(np.sum(pos / max_pos)) + np.log(max_pos)
        log_sum_neg = np.log(np.sum(neg / max_neg)) + np.log(max_neg)
        log_sum_all = np.log(np.exp(log_sum_pos) + np.exp(log_sum_neg))
        return log_sum_all / beta
    
    def obj_and_grad(x):
        val = smooth_max_abs_ac(x)
        # Numerical gradient
        eps = 1e-5
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (smooth_max_abs_ac(x_plus) - smooth_max_abs_ac(x_minus)) / (2 * eps)
        return val, grad
    
    # Since numerical gradient is O(N) evaluations of FFT, and N=400, this is 400 FFTs per gradient step.
    # L-BFGS-B might take ~100 steps, so 40,000 FFTs. Each FFT is O(N log N) ~ 400*9 ~ 3600 ops. Total ~ 1.4e8 ops.
    # This should be fast enough in NumPy.
    
    # Let's try a few random initializations.
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization with sum 1
        x0 = np.random.randn(N)
        x0 = x0 / np.sum(x0)  # Normalize to sum 1
        # Ensure no zeros for numerical stability? Not necessary.
        
        # Optimize
        try:
            res = minimize(obj_and_grad, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5})
            x_opt = res.x
            # Normalize again to ensure sum 1
            x_opt = x_opt / np.sum(x_opt)
            score = _score(x_opt)
            if score < best_score:
                best_score = score
                best_x = x_opt.copy()
        except Exception:
            pass
    
    # If no optimization succeeded, return a default
    if best_x is None:
        # Try a simple pattern: alternating signs
        x = np.ones(N)
        x[::2] = -1.0
        # Normalize
        x = x / np.sum(x)
        best_x = x
        best_score = _score(x)
    
    # Final polish: coordinate descent on the original (non-smooth) max|ac| objective
    # with sum(x)=1 constraint.
    if time.time() < deadline - 10:
        x = best_x.copy()
        # Coordinate descent: flip or adjust individual coordinates
        # Since we want to minimize max|ac|, we can try to reduce the peak.
        # This is complex. Let's just return the L-BFGS-B result.
        pass
    
    # The score is max|ac| / (sum(x))^2. Since we normalized sum(x)=1, score = max|ac|.
    # But the task asks for the sequence of signed step heights. The score is the upper bound on C3'.
    # The evaluator measures the raw value (lower is better).
    # So we return the sequence x. The evaluator will compute the score.
    
    # Ensure the output is a sequence of signed step heights.
    # The seed returned np.ones(N). We return a numpy array.
    
    return best_x
# EVOLVE-BLOCK-END
