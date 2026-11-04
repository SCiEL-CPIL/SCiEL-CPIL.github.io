"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))

def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))

def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _construct():
    deadline = time.time() + 145.0
    
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    rng = np.random.default_rng(42)
    
    # Strategy: Direct optimization of the magnitude spectrum |X|^2.
    # Key Insight: The autoconvolution is c = IFFT(|X|^2).
    # The metric is max |c_k| / (sum x)^2 = max |c_k| / P[0], where P = |X|^2.
    # We want to minimize this ratio.
    # This is equivalent to finding a power spectrum P such that its inverse FFT is "flat"
    # (small peak) relative to its DC component P[0].
    #
    # We parameterize P directly. P must be non-negative and Hermitian symmetric (P_k = P_{N-k}).
    # We can optimize the vector P (length N) subject to P >= 0.
    # The objective is f(P) = max_k | IFFT(P)_k | / P[0].
    # This is a non-convex problem due to the max and the ratio, but we can use gradient descent
    # on a smoothed version or direct numerical optimization.
    #
    # Let's use scipy.optimize.minimize on the log-magnitude or directly on P.
    # Since P must be positive, we can optimize log(P) or just P with bounds.
    # However, the max function is non-smooth. We can use a soft-max approximation:
    # soft_max(y) = log(sum(exp(y/tau))) * tau.
    # But we have abs. So soft_max_abs(y) = soft_max(|y|).
    #
    # Let's try a simpler approach:
    # 1. Initialize P to be flat (P_k = 1 for all k). This corresponds to x being an impulse (up to scaling).
    #    Wait, if P is flat, IFFT(P) is a delta at 0. So c_0 = 1, c_k = 0 for k != 0.
    #    max |c| = 1. P[0] = 1. Metric = 1.
    #    This is the worst case (relative to 1/N).
    #
    # 2. Initialize P to be a low-pass filter. E.g., P_k = 1 for k < K, 0 otherwise.
    #    This corresponds to a sinc-like x.
    #
    # Let's try to optimize P using gradient descent on the smoothed metric.
    
    def _metric_from_P(P):
        """Compute metric given power spectrum P (length N, real, non-negative)."""
        # Ensure P is symmetric for real x?
        # The input to IFFT for autoconvolution is P.
        # If P is real and symmetric (P[k] == P[N-k]), then c = IFFT(P) is real.
        # We enforce symmetry in the optimization.
        c = np.fft.irfft(P, _FFT_SIZE)
        c = c[:2 * N - 1]
        max_abs = np.max(np.abs(c))
        P0 = P[0]
        if P0 < 1e-12:
            return float('inf')
        return max_abs / P0

    def _soft_max_abs(y, tau=0.1):
        """Smooth approximation of max |y|."""
        # Use log-sum-exp trick for stability
        y_abs = np.abs(y)
        m = np.max(y_abs)
        if m == 0:
            return 0.0
        # Soft max of y_abs
        # log(sum(exp(y_abs/tau))) * tau
        # = (m + tau * log(sum(exp((y_abs - m)/tau))))
        shifted = (y_abs - m) / tau
        # Clip to avoid overflow in exp
        shifted = np.clip(shifted, -100, 100)
        return m + tau * np.log(np.sum(np.exp(shifted)))

    def _objective_P(log_P):
        """Objective function for optimization. log_P is the variable."""
        P = np.exp(log_P)
        # Enforce symmetry: P[k] = P[N-k]
        # We only optimize the first half + DC + Nyquist (if N even)
        # Actually, let's just use the full P and average with its reverse in the metric calculation?
        # No, better to parameterize only the independent components.
        # For now, let's just use the full vector and assume the optimizer keeps it symmetric?
        # Or, we can define the metric using the symmetrized P.
        P_sym = 0.5 * (P + P[::-1])
        # Ensure P[0] is positive (it is, exp is always positive)
        c = np.fft.irfft(P_sym, _FFT_SIZE)
        c = c[:2 * N - 1]
        max_abs = _soft_max_abs(c, tau=0.5)
        P0 = P_sym[0]
        return max_abs / P0

    def _objective_P_grad(log_P):
        """Gradient of the objective function."""
        P = np.exp(log_P)
        P_sym = 0.5 * (P + P[::-1])
        c = np.fft.irfft(P_sym, _FFT_SIZE)
        c = c[:2 * N - 1]
        
        # Gradient of soft_max_abs(c) w.r.t. c
        y_abs = np.abs(c)
        m = np.max(y_abs)
        if m == 0:
            grad_c = np.zeros_like(c)
        else:
            shifted = (y_abs - m) / 0.5
            shifted = np.clip(shifted, -100, 100)
            weights = np.exp(shifted) / np.sum(np.exp(shifted))
            # d/d|c| = weights
            # d|c|/dc = sign(c)
            grad_c = weights * np.sign(c)
        
        # Gradient of c w.r.t. P_sym
        # c = IFFT(P_sym)
        # This is a linear transformation.
        # We can compute the gradient via FFT.
        # d(c_k)/d(P_sym_j) = 1/_FFT_SIZE * exp(2j pi k j / _FFT_SIZE)
        # But c is real, P_sym is real.
        # Let's use the adjoint.
        # grad_P_sym = FFT(grad_c) ? No, IFFT is the inverse.
        # If c = IFFT(P), then P = FFT(c).
        # The Jacobian is the FFT matrix.
        # grad_P_sym = FFT(grad_c) (with appropriate scaling and conjugation).
        # Actually, for real signals, we can just compute:
        # grad_P_sym = np.fft.rfft(grad_c, _FFT_SIZE) * _FFT_SIZE # ?
        # Let's check: <grad_c, IFFT(P)> = <FFT(grad_c), P> (Parseval)
        # So grad_P_sym = np.fft.rfft(grad_c, _FFT_SIZE)
        # Note: grad_c is length 2N-1, padded to _FFT_SIZE.
        grad_c_padded = np.zeros(_FFT_SIZE)
        grad_c_padded[:2 * N - 1] = grad_c
        grad_P_sym = np.fft.rfft(grad_c_padded, _FFT_SIZE)
        
        # Objective is max_abs / P0
        # f = A / P0
        # df/dP0 = -A / P0^2
        # df/dP_j = (1/P0) * grad_A/dP_j
        # grad_A = grad_P_sym
        # So grad_f = (1/P0) * grad_P_sym - (A/P0^2) * e_0
        # where e_0 is 1 at index 0, 0 elsewhere.
        
        A = _soft_max_abs(c, tau=0.5)
        P0 = P_sym[0]
        
        grad_f_P = (1.0 / P0) * grad_P_sym
        grad_f_P[0] -= A / (P0 * P0)
        
        # Gradient w.r.t. P (before symmetrization)
        # P_sym = 0.5 * (P + P[::-1])
        # dP_sym_j / dP_k = 0.5 * (delta_jk + delta_{j, N-k})
        # grad_P_k = sum_j grad_f_P_j * dP_sym_j / dP_k
        # = 0.5 * (grad_f_P_k + grad_f_P_{N-k})
        grad_P = 0.5 * (grad_f_P + grad_f_P[::-1])
        
        # Gradient w.r.t. log_P
        # P = exp(log_P)
        # dP_k / dlog_P_k = P_k
        grad_log_P = grad_P * P
        
        return grad_log_P

    # Optimization
    # We need to initialize log_P.
    # Start with P = ones (impulse). log_P = zeros.
    # Or start with a low-pass P.
    
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        if i == 0:
            # Flat spectrum
            P_init = np.ones(N)
        elif i == 1:
            # Low-pass
            P_init = np.ones(N)
            P_init[N//4:] = 0.1
        elif i == 2:
            # High-pass
            P_init = np.ones(N)
            P_init[N//4:] = 1.0
            P_init[:N//4] = 0.1
            P_init[0] = 1.0 # Keep DC
        else:
            # Random
            P_init = rng.uniform(0.1, 1.0, size=N)
            P_init[0] = 1.0
        
        # Symmetrize
        P_init = 0.5 * (P_init + P_init[::-1])
        P_init[0] = np.max(P_init[0], 0.1)
        
        log_P_init = np.log(P_init)
        
        # Use BFGS or L-BFGS-B
        # L-BFGS-B is good for large N.
        try:
            res = opt.minimize(
                _objective_P,
                log_P_init,
                method='L-BFGS-B',
                jac=_objective_P_grad,
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8, 'disp': False}
            )
            log_P_opt = res.x
            P_opt = np.exp(log_P_opt)
            P_opt = 0.5 * (P_opt + P_opt[::-1])
            
            # Construct x from P_opt
            # We need X such that |X|^2 = P_opt.
            # We can choose X = sqrt(P_opt) * exp(i * theta).
            # Any theta works for the metric, but we need x to be real.
            # If X is Hermitian, x is real.
            # Let's choose theta = 0. Then X = sqrt(P_opt).
            # X must be Hermitian. sqrt(P_opt) is real and symmetric, so it is Hermitian.
            X_opt = np.sqrt(P_opt)
            x_opt = np.fft.irfft(X_opt, N)
            
            # Normalize x to sum = N (optional, metric is scale invariant)
            S = np.sum(x_opt)
            if abs(S) > 1e-6:
                x_opt = x_opt * (N / S)
            
            val_opt = _metric(x_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_x = x_opt
                
        except Exception as e:
            pass
            
    return best_x

def construct():
    return _construct()
# EVOLVE-BLOCK-END
