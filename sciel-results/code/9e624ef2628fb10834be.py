import numpy as np
from scipy.optimize import minimize
import time

N = 400

def _autocorr_fft(x):
    """Compute linear autocorrelation of x using FFT."""
    # x is expected to be length N
    # Pad to 2N to avoid circular wrap-around
    n_fft = 2 * len(x)
    X = np.fft.fft(x, n=n_fft)
    # Power spectrum
    P = X * np.conj(X)
    # Inverse FFT to get circular autocorrelation
    R = np.fft.ifft(P).real
    # The first N elements of R correspond to lags 0 to N-1
    # R[0] is the DC term (sum x^2)
    # R[k] for k>0 is sum x[i] x[i+k]
    # Due to symmetry, R[-k] = R[k]
    # We return the vector of R(tau) for tau = 0, 1, ..., N-1
    return R[:len(x)]

def _objective_and_grad(x, beta=10.0):
    """
    Objective function for continuous relaxation.
    Minimizes the LogSumExp of the absolute values of off-peak autocorrelations.
    """
    # x is a flat array of length N
    R = _autocorr_fft(x)
    # R[0] is the main lobe. We ignore it.
    off_peak = np.abs(R[1:])
    
    # LSE approximation of max
    # max(off_peak) ~ 1/beta * log(sum(exp(beta * off_peak)))
    # To avoid overflow, subtract max
    m = np.max(off_peak)
    if m == 0:
        return 0.0, np.zeros_like(x)
    
    # Compute LSE
    val = (1.0 / beta) * (m + np.log(np.sum(np.exp(beta * (off_peak - m)))))
    
    # Gradient
    # d/dx LSE = 1/beta * d/dx log(sum exp(beta * |R_tau|))
    # = 1/beta * 1/sum_exp * sum exp(beta * |R_tau|) * beta * sign(R_tau) * dR_tau/dx
    # = sum_exp_weights * sign(R_tau) * dR_tau/dx
    # where weights are softmax(beta * |R_tau|)
    
    exp_terms = np.exp(beta * (off_peak - m))
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp  # These sum to 1
    
    # We need dR_tau/dx
    # R_tau = sum_{i=0}^{N-1-tau} x_i x_{i+tau}
    # dR_tau/dx_j = x_{j-tau} (if j-tau in range) + x_{j+tau} (if j+tau in range)
    # This is a convolution-like operation.
    # Instead of computing the full gradient analytically (which is O(N^2) if done naively for all tau),
    # we can use the fact that the gradient of the sum of weighted R_taus is a weighted sum of their gradients.
    
    # Let G be the gradient vector.
    # G_j = sum_{tau=1}^{N-1} weights[tau-1] * sign(R_tau) * (x_{j-tau} + x_{j+tau})
    # This can be computed efficiently.
    # Let S_tau = weights[tau-1] * sign(R_tau) for tau=1..N-1
    # Then G is the "autocorrelation" of x with the sequence S (with appropriate signs/shifts).
    
    # Construct S array of length N
    S = np.zeros(N)
    # R[tau] corresponds to index tau-1 in off_peak
    # sign(R[tau])
    signs = np.sign(R[1:])
    S[1:] = weights * signs
    
    # G_j = sum_{tau} S_tau * (x_{j-tau} + x_{j+tau})
    # This is equivalent to:
    # G = convolve(x, S) + convolve(x, S[::-1]) ?
    # Let's check indices.
    # Term x_{j-tau}: for fixed j, sum over tau where j-tau >= 0.
    # This is like (x * S)[j] if we define convolution appropriately.
    
    # Using FFT for gradient computation is also fast.
    # G = IFFT( FFT(x) * (FFT(S) + FFT(S[::-1])) ) ... roughly.
    # Actually, simpler:
    # G_j = sum_{k} x_k S_{j-k} + sum_{k} x_k S_{k-j}
    # This is the circular autocorrelation of x and S.
    
    # Let's just do it directly for N=400, it's fast enough.
    # But we are in a loop, so FFT is better.
    
    # Gradient of R_tau w.r.t x is the "cross-correlation" of x with delta(tau).
    # Summing weighted gradients:
    # G = sum_tau w_tau sign(R_tau) (shift(x, tau) + shift(x, -tau))
    
    # We can compute this via FFT:
    # G = IFFT( FFT(x) * (FFT(S) + FFT(S)[::-1]) ) ? No.
    
    # Let's use a simpler, slightly slower but robust method for gradient:
    # Since N=400, O(N^2) is 160,000 ops. Very fast.
    grad = np.zeros(N)
    for tau in range(1, N):
        w = S[tau]
        if w == 0: continue
        # dR_tau/dx_j = x_{j-tau} + x_{j+tau}
        # for j in 0..N-1
        # x_{j-tau} is non-zero if j >= tau
        # x_{j+tau} is non-zero if j <= N-1-tau
        
        # Vectorized update
        # Part 1: x_{j-tau}
        grad[tau:] += w * x[:-tau]
        # Part 2: x_{j+tau}
        grad[:N-tau] += w * x[tau:]
        
    return val, grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # 1. Continuous Optimization
    # Initialize with random signs
    x0 = np.random.choice([-1.0, 1.0], N)
    
    # Run L-BFGS-B
    try:
        res = minimize(
            _objective_and_grad, 
            x0, 
            jac=True, 
            method='L-BFGS-B', 
            bounds=[(-1.0, 1.0)] * N,
            options={'maxiter': 100, 'maxfun': 500, 'ftol': 1e-6, 'gtol': 1e-5}
        )
        x_opt = res.x
    except Exception:
        x_opt = x0
    
    # Round to binary
    x_bin = np.sign(x_opt)
    # Ensure no zeros (if any, assign random sign)
    x_bin[x_bin == 0] = 1.0
    
    # 2. Local Search Refinement
    # We want to minimize max |R(tau)| for tau != 0
    
    def get_psl(x):
        R = _autocorr_fft(x)
        return np.max(np.abs(R[1:]))
    
    current_x = x_bin.copy()
    current_psl = get_psl(current_x)
    
    # Local search loop
    # We try to flip bits that reduce the PSL.
    # To be efficient, we identify the lag with max PSL and try to fix it.
    
    while time.time() < deadline:
        R = _autocorr_fft(current_x)
        abs_R = np.abs(R[1:])
        max_idx = np.argmax(abs_R) + 1  # tau index
        max_val = abs_R[max_idx - 1]
        
        # If PSL is very low, break?
        if max_val < 20: 
            break
            
        # Determine which flips would reduce R[max_idx]
        # R[tau] = sum x_i x_{i+tau}
        # dR[tau]/dx_j = x_{j-tau} + x_{j+tau}
        # We want to change R[tau] towards 0.
        # If R[tau] > 0, we want to decrease it. So flip x_j if dR/dx_j > 0.
        # If R[tau] < 0, we want to increase it. So flip x_j if dR/dx_j < 0.
        
        sign_R = np.sign(R[max_idx])
        
        # Calculate derivative for all j
        # dR_j = x_{j-tau} + x_{j+tau}
        dR = np.zeros(N)
        tau = max_idx
        dR[tau:] += current_x[:-tau]
        dR[:N-tau] += current_x[tau:]
        
        # We want to flip x_j such that the change in R is in the desired direction.
        # Change in R if x_j -> -x_j is -2 * x_j * dR_j.
        # We want -2 * x_j * dR_j * sign(R) < 0  => x_j * dR_j * sign(R) > 0.
        # So we look for j where x_j * dR_j * sign_R > 0.
        
        candidates = np.where((current_x * dR * sign_R) > 0)[0]
        
        if len(candidates) == 0:
            # Try random flip
            j = np.random.randint(N)
        else:
            # Pick the candidate that gives the largest reduction in R[tau]
            # Reduction is proportional to x_j * dR_j * sign_R
            scores = current_x[candidates] * dR[candidates] * sign_R
            j = candidates[np.argmax(scores)]
            
        # Flip
        new_x = current_x.copy()
        new_x[j] *= -1
        
        # Check if this flip improves the GLOBAL PSL
        # This is expensive to check every time, but N=400 is small.
        new_psl = get_psl(new_x)
        
        if new_psl < current_psl:
            current_x = new_x
            current_psl = new_psl
        else:
            # If no improvement, try a few random flips to escape local minima
            for _ in range(5):
                j = np.random.randint(N)
                new_x = current_x.copy()
                new_x[j] *= -1
                new_psl = get_psl(new_x)
                if new_psl < current_psl:
                    current_x = new_x
                    current_psl = new_psl
                    break
                    
    return current_x
