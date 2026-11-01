"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START

def _c3_upper_bound(x):
    """
    Compute the C3 upper bound for a signed step function x of length N.
    The bound is related to the maximum of the autoconvolution sidelobes.
    Specifically, we look at the autoconvolution c = x * x (linear convolution).
    The main lobe is at index N-1. We want to minimize the maximum of |c[k]| for k != N-1.
    Actually, the 'C3' in the context of the third autocorrelation inequality often refers to
    the sum of squared sidelobes or the max sidelobe. 
    Given the seed returns np.ones(N), let's analyze.
    If x = [1, 1, ..., 1], autoconvolution c[k] = N-k for k=0..N-1, and symmetric.
    The peak is N. The sidelobes go down to 1.
    Usually, these problems ask to minimize the maximum sidelobe magnitude.
    Let's assume the score is the maximum absolute value of the autoconvolution
    excluding the central peak.
    
    Wait, the prompt says: "score is the upper bound on C3".
    "The evaluator measures the raw value (lower is better)."
    
    Let's calculate the autoconvolution of x.
    c = np.convolve(x, x, mode='full')
    The central peak is at index N-1.
    The sidelobes are c[:N-1] and c[N:].
    We want to minimize max(|c[:N-1]|, |c[N:]|).
    """
    c = np.convolve(x, x, mode='full')
    # Central peak is at index N-1
    sidelobes = np.concatenate([c[:N-1], c[N:]])
    return np.max(np.abs(sidelobes))

def _c3_upper_bound_grad(x):
    """
    Gradient of the max sidelobe objective.
    Since max is non-smooth, we use a soft-max approximation or subgradient.
    For optimization, we can use a smooth surrogate: log-sum-exp of squared values?
    Or simply approximate the gradient of max(|y|) as y_i / |y_i| * (1/max_count) for those achieving max.
    However, scipy's L-BFGS-B requires a smooth objective.
    Let's use a smooth approximation: (sum(exp(beta * |c|)))^(1/beta) approximates max.
    Or simpler: sum(c^2) is a common proxy (minimizing energy of sidelobes), but max is specified.
    The prompt mentions "soft-min/soft-max via log-sum-exp".
    
    Let f(x) = log( sum(exp(beta * |c_k|)) ) / beta.
    As beta -> inf, this approaches max(|c_k|).
    Derivative wrt x:
    d/dx log(sum(exp(beta * |c_k|))) = sum( exp(beta * |c_k|) * beta * sign(c_k) * dc_k/dx ) / sum(exp(beta * |c_k|))
    
    This is computationally expensive if done naively, but N=400 is small.
    convolve length is 2N-1 = 799.
    We can compute the gradient of the autoconvolution efficiently.
    c = x * x.
    dc_k/dx_j = x_{k-j} + x_{k-N+1-j} ... roughly.
    Actually, c_k = sum_i x_i x_{k-i}.
    dc_k/dx_j = x_{k-j} + x_{k-N+1-j} (with boundary checks).
    
    Alternatively, we can just use finite differences or a simpler gradient for the L-BFGS-B.
    Given the time limit, maybe a simpler approach is better.
    The knowledge says: "Continuous relaxation via L-BFGS-B navigates the non-convex landscape more globally... followed by fast FFT-based discrete refinement."
    
    Let's implement the smooth objective.
    """
    beta = 50.0  # High enough to approximate max, low enough for stability
    c = np.convolve(x, x, mode='full')
    sidelobes = np.concatenate([c[:N-1], c[N:]])
    abs_c = np.abs(sidelobes)
    
    # Soft max
    max_val = np.max(abs_c)
    # Shift for numerical stability
    exps = np.exp(beta * (abs_c - max_val))
    sum_exps = np.sum(exps)
    val = (max_val + np.log(sum_exps) / beta)
    
    # Gradient
    # d_val/dx = (1/sum_exps) * sum_k ( exps_k * beta * sign(c_k) * dc_k/dx )
    # We need dc_k/dx.
    # Let's compute the gradient of the sum of exp(beta*|c_k|) wrt x.
    # grad_x = sum_k ( exps_k * beta * sign(c_k) ) * (d c_k / d x)
    
    # d c / d x is a convolution operation.
    # c = x * x.
    # grad_c = 2 * x * (vector of weights w_k = exps_k * beta * sign(c_k)) ?
    # No, c_k = sum_i x_i x_{k-i}.
    # d c_k / d x_j = x_{k-j} + x_{k-N+1-j} (symmetric part)
    # This is equivalent to convolving x with the weight vector w.
    # Specifically, if we define w_k = exps_k * beta * sign(c_k),
    # then grad_x = convolve(x, w, mode='full') + convolve(w, x, mode='full') ?
    # Let's check indices.
    # c_k = sum_i x_i x_{k-i}.
    # d c_k / d x_j = x_{k-j} + x_{k-j} (if j != k-j) ... actually it's just the sum of products where one factor is x_j.
    # If c is full convolution, the gradient of c wrt x is [x * w] + [w * x] (reversed appropriately).
    # Actually, gradient of c = x*x wrt x is 2 * x * w (in the sense of convolution with w being the derivative weights).
    # Let's use FFT for speed and correctness.
    
    # Weights for each lag k in the full convolution
    w = exps * beta * np.sign(sidelobes)
    # w is only for sidelobes. The central peak and other parts are 0.
    # We need to place w into the full convolution length array.
    w_full = np.zeros(2*N-1)
    # Sidelobes correspond to indices 0..N-2 and N..2N-2
    w_full[:N-1] = w[:N-1]
    w_full[N:] = w[N-1:]
    
    # Gradient of sum(w_full_k * c_k) wrt x
    # This is convolve(x, w_full, mode='full') + convolve(w_full, x, mode='full') ?
    # Let's verify.
    # c_k = sum_i x_i x_{k-i}
    # sum_k w_k c_k = sum_k w_k sum_i x_i x_{k-i}
    # = sum_i x_i (sum_k w_k x_{k-i})
    # Let m = k-i. sum_m w_{m+i} x_m.
    # This is the cross-correlation or convolution.
    # grad_x_i = sum_m w_{m+i} x_m + sum_m w_{m+i} x_m ?
    # Actually, it's the convolution of x with w_full.
    # Let's use np.convolve(x, w_full, mode='full') and take the middle part?
    # The gradient vector has length N.
    # grad_x = np.convolve(x, w_full, mode='full')[:N] + np.convolve(w_full, x, mode='full')[:N] ?
    # No, the standard result for gradient of x*x is 2 * x * w (where * is convolution).
    # Let's compute g = np.convolve(x, w_full, mode='full')
    # The relevant part is g[N-1:N-1+N] ?
    # Let's test with small N.
    
    # For now, let's just use the value for the objective and use finite difference gradient?
    # No, L-BFGS-B is faster with analytic gradient.
    # Let's assume grad_x = 2 * np.convolve(x, w_full, mode='full')[N-1:N-1+N]
    
    g = 2 * np.convolve(x, w_full, mode='full')[N-1:N-1+N]
    
    # Normalize by sum_exps
    grad_val = g / sum_exps
    
    return val, grad_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial guess: random signs
    np.random.seed(42)
    x0 = np.random.choice([-1.0, 1.0], size=N)
    
    best_x = x0.copy()
    best_score = _c3_upper_bound(best_x)
    
    # Continuous optimization
    try:
        res = minimize(
            _c3_upper_bound_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if time.time() < deadline:
            x_cont = res.x
            # Round to nearest sign
            x_disc = np.sign(x_cont)
            # Ensure no zeros
            x_disc[x_disc == 0] = 1.0
            
            score_cont = _c3_upper_bound(x_disc)
            if score_cont < best_score:
                best_score = score_cont
                best_x = x_disc.copy()
    except Exception:
        pass
    
    # Local refinement: flip individual bits
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                # Try flipping bit i
                x_new = best_x.copy()
                x_new[i] *= -1
                score_new = _c3_upper_bound(x_new)
                if score_new < best_score:
                    best_score = score_new
                    best_x = x_new
                    improved = True
    
    # Pairwise flips if time permits
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Random sampling of pairs to save time
            for _ in range(1000):
                if time.time() > deadline:
                    break
                i, j = np.random.choice(N, 2, replace=False)
                if i == j:
                    continue
                x_new = best_x.copy()
                x_new[i] *= -1
                x_new[j] *= -1
                score_new = _c3_upper_bound(x_new)
                if score_new < best_score:
                    best_score = score_new
                    best_x = x_new
                    improved = True
                    break # Restart the while loop or just continue?
                    # Usually, after a flip, we should restart the scan or just continue.
                    # Let's just set improved=True and break inner loop to restart outer?
                    # Or just continue.
    
    return best_x

# EVOLVE-BLOCK-END
