"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import scipy
import numpy as np
import time

N = 600
# Interval width on [-1/4, 1/4]
h = (1/4 - (-1/4)) / N  # 1/2 / 600 = 1/1200

# EVOLVE-BLOCK-START
def _score(f):
    """Compute the normalized maximum of the autoconvolution."""
    s = np.sum(f) * h
    if s == 0:
        return np.inf
    c = np.convolve(f, f, mode='full')
    max_conv = np.max(c) * h
    return max_conv / (s * s)

def _softmax(x):
    """Stable softmax."""
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)

def _objective_and_grad(x, tau):
    """
    Compute the smooth max objective and its gradient with respect to x.
    f = softmax(x) is the probability distribution (sums to 1).
    We minimize max(h * convolve(f, f)).
    Surrogate: tau * log(sum(exp(h * convolve(f, f) / tau)))
    """
    f = _softmax(x)
    c = np.convolve(f, f, mode='full')
    z = h * c
    
    # Stable log-sum-exp
    z_max = np.max(z)
    log_sum_exp = z_max + np.log(np.sum(np.exp((z - z_max) / tau)))
    obj = tau * log_sum_exp
    
    # Gradient of obj w.r.t z
    p = np.exp((z - z_max) / tau) / np.sum(np.exp((z - z_max) / tau))
    
    # Gradient of z w.r.t f
    # z_k = h * sum_i f_i f_{k-i}
    # dz_k/df_j = h * (f_{k-j} + f_{j-k}) (with bounds)
    # This is equivalent to 2 * h * convolve(f, f) but careful with indices.
    # Actually, grad_f = h * (convolve(p, f) + convolve(f, p))
    # Since convolve is commutative, grad_f = 2 * h * convolve(p, f)
    # Wait, p is defined on the same grid as z (length 2N-1).
    # f is length N.
    # convolve(p, f) will have length 3N-2. We only need the part corresponding to f's indices?
    # No, grad_f has length N.
    # Let's compute grad_f directly.
    # grad_f_j = h * sum_k p_k (f_{k-j} + f_{j-k})
    # Term 1: sum_k p_k f_{k-j}. Let m = k-j => k = m+j.
    # Sum over k where k-j is in [0, N-1].
    # This is (p * f)_j?
    # (p * f)_j = sum_m p_m f_{j-m}.
    # If we reverse p, (p_rev * f)_j = sum_m p_{-m} f_{j-m} = sum_m p_m f_{j+m}?
    # Let's use np.convolve.
    # convolve(p, f, mode='same')? No, sizes are different.
    # Let's just compute it via FFT or explicit loop if N is small? N=600 is small.
    # But let's try to be vectorized.
    # grad_f_j = h * ( (p * f)_j + (f * p)_j )?
    # (p * f)_k = sum_i p_i f_{k-i}.
    # We want sum_k p_k f_{k-j}.
    # Let i = k-j => k = i+j.
    # sum_i p_{i+j} f_i.
    # This is the cross-correlation of p and f, or convolution with flipped p.
    # corr_p_f_j = sum_i p_{i+j} f_i = (p_rev * f)_j?
    # p_rev[k] = p[-k].
    # (p_rev * f)_j = sum_k p_rev[k] f_{j-k} = sum_k p_{-k} f_{j-k}.
    # Let m = -k => k = -m.
    # sum_m p_m f_{j+m}.
    # This is not quite sum_i p_{i+j} f_i.
    # sum_i p_{i+j} f_i is (p * f) with shift?
    # Actually, `np.convolve(p, f)` computes `sum_i p_i f_{k-i}`.
    # We want `sum_k p_k f_{k-j}`.
    # This is `np.correlate(p, f, mode='full')[j]`?
    # `np.correlate(a, b)[k] = sum_n a_{n+k} b_n`.
    # So `np.correlate(p, f, mode='full')[j] = sum_n p_{n+j} f_n`.
    # This matches our term!
    
    term1 = np.correlate(p, f, mode='full')
    # We need the central part corresponding to f indices 0..N-1.
    # The full correlation has length len(p) + len(f) - 1 = (2N-1) + N - 1 = 3N-2.
    # The indices of f are 0..N-1.
    # The correlation index j corresponds to sum_n p_{n+j} f_n.
    # For j=0, sum_n p_n f_n.
    # For j=N-1, sum_n p_{n+N-1} f_n.
    # The valid range of j for which the overlap is significant?
    # Actually, we just take the slice of `term1` that corresponds to the gradient of f.
    # The gradient `grad_f` has length N.
    # `term1` has length 3N-2.
    # The center of `term1` corresponds to j=0?
    # Let's check indices.
    # `np.correlate(p, f)` output index `k` is `sum_n p[n+k] f[n]`.
    # We want `grad_f[j]` to correspond to `sum_k p[k] f[k-j]`?
    # My previous derivation: `grad_f_j = h * sum_k p_k (f_{k-j} + f_{j-k})`.
    # Term 1: `sum_k p_k f_{k-j}`. Let `i = k-j`. `k = i+j`.
    # `sum_i p_{i+j} f_i`.
    # This is `np.correlate(p, f, mode='full')[j]`?
    # `np.correlate(p, f)[k] = sum_n p[n+k] f[n]`.
    # Yes, if we set `k=j`, we get `sum_n p[n+j] f[n]`.
    # So `term1[j]` is the value for `grad_f[j]`?
    # But `term1` has length 3N-2.
    # The valid `j` for `grad_f` are 0..N-1.
    # Does `term1[0..N-1]` give the correct values?
    # For `j=0`, `sum_n p[n] f[n]`.
    # For `j=N-1`, `sum_n p[n+N-1] f[n]`.
    # This seems correct.
    
    # However, `p` is length 2N-1. `f` is length N.
    # `np.correlate` with `mode='full'` returns length `len(p) + len(f) - 1`.
    # The slice `term1[0:N]` should correspond to `j=0` to `j=N-1`.
    
    term1_slice = term1[:N]
    
    # Term 2: `sum_k p_k f_{j-k}`.
    # Let `i = j-k` => `k = j-i`.
    # `sum_i p_{j-i} f_i`.
    # This is `np.correlate(p, f, mode='full')[-1 - j]`? Or similar.
    # Actually, `sum_k p_k f_{j-k}` is the convolution `(p * f)_j`.
    # `np.convolve(p, f, mode='full')[j] = sum_k p_k f_{j-k}`.
    # So `term2 = np.convolve(p, f, mode='full')`.
    # We need `term2[0:N]`?
    # `convolve(p, f)` has length `2N-1 + N - 1 = 3N-2`.
    # Index `j` in `0..N-1` corresponds to `sum_k p_k f_{j-k}`.
    # This is correct.
    
    term2 = np.convolve(p, f, mode='full')
    term2_slice = term2[:N]
    
    grad_f = h * (term1_slice + term2_slice)
    
    # Now chain rule to x
    # f = softmax(x)
    # df_i/dx_j = f_i (delta_ij - f_j)
    # grad_x_j = sum_i grad_f_i df_i/dx_j
    # = sum_i grad_f_i f_i (delta_ij - f_j)
    # = f_j sum_i grad_f_i f_i - f_j^2 sum_i grad_f_i? No.
    # = f_j (sum_i grad_f_i f_i) - f_j (sum_i grad_f_i f_i)?
    # Let's expand:
    # sum_i grad_f_i f_i delta_ij = grad_f_j f_j.
    # sum_i grad_f_i f_i (-f_j) = -f_j sum_i grad_f_i f_i.
    # So grad_x_j = f_j * (grad_f_j - sum_i grad_f_i f_i).
    
    dot_prod = np.dot(grad_f, f)
    grad_x = f * (grad_f - dot_prod)
    
    return obj, grad_x

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    np.random.seed(42)
    
    best_score = np.inf
    best_f = None
    
    # Try different taus
    taus = [0.1, 0.05, 0.01, 0.005]
    
    # Try different initializations
    inits = []
    # Uniform
    inits.append(np.zeros(N))
    # Gaussian bump
    x_grid = np.linspace(-1/4, 1/4, N, endpoint=False)
    inits.append(np.log(np.exp(-100 * x_grid**2) + 1e-10))
    # Random
    inits.append(np.random.randn(N))
    # Sparse random
    f_sparse = np.zeros(N)
    f_sparse[np.random.choice(N, 100, replace=False)] = 1.0
    inits.append(np.log(f_sparse + 1e-10))
    
    for tau in taus:
        for x0 in inits:
            if time.time() > deadline - 10:
                break
            
            x0 = x0.copy()
            # Normalize x0? Softmax is shift-invariant, so it's fine.
            
            try:
                res = scipy.optimize.minimize(
                    _objective_and_grad,
                    x0,
                    args=(tau,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 100, 'maxfun': 200, 'disp': False}
                )
            except Exception:
                continue
                
            if res.success:
                f = _softmax(res.x)
                score = _score(f)
                if score < best_score:
                    best_score = score
                    best_f = f.copy()
    
    if best_f is None:
        # Fallback to uniform
        best_f = np.ones(N) / N
        best_score = _score(best_f)
        
    return best_f.tolist()
# EVOLVE-BLOCK-END
