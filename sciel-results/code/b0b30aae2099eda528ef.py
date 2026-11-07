"""Optimized program for the second autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _compute_value(h):
    """Compute the objective value for given heights h."""
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0.0, None)
    if not np.all(np.isfinite(h)):
        return 0.0
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    
    l2 = (w / 3.0) * np.sum(y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf < 1e-30:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _soft_norm_inf(g, beta):
    """Compute soft max via log-sum-exp for numerical stability."""
    max_g = np.max(g)
    if max_g < 1e-30:
        return 1e-30
    shifted = beta * (g - max_g)
    log_sum_exp = max_g + (1.0/beta) * np.log(np.sum(np.exp(shifted)))
    return log_sum_exp

def _smooth_objective(h, beta):
    """
    Smooth objective: maximize l2 / (norm_1 * norm_inf_smooth)
    where norm_inf_smooth = (1/beta) * log(sum(exp(beta*g_i)))
    We minimize the negative of this.
    h is nonnegative.
    """
    h = np.maximum(h, 0.0)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    
    l2 = (w / 3.0) * np.sum(y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2)
    norm_1 = np.sum(g) / (m + 1)  # g >= 0 since h >= 0
    norm_inf_smooth = _soft_norm_inf(g, beta)
    
    if norm_1 * norm_inf_smooth < 1e-30:
        return 0.0
    
    val = l2 / (norm_1 * norm_inf_smooth)
    return -val  # minimize negative

def _gradient_smooth(h, beta):
    """
    Analytical gradient for the smooth objective using FFT-based convolution.
    This is much faster than finite differences.
    """
    h = np.maximum(h, 0.0)
    g = np.convolve(h, h)
    m = len(g)
    
    # Compute l2 and its gradient
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    
    # l2 = (w/3) * sum(y[i]^2 + y[i]*y[i+1] + y[i+1]^2) for i in 0..m
    # d(l2)/d(y[i]) for i in 0..m+1:
    #   For term y[i]^2: contributes to y[i]
    #   For term y[i]*y[i+1]: contributes to y[i] and y[i+1]
    #   For term y[i+1]^2: contributes to y[i+1]
    
    # Let's compute dy_dg first, then dval_dg, then dval_dh
    # y[1:m+1] = g, so dy/dg = I (identity) for indices 1..m
    
    # Compute d(l2)/d(y) for all y indices
    dl2_dy = np.zeros(m + 2)
    for i in range(m + 1):
        # Term: y[i]^2 -> 2*y[i]
        dl2_dy[i] += 2.0 * y[i]
        # Term: y[i]*y[i+1] -> y[i+1] for y[i], y[i] for y[i+1]
        dl2_dy[i] += y[i+1]
        dl2_dy[i+1] += y[i]
        # Term: y[i+1]^2 -> 2*y[i+1]
        dl2_dy[i+1] += 2.0 * y[i+1]
    
    dl2_dy *= w / 3.0
    
    # d(l2)/d(g[j]) for j in 0..m-1:
    #   g[j] = y[j+1]
    #   So d(l2)/d(g[j]) = dl2_dy[j+1]
    dl2_dg = dl2_dy[1:m+1].copy()
    
    # d(norm_1)/d(g[j]) = 1/(m+1) for all j
    dnorm1_dg = np.ones(m) / (m + 1)
    
    # d(norm_inf_smooth)/d(g[j]) = exp(beta * g[j]) / sum(exp(beta * g[k]))
    max_g = np.max(g)
    exp_shifted = np.exp(beta * (g - max_g))
    sum_exp = np.sum(exp_shifted)
    dnorminf_dg = exp_shifted / sum_exp
    
    # Now val = l2 / (norm_1 * norm_inf_smooth)
    # d(val)/d(g[j]) = [dl2_dg[j] * norm_1 * norm_inf_smooth - l2 * (dnorm1_dg[j] * norm_inf_smooth + norm_1 * dnorminf_dg[j])] / (norm_1 * norm_inf_smooth)^2
    
    norm_1 = np.sum(g) / (m + 1)
    norm_inf_smooth = _soft_norm_inf(g, beta)
    denom = (norm_1 * norm_inf_smooth) ** 2
    
    if denom < 1e-30:
        return np.zeros(N)
    
    l2 = (w / 3.0) * np.sum(y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2)
    
    dval_dg = (dl2_dg * norm_1 * norm_inf_smooth - l2 * (dnorm1_dg * norm_inf_smooth + norm_1 * dnorminf_dg)) / denom
    
    # d(val)/d(h[i]) = sum_j d(val)/d(g[j]) * d(g[j])/d(h[i])
    # g = convolve(h, h), so g[j] = sum_k h[k]*h[j-k]
    # d(g[j])/d(h[i]) = h[j-i] if 0 <= j-i < N, else 0
    # Equivalently, d(g)/d(h[i]) is the i-th row of the convolution matrix
    
    # Using FFT for efficiency:
    # g = ifft(fft(h)^2)
    # d(g[j])/d(h[i]) = h[j-i] (circular, but we use linear convolution)
    
    # For linear convolution: g[j] = sum_{k=max(0,j-N+1)}^{min(N-1,j)} h[k]*h[j-k]
    # d(g[j])/d(h[i]) = h[j-i] if i <= j and j-i < N, plus h[i-j] if j < i and i-j < N
    # Actually, by symmetry: d(g[j])/d(h[i]) = h[j-i] if 0 <= j-i < N, plus h[i-j] if 0 <= i-j < N
    # But since g is symmetric and h is the same in both factors:
    # d(g[j])/d(h[i]) = h[j-i] + h[i-j] if both indices valid, but actually it's just:
    # g[j] = sum_k h[k] * h[j-k]
    # d(g[j])/d(h[i]) = h[j-i] (from first factor) + h[i-j] (from second factor, when k=j-i in the sum for the other term)
    # Wait, let me be more careful.
    # g[j] = sum_{k=0}^{N-1} h[k] * h[j-k] (where h[j-k]=0 if j-k not in [0,N-1])
    # d(g[j])/d(h[i]) = sum_k [delta_{k,i} * h[j-k] + h[k] * delta_{j-k,i}]
    #                  = h[j-i] + h[i-j] (if both indices valid)
    # But since h is the same array, and the convolution is linear:
    # Actually, for linear convolution, if we write g = h * h (convolution), then:
    # d(g[j])/d(h[i]) = h[j-i] if 0 <= j-i < N (from differentiating the first h)
    #                 + h[i-j] if 0 <= i-j < N (from differentiating the second h)
    # But h[i-j] = h[|i-j|] only if we think of it as absolute, but actually it's h[i-j] which may be out of bounds.
    # Let me just compute it directly using the fact that:
    # d(g)/d(h[i]) is a vector where element j is h[j-i] + h[i-j] (with appropriate zero-padding)
    
    # More efficiently: d(g)/d(h[i]) = convolve(delta_i, h) + convolve(h, delta_i)
    # where delta_i is a vector with 1 at position i and 0 elsewhere
    # But convolve(delta_i, h) = h shifted by i, and convolve(h, delta_i) = h shifted by i (same thing for linear conv)
    # Actually, convolve(a, b) is not necessarily symmetric in the sense of shifts...
    # Let me just compute dval_dh directly:
    # dval_dh[i] = sum_j dval_dg[j] * d(g[j])/d(h[i])
    # d(g[j])/d(h[i]) = h[j-i] if 0 <= j-i < N, else 0, PLUS h[i-j] if 0 <= i-j < N, else 0
    # But since g[j] = sum_k h[k]h[j-k], and we're differentiating w.r.t. h[i]:
    # The term where k=i gives h[j-i]
    # The term where j-k=i, i.e., k=j-i, gives h[j-i] (same!)
    # Wait, no. Let me redo this.
    # g[j] = sum_{k=0}^{N-1} h[k] * h[j-k]
    # d(g[j])/d(h[i]) = sum_k [delta_{k,i} * h[j-k] + h[k] * delta_{j-k,i}]
    #                  = h[j-i] (from first term, when k=i)
    #                  + h[j-i] (from second term, when j-k=i, i.e., k=j-i)
    # But only if the indices are valid!
    # First term: valid if 0 <= j-i < N, i.e., i <= j < i+N
    # Second term: valid if 0 <= j-i < N, i.e., same condition
    # So d(g[j])/d(h[i]) = 2 * h[j-i] if i <= j < i+N, else 0?
    # No, that's not right either. Let me think again.
    # When k=i in the sum: the term is h[i] * h[j-i], derivative w.r.t. h[i] is h[j-i] (if j-i is valid)
    # When j-k=i in the sum, i.e., k=j-i: the term is h[j-i] * h[i], derivative w.r.t. h[i] is h[j-i] (if j-i is valid)
    # But these are the SAME term in the sum! When k=i and k=j-i, that means i = j-i, i.e., j=2i.
    # So in general, the two contributions are from different k values unless j=2i.
    # Let me just compute it numerically for correctness:
    
    dval_dh = np.zeros(N)
    for i in range(N):
        # d(g[j])/d(h[i]) for all j
        # g[j] = sum_k h[k] * h[j-k]
        # d(g[j])/d(h[i]) = h[j-i] (if 0<=j-i<N) + h[j-i] (if 0<=j-i<N) ... no
        # Let me just do: d(g[j])/d(h[i]) = sum_k (delta_{k,i}*h[j-k] + h[k]*delta_{k,j-i})
        # = h[j-i] (if 0<=j-i<N) + h[j-i] (if 0<=j-i<N)
        # Wait, delta_{k,j-i} means k=j-i, so the term is h[j-i]*h[i]... no
        # The sum is over k. For a fixed j and i:
        # Term 1: delta_{k,i} * h[j-k] -> this is nonzero only when k=i, giving h[j-i] (if j-i in [0,N-1])
        # Term 2: h[k] * delta_{j-k,i} -> this is nonzero only when j-k=i, i.e., k=j-i, giving h[j-i] (if j-i in [0,N-1])
        # So d(g[j])/d(h[i]) = h[j-i] + h[j-i] = 2*h[j-i] if 0<=j-i<N, else 0
        # But wait, that can't be right because when i=j-i (i.e., j=2i), we're double-counting the same term.
        # Actually no, in the sum sum_k h[k]*h[j-k], when k=i, we get h[i]*h[j-i].
        # When k=j-i, we get h[j-i]*h[i].
        # These are the same product, but they appear as separate terms in the sum unless i=j-i.
        # When i != j-i, they are two different terms in the sum, each contributing h[j-i] to the derivative.
        # When i = j-i (j=2i), it's the same term, and the derivative is h[i] (not 2*h[i]).
        # Hmm, this is getting complicated. Let me just use the fact that:
        # d(g)/d(h[i]) = convolve(h, e_i) + convolve(e_i, h) where e_i is unit vector
        # But for linear convolution, convolve(h, e_i)[j] = h[j-i] (if valid)
        # And convolve(e_i, h)[j] = h[j-i] (if valid)
        # So d(g[j])/d(h[i]) = h[j-i] + h[j-i] = 2*h[j-i] (if 0<=j-i<N)
        # But this double-counts when the same h value is used in both factors.
        # Actually, I think the correct formula is:
        # d(g[j])/d(h[i]) = h[j-i] if 0<=j-i<N (from first factor)
        #                  + h[i-j] if 0<=i-j<N (from second factor)
        # But h[i-j] = h[|i-j|] only if we consider the array indexed from 0.
        # Let me just compute it directly:
        for j in range(m):
            # d(g[j])/d(h[i])
            # From first factor: h[j-i] if 0 <= j-i < N
            # From second factor: h[i-j] if 0 <= i-j < N (but h is defined on [0,N-1], so h[i-j] means h at index i-j)
            # Actually, the second factor is h[j-k] where k is the summation index.
            # When we differentiate h[k] w.r.t. h[i], we get delta_{k,i}, so the term is h[j-i] if j-i is valid.
            # When we differentiate h[j-k] w.r.t. h[i], we get delta_{j-k,i} = delta_{k,j-i}, so the term is h[j-i] if j-i is valid.
            # So both give h[j-i] if 0 <= j-i < N.
            # But wait, that means d(g[j])/d(h[i]) = 2*h[j-i] if 0<=j-i<N, else 0.
            # Let me verify with a simple example: h = [a, b], g = [a^2, 2ab, b^2]
            # d(g[0])/d(a) = 2a. Formula: 2*h[0-0] = 2*h[0] = 2a. Correct.
            # d(g[1])/d(a) = 2b. Formula: 2*h[1-0] = 2*h[1] = 2b. Correct.
            # d(g[2])/d(a) = 0. Formula: 2*h[2-0] = 2*h[2], but h[2] is out of bounds, so 0. Correct.
            # d(g[1])/d(b) = 2a. Formula: 2*h[1-1] = 2*h[0] = 2a. Correct.
            # So the formula is: d(g[j])/d(h[i]) = 2*h[j-i] if 0<=j-i<N, else 0.
            
            if 0 <= j - i < N:
                dval_dh[i] += dval_dg[j] * 2.0 * h[j - i]
    
    return -dval_dh  # We want to minimize -val, so gradient is -dval/dh

def _generate_initializations():
    """Generate diverse initializations."""
    inits = []
    
    # Uniform
    inits.append(np.ones(N))
    
    # Spike at center
    for center in [N//4, N//2, 3*N//4]:
        h = np.zeros(N)
        h[center] = 1.0
        inits.append(h)
    
    # Gaussian bumps
    for center in [N//4, N//2, 3*N//4]:
        for width in [2, 5, 10]:
            x = np.arange(N)
            h = np.exp(-0.5 * ((x - center) / width) ** 2)
            inits.append(h)
    
    # Two bumps
    for c1, c2 in [(N//4, 3*N//4), (N//3, 2*N//3)]:
        for w1, w2 in [(3, 3), (5, 5), (2, 8)]:
            x = np.arange(N)
            h = np.exp(-0.5 * ((x - c1) / w1) ** 2) + np.exp(-0.5 * ((x - c2) / w2) ** 2)
            inits.append(h)
    
    # Random
    rng = np.random.RandomState(42)
    for _ in range(5):
        h = rng.rand(N)
        inits.append(h)
    
    return inits

def construct():
    global START_TIME, TIME_LIMIT
    
    best_h = np.ones(N, dtype=np.float64)
    best_val = _compute_value(best_h)
    
    # Generate diverse initializations
    inits = _generate_initializations()
    
    # Multi-stage optimization with increasing beta
    betas = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0]
    
    # For each initialization, run multi-stage optimization
    for init_h in inits:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        current_h = np.maximum(init_h, 0.0)
        
        for beta in betas:
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            
            def obj(x):
                return _smooth_objective(x, beta)
            
            def grad(x):
                return _gradient_smooth(x, beta)
            
            try:
                result = minimize(
                    obj,
                    current_h,
                    jac=grad,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 2000}
                )
                
                candidate = np.maximum(result.x, 0.0)
                val = _compute_value(candidate)
                if val > best_val:
                    best_val = val
                    best_h = candidate.copy()
                current_h = candidate
            except Exception:
                pass
            
            # Check time
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
    
    # Final high-precision polish with very high beta
    if time.time() - START_TIME < TIME_LIMIT - 5:
        beta_final = 1000.0
        
        def obj_final(x):
            return _smooth_objective(x, beta_final)
        
        def grad_final(x):
            return _gradient_smooth(x, beta_final)
        
        try:
            result = minimize(
                obj_final,
                best_h,
                jac=grad_final,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 5000}
            )
            
            candidate = np.maximum(result.x, 0.0)
            val = _compute_value(candidate)
            if val > best_val:
                best_val = val
                best_h = candidate.copy()
        except Exception:
            pass
    
    # Final polish: try a few random perturbations from best
    if time.time() - START_TIME < TIME_LIMIT - 2:
        rng = np.random.RandomState(123)
        for _ in range(20):
            if time.time() - START_TIME > TIME_LIMIT - 1:
                break
            perturbation = rng.randn(N) * 0.05
            candidate = np.maximum(best_h + perturbation, 0.0)
            val = _compute_value(candidate)
            if val > best_val:
                best_val = val
                best_h = candidate.copy()
    
    # Ensure nonnegative and finite
    best_h = np.maximum(best_h, 0.0)
    best_h = np.where(np.isfinite(best_h), best_h, 0.0)
    
    return list(best_h)

# Ensure the function is defined at module level for the evaluator
if __name__ == "__main__":
    result = construct()
    print(f"Value: {_compute_value(result)}")
