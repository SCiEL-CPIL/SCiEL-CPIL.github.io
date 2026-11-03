import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 600
TIME_LIMIT = 145  # Safety margin for 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def smooth_max_conv(f, beta=50.0):
    """
    Approximate max(conv) using log-sum-exp.
    Returns (value, gradient)
    """
    conv = np.convolve(f, f, mode='full')
    # Scale to avoid overflow in exp
    max_c = np.max(conv)
    if max_c < 1e-12:
        return 1e-12, np.zeros(N)
    
    # log-sum-exp trick
    # LSE = max_c + log(sum(exp((conv - max_c)/beta)))
    # We want d/d f_j of LSE.
    # Let u = (conv - max_c) / beta
    # LSE = max_c + beta * log(sum(exp(u)))
    # d LSE / d f_j = beta * (sum_k exp(u_k) * d u_k / d f_j) / sum_k exp(u_k)
    # d u_k / d f_j = (1/beta) * d conv_k / d f_j
    # So d LSE / d f_j = (sum_k exp(u_k) * d conv_k / d f_j) / sum_k exp(u_k)
    
    u = (conv - max_c) / beta
    exp_u = np.exp(u)
    sum_exp = np.sum(exp_u)
    
    # Weights for each lag k in the max approximation
    weights = exp_u / sum_exp
    
    # Gradient of conv wrt f
    # conv[k] = sum_i f[i] f[k-i]
    # d conv[k] / d f[j] = f[k-j] + f[j-k] (with bounds)
    # We need sum_k weights[k] * (f[k-j] + f[j-k])
    # This is a convolution of weights with f, plus reverse.
    
    # Let g = weights
    # term1 = sum_k g[k] f[k-j] -> this is (g * f)[j] in circular conv? No, linear.
    # Actually, let's just compute it efficiently.
    # grad[j] = sum_k weights[k] * (f[k-j] + f[j-k])
    # Note that f[k-j] is non-zero only if 0 <= k-j < N.
    # This is equivalent to the cross-correlation of f with weights.
    
    # Using FFT for speed since N=600 is small but we call this many times.
    # Actually, N=600 direct loop is 600*1200 ~ 720k ops, which is fast in numpy if vectorized.
    # But let's try to vectorize.
    
    # d conv_k / d f_j = f[k-j] (if k>=j) + f[j-k] (if j>=k)
    # Let's create a matrix M where M[k,j] = f[k-j] + f[j-k]
    # This is too big (1200x600).
    
    # Alternative:
    # grad[j] = sum_{k} weights[k] f[k-j] + sum_{k} weights[k] f[j-k]
    # Let A[j] = sum_{k} weights[k] f[k-j]
    # Let B[j] = sum_{k} weights[k] f[j-k]
    # Note B[j] is just A reversed? No.
    # A[j] is the correlation of f with weights.
    # A[j] = sum_k weights[k] f[k-j]. Let m = k-j. k = j+m.
    # A[j] = sum_m weights[j+m] f[m].
    # This is the cross-correlation of weights and f.
    
    # We can compute this using np.correlate or FFT.
    # np.correlate(weights, f, mode='full') gives sum_k weights[k] f[k + n - offset]...
    # Let's use FFT for robustness and speed.
    size = 1024
    F_f = np.fft.rfft(f, n=size)
    F_w = np.fft.rfft(weights, n=size)
    
    # Correlation C[j] = sum_k w[k] f[k-j]
    # In frequency domain: IFFT( F_w * conj(F_f) )? 
    # Let's check:
    # corr = ifft(fft(w) * conj(fft(f)))
    # This gives sum_k w[k] f[k-j] (circular).
    # Since supports are small enough, linear approx holds if size is large.
    
    corr = np.fft.irfft(F_w * np.conj(F_f), n=size)
    # corr[j] corresponds to sum_k w[k] f[k-j] ?
    # Standard definition: (w * f)[n] = sum_k w[k] f[n-k]
    # Correlation: <w, f>_n = sum_k w[k] f[k-n]
    # Let's verify indices.
    # We want grad[j] = sum_k w[k] (f[k-j] + f[j-k])
    # Term 1: sum_k w[k] f[k-j]
    # Term 2: sum_k w[k] f[j-k]
    
    # Let's compute term1 and term2 separately to be safe.
    # term1[j] = sum_k w[k] f[k-j]
    # term2[j] = sum_k w[k] f[j-k]
    
    # Using numpy correlate:
    # np.correlate(a, v, mode) computes sum_i a[i] v[i-n]
    # So np.correlate(weights, f, 'full')[n] = sum_i weights[i] f[i-n]
    # We want sum_k weights[k] f[k-j]. This matches with n=j.
    # So term1 = np.correlate(weights, f, 'full')[offset + j]
    
    # For term2: sum_k weights[k] f[j-k]
    # Let m = j-k => k = j-m.
    # sum_m weights[j-m] f[m]
    # This is sum_m f[m] weights[j-m]
    # This is the same as term1 if weights is symmetric? No.
    # It is the correlation of f and weights.
    # np.correlate(f, weights, 'full')[n] = sum_i f[i] weights[i-n]
    # We want sum_m f[m] weights[j-m]. This matches with n=j.
    # So term2 = np.correlate(f, weights, 'full')[offset + j]
    
    # Wait, np.correlate(a, v) is sum a[i] v[i-n].
    # term1: sum_k w[k] f[k-j]. Let i=k. sum_i w[i] f[i-j]. This is correlate(w, f) at index j?
    # np.correlate(w, f, 'full') has length len(w)+len(f)-1.
    # Index n in 'full' output corresponds to shift n - (len(f)-1) ?
    # Documentation: c[n] = sum_m a[m] v[m-n] for mode='full'?
    # Actually, numpy correlate is:
    # c[n] = sum_m a[m] * v[m - n] for 0 <= n < len(a) + len(v) - 1 ?
    # Let's just use the FFT approach which is cleaner for linear conv/corr.
    
    # Linear correlation of w and f:
    # corr_wf = ifft(fft(w) * conj(fft(f)))
    # This gives circular correlation.
    # Let's stick to the direct vectorized sum if it's fast enough, or use the FFT carefully.
    
    # Given N=600, direct loop over j (600) and inner sum (1200) is 720k ops.
    # In Python, a loop of 600 iterations with numpy vector ops inside is fast.
    
    grad = np.zeros(N)
    for j in range(N):
        # indices for f[k-j] where 0 <= k-j < N => j <= k < N+j
        # k ranges from max(0, j) to min(len(conv)-1, N+j-1)
        # But weights[k] is defined for k in 0..len(conv)-1.
        # f[idx] is defined for idx in 0..N-1.
        
        # Term 1: sum_k weights[k] * f[k-j]
        # k-j must be in [0, N-1] => k in [j, j+N-1]
        k_start = j
        k_end = min(len(weights), j + N)
        if k_start < k_end:
            idx_f = np.arange(k_start, k_end) - j
            grad[j] += np.dot(weights[k_start:k_end], f[idx_f])
            
        # Term 2: sum_k weights[k] * f[j-k]
        # j-k must be in [0, N-1] => k in [j-(N-1), j]
        k_start2 = max(0, j - (N - 1))
        k_end2 = min(len(weights), j + 1)
        if k_start2 < k_end2:
            idx_f2 = j - np.arange(k_start2, k_end2)
            grad[j] += np.dot(weights[k_start2:k_end2], f[idx_f2])
            
    value = max_c + beta * np.log(sum_exp)
    return value, grad

def objective_and_grad(f, beta=50.0):
    """
    Objective: Smooth max of conv / s^2
    We optimize f on the simplex.
    To handle the simplex constraint, we can use a change of variables or just project.
    However, L-BFGS-B requires box constraints.
    We can map from unconstrained space to simplex?
    Or just use f >= 0 and fix sum(f)=1 by normalizing inside?
    Normalizing inside makes the gradient complicated.
    
    Better: Optimize on the simplex directly using a method that supports it?
    scipy.optimize.minimize doesn't support simplex constraints natively except via bounds (which is box).
    
    Strategy:
    1. Start with f on simplex.
    2. Use L-BFGS-B with bounds [0, inf].
    3. Inside the objective, normalize f to sum 1?
       If we normalize f inside the objective, the gradient returned must be consistent with the raw f.
       Let g = f / sum(f).
       Obj = SmoothMax(g) / sum(g)^2 ? No, sum(g)=1, so Obj = SmoothMax(g).
       But g depends on f.
       d Obj / d f_j = sum_k (d Obj / d g_k) * (d g_k / d f_j)
       d g_k / d f_j = (delta_kj * S - f_k) / S^2
       This is computable.
       
       However, this changes the geometry.
       
    Alternative:
    Just optimize f with f >= 0, and penalize deviation from sum(f)=1?
    Or, since the problem is scale invariant (score is invariant to scaling f),
    we can fix sum(f)=1 by construction?
    If we use L-BFGS-B, it will move f. We can project f back to simplex after each step?
    L-BFGS-B doesn't allow projection in the step.
    
    Let's try a different approach:
    Optimize log(f) + small constant? No, f can be 0.
    
    Let's use the **Dinkelbach** or just simple **Projected Gradient** with the smooth max.
    But the prompt asks for a different strategy.
    The current program uses Parametric DE + Local Coord Descent.
    
    New Strategy:
    1. Use **L-BFGS-B** on the **logit** representation?
       Let f_i = exp(y_i) / sum(exp(y_j)).
       This ensures f_i > 0 and sum(f)=1.
       We optimize y (unconstrained).
       Gradient w.r.t y_i is easy to compute from gradient w.r.t f_i.
       
       Let F(y) = SmoothMax(softmax(y)).
       d F / d y_i = sum_k (d F / d f_k) * (d f_k / d y_i)
       f_k = exp(y_k) / S
       d f_k / d y_i = f_k (delta_ki - f_i)
       
       So grad_y_i = sum_k grad_f_k * f_k (delta_ki - f_i)
       = f_i * (sum_k grad_f_k f_k) - f_i * (sum_k grad_f_k f_k) ? No.
       = f_i * grad_f_i - f_i * (sum_k grad_f_k f_k)
       
       Let G = dot(grad_f, f).
       grad_y_i = f_i * (grad_f_i - G)
       
       This is a standard softmax gradient.
       
       This removes the simplex constraint and non-negativity constraint (since exp > 0).
       We just need to handle the case where f becomes very small (underflow).
       For N=600, this should be fine.
       
       This is a structurally different optimization: Unconstrained optimization on the log-softmax manifold.
    """
    # f is the raw vector passed by L-BFGS-B? No, we pass y.
    # Let's define the function to take y.
    pass

def construct():
    start_time = time.time()
    
    # Initial guess: Uniform distribution
    # y = log(f). If f is uniform, y is constant.
    # Let's start with a few random initializations.
    
    best_score = np.inf
    best_f = None
    
    # We will run L-BFGS-B on y variables.
    # f = softmax(y)
    
    def obj_and_grad_y(y):
        # Check time
        if time.time() - start_time > TIME_LIMIT - 5:
            raise TimeoutError()
            
        # Softmax
        y_max = np.max(y)
        exp_y = np.exp(y - y_max)
        f = exp_y / np.sum(exp_y)
        
        # Compute Smooth Max and Gradient wrt f
        val, grad_f = smooth_max_conv(f, beta=50.0)
        
        # The score is val / sum(f)^2. Since sum(f)=1, score = val.
        # But wait, the original score is max(conv) / s^2.
        # If we fix sum(f)=1, score = max(conv).
        # So we minimize max(conv).
        
        # Gradient wrt y
        # grad_y_i = f_i * (grad_f_i - sum(grad_f * f))
        G = np.dot(grad_f, f)
        grad_y = f * (grad_f - G)
        
        return val, grad_y

    # Run multiple restarts
    n_restarts = 10
    max_iter_per_restart = 500
    
    for r in range(n_restarts):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        # Initialize y
        if r == 0:
            y0 = np.zeros(N)
        else:
            # Random initialization
            y0 = np.random.randn(N) * 0.1
        
        try:
            res = minimize(
                obj_and_grad_y, 
                y0, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': max_iter_per_restart, 'gtol': 1e-6}
            )
            
            if res.success or res.fun < best_score:
                y_opt = res.x
                # Convert back to f
                y_max = np.max(y_opt)
                f_opt = np.exp(y_opt - y_max)
                f_opt = f_opt / np.sum(f_opt)
                
                # Verify score
                score = compute_score(f_opt)
                if score < best_score:
                    best_score = score
                    best_f = f_opt
                    
        except TimeoutError:
            break
        except Exception as e:
            pass

    if best_f is None:
        # Fallback to uniform
        best_f = np.ones(N) / N
        best_score = compute_score(best_f)
        
    # Final refinement: Local coordinate descent on the simplex?
    # Or just return the L-BFGS result.
    # Given the time, L-BFGS with softmax should be strong.
    
    # Ensure non-negative and normalized
    best_f = np.maximum(best_f, 0.0)
    s = np.sum(best_f)
    if s > 1e-15:
        best_f = best_f / s
        
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
