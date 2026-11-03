"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # We are looking for a signed step function (vector x of length N)
    # that minimizes max |autoconv(x)| / (sum(x)^2)
    # Let A = autoconv(x). We want to minimize max |A_i| / S^2 where S = sum(x).
    # This is equivalent to minimizing t such that |A_i| <= t * S^2 for all i.
    
    # Since the problem is homogeneous, we can fix S = 1 (or any constant) and minimize max |A_i|.
    # However, x must be a "signed step function". The prompt says "sequence of signed step heights".
    # The seed returns np.ones(N). This implies x_i can be any real number? Or just +/- 1?
    # "Signed step functions" usually implies values are constant on steps, but here N is fixed and we return a sequence.
    # Given the seed is np.ones(N), and the task mentions "signed equal-width step functions", it's likely we can choose any real values for each element, 
    # but the structure of "step functions" might imply piecewise constant. However, the interface just asks for a sequence.
    # Let's assume we can choose any real vector x of length N.
    # If x is any real vector, we can scale it. Let sum(x) = 1.
    # Then we minimize max |A_i| where A = np.convolve(x, x, mode='full')? 
    # Wait, autoconvolution of a step function. 
    # The prompt says "autoconvolution". For a discrete signal x, autoconvolution is usually (x * x)[k] = sum x_i x_{i+k}.
    # Let's check the definition. "maximum absolute autoconvolution divided by squared integral".
    # Integral of a step function is sum(x) * width. If width is 1, integral is sum(x).
    # So we minimize max |C_k| / (sum(x))^2.
    
    # Strategy:
    # 1. Start with a random sign vector or a smooth vector.
    # 2. Optimize the continuous vector x to minimize a smooth surrogate of max |C_k| subject to sum(x)=1.
    # 3. The objective is non-smooth due to max and abs. We can use a log-sum-exp approximation or just optimize the maximum directly using subgradients or a soft-max.
    # Let f(x) = log-sum-exp(C_k^2) / (2 * (sum x)^2) ? No, we want max |C_k|.
    # Let's minimize g(x) = log-sum-exp(alpha * C_k^2) to approximate max C_k^2.
    # Actually, we want to minimize max |C_k|. Let M = max |C_k|.
    # We can minimize M subject to sum(x)=1.
    # Using an auxiliary variable t: minimize t subject to C_k^2 <= t^2, sum(x)=1.
    # This is a convex problem if we fix the signs? No, C_k is quadratic in x, so C_k^2 is quartic. Not convex.
    
    # Alternative: Use gradient descent on a soft-max surrogate.
    # Surrogate: S(x) = (1/beta) * log( sum_k exp(beta * C_k^2) )
    # As beta -> inf, S(x) -> max C_k^2. Minimizing S(x) minimizes max C_k^2, which is equivalent to minimizing max |C_k|.
    # Constraint: sum(x) = 1.
    
    # Let's implement a custom gradient descent or use scipy.optimize.minimize with a smooth objective.
    
    # Initial guess: 
    # A known good structure for minimizing autoconvolution peak is related to Barker sequences or similar.
    # But we can use continuous optimization first.
    
    # Let's try a few random initializations and pick the best one.
    
    best_x = None
    best_val = float('inf')
    
    # Precompute indices for autoconvolution?
    # C_k = sum_{i} x_i x_{i+k} for k in -(N-1)..(N-1).
    # Using FFT is faster for large N, but N=400 is small. Direct computation is O(N^2) = 160k ops, which is fast.
    
    def autoconv(x):
        return np.convolve(x, x, mode='full')
    
    def objective(x_flat):
        x = x_flat
        s = np.sum(x)
        if s == 0:
            return 1e10
        c = autoconv(x)
        # We want to minimize max |c| / s^2
        # Let's use a soft-max on c^2 to approximate max c^2.
        # Then divide by s^2.
        # But s is fixed to 1 in the optimization if we use a constraint.
        # Let's use scipy minimize with constraint sum(x)=1.
        
        beta = 10.0 # temperature for soft-max
        # Avoid overflow in exp
        c2 = c * c
        max_c2 = np.max(c2)
        if max_c2 == 0:
            return 0
        # log-sum-exp trick
        shifted = c2 - max_c2
        lse = max_c2 + np.log(np.sum(np.exp(shifted)))
        val = lse / (s * s)
        return val
    
    def grad_objective(x_flat):
        x = x_flat
        s = np.sum(x)
        if s == 0:
            return np.zeros_like(x)
        c = autoconv(x)
        c2 = c * c
        beta = 10.0
        max_c2 = np.max(c2)
        shifted = c2 - max_c2
        weights = np.exp(shifted)
        weights /= np.sum(weights)
        
        # Gradient of lse = max_c2 + log(sum(exp(shifted)))
        # d(lse)/d(c2_j) = weights_j
        # d(c2_j)/dx = 2 * c_j * (d c_j / dx)
        
        # c_j = sum_i x_i x_{i+j}
        # d c_j / dx_k = x_{k-j} + x_{k+j} (with bounds check)
        
        # This gradient is complex to compute directly. Let's use numerical gradient or a simpler approach.
        # Given N=400, numerical gradient might be too slow if we do many iterations.
        # Let's try to use the fact that we can just use scipy.optimize.minimize with method='L-BFGS-B' and finite differences?
        # No, 400 dimensions, finite differences is 401 function evals per gradient. Too slow for many iterations.
        
        # Let's derive the gradient properly.
        # Let L = lse / s^2
        # dL/dx = (1/s^2) * d(lse)/dx - 2/s^3 * (sum x) * lse ? No, s is sum(x), so ds/dx_i = 1.
        # d(lse)/dx_i = sum_j weights_j * d(c2_j)/dx_i
        # d(c2_j)/dx_i = 2 c_j * (x_{i-j} + x_{i+j}) (if indices valid)
        
        # Let's compute d(c2_j)/dx efficiently.
        # c = autoconv(x)
        # dc/dx is a convolution matrix.
        # d(c2)/dx = 2 * c * (dc/dx)
        
        # Let's just use a simpler heuristic: 
        # Minimize max |c| directly using a subgradient method?
        # Or just use a random search + local polish?
        
        # Let's try a different approach: 
        # The optimal solution for minimizing max |autoconv| / (sum)^2 for signed steps is often related to making the autoconvolution as flat as possible.
        # A known sequence is the Barker sequence, but those are for +/- 1.
        # If we allow real values, we can do better.
        
        # Let's use a gradient descent with a simple gradient approximation or just use scipy's minimize with a smooth objective and let it handle the gradient numerically? 
        # No, let's implement the gradient.
        
        # Indices for c: k = 0..2N-2 corresponds to lag = k - (N-1)
        # c_k = sum_{i} x_i x_{i + (k - N + 1)}
        
        # Let's compute the gradient of the soft-max objective.
        # grad_lse[i] = sum_j weights_j * 2 * c_j * (x_{i-j+N-1} + x_{i+j-N-1}) (with checks)
        
        # This is O(N^2) per gradient. N=400, so 160k ops. Acceptable.
        
        lags = np.arange(0, 2*N-1) - (N-1)
        
        grad_c2 = np.zeros_like(x)
        
        # For each lag j, c_j = sum x_i x_{i+j}
        # d(c_j^2)/dx_i = 2 c_j (x_{i-j} + x_{i+j})
        
        # We can compute this using convolutions.
        # Let w = weights * 2 * c  (length 2N-1)
        # Then grad_lse = w * (x shifted) ... 
        # Specifically, sum_j w_j (x_{i-j} + x_{i+j})
        # = sum_j w_j x_{i-j} + sum_j w_j x_{i+j}
        # Let w_pos[j] = w_{j+N-1} for j>=0? No, w is indexed by k=0..2N-2.
        # Let's define w_lag[l] where l is the lag.
        # w_lag has length 2N-1, centered at 0.
        
        w_lag = weights * 2 * c
        # w_lag[k] corresponds to lag = k - (N-1)
        
        # Term 1: sum_l w_lag[l] x_{i-l}
        # This is a convolution of x and w_lag?
        # Let's just do it with loops or numpy shifts for clarity, N is small.
        
        # Actually, let's just use a numerical gradient for the first version to ensure correctness, 
        # and if it's too slow, switch to analytic.
        # 400 dims, 100 iterations -> 40000 function evals. 
        # Each function eval is O(N^2) = 160k ops.
        # 40000 * 160k = 6.4e9 ops. Too slow in Python.
        
        # We MUST use analytic gradient or a smarter method.
        
        # Let's implement the analytic gradient.
        
        # grad_lse[i] = sum_{k=0}^{2N-2} weights[k] * 2 * c[k] * (x[i - (k - N + 1)] + x[i + (k - N + 1)])
        
        # Let l = k - N + 1. l ranges from -(N-1) to N-1.
        # term1[i] = sum_l w_l x_{i-l}
        # term2[i] = sum_l w_l x_{i+l}
        
        # We can compute term1 and term2 using convolve.
        # Let w_rev be w_lag reversed?
        
        # Let's use np.convolve for efficiency.
        # x is length N. w_lag is length 2N-1.
        # We need the "full" convolution but only the part that aligns with x.
        
        # Actually, let's just use a simple loop for the gradient since N=400 is small enough for a few iterations if we are careful.
        # Or better, use FFT?
        
        # Let's try a simpler optimization: 
        # Minimize the L2 norm of c? No, we want max.
        
        # Let's use the analytic gradient.
        
        w_lag = weights * 2 * c
        # w_lag[k] is for lag l = k - (N-1)
        
        # Compute term1: sum_l w_l x_{i-l}
        # This is equivalent to convolving x with a flipped w_lag?
        
        # Let's create an array w_full of length 2N-1, centered.
        # We want to compute y[i] = sum_l w_l x_{i-l}
        # This is a cross-correlation.
        
        # Let's just use a loop for the gradient calculation. N=400, 2N-1=799 terms.
        # 400 * 799 = 319,600 ops per gradient.
        # 100 iterations -> 32 million ops. This is fast in NumPy if vectorized, but in a Python loop it might be slow.
        # Let's vectorize it.
        
        # We can compute term1 and term2 using np.convolve with mode='same' or similar?
        # Let's define w_centered such that w_centered[N-1] = w_0 (lag 0).
        # w_lag is already in order k=0..2N-2, so w_lag[N-1] is lag 0.
        
        # term1[i] = sum_l w_lag[l+N-1] x_{i-l}
        # Let v = w_lag.
        # term1 = np.convolve(x, v, mode='full') ? No.
        
        # Let's stick to a simpler method: 
        # Use scipy.optimize.minimize with method='Powell' or 'Nelder-Mead'? No, too slow for 400 dims.
        
        # Let's use a random restart with a simple local search on signs?
        # If we restrict to x_i in {-1, 1}, it's a combinatorial problem.
        # But the task allows "signed step heights", implying real values.
        
        # Let's try this: 
        # 1. Start with x = ones(N).
        # 2. Optimize using gradient descent with the analytic gradient.
        
        def compute_grad(x):
            s = np.sum(x)
            c = autoconv(x)
            c2 = c * c
            beta = 10.0
            max_c2 = np.max(c2)
            shifted = c2 - max_c2
            weights = np.exp(shifted)
            weights /= np.sum(weights)
            
            w_lag = weights * 2 * c
            
            # Compute grad_lse
            # grad_lse[i] = sum_l w_lag[l] * (x[i-l] + x[i+l])
            # where l = k - (N-1)
            
            # Let's use FFT for speed.
            # This is a convolution.
            # Let's pad x and w_lag to length 4N.
            
            # Actually, for N=400, a simple loop might be okay if we do it in NumPy.
            # Let's try to vectorize the sum over l.
            
            # We can create a matrix of shifts? No, too much memory.
            
            # Let's use the fact that this is a convolution.
            # term1[i] = sum_l w_l x_{i-l}
            # This is (x * w_rev)[i] where w_rev is w_lag reversed?
            
            # Let's just use np.convolve.
            # We want the part of the convolution that corresponds to i=0..N-1.
            
            # Let's define w_pad of length 2N-1, centered at N-1.
            # w_pad[N-1] = w_0 (lag 0)
            # w_pad[N-1+l] = w_l (lag l)
            # w_pad[N-1-l] = w_-l (lag -l)
            
            # w_lag is indexed k=0..2N-2, lag = k - (N-1).
            # So w_lag[k] corresponds to lag k - (N-1).
            # So w_lag is already in the order we need for a centered array if we consider index N-1 as lag 0.
            
            # Let's compute term1 = convolve(x, w_lag, mode='full')
            # The result will have length N + (2N-1) - 1 = 3N-2.
            # The center of the result is at index (N-1) + (N-1) = 2N-2?
            
            # This is getting complicated. Let's just use a simple loop for the gradient.
            # N=400, 799 lags.
            # grad = np.zeros(N)
            # for k in range(2*N-1):
            #     l = k - (N-1)
            #     w = w_lag[k]
            #     if w == 0: continue
            #     # x[i-l] and x[i+l]
            #     # We can use slicing.
            
            # Let's try to vectorize using np.roll?
            # No, l varies.
            
            # Okay, let's use a different strategy: 
            # Minimize the max |c| using a binary search on t and checking feasibility?
            # Feasibility: is there x with sum(x)=1 and |c_k| <= t for all k?
            # This is a convex feasibility problem? No, c_k is quadratic.
            
            # Let's go back to the analytic gradient with a simple loop. It should be fast enough for 100 iterations.
            
            grad_lse = np.zeros(N)
            for k in range(2*N-1):
                w = w_lag[k]
                if w == 0:
                    continue
                l = k - (N-1)
                # x[i-l]
                # i ranges from 0 to N-1.
                # i-l ranges from -l to N-1-l.
                # Valid i are those where 0 <= i-l < N and 0 <= i+l < N.
                
                # Let's compute the contributions.
                # Term 1: sum_i x[i] * w * (x[i-l] if valid else 0)
                # This is a shift of x.
                
                # Let's just use numpy arrays for the shifts.
                # x_shifted1[i] = x[i-l] if valid else 0
                # x_shifted2[i] = x[i+l] if valid else 0
                
                # We can create these arrays quickly.
                
                # For l > 0:
                # x[i-l] is valid for i >= l. So x_shifted1[l:] = x[:-l]
                # x[i+l] is valid for i < N-l. So x_shifted2[:N-l] = x[l:]
                
                # For l < 0:
                # Let m = -l > 0.
                # x[i+m] is valid for i < N-m. x_shifted1[:N-m] = x[m:]
                # x[i-m] is valid for i >= m. x_shifted2[m:] = x[:-m]
                
                # For l == 0:
                # x_shifted1 = x, x_shifted2 = x.
                
                if l == 0:
                    grad_lse += w * (x + x)
                elif l > 0:
                    # x[i-l]
                    x1 = np.zeros(N)
                    x1[l:] = x[:-l]
                    # x[i+l]
                    x2 = np.zeros(N)
                    x2[:N-l] = x[l:]
                    grad_lse += w * (x1 + x2)
                else:
                    m = -l
                    # x[i-m]
                    x1 = np.zeros(N)
                    x1[m:] = x[:-m]
                    # x[i+m]
                    x2 = np.zeros(N)
                    x2[:N-m] = x[m:]
                    grad_lse += w * (x1 + x2)
            
            # Now, grad of objective = (1/s^2) * grad_lse - (2/s^3) * lse * ones(N)
            # Wait, objective = lse / s^2
            # d(obj)/dx = (s^2 * grad_lse - lse * 2 * s * 1) / s^4
            # = grad_lse / s^2 - 2 * lse / s^3
            
            lse_val = max_c2 + np.log(np.sum(weights))
            
            grad_obj = grad_lse / (s * s) - 2 * lse_val / (s * s * s) * np.ones(N)
            
            return grad_obj
        
        # Optimization loop
        x = np.ones(N)
        x /= np.sum(x)
        
        lr = 0.01
        for i in range(50):
            if time.time() > deadline - 10:
                break
            
            obj_val = objective(x)
            if obj_val < best_val:
                best_val = obj_val
                best_x = x.copy()
            
            grad = compute_grad(x)
            
            # Gradient step
            x = x - lr * grad
            
            # Project back to sum(x)=1
            x = x / np.sum(x)
            
            # Optional: clip x to avoid large values?
        
        # Final evaluation
        if best_x is not None:
            s = np.sum(best_x)
            c = autoconv(best_x)
            val = np.max(np.abs(c)) / (s * s)
            if val < best_val:
                best_val = val
        
        return best_x

# EVOLVE-BLOCK-END
