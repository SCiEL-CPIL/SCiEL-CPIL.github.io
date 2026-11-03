import numpy as np
from scipy.optimize import minimize

N = 400

def construct():
    # We optimize the values of f directly.
    # Objective: Minimize max |f*f| subject to sum(f) = 1.
    # We use a soft-max approximation to handle the max.
    
    def objective(f):
        # f is the vector of heights
        # Compute autoconvolution
        autoconv = np.convolve(f, f, mode='full')
        # Use soft-max to approximate max absolute value
        # alpha controls the sharpness
        alpha = 50.0
        # Avoid overflow/underflow by scaling
        # Note: autoconv values can be large or small
        # We want to minimize the peak.
        # If sum(f)=1, the peak should be small.
        
        # To ensure numerical stability, we can subtract the mean or scale
        # But simply:
        abs_auto = np.abs(autoconv)
        # Soft max: log(sum(exp(alpha * x))) / alpha approximates max(x)
        # We add a large constant to avoid underflow? No, just use log-sum-exp trick
        max_val = np.max(abs_auto)
        if max_val < 1e-10:
            return 0.0
        # Normalize by max to keep exponents in range
        exp_vals = np.exp(alpha * (abs_auto - max_val))
        log_sum = np.log(np.sum(exp_vals))
        soft_max = (log_sum + alpha * max_val) / alpha
        return soft_max

    def gradient(f):
        # Gradient of soft-max of |f*f|
        # This is complex to derive analytically for the soft-max of abs.
        # Let's use finite differences or let scipy handle it?
        # L-BFGS-B uses finite differences if jac is not provided, which is slow.
        # Let's provide the gradient for the sum of squares? No, that minimizes L2 norm, not max.
        
        # Alternative: Use SLSQP with finite differences? Also slow.
        
        # Let's try a different objective: Minimize the L2 norm of the autoconvolution?
        # No, the task is "maximum absolute autoconvolution".
        
        # Let's use a simpler heuristic that is "from scratch":
        # Random search with local refinement.
        pass

    # Given the complexity of the gradient, let's try a different strategy:
    # Minimize the peak by adjusting the function in the time domain using a "water-filling" or "gradient descent" approach on the peak.
    
    # Strategy:
    # 1. Initialize f with random values.
    # 2. Normalize sum to 1.
    # 3. Iterate:
    #    a. Compute autoconv.
    #    b. Find peak index k*.
    #    c. Compute gradient of (f*f)[k*] wrt f.
    #       (f*f)[k] = sum_i f[i] f[k-i]
    #       d/d f[j] (f*f)[k] = f[k-j] + f[j-k] (if indices valid)
    #    d. Update f[j] -= learning_rate * gradient[j]
    #    e. Re-normalize sum to 1.
    #    f. Optionally, project onto a box (e.g., [-A, A]) or keep signs?
    #       The task says "signed step heights", doesn't restrict magnitude.
    
    rng = np.random.default_rng(42)
    # Start with random real values
    f = rng.standard_normal(N)
    f = f / np.sum(f) # Normalize sum to 1
    
    learning_rate = 0.01
    num_iters = 200
    
    for _ in range(num_iters):
        autoconv = np.convolve(f, f, mode='full')
        max_idx = np.argmax(np.abs(autoconv))
        
        # Compute gradient of autoconv[max_idx] wrt f
        # Let k = max_idx
        # g[j] = f[k-j] + f[j-k] (with bounds checking)
        k = max_idx
        grad = np.zeros(N)
        
        # Efficiently compute the gradient
        # f[k-j] is valid if 0 <= k-j < N => k-N < j <= k
        # f[j-k] is valid if 0 <= j-k < N => k <= j < k+N
        
        # We can use convolution again or slicing
        # grad = np.convolve(f, np.ones(1), mode='full')? No.
        
        # Let's construct the gradient vector
        # Term 1: f[k-j]
        # j ranges from max(0, k-N+1) to min(N-1, k)
        # Actually, simpler:
        # The gradient of the circular convolution of f with itself at index k is 2 * f[-k] (circular).
        # For linear convolution, it's the sum of two shifted copies.
        
        # Let's just compute it directly for clarity and correctness, N=400 is small.
        for j in range(N):
            val = 0.0
            if 0 <= k - j < N:
                val += f[k - j]
            if 0 <= j - k < N:
                val += f[j - k]
            grad[j] = val
            
        # Update f
        f -= learning_rate * grad
        
        # Re-normalize to keep sum(f) = 1
        # This step is crucial to maintain the constraint.
        # However, simple re-normalization might undo the gradient step partially.
        # Better: Project onto the hyperplane sum(f)=1.
        # f = f - (sum(f) - 1) / N * ones
        sum_f = np.sum(f)
        f -= (sum_f - 1.0) / N
        
        # Optional: Clip values to prevent explosion?
        # The problem doesn't specify bounds, but large values might cause numerical issues.
        # However, since we normalize sum, the average is 1/N.
        
    # Final calculation
    s = np.sum(f)
    if abs(s) < 1e-10:
        return np.ones(N)
    autoconv = np.convolve(f, f, mode='full')
    max_abs = np.max(np.abs(autoconv))
    score = max_abs / (s * s)
    
    return f
