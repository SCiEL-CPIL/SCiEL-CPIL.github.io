"""Signed autocorrelation maximum absolute-value variant."""
import numpy as np
import time

N = 400

def construct():
    """
    Returns a sequence of signed step heights that minimizes the ratio:
    max_k |autocorrelation(k)| / (sum(x))^2
    
    Analysis:
    Let x be a sequence of length N.
    Autocorrelation A(k) = sum_i x_i x_{i+k}.
    Integral I = sum_i x_i.
    Objective: J = max_k |A(k)| / I^2.
    
    Note that A(0) = sum_i x_i^2.
    By Cauchy-Schwarz, I^2 = (sum_i x_i)^2 <= (sum_i x_i^2) * N = A(0) * N.
    Therefore, A(0) / I^2 >= 1/N.
    
    Since max_k |A(k)| >= |A(0)|, we have J >= A(0)/I^2 >= 1/N.
    
    The lower bound 1/N is achieved if and only if:
    1. x_i is constant for all i (so I^2 = N * A(0) is tight).
    2. The maximum absolute autocorrelation is attained at k=0 (or tied with it).
    
    For x = [1, 1, ..., 1]:
    I = N.
    A(0) = N.
    A(k) = N - k for k > 0.
    max |A(k)| = N (at k=0).
    J = N / N^2 = 1/N.
    
    This is the global minimum.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Baseline solution: all ones
    base_x = np.ones(N)
    base_score = 1.0 / N  # Exact value for all ones
    
    best_x = base_x.copy()
    best_score = base_score
    
    # Try to improve using gradient descent on a smooth approximation
    # of the max absolute autoconvolution, normalized by sum^2.
    
    # We optimize f(x) = max_k |A(k)| / (sum x)^2
    # Since scaling x by c scales A(k) by c^2 and (sum x)^2 by c^2,
    # the objective is scale-invariant. We can fix sum(x) = 1 to simplify.
    # Then we minimize max_k |A(k)| subject to sum(x) = 1.
    # The value of the objective will then be max_k |A(k)|.
    
    # Start with normalized ones
    x = np.ones(N) / N
    
    # Gradient descent with smooth max
    eta = 0.01
    beta = 5.0
    steps = 500
    
    def compute_objective_and_grad(x, beta):
        """
        Compute smooth max |autoconv| and its gradient w.r.t x.
        x is assumed to have sum 1.
        """
        n = len(x)
        size = 2 * n - 1
        fft_size = 1
        while fft_size < size:
            fft_size *= 2
            
        # Compute autoconvolution
        X = np.fft.fft(x, n=fft_size)
        C = np.fft.ifft(X * X)
        c = C[:size].real  # Should be real
        
        abs_c = np.abs(c)
        max_abs_c = np.max(abs_c)
        
        if max_abs_c < 1e-15:
            return 0.0, np.zeros_like(x)
        
        # Softmax approximation of max
        shifted = beta * (abs_c - max_abs_c)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        
        M = max_abs_c + (1.0 / beta) * np.log(sum_exp)
        
        # Gradient of M w.r.t c
        # dM/dc_i = sign(c_i) * exp(beta * |c_i|) / (beta * sum_exp)
        # Note: d|c_i|/dc_i = sign(c_i)
        # Let w_i = exp(beta * |c_i|) * sign(c_i) / (beta * sum_exp)
        # But we used shifted exp. Let's be careful.
        # M = max_c + (1/beta) log(sum_j exp(beta(|c_j| - max_c)))
        # = max_c + (1/beta) log(sum_j exp(beta|c_j|) / exp(beta max_c))
        # = max_c - max_c + (1/beta) log(sum_j exp(beta|c_j|))
        # Wait, log(A/B) = log A - log B.
        # So M = (1/beta) log(sum_j exp(beta|c_j|)).
        
        # Let's recompute weights based on unshifted exp to be safe, 
        # or stick to the shifted version which is numerically stable.
        # M = max_c + (1/beta) log(S) where S = sum exp(beta(|c_j| - max_c))
        # dM/dc_i = (1/beta) * (1/S) * dS/dc_i
        # dS/dc_i = exp(beta(|c_i| - max_c)) * beta * sign(c_i)
        # dM/dc_i = sign(c_i) * exp(beta(|c_i| - max_c)) / S
        
        weights_c = (np.sign(c) * exp_shifted) / sum_exp
        
        # Gradient of c w.r.t x
        # c_k = sum_j x_j x_{k-j}
        # dc_k/dx_i = x_{k-i} + x_{k-i} ? No.
        # c_k = sum_{j=0}^{k} x_j x_{k-j} (for linear conv, assuming 0 padding)
        # Actually, for linear convolution c = x * x, 
        # c_k = sum_{j} x_j x_{k-j} where x_m = 0 for m<0 or m>=N.
        # dc_k/dx_i = x_{k-i} + x_{k-i} if i != k-i?
        # Let's use the property: grad_x (sum_k w_k c_k) = 2 * (w * x)
        # where * is linear convolution.
        
        # Compute convolution of weights_c with x
        W = np.fft.fft(weights_c, n=fft_size)
        grad_c_x = np.fft.ifft(W * X)
        grad_x = 2.0 * grad_c_x[:n].real
        
        return M, grad_x
    
    for step in range(steps):
        if time.time() > deadline:
            break
        
        M, grad_x = compute_objective_and_grad(x, beta)
        
        # Project gradient to subspace orthogonal to all-ones (to maintain sum=1)
        # grad_proj = grad_x - mean(grad_x)
        grad_mean = np.mean(grad_x)
        grad_proj = grad_x - grad_mean
        
        # Update
        x_new = x - eta * grad_proj
        
        # Renormalize to sum=1
        s = np.sum(x_new)
        if abs(s) < 1e-10:
            x_new = x
        else:
            x_new = x_new / s
        
        # Evaluate true objective
        # Objective is max |A(k)| / (sum x)^2
        # Since sum x_new = 1, objective is max |A(k)|
        
        n = len(x_new)
        size = 2 * n - 1
        fft_size = 1
        while fft_size < size:
            fft_size *= 2
        
        X_new = np.fft.fft(x_new, n=fft_size)
        C_new = np.fft.ifft(X_new * X_new)
        c_new = C_new[:size].real
        true_val = np.max(np.abs(c_new))
        
        if true_val < best_score:
            best_score = true_val
            best_x = x_new.copy()
        
        x = x_new
        
        # Anneal
        if step % 50 == 0:
            beta *= 1.2
            eta *= 0.95
    
    # Return the best sequence found
    # The evaluator likely normalizes or computes the ratio itself.
    # The problem states "score is the upper bound on C3' (maximum absolute autoconvolution)"
    # but also "evaluator measures the raw value (lower is better)".
    # And "ratio: max_k |autocorrelation(k)| / (sum(x))^2".
    # If I return x with sum=1, the ratio is max|A|.
    # If I return x with sum=N (like base), the ratio is max|A|/N^2.
    # The base program returns np.ones(N).
    # Ratio for np.ones(N) is 1/N.
    # My optimized x has sum=1. Let y = x * N.
    # Then sum(y) = N.
    # A_y(k) = N^2 A_x(k).
    # Ratio = N^2 max|A_x| / N^2 = max|A_x|.
    # So scaling doesn't change the ratio.
    # I should return the sequence that minimizes the ratio.
    # Since the ratio is scale invariant, I can return x (sum=1) or x*N (sum=N).
    # The seed returns np.ones(N). To be consistent with "step heights", 
    # and given the seed uses 1s, it's safer to return a sequence with similar magnitude.
    # However, the ratio is what matters.
    # Let's return x * N to have sum N, similar to seed.
    
    final_x = best_x * N
    
    return final_x
