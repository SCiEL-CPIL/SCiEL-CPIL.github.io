"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START

def _compute_autoconv(x):
    """Compute linear autoconvolution of x using FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    X = np.fft.rfft(x, size)
    c = np.fft.irfft(X * X, size)[:2 * n - 1]
    return c

def _score_and_grad(x):
    """
    Compute the score and the gradient of the score w.r.t x.
    Score = max(c) / (sum(x))^2
    Gradient derived via quotient rule.
    """
    n = len(x)
    c = _compute_autoconv(x)
    integral = np.sum(x)
    
    if integral <= 1e-12:
        return np.inf, np.zeros(n)
        
    peak_val = np.max(c)
    peak_idx = np.argmax(c)
    
    score = peak_val / (integral * integral)
    
    # Gradient of numerator N = c[peak_idx]
    # c[k] = sum_i x[i] x[k-i]
    # d c[k] / d x[i] = x[k-i] (if k-i in [0, n-1]) else 0
    # Note: x is defined on [0, n-1].
    # The term x[k-i] is non-zero only if 0 <= k-i < n => k-n+1 <= i <= k.
    
    grad_N = np.zeros(n)
    # Indices i such that 0 <= peak_idx - i < n
    # i <= peak_idx and i >= peak_idx - (n - 1)
    start_i = max(0, peak_idx - (n - 1))
    end_i = min(n - 1, peak_idx)
    
    # For i in [start_i, end_i], the contribution is x[peak_idx - i]
    # Let j = peak_idx - i. As i goes from start to end, j goes from peak_idx-start to peak_idx-end.
    # x indices are j.
    # We need to fill grad_N[i] = x[peak_idx - i]
    
    # Efficient vectorized assignment:
    # i ranges from start_i to end_i
    # j = peak_idx - i
    # j ranges from peak_idx - start_i to peak_idx - end_i
    # Note: peak_idx - start_i is the max j, peak_idx - end_i is the min j.
    # Let's just use a loop or fancy indexing for clarity and safety, N=600 is small enough for this part to not be bottleneck if done right.
    # Actually, simple slicing is best.
    
    # Let's construct the array of x values needed.
    # j_min = peak_idx - end_i
    # j_max = peak_idx - start_i
    # The values are x[j_min : j_max+1]
    # But they need to be assigned to grad_N[start_i : end_i+1] in reverse order?
    # Let's check:
    # i = start_i -> j = peak_idx - start_i = j_max. So grad_N[start_i] = x[j_max]
    # i = end_i   -> j = peak_idx - end_i   = j_min. So grad_N[end_i]   = x[j_min]
    # So grad_N[start_i:end_i+1] = x[j_min:j_max+1][::-1]
    
    j_min = peak_idx - end_i
    j_max = peak_idx - start_i
    
    grad_N[start_i:end_i + 1] = x[j_min:j_max + 1][::-1]
    
    # Gradient of denominator D = I^2
    # d D / d x[i] = 2 I
    # Grad S = (Grad N * D - N * Grad D) / D^2
    # D = I^2, D^2 = I^4
    # Grad S = (grad_N * I^2 - peak_val * 2 * I) / I^4
    # Grad S = grad_N / I^2 - (2 * peak_val) / I^3
    
    grad_S = grad_N / (integral * integral) - (2.0 * peak_val) / (integral**3)
    
    return score, grad_S

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initialization: A smooth, non-negative function
    # Optimal shapes for this type of problem are often close to cosines or plateaus.
    # Let's try a raised cosine window as a good starting point.
    n = np.arange(N)
    x = 0.5 * (1 + np.cos(2 * np.pi * (n - N//2) / N))
    # Ensure non-negative (it is by construction)
    # Normalize to have sum = 1 to avoid scaling issues initially, though score is scale invariant.
    # Actually, score is scale invariant: (k*x)*(k*x) / (sum(k*x))^2 = k^2 * c / (k^2 * I^2) = c/I^2.
    # So we can keep any scale.
    
    # Check if we have time for a quick parametric search for better init
    # But PGD is robust. Let's just run PGD.
    
    best_x = x.copy()
    best_score, _ = _score_and_grad(best_x)
    
    # Adaptive step size
    # The Lipschitz constant of the gradient is related to the norm of the Hessian.
    # For convolution, it's bounded by something like 2 * ||x||_2^2 / I^4 ... 
    # Let's use a simple backtracking line search or a fixed small step with decay.
    # Given N=600, we can afford many iterations.
    
    # We will use a simple gradient descent with adaptive step size.
    # Step size eta.
    # If score increases, reduce eta. If it decreases, keep or slightly increase.
    
    eta = 0.01
    min_eta = 1e-6
    max_eta = 0.1
    
    # Pre-compute FFT size for speed
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2
        
    # Optimization Loop
    no_improve_count = 0
    last_improve_time = start_time
    
    while time.time() < deadline:
        score, grad = _score_and_grad(x)
        
        if score < best_score:
            best_score = score
            best_x = x.copy()
            last_improve_time = time.time()
            no_improve_count = 0
            # Slightly increase step size if we are making good progress?
            # Or just keep it.
        else:
            no_improve_count += 1
            # If no improvement for a while, reduce step size
            if no_improve_count > 10:
                eta *= 0.9
                if eta < min_eta:
                    # If step size is too small, we are stuck.
                    # Try a random restart or perturbation?
                    # Or break if we've been stuck for a long time.
                    if time.time() - last_improve_time > 20:
                        break
                no_improve_count = 0
        
        # Gradient Descent Step
        # x_new = x - eta * grad
        x_temp = x - eta * grad
        
        # Project to non-negative orthant
        x_temp = np.maximum(0, x_temp)
        
        # Check if step was too large (score increased significantly)
        # We already checked score in the next loop, but we can do a quick check here
        # to adjust eta immediately.
        # However, the "no_improve" logic handles this.
        
        x = x_temp
        
        # Periodically, if stuck, try a small random perturbation to escape local minima
        # The landscape is not strictly convex, but for this specific problem, 
        # the global minimum is likely unique or well-behaved.
        # A "soft" restart: if stuck for > 10s, add small noise.
        if time.time() - last_improve_time > 10:
            # Add small noise to non-zero elements
            noise = np.random.randn(N) * 0.01
            x = np.maximum(0, x + noise)
            last_improve_time = time.time() # Reset timer to give it a chance
            eta = max(eta * 0.5, min_eta) # Reduce step size for stability after noise

    # Final polish: A few steps of very fine tuning
    # Ensure we return the best found
    return list(best_x)

# EVOLVE-BLOCK-END
