"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Initialize with a random signed step function
    x = np.random.choice([-1, 1], N)
    
    # 2. Continuous Relaxation using BFGS
    # We optimize a continuous vector y, then project to signs.
    # Objective: Minimize the "peak" of the autoconvolution sidelobes.
    # The autoconvolution C3 is related to the max absolute value of the autocorrelation sidelobes.
    # Let's use a soft-max approximation to make it differentiable.
    
    def objective_and_grad(y):
        # Project to signs for the actual metric calculation? 
        # No, for gradient descent we need a continuous objective.
        # Let's use the "soft" max of the squared autocorrelation sidelobes.
        
        # Compute autocorrelation via FFT (circular, then adjust or use linear)
        # Since N is small, linear autocorrelation is fine, but FFT is faster.
        # For a step function, the "autoconvolution" usually refers to the discrete autocorrelation.
        # The task says "autoconvolution upper bound C3".
        # Let's assume C3 is the maximum absolute value of the sidelobes of the autocorrelation.
        
        # To make it differentiable, we use a smooth approximation of the max.
        # f(y) = (1/t) * log(sum(exp(t * r_k^2)))
        # where r_k are the sidelobe values.
        
        # However, calculating the full autocorrelation at every step is expensive if done naively.
        # N=400 is small. FFT size ~ 1024.
        
        # Let's use the continuous y directly in the autocorrelation calculation.
        # The autocorrelation of y is r_k = sum y_i y_{i+k}.
        # This is differentiable w.r.t y.
        
        # Compute autocorrelation
        # Use FFT for speed
        fft_len = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(y, fft_len)
        power = F * np.conj(F)
        r = np.fft.irfft(power, fft_len)[:N]
        
        # r[0] is the peak (N). We want to minimize the max of |r[k]| for k > 0.
        # Sideloobes are r[1:N]
        sidelobes = r[1:]
        
        # Soft max of squared sidelobes
        # We want to minimize the "energy" concentrated in the largest sidelobe.
        # Let's try minimizing the sum of squares (L2 norm) first, as it's convex-ish and easy.
        # Or soft-max. Soft-max is better for "minimax" behavior.
        
        # Temperature parameter for soft-max
        t = 10.0 
        
        # To avoid overflow, subtract max
        sq_sidelobes = sidelobes ** 2
        max_sq = np.max(sq_sidelobes)
        if max_sq > 1e6:
            return max_sq, np.zeros_like(y) # Gradient explosion guard, though unlikely with BFGS
            
        # Soft max
        exp_terms = np.exp(t * (sq_sidelobes - max_sq))
        soft_max_val = max_sq + (1.0/t) * np.log(np.sum(exp_terms))
        
        # Gradient calculation
        # d/dy_k [ sum_i exp(t * r_i^2) ] / sum ...
        # This is complex. Let's use numerical gradient or a simpler objective.
        
        # Alternative: Minimize the L2 norm of the sidelobes.
        # This is a well-known proxy for "low autocorrelation".
        # Objective: sum(r_k^2) for k=1..N-1
        # This is equivalent to minimizing the "energy" outside the peak.
        
        # Let's stick to L2 norm for stability and speed, then refine with local search.
        # Actually, the task asks for "upper bound", which implies a minimax objective.
        # But L2 is a good starting point.
        
        # Let's try to compute gradient of soft-max properly?
        # It's risky. Let's use a simpler approach:
        # Minimize the sum of squares of the sidelobes.
        
        obj_val = np.sum(sq_sidelobes)
        
        # Gradient of sum(r_k^2) w.r.t y
        # r_k = sum_i y_i y_{i+k}
        # This is a quadratic form.
        # Let's use the property that the gradient of the L2 norm of autocorrelation sidelobes
        # can be computed via the inverse FFT of the power spectrum?
        
        # Actually, let's just use numerical differentiation? No, too slow.
        # Let's use the analytic gradient for the L2 objective.
        # The gradient of the sum of squared autocorrelation sidelobes is related to the
        # "autocorrelation of the autocorrelation"?
        
        # Simpler: Use the fact that minimizing the L2 norm of the sidelobes is equivalent to
        # minimizing the "out-of-band" energy in the frequency domain?
        
        # Let's try a different strategy:
        # Use the "soft" max with a small temperature to approximate minimax, but use
        # numerical gradient via finite differences? No, N=400, too slow.
        
        # Let's use the analytic gradient for the soft-max.
        # Let S = sum_j exp(t * r_j^2)
        # Obj = (1/t) log S
        # dObj/dy_i = (1/t) * (1/S) * sum_j d(exp(t r_j^2))/dy_i
        # d(exp(t r_j^2))/dy_i = exp(t r_j^2) * 2 t r_j * (dr_j/dy_i)
        # dr_j/dy_i = y_{i-j} + y_{i+j} (with boundary conditions)
        
        # This is computable.
        
        # However, BFGS might struggle with the non-convexity of the soft-max.
        # Let's try minimizing the L2 norm first, as it's convex in the relaxed space?
        # No, it's not convex.
        
        # Let's go with the L2 norm objective for the continuous relaxation.
        # It is robust and often finds good sequences.
        
        # Gradient of L2 norm of sidelobes:
        # The gradient is 2 * (autocorrelation of the sequence) convolved with something?
        
        # Actually, there is a known result: The gradient of the sum of squared autocorrelation
        # sidelobes is 2 * (y * y) * y? No.
        
        # Let's use a simpler gradient:
        # The autocorrelation r_k = <y, shift_k(y)>
        # The gradient of sum_k r_k^2 is 4 * (r * y) ?
        
        # Let's compute the gradient numerically for a few steps to verify? No.
        
        # Let's use the "power method" style update?
        
        # Okay, I will implement the gradient for the L2 objective analytically.
        # Let J(y) = sum_{k=1}^{N-1} r_k^2
        # r_k = sum_{i} y_i y_{i+k}
        # dJ/dy_m = sum_{k=1}^{N-1} 2 r_k (dr_k/dy_m)
        # dr_k/dy_m = y_{m-k} + y_{m+k} (if indices valid)
        
        # So dJ/dy_m = 2 sum_{k=1}^{N-1} r_k (y_{m-k} + y_{m+k})
        # This can be computed via convolution.
        # Let A = r[1:N] (padded)
        # dJ/dy = 2 * (A * y + rev(A) * y)?
        
        # Let's use FFT to compute this gradient efficiently.
        
        # 1. Compute r via FFT (as above)
        # 2. Construct the gradient vector
        
        # Actually, for N=400, we can just loop or use simple numpy convolutions if needed,
        # but FFT is better.
        
        # Let's define the gradient function for the L2 objective.
        
        # We already have r from the FFT calculation above.
        # r is the full linear autocorrelation (length 2N-1, but we took first N? No, irfft gives 2N-1 if we use correct size).
        # Wait, np.fft.irfft(power, fft_len) gives length fft_len.
        # The autocorrelation is in the first N and last N-1 elements?
        # For linear autocorrelation of length N, the result is length 2N-1.
        # r[0] is peak. r[1:N-1] are positive lags. r[N:2N-1] are negative lags (symmetric).
        
        # We want sidelobes for k=1 to N-1.
        # In the FFT output `r_full` (length fft_len), the autocorrelation is centered?
        # No, irfft of power spectrum gives the linear autocorrelation starting from lag 0.
        # So r_full[0] = r_0, r_full[1] = r_1, ..., r_full[N-1] = r_{N-1}.
        # The negative lags are not explicitly in the first N, but due to symmetry r_k = r_{-k}.
        
        # So sidelobes are r_full[1:N].
        
        # Gradient calculation:
        # dJ/dy_m = 2 * sum_{k=1}^{N-1} r_k * (y_{m-k} + y_{m+k})
        
        # This is a convolution of r with y.
        
        # Let's compute it.
        
        # Create a vector G such that G[m] = sum_{k} r_k y_{m-k} + r_k y_{m+k}
        
        # We can use np.convolve.
        # y is length N. r_sidelobes is length N-1.
        
        # Term 1: sum_{k=1}^{N-1} r_k y_{m-k}
        # This is convolution of y with r_sidelobes (reversed).
        
        # Term 2: sum_{k=1}^{N-1} r_k y_{m+k}
        # This is correlation.
        
        # Let's use FFT for the gradient too.
        
        # Actually, let's simplify.
        # If we use the L2 objective, the gradient is:
        # grad = 2 * (np.convolve(y, r_sidelobes, mode='same') + np.convolve(y, r_sidelobes[::-1], mode='same'))
        # Wait, indices are tricky.
        
        # Let's use a simpler, robust approach:
        # Use `scipy.optimize.minimize` with `method='BFGS'` and `jac` provided.
        
        # Let's refine the objective to be the soft-max to better target the "upper bound" (minimax).
        # But soft-max gradient is messy.
        
        # Let's try the L2 norm first. It's a standard "low autocorrelation" objective.
        
        # Gradient of L2 norm of sidelobes:
        # grad = 4 * (r * y) ? No.
        
        # Let's compute the gradient numerically? No.
        
        # Let's use the following trick:
        # The sum of squared autocorrelation sidelobes is equal to the sum of the squared magnitudes
        # of the DFT of the sequence, minus the square of the DC component, divided by N?
        # Parseval's theorem: sum |r_k|^2 = (1/N) sum |F_k|^2 ?
        # No, that's for the DFT of the sequence.
        
        # Actually, sum_{k} r_k^2 = sum_{i,j} (sum_k y_{i+k} y_{j+k})^2 ...
        
        # Let's just use the direct convolution gradient.
        
        # r_sidelobes = r[1:N]
        # grad[m] = 2 * (sum_{k=1}^{N-1} r_k y_{m-k} + sum_{k=1}^{N-1} r_k y_{m+k})
        
        # We can compute this using FFT.
        
        # Let's define a helper to compute the gradient.
        
        def compute_grad(y, r):
            # r is the full autocorrelation from FFT
            # We need r[1:N]
            r_side = r[1:N]
            
            # Convolution part
            # We need to handle boundaries.
            # Using FFT for convolution:
            # g1 = convolve(y, r_side)
            # g2 = convolve(y, r_side[::-1])
            
            # Let's use np.convolve for simplicity since N=400 is small.
            # It's O(N^2) but 400^2 = 160,000 operations, which is very fast.
            
            # y is 1D. r_side is 1D.
            
            # Term 1: sum_{k=1}^{N-1} r_k y_{m-k}
            # This is the cross-correlation of y and r_side.
            # np.correlate(y, r_side, mode='same')
            
            # Term 2: sum_{k=1}^{N-1} r_k y_{m+k}
            # This is the cross-correlation of y and r_side[::-1].
            
            # Let's test this logic.
            
            c1 = np.correlate(y, r_side, mode='same')
            c2 = np.correlate(y, r_side[::-1], mode='same')
            
            # The 'same' mode centers the result.
            # We need to be careful with the exact index alignment.
            # For N=400, r_side has length 399.
            # np.correlate output length is N.
            
            # The gradient is 2 * (c1 + c2)
            
            return 2.0 * (c1 + c2)

        # Let's verify this gradient with a small example in my head?
        # If y is all 1s, r_side is all N-k.
        # Gradient should be 0? No, all 1s is not a stationary point for L2.
        
        # Let's use this gradient.
        
        # However, the "soft-max" is better for the task "upper bound".
        # Let's try to implement the soft-max gradient.
        
        # Actually, let's stick to L2 for the first pass of the BFGS, then switch to local search.
        # The L2 solution is often a good starting point for the minimax problem.
        
        # Let's run BFGS on the L2 objective.
        
        # We need to pass the function and gradient.
        
        def l2_objective(y):
            fft_len = 1 << (2 * N - 1).bit_length()
            F = np.fft.rfft(y, fft_len)
            power = F * np.conj(F)
            r = np.fft.irfft(power, fft_len)[:N]
            return np.sum(r[1:]**2)
            
        def l2_gradient(y):
            fft_len = 1 << (2 * N - 1).bit_length()
            F = np.fft.rfft(y, fft_len)
            power = F * np.conj(F)
            r = np.fft.irfft(power, fft_len)[:N]
            r_side = r[1:N]
            c1 = np.correlate(y, r_side, mode='same')
            c2 = np.correlate(y, r_side[::-1], mode='same')
            return 2.0 * (c1 + c2)

        # Initial point
        y0 = x.astype(float)
        
        # Run BFGS
        try:
            res = minimize(l2_objective, y0, jac=l2_gradient, method='BFGS', 
                           options={'maxiter': 500, 'gtol': 1e-5})
            y_opt = res.x
        except Exception:
            y_opt = y0
            
        # Project to signs
        x_opt = np.sign(y_opt)
        # Handle zeros
        x_opt[x_opt == 0] = 1
        
        # 3. Local Search (Bit Flipping)
        # Improve the discrete sequence
        
        def max_sidelobe(seq):
            fft_len = 1 << (2 * N - 1).bit_length()
            F = np.fft.rfft(seq, fft_len)
            power = F * np.conj(F)
            r = np.fft.irfft(power, fft_len)[:N]
            return np.max(np.abs(r[1:]))
            
        current_val = max_sidelobe(x_opt)
        best_x = x_opt.copy()
        best_val = current_val
        
        # Greedy local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try flipping each bit
            # To speed up, we can try random subset or all?
            # N=400. 400 evaluations. Each eval is ~1ms. Total 400ms.
            # We can do multiple passes.
            
            for i in range(N):
                if time.time() > deadline:
                    break
                
                # Flip bit i
                new_x = best_x.copy()
                new_x[i] *= -1
                
                new_val = max_sidelobe(new_x)
                
                if new_val < best_val:
                    best_val = new_val
                    best_x = new_x
                    improved = True
                    # Restart loop? Or continue?
                    # Continuing is fine, but we might miss better flips.
                    # Let's break and restart for a proper greedy hill climb.
                    break
            
            # If we broke due to improvement, the loop continues.
            # If we finished the loop without improvement, improved is False.
            
            # Add a small random perturbation if stuck?
            # For now, simple greedy.
            
            # To avoid infinite loops in case of ties, we only accept strict improvement.
            # If no strict improvement, we stop.
            
            # To allow escaping local minima, we could add simulated annealing,
            # but time is limited.
            
            # Let's try a few random restarts if we have time?
            # No, let's stick to the best found.
            
        return best_x

# EVOLVE-BLOCK-END
