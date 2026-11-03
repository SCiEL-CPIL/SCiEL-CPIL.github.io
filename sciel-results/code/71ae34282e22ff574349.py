import numpy as np
import time
import scipy.optimize as so

N = 600

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0  # safety margin
    h_width = (1.0 / 2.0) / N
    # Midpoints of the 600 intervals in [-0.25, 0.25]
    # Intervals: [-0.25, -0.25 + h_width), ..., [0.25 - h_width, 0.25]
    # Midpoint i: -0.25 + (i + 0.5) * h_width
    midpoints = -0.25 + (np.arange(N) + 0.5) * h_width
    # Map to angle theta in [0, 2pi) for z = e^(i * 2 * pi * x) ? 
    # The prompt says z = e^(i theta) where theta maps interval to [0, 2pi).
    # Let's assume x in [-0.25, 0.25] maps to theta in [0, 2pi).
    # theta = 2 * pi * (x + 0.25) / 0.5 = 2 * pi * (2x + 0.5) = 4*pi*x + pi
    # Or simply map x to [0, 1) then to [0, 2pi).
    # Let's use theta = 2 * pi * (x - min_x) / (max_x - min_x)
    # min_x = -0.25, max_x = 0.25, range = 0.5
    # theta = 2 * pi * (x + 0.25) / 0.5 = 4 * pi * (x + 0.25)
    thetas = 4.0 * np.pi * (midpoints + 0.25)
    
    def score_from_heights(heights):
        # heights: array of 600 floats
        # Compute linear autoconvolution
        M = 1
        while M < 2 * N - 1:
            M *= 2
        F = np.fft.rfft(heights, M)
        conv = np.fft.irfft(F * F.conj(), M)
        peak = np.max(conv[:2*N-1])
        l1 = np.sum(heights) * h_width
        if l1 == 0:
            return float('inf')
        # Normalized max: peak / l1^2
        return peak / (l1 * l1)

    def evaluate_params(params):
        # params: flattened array of [real(c0), imag(c0), real(c1), imag(c1), ...]
        K = len(params) // 2
        coeffs = params[0::2] + 1j * params[1::2]
        
        # Compute |sum c_k e^(i k theta)|^2
        # z = e^(i theta)
        # P(z) = sum c_k z^k
        # We can compute this efficiently using powers of z
        z = np.exp(1j * thetas)
        # Vectorized polynomial evaluation
        # h = |P(z)|^2
        # P(z) = c_0 + c_1 z + c_2 z^2 + ...
        # We can use np.polynomial or just loop/unroll if K is small.
        # For K up to 10, loop is fine.
        P_val = np.zeros(N, dtype=complex)
        z_power = np.ones(N, dtype=complex)
        for k in range(K + 1):
            P_val += coeffs[k] * z_power
            z_power *= z
            
        heights = np.abs(P_val) ** 2
        # Ensure non-negative (it is by construction)
        # Optional: clip tiny negatives if any (shouldn't happen)
        
        val = score_from_heights(heights)
        return val

    best_val = float('inf')
    best_heights = np.ones(N) / N # fallback uniform
    
    # Try different degrees K
    # K=0 is constant. K=1 is linear in cos/sin squared.
    for K in [2, 4, 6, 8]:
        if time.time() > deadline - 20:
            break
            
        n_params = 2 * (K + 1)
        
        # Bounds for coefficients
        # We can scale them. Let's use [-1, 1] for real and imag parts.
        # Differential Evolution will search this space.
        bounds = [(-1.0, 1.0)] * n_params
        
        # Objective function wrapper for DE
        def obj_de(params):
            return evaluate_params(params)
            
        try:
            # Use DE
            # maxiter and popsize need to be balanced with time.
            # 120s limit. Let's allow ~10-20s per K.
            result = so.differential_evolution(
                obj_de, 
                bounds, 
                maxiter=50, 
                popsize=10, 
                tol=0.01, 
                seed=42,
                update_interval=1,
                polish=True
            )
            
            if result.fun < best_val:
                best_val = result.fun
                # Recover heights
                coeffs = result.x[0::2] + 1j * result.x[1::2]
                z = np.exp(1j * thetas)
                P_val = np.zeros(N, dtype=complex)
                z_power = np.ones(N, dtype=complex)
                for k in range(K + 1):
                    P_val += coeffs[k] * z_power
                    z_power *= z
                best_heights = np.abs(P_val) ** 2
                
        except Exception as e:
            pass

    # If DE didn't find anything good, or if we want to polish the best found
    # We can run a local optimization on the best heights found?
    # The directive asks for the structural change to be the main driver.
    # However, adding a local polish on the discrete vector might help.
    # But we must stick to the "sparse polynomial" representation?
    # The directive says "Represent the step function as...". 
    # It doesn't explicitly forbid polishing the resulting vector, 
    # but the "search space" reduction is the key.
    # Let's stick to the parametric search result to be safe and fast.
    
    # Normalize output? 
    # The score is scale-invariant. 
    # The evaluator expects a sequence of nonnegative step heights.
    # We return the heights.
    
    out = np.maximum(best_heights, 0.0)
    s = np.sum(out)
    if s > 0:
        out = out / s # Normalize L1 to 1 for consistency, though score is invariant
    
    return [float(x) for x in out]
# EVOLVE-BLOCK-END
