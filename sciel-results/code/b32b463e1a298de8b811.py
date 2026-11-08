"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 600
# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # The domain is [-1/4, 1/4] with N=600 intervals.
    # dx = 0.5 / N
    dx = 0.5 / N
    
    # Objective function: Normalized maximum of autoconvolution
    # We want to minimize max(conv) / (integral^2)
    # Since integral is fixed by normalization in the final step, 
    # we can just minimize max(conv) for a fixed L1 norm, or minimize max(conv) / (sum(x)*dx)^2
    
    def objective(x):
        # x is a vector of nonnegative step heights
        # Compute autoconvolution using FFT
        n_fft = 2 * N
        x_pad = np.zeros(n_fft)
        x_pad[:N] = x
        
        X = np.fft.fft(x_pad)
        conv = np.fft.ifft(np.abs(X)**2).real
        
        # The linear autoconvolution is in the first N elements and last N-1 elements
        # Actually, for linear convolution of two length-N sequences, the result has length 2N-1
        # conv[0:N] gives the first part, and conv[N-1:2*N-1] gives the overlapping part
        # The max is over the whole valid range.
        
        s = np.sum(x)
        norm = (s * dx)**2
        if norm < 1e-15:
            return 1e15
        
        return np.max(conv) / norm

    # Strategy: Search over symmetric parametric shapes
    # The optimal function is likely symmetric around 0.
    # We can parameterize x[i] as f(i - N/2) where f is a symmetric function.
    
    # Candidate families:
    # 1. Uniform (baseline)
    # 2. Triangular (tent)
    # 3. Gaussian (width parameter)
    # 4. Cosine bell (width parameter)
    # 5. Box (width parameter)
    
    def generate_shape(shape_id, *params):
        # Returns a vector of length N
        centers = np.arange(N) - (N - 1) / 2.0
        if shape_id == 0:
            # Uniform
            return np.ones(N)
        elif shape_id == 1:
            # Triangular: max(0, 1 - |x|/width)
            width = params[0]
            return np.maximum(0, 1 - np.abs(centers) / width)
        elif shape_id == 2:
            # Gaussian: exp(-x^2 / (2*width^2))
            width = params[0]
            return np.exp(-centers**2 / (2 * width**2))
        elif shape_id == 3:
            # Cosine bell: cos(pi * x / width) for |x| < width, else 0
            width = params[0]
            mask = np.abs(centers) < width
            vals = np.zeros(N)
            vals[mask] = np.cos(np.pi * centers[mask] / width)
            return np.maximum(0, vals)
        elif shape_id == 4:
            # Box: 1 if |x| < width, else 0
            width = params[0]
            return (np.abs(centers) < width).astype(float)
        else:
            return np.ones(N)

    best_x = np.ones(N)
    best_score = objective(best_x)
    
    # Grid search over parameters for each shape
    # Width is in units of "centers" (i.e., number of intervals from center)
    # Max width is N/2 = 300
    
    # We will use a coarse grid first, then refine the best candidates.
    
    shapes_params = [
        (0, []), # Uniform
        (1, [100, 150, 200, 250, 300]), # Triangular
        (2, [50, 100, 150, 200, 250, 300]), # Gaussian
        (3, [100, 150, 200, 250, 300]), # Cosine
        (4, [100, 150, 200, 250, 300]) # Box
    ]
    
    candidates = []
    
    for shape_id, widths in shapes_params:
        if not widths:
            x = generate_shape(shape_id)
            candidates.append(x)
        else:
            for w in widths:
                x = generate_shape(shape_id, w)
                # Ensure non-negative
                x = np.maximum(0, x)
                candidates.append(x)
                
                if time.time() > deadline - 10:
                    break
        if time.time() > deadline - 10:
            break
    
    # Evaluate all candidates
    for x in candidates:
        if time.time() > deadline - 10:
            break
        score = objective(x)
        if score < best_score:
            best_score = score
            best_x = x.copy()
            
    # Refine the best candidate using local optimization on the full vector
    # But constrained to be symmetric? 
    # Actually, let's just do a local polish on the best_x using L-BFGS-B 
    # but with a smooth surrogate for the max to help gradients.
    
    # Smooth max approximation: log-sum-exp
    def smooth_objective(x):
        n_fft = 2 * N
        x_pad = np.zeros(n_fft)
        x_pad[:N] = x
        X = np.fft.fft(x_pad)
        conv = np.fft.ifft(np.abs(X)**2).real
        s = np.sum(x)
        norm = (s * dx)**2
        if norm < 1e-15:
            return 1e15
        # Use log-sum-exp for smooth max
        # We want to approximate max(conv)
        # Let M = max(conv). log-sum-exp = log(sum(exp(conv - M))) + M
        # To avoid overflow, subtract max first
        c_max = np.max(conv)
        lse = c_max + np.log(np.sum(np.exp(conv - c_max)))
        return lse / norm

    # We will try to polish best_x
    # Since the objective is non-smooth, L-BFGS-B might struggle.
    # However, the smooth surrogate is differentiable.
    # Let's try a few iterations of L-BFGS-B on the smooth surrogate starting from best_x.
    
    try:
        if time.time() < deadline - 20:
            # Use the smooth objective for gradient-based refinement
            # Note: The smooth objective is an upper bound on the true objective.
            # Minimizing it should still reduce the true objective.
            
            # We need to be careful with bounds. x >= 0.
            bounds = [(0, None)] * N
            
            # We will run minimize on the smooth objective
            # But we must check the true objective at the end.
            
            result = minimize(smooth_objective, best_x, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 1000, 'maxfun': 5000, 'gtol': 1e-6})
            
            if result.fun < smooth_objective(best_x):
                # Check true objective
                true_score = objective(result.x)
                if true_score < best_score:
                    best_x = result.x
                    best_score = true_score
    except Exception:
        pass

    # If time permits, try a few random symmetric perturbations of the best shape
    if time.time() < deadline - 10:
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            # Add small noise
            noise = np.random.normal(0, 0.01, N)
            # Make it symmetric
            noise = (noise + noise[::-1]) / 2
            x_trial = best_x + noise
            x_trial = np.maximum(0, x_trial)
            score = objective(x_trial)
            if score < best_score:
                best_score = score
                best_x = x_trial

    # Final check
    best_x = np.maximum(0, best_x)
    
    return list(best_x)
# EVOLVE-BLOCK-END
