"""
Strategy: STRUCTURED BASIS + NELLER-MEAD + GREEDY REFINEMENT

The current approach uses random smooth bumps (Gaussians/Cosines) which are 
"blobby" and tend to have high peak autocorrelation. 
Theoretical bounds on C1 are often approached by functions that are 
"flat" in the time domain (like a box) or have specific comb-like structures 
that distribute the autocorrelation energy more evenly.

This new strategy:
1. Represents f as a linear combination of a small set of structured basis 
   functions: 
   - A global box (flat top)
   - Gray-code-like oscillatory patterns (to create comb structures)
   - A few Gaussian bumps for local flexibility.
2. Uses Nelder-Mead to optimize the coefficients of this basis. This is 
   robust to non-smoothness and local minima compared to gradient methods 
   on the raw 600-dim vector.
3. Uses a Greedy Coordinate Descent (GCD) pass on the raw 600-dim vector 
   as a final polish. GCD is very effective for sparsity/peak reduction 
   problems and is fast enough for 600 dimensions.

This is a structural change from "random search + sensitivity gradient" to 
"parametric structured basis + simplex + coordinate descent".
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 600
TIME_LIMIT = 135  # safety margin before 150s kill

def compute_score_fft(f):
    """Compute the normalized maximum of the autoconvolution using FFT for speed."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    # FFT length must be >= 2N-1
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    F = np.fft.rfft(f, n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft)[:2 * N - 1]
    return np.max(conv) / (s * s)

def compute_score_exact(f):
    """Compute the normalized maximum of the autoconvolution exactly."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def generate_basis_functions():
    """
    Generate a set of structured basis functions.
    1. Box function (flat top)
    2. Gray-code-like oscillations (create comb structures)
    3. A few Gaussians for local adjustment
    """
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    basis = []
    
    # 1. Box function (centered, width 0.4)
    box = np.zeros(N)
    mask = np.abs(x) < 0.2
    box[mask] = 1.0
    basis.append(box)
    
    # 2. Gray-code-like / Walsh-like patterns
    # These create high-frequency structures that can reduce peak autocorrelation
    # We use a few low-order Walsh functions (Haar-like)
    def walsh(n, x):
        # Simple implementation of Walsh functions
        # Rademacher functions
        if n == 0:
            return np.ones_like(x)
        k, m = n, 0
        while k > 0:
            if k % 2 == 1:
                m += 1
            k //= 2
        # Walsh function W_n(x)
        # This is a simplified version, we just want orthogonal-ish oscillatory patterns
        # Let's use sine waves with specific frequencies instead for stability
        pass
        
    # Let's use specific frequencies that are integer multiples of the fundamental
    # f(x) = cos(2*pi*k*x)
    for k in [2, 4, 6, 8, 10]:
        # Ensure non-negative by taking absolute value or shifting
        # We want non-negative basis functions. 
        # Let's use "combs": 1 + a*cos(2*pi*k*x) with a small enough a
        # Or better: max(0, 1 + 0.5*cos(2*pi*k*x))
        comb = np.maximum(0, 1.0 + 0.5 * np.cos(2 * np.pi * k * x))
        basis.append(comb)
        
    # 3. Gaussians at different positions
    for center in [-0.15, 0.0, 0.15]:
        sigma = 0.05
        gauss = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        basis.append(gauss)
        
    # 4. A "plateau" in the middle
    plat = np.zeros(N)
    mask = np.abs(x) < 0.1
    plat[mask] = 1.0
    basis.append(plat)
    
    return np.array(basis)

def param_to_step(params, basis):
    """Convert parameters to a step function."""
    # params[0] is the scale/offset, params[1:] are coefficients
    # We want f = max(0, sum(c_i * b_i))
    coeffs = params[1:]
    f = np.zeros(N)
    for c, b in zip(coeffs, basis):
        f += c * b
    return np.maximum(f, 0.0)

def objective(params, basis):
    """Objective function for Nelder-Mead."""
    f = param_to_step(params, basis)
    s = np.sum(f)
    if s < 1e-12:
        return 1e6
    score = compute_score_fft(f)
    return score

def greedy_coordinate_descent(f, max_iter=100, time_limit=10):
    """
    Greedy Coordinate Descent for minimizing peak autocorrelation.
    For each coordinate j, find the optimal value of f[j] given other f[i] fixed.
    """
    start_time = time.time()
    current_f = f.copy()
    current_score = compute_score_fft(current_f)
    best_f = current_f.copy()
    best_score = current_score
    
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
        
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        improved = False
        for j in range(N):
            if time.time() - start_time > time_limit:
                break
            
            # Compute convolution contribution of f[j] to the peak
            # conv[k] = sum_i f[i] f[k-i]
            # Changing f[j] affects conv[k] for k where j is involved.
            # conv[k] = f[j]*f[k-j] + f[k-j]*f[j] + ... (other terms)
            # Actually, let's just evaluate the score for a few candidate values of f[j]
            
            # Current value
            f_j = current_f[j]
            
            # Try reducing f[j] if it contributes to the peak
            # Sensitivity: d(conv[k])/df[j] = f[k-j]
            # The peak is at k_max.
            # If we reduce f[j], conv[k_max] decreases by 2*f[j]*f[k_max-j] (approx)
            # But other peaks might increase? No, reducing f[j] reduces all conv[k] where j is involved.
            # Wait, conv is symmetric.
            # Let's try a few values: 0, f_j * 0.5, f_j * 0.1
            
            candidates = [0.0, f_j * 0.5, f_j * 0.1, f_j]
            best_val = current_score
            best_fj = f_j
            
            for fj in candidates:
                if fj == f_j:
                    continue
                current_f[j] = fj
                s = np.sum(current_f)
                if s < 1e-12:
                    current_f[j] = f_j
                    continue
                F = np.fft.rfft(current_f, n=n_fft)
                conv = np.fft.irfft(F * F, n=n_fft)[:2 * N - 1]
                score = np.max(conv) / (s * s)
                if score < best_val:
                    best_val = score
                    best_fj = fj
            
            current_f[j] = best_fj
            if best_fj != f_j:
                improved = True
                current_score = best_val
                if current_score < best_score:
                    best_score = current_score
                    best_f = current_f.copy()
            
        if not improved:
            break
            
    return best_f, best_score

def construct():
    start_time = time.time()
    
    # Generate basis
    basis = generate_basis_functions()
    n_basis = len(basis)
    
    # Initial parameters: all zeros except a small positive offset
    # We optimize coefficients c_i
    # f = max(0, sum c_i b_i)
    # Let's start with a flat function
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    f_init = np.ones(N)
    
    # We can initialize coefficients to roughly match f_init
    # But Nelder-Mead works better with a small parameter space.
    # Let's just optimize the coefficients directly.
    
    # Initialize params: [offset, c1, c2, ...]
    # Actually, let's drop the offset and just use coefficients.
    # f = max(0, sum c_i b_i)
    
    # Start with coefficients that produce a flat function
    # Since basis includes a box, let's set c_box = 1, others 0
    init_params = np.zeros(n_basis)
    # Find the box index (first one)
    init_params[0] = 1.0
    
    best_f = f_init.copy()
    best_score = compute_score_fft(best_f)
    
    # Nelder-Mead optimization
    # We have n_basis parameters.
    # Method: Nelder-Mead is good for low-dimensional non-smooth problems.
    
    def obj_wrapper(params):
        return objective(params, basis)
    
    # Run Nelder-Mead
    try:
        res = minimize(obj_wrapper, init_params, method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6, 'disp': False})
        if res.fun < best_score:
            best_score = res.fun
            best_f = param_to_step(res.x, basis)
    except Exception as e:
        pass
    
    # Check time
    if time.time() - start_time > TIME_LIMIT - 20:
        return [float(x) for x in best_f]
    
    # Try a few different initializations for Nelder-Mead
    # 1. Comb-like
    init_params2 = np.zeros(n_basis)
    init_params2[1] = 1.0  # First comb
    try:
        res2 = minimize(obj_wrapper, init_params2, method='Nelder-Mead',
                        options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6, 'disp': False})
        if res2.fun < best_score:
            best_score = res2.fun
            best_f = param_to_step(res2.x, basis)
    except Exception:
        pass
    
    # 2. Gaussian-like
    init_params3 = np.zeros(n_basis)
    init_params3[-3] = 1.0  # Middle Gaussian
    try:
        res3 = minimize(obj_wrapper, init_params3, method='Nelder-Mead',
                        options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6, 'disp': False})
        if res3.fun < best_score:
            best_score = res3.fun
            best_f = param_to_step(res3.x, basis)
    except Exception:
        pass
        
    # Greedy Coordinate Descent on the best f
    if time.time() - start_time < TIME_LIMIT - 10:
        refined_f, refined_score = greedy_coordinate_descent(best_f, max_iter=50, time_limit=TIME_LIMIT - time.time() - 5)
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Final exact check
    final_score = compute_score_exact(best_f)
    if final_score < best_score:
        best_score = final_score
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    start = time.time()
    f = construct()
    elapsed = time.time() - start
    score = compute_score_exact(f)
    print(f"Score: {score}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
    print(f"Time: {elapsed:.2f}s")
