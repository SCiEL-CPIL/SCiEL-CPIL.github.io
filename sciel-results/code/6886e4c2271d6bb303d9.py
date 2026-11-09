import time
import numpy as np
from scipy.special import roots_genlaguerre
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
from scipy.fft import fft, ifft, fftfreq

def construct():
    """
    Returns the 12 positive double-root locations for the Laguerre combination
    using numerical optimization to minimize the Fourier sign uncertainty bound.
    """
    start_time = time.time()
    deadline = 600.0
    margin = 10.0 # Safety margin
    
    # Initial guess: Zeros of L_12^(1)(x)
    try:
        x0 = roots_genlaguerre(12, 1)
        x0 = np.sort(x0)
    except Exception:
        # Fallback if scipy fails
        x0 = np.linspace(1, 12, 12)
        
    # The objective function: 
    # We model the uncertainty bound via the L1 norm of the Fourier transform 
    # of the radial profile associated with the roots.
    # Let f(r) = exp(-r^2/2) * prod_{i=1}^{12} (r^2 - x_i)^2
    # The Fourier transform in 1D (radial reduction for n=1) involves an integral.
    # However, a simpler proxy for "uncertainty" in this context is often 
    # related to the spread or the L1 norm of the transform.
    # To ensure speed, we approximate the L1 norm using a Discrete Fourier Transform (DFT)
    # on a sufficiently fine grid, or use quad if the grid is too coarse.
    # Given 12 roots, the function is a polynomial of degree 24 times a Gaussian.
    
    # Grid setup for DFT approximation
    # The function decays as exp(-r^2/2). Significant support is r in [-10, 10].
    # We use a symmetric grid.
    N = 4096
    L = 20.0 # Half-width
    r = np.linspace(-L, L, N)
    dr = r[1] - r[0]
    
    def objective(params):
        """
        params: array of 12 root locations x_i
        Returns: Approximated L1 norm of the Fourier transform of f(r)
        """
        # Ensure roots are positive and sorted to avoid symmetry issues in product
        # The problem states "positive double-root locations"
        # We constrain optimization to x_i > 0
        
        # Compute f(r)
        # f(r) = exp(-r^2/2) * prod (r^2 - x_i)^2
        # To avoid overflow/underflow in product, we can compute log? 
        # But roots are small (< 20 usually), r is bounded.
        # Direct computation is fine for N=4096.
        
        # Efficient computation of product
        # f(r) = exp(-r^2/2)
        f_vals = np.exp(-0.5 * r**2)
        
        # Multiply by (r^2 - x_i)^2 for each i
        # This is O(N * 12)
        r2 = r**2
        for x_i in params:
            diff = r2 - x_i
            f_vals *= diff * diff
            
        # Compute Fourier Transform
        # The continuous FT is integral f(r) e^{-i k r} dr
        # DFT approximation: sum f(r_j) e^{-i k r_j} dr
        # We want the L1 norm: sum |FT(k)| dk
        # Using FFT:
        # Note: scipy.fft.fft assumes input is sampled uniformly.
        # The frequency spacing dk = 2 * pi / (N * dr)
        
        F = fft(f_vals)
        # The FFT computes sum f_j exp(-2 pi i j k / N)
        # Our continuous FT kernel is exp(-i k r)
        # r_j = -L + j * dr
        # exp(-i k r_j) = exp(-i k (-L + j dr)) = exp(i k L) exp(-i k j dr)
        # The term exp(i k L) is a phase shift, doesn't affect magnitude.
        # The frequency bins in FFT correspond to k_m = 2 * pi * m / (N * dr)
        
        # We only need the magnitude |F(k)|
        abs_F = np.abs(F)
        
        # Scale by dr to approximate the integral
        l1_norm = np.sum(abs_F) * dr
        
        # Normalize? The bound is an upper bound. We just want to minimize this value.
        return l1_norm

    # Constraint: x_i > 0
    # We use differential evolution or L-BFGS-B with bounds.
    # Differential evolution is robust for non-smooth or noisy objectives, 
    # but this objective is smooth. L-BFGS-B is faster per iteration.
    # However, the landscape might have local minima. 
    # Let's try L-BFGS-B first with the Laguerre roots as start.
    
    bounds = [(0.01, 50.0) for _ in range(12)]
    
    best_result = None
    best_val = float('inf')
    
    # Try L-BFGS-B
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 500, 'ftol': 1e-8})
        if res.success:
            best_val = res.fun
            best_result = res.x
        else:
            # Even if not fully converged, take the best point found if it's better
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x
    except Exception:
        pass

    # If L-BFGS-B failed or didn't improve, try a few random restarts with Nelder-Mead
    # or Differential Evolution if time permits.
    # Given the 600s limit, and objective eval taking ~0.01s, we can do ~1000 evals.
    # L-BFGS-B usually converges in < 100 evals.
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed < deadline - margin - 30: # Reserve 30s for safety
        # Try one random restart to escape local minima
        rng = np.random.default_rng(42)
        x_rand = rng.uniform(0.5, 15.0, size=12)
        x_rand = np.sort(x_rand)
        
        try:
            res2 = minimize(objective, x_rand, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-8})
            if res2.fun < best_val:
                best_val = res2.fun
                best_result = res2.x
        except Exception:
            pass
            
    # Finalize result
    if best_result is not None:
        # Ensure positive and sorted
        result = np.sort(np.abs(best_result))
        # Ensure exactly 12
        if len(result) == 12 and all(r > 0 for r in result):
            return [float(r) for r in result]
            
    # Fallback to seed if optimization failed or produced invalid results
    return [float(r) for r in x0]

# Keep the seed function name and signature
# The prompt asks to "Keep the function names and signatures of the seed program."
# The seed program has `construct`.
