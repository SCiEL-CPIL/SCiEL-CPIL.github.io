"""Refined Cohn-Goncalves Laguerre-polynomial construction optimizer."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_laguerre

# EVOLVE-BLOCK-START

def _compute_c4_bound(roots, N=2048):
    """
    Compute the C4 upper bound for the given roots.
    The bound is proportional to || \hat{f} ||_1 where f is the indicator of the set
    where the Laguerre polynomial P(x) > 0.
    
    We approximate the integral numerically.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != 12:
        return 1e9
    
    # Ensure roots are positive and sorted
    roots = np.sort(roots)
    if np.any(roots <= 0):
        return 1e9
    
    # Determine the support range. 
    # For Laguerre polynomials, the roots are in (0, infinity).
    # The polynomial oscillates between roots.
    # We need to define the grid carefully.
    
    # The "positive set" S is the union of intervals where P(x) > 0.
    # For a polynomial with double roots, the sign doesn't change at the roots.
    # Wait, double roots mean (x-r)^2, so the sign is the same on both sides.
    # So the sign only changes if there are odd multiplicity roots?
    # The problem states "12 double roots".
    # If all roots are double, the polynomial is non-negative or non-positive everywhere?
    # No, Laguerre polynomials of degree m have m simple roots.
    # The "Cohn-Goncalves construction" likely refers to a specific polynomial 
    # formed from Laguerre polynomials that has double roots.
    # Let's assume the polynomial is P(x) = \prod_{i=1}^{12} (x - r_i)^2 * (some positive factor)
    # Or perhaps it's a linear combination of Laguerre polynomials that happens to have double roots.
    
    # Actually, the standard Cohn-Goncalves construction for uncertainty principles 
    # often involves the function f(x) = e^{-x/2} L_m(x) or similar.
    # The "sign uncertainty" usually refers to the measure of the set where 
    # f(x) and \hat{f}(x) have different signs, or similar.
    
    # However, the task says "upper bound on C4" and "score is the upper bound".
    # Let's look at the inspiration: "Direct L1-Norm Minimization of the Fourier Transform".
    # This suggests the bound is related to \int |\hat{f}(t)| dt.
    
    # Let's model the function f(x).
    # If the roots are double, f(x) doesn't change sign at the roots.
    # This implies the "sign" is constant? That would make sign uncertainty trivial.
    # Perhaps the "roots" refer to the zeros of the derivative, or the turning points?
    # Or maybe the polynomial is of the form \sum c_k L_k(x) and we are tuning it 
    # to have specific roots?
    
    # Let's re-read carefully: "Tune the refined Cohn-Goncalves Laguerre-polynomial construction... 
    # notebook supplies 12 double roots... verifies root constraints... computes the last sign change."
    
    # "Last sign change" implies the polynomial DOES change sign.
    # If it has 12 double roots, it doesn't change sign at those points.
    # Maybe the "double roots" are in the squared modulus |f|^2?
    # If |f(x)|^2 has double roots, then f(x) has simple roots.
    # This is consistent with "sign change".
    # So, let's assume f(x) has simple roots at r_1, ..., r_12.
    # And f(x) is related to Laguerre polynomials.
    
    # A common construction is f(x) = e^{-x/2} P(x) where P is a polynomial.
    # Or f(x) is a sum of Laguerre functions.
    
    # Let's assume the function is f(x) = \prod_{i=1}^{12} (x - r_i) * e^{-x/2} (or similar decay)
    # But we need to normalize it.
    
    # Let's try a specific model:
    # f(x) = e^{-x/2} \prod_{i=1}^{12} (x - r_i)
    # This function is zero at r_i.
    # The Fourier transform \hat{f}(t) can be computed.
    # The C4 bound might be \frac{\|\hat{f}\|_1}{\|f\|_1} or similar.
    
    # Let's compute \|\hat{f}\|_1 / \|f\|_2^2 or something similar.
    # The inspiration says "L1-Norm Minimization of the Fourier Transform".
    # So we want to minimize \|\hat{f}\|_1.
    
    # Grid for x: [0, X_max]
    # X_max should cover the largest root plus some tail.
    max_root = np.max(roots)
    X_max = max_root * 2 + 10
    
    x = np.linspace(0, X_max, N)
    dx = x[1] - x[0]
    
    # Compute f(x)
    # To avoid overflow/underflow with the product, use log-sum-exp or just direct product with clipping.
    # Since roots are positive and x is positive, (x-r) can be negative.
    # The product will oscillate in sign.
    
    # Let's build the polynomial part
    # P(x) = prod (x - r_i)
    # We can compute this iteratively
    P = np.ones_like(x)
    for r in roots:
        P = P * (x - r)
        
    # Decay factor
    decay = np.exp(-x / 2.0)
    
    f = P * decay
    
    # Normalize f?
    # The bound usually depends on the shape, not the scale, if it's a ratio.
    # If it's just ||\hat{f}||_1, then scaling f scales the result.
    # We must fix the norm of f.
    # Let's fix ||f||_2 = 1.
    norm_f = np.sqrt(np.sum(f**2) * dx)
    if norm_f < 1e-12:
        return 1e9
    f = f / norm_f
    
    # Compute Fourier Transform
    # \hat{f}(t) = \int f(x) e^{-2\pi i t x} dx
    # Use FFT.
    # We need t-grid.
    # The bandwidth is related to the support size.
    # Let's use a sufficiently large FFT.
    n_fft = 4 * N
    f_pad = np.zeros(n_fft)
    f_pad[:N] = f
    # fftshift to center? No, f is defined on [0, X_max].
    # Standard FFT assumes x starts at 0.
    F = np.fft.fft(f_pad) * dx
    freqs = np.fft.fftfreq(n_fft, d=dx)
    
    # We only care about positive frequencies? Or all?
    # L1 norm is over all t.
    # Due to symmetry for real f, \hat{f}(-t) = \hat{f}(t)^*.
    # So ||\hat{f}||_1 = 2 * \int_0^\infty |\hat{f}(t)| dt (approx).
    
    # We need to integrate |\hat{f}(t)| over the frequency range covered by FFT.
    # The max frequency in FFT is 1/(2*dx).
    # 1/(2*dx) = N / (2 * X_max)
    # For N=2048, X_max ~ 50, max_freq ~ 20.
    # This should be enough for the main lobe.
    
    # Compute L1 norm
    l1_norm = np.sum(np.abs(F)) * dx  # dx here is dt?
    # dt = 1/(n_fft * dx_x)
    # The FFT output F[k] approximates \int f(x) e^{-2\pi i k T x} dx
    # The spacing in frequency is dt = 1/(n_fft * dx_x)
    dt = 1.0 / (n_fft * dx)
    l1_norm = np.sum(np.abs(F)) * dt
    
    return l1_norm

def _objective(roots):
    """
    Objective function to minimize.
    Adds penalties for invalid roots.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != 12:
        return 1e9
    if np.any(roots <= 0):
        return 1e9
    if np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
        return 1e9
    
    # Sort roots
    roots = np.sort(roots)
    
    # Check for very small gaps (numerical instability)
    gaps = np.diff(roots)
    if np.any(gaps < 1e-6):
        return 1e9
        
    val = _compute_c4_bound(roots, N=1024) # Use smaller N for speed during optimization
    
    # Add a small penalty for large roots to keep them in a reasonable range
    # Laguerre roots typically scale with m. For m=12, roots are O(10).
    if np.max(roots) > 100:
        val += (np.max(roots) - 100) ** 2 * 0.1
        
    return val

def construct():
    """
    Construct the 12 positive double-root locations for the Laguerre 
    combination (m=12, n=1).
    
    Returns a list of 12 positive real numbers representing the root locations.
    """
    deadline = time.time() + 600.0
    time_limit_per_eval = 5.0 # seconds per evaluation roughly
    
    # Initial guess: Roots of Laguerre polynomial L_12(x)?
    # Or simply uniform spacing.
    # Roots of L_12(x) are in (0, 24) roughly.
    # Let's try to get initial roots from scipy if possible, or just uniform.
    
    # Let's use the roots of the Laguerre polynomial L_12(x) as a starting point.
    # scipy.special.roots_laguerre(12)
    try:
        from scipy.special import roots_laguerre
        r, w = roots_laguerre(12)
        initial_roots = r
    except ImportError:
        initial_roots = np.linspace(1, 24, 12)
        
    best_roots = initial_roots.copy()
    best_score = _objective(best_roots)
    
    # Multi-start local optimization
    # We will generate several random perturbations of the initial roots
    # and run Nelder-Mead on each.
    
    num_starts = 20
    rng = np.random.default_rng(42)
    
    for i in range(num_starts):
        if time.time() > deadline - 30.0:
            break
            
        # Generate a perturbed initial guess
        # Scale and shift
        scale = rng.uniform(0.5, 1.5)
        shift = rng.uniform(-2, 2)
        noise = rng.normal(0, 0.5, 12)
        
        candidate = initial_roots * scale + shift + noise
        # Ensure positive
        candidate = np.maximum(candidate, 0.1)
        # Ensure sorted
        candidate = np.sort(candidate)
        
        # Run local minimization
        try:
            res = minimize(
                _objective, 
                candidate, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_roots = res.x
        except Exception:
            pass
            
    # Final polish with higher resolution
    if time.time() < deadline - 10.0:
        try:
            # Define a higher-res objective
            def _objective_high_res(roots):
                roots = np.asarray(roots, dtype=float)
                if len(roots) != 12 or np.any(roots <= 0):
                    return 1e9
                roots = np.sort(roots)
                if np.any(np.diff(roots) < 1e-6):
                    return 1e9
                return _compute_c4_bound(roots, N=2048)
            
            res = minimize(
                _objective_high_res, 
                best_roots, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res.fun < best_score:
                best_roots = res.x
        except Exception:
            pass

    # Final cleanup
    roots = np.sort(best_roots)
    roots = np.maximum(roots, 1e-6)
    
    # Ensure strictly increasing
    for i in range(1, len(roots)):
        if roots[i] <= roots[i-1]:
            roots[i] = roots[i-1] + 1e-6
            
    return [float(r) for r in roots]

# EVOLVE-BLOCK-END
