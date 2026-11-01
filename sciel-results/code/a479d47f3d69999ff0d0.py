"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard: evaluator kills after 210s wall-clock.
# We aim to finish well before that.
DEADLINE_SECONDS = 190.0  # Safety margin

def _hermite_polynomials(x, max_n=12):
    """
    Compute probabilists' Hermite polynomials H_0..H_max_n at points x.
    Probabilists' Hermite: He_0=1, He_1=x, He_{n+1}=x*He_n - n*He_{n-1}
    """
    H = np.zeros((max_n + 1, len(x)))
    H[0, :] = 1.0
    if max_n >= 1:
        H[1, :] = x
    for n in range(2, max_n + 1):
        H[n, :] = x * H[n-1, :] - (n-1) * H[n-2, :]
    return H

def _compute_upper_bound(c0, c4, c8, c12, num_points=20000, x_range=15.0):
    """
    Compute the upper bound on C4 = A(f)*A(F) for the given Hermite combination.
    
    The test function is g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    We need to compute:
      A(f) = sup |f(x)| / ||f||_2  (or similar normalization)
      A(F) = sup |F(ξ)| / ||F||_2
    
    Actually, for the uncertainty inequality, we're looking at:
      A(f) * A(F) where A(f) = sup_x |f(x)| / (∫|f|^2)^{1/2}
    
    But the task says "upper bound on C4". Let me think about this more carefully.
    
    In the context of the Fourier uncertainty inequality, for a function f:
      A(f) = sup_x |f(x)| / ||f||_2
      A(F) = sup_ξ |F(ξ)| / ||F||_2
    
    The product A(f)*A(F) is bounded below by some constant. The task is to MINIMIZE
    this product by choosing the coefficients.
    
    Wait, re-reading: "reduce the upper bound on A(f)*A(Fourier(f))"
    
    Actually, I think the task is about finding a test function that achieves a low value
    of the product, which gives a tight lower bound on the uncertainty constant.
    
    Let me reconsider. The standard approach:
    - g(x) = P(x) * exp(-x^2/2) where P is a polynomial
    - Compute ||g||_∞ / ||g||_2 and ||ĝ||_∞ / ||ĝ||_2
    - The product gives a value, and we want to minimize it
    
    For the Fourier transform convention: F(ξ) = ∫ f(x) e^{-2πixξ} dx
    Or possibly: F(ξ) = ∫ f(x) e^{-ixξ} dx
    
    Let me use the unitary convention: F(ξ) = (1/√(2π)) ∫ f(x) e^{-ixξ} dx
    
    Actually, for Hermite functions, the natural convention is:
    H_n(x) e^{-x^2/2} are eigenfunctions of the Fourier transform (with appropriate normalization)
    
    Let me use: F(ξ) = ∫_{-∞}^{∞} f(x) e^{-ixξ} dx / √(2π)
    
    For f(x) = e^{-x^2/2}, F(ξ) = e^{-ξ^2/2}
    For f(x) = H_n(x) e^{-x^2/2}, F(ξ) = i^n H_n(ξ) e^{-ξ^2/2} (up to normalization)
    
    More precisely, with the convention F(ξ) = (1/√(2π)) ∫ f(x) e^{-ixξ} dx:
    If f_n(x) = (2^n n! √π)^{-1/2} H_n(x) e^{-x^2/2}, then F_n(ξ) = i^n f_n(ξ)
    
    But we're using unnormalized H_n(x) e^{-x^2/2}.
    
    Let me just compute numerically to be safe.
    
    The score is the upper bound on C4, which I believe is the value of A(f)*A(F) for our test function.
    Lower is better.
    """
    start_time = time.time()
    
    # Create evaluation grid
    x = np.linspace(-x_range, x_range, num_points)
    dx = x[1] - x[0]
    
    # Compute Hermite polynomials
    H = _hermite_polynomials(x, max_n=12)
    
    # Construct g(x)
    g = (c0 * H[0] + c4 * H[4] + c8 * H[8] + c12 * H[12]) * np.exp(-x**2 / 2.0)
    
    # Compute L2 norm of g
    norm_g_sq = np.trapz(g**2, x)
    if norm_g_sq <= 0:
        return 1e10
    
    norm_g = np.sqrt(norm_g_sq)
    
    # A(f) = sup |g(x)| / ||g||_2
    A_f = np.max(np.abs(g)) / norm_g
    
    # Compute Fourier transform of g
    # Use FFT for efficiency
    # We need to be careful with the FFT normalization
    
    # The grid is from -x_range to x_range with num_points points
    # FFT assumes the signal starts at 0, so we need to shift
    g_shifted = np.roll(g, num_points // 2)
    
    # FFT
    G = np.fft.fft(g_shifted) * dx
    G = np.fft.fftshift(G)
    
    # The frequency grid
    freqs = np.fft.fftfreq(num_points, d=dx)
    freqs = np.fft.fftshift(freqs)
    
    # With the convention F(ξ) = ∫ g(x) e^{-ixξ} dx
    # The FFT gives us F at frequencies 2π*k/dx, but we need to be careful
    # Actually, np.fft.fft computes sum g[j] * exp(-2πi*j*k/N)
    # If our grid is x_j = -x_range + j*dx, then this corresponds to
    # F(2π*k/(N*dx)) approximately, with a phase factor
    
    # For the magnitude, the phase doesn't matter
    norm_G_sq = np.sum(np.abs(G)**2) * dx**2 / num_points  # Parseval-like
    # Actually, let's use Parseval's theorem more carefully
    # ||F||_2^2 = 2π ||g||_2^2 for the convention F(ξ) = ∫ g(x) e^{-ixξ} dx
    # But with discrete FFT, let's just compute directly
    
    # Let me recompute using a more direct approach
    # F(ξ_k) ≈ ∫ g(x) e^{-ixξ_k} dx ≈ dx * sum g(x_j) e^{-ix_j ξ_k}
    
    # The frequencies from FFT are ω_k = 2π*k/(N*dx) for k=0,...,N-1
    # After fftshift, they are centered
    
    # Actually, let me just use the fact that for our specific functions,
    # the Fourier transform has a known form. But to be safe, let's do it numerically.
    
    # Recompute Fourier transform more carefully
    # We want F(ξ) = ∫ g(x) e^{-ixξ} dx
    # On our grid, this is approximated by dx * sum_j g(x_j) e^{-i x_j ξ_k}
    
    # The FFT gives us sum_j g_shifted[j] e^{-2πi jk/N}
    # g_shifted[j] = g(x_j - x_range) approximately (with the roll)
    
    # This is getting complicated. Let me use a simpler approach:
    # Since our function is a Hermite-Gaussian, its Fourier transform is also a Hermite-Gaussian.
    # Specifically, if g(x) = sum_n c_n H_n(x) e^{-x^2/2}, then
    # G(ξ) = sum_n c_n i^n H_n(ξ) e^{-ξ^2/2} (with the convention F(ξ) = (1/√(2π))∫g e^{-ixξ}dx)
    
    # Wait, let me verify this. With F(ξ) = (1/√(2π)) ∫ g(x) e^{-ixξ} dx:
    # For g(x) = e^{-x^2/2}: F(ξ) = e^{-ξ^2/2}
    # For g(x) = H_1(x) e^{-x^2/2} = x e^{-x^2/2}: F(ξ) = -i ξ e^{-ξ^2/2}... 
    # Actually, d/dx e^{-x^2/2} = -x e^{-x^2/2}, so H_1(x) e^{-x^2/2} = -d/dx e^{-x^2/2}
    # F(-d/dx f) = iξ F(ξ), so F(H_1 e^{-x^2/2}) = -iξ e^{-ξ^2/2}... 
    # Hmm, let me just use the known result: F(H_n(x) e^{-x^2/2}) = i^n H_n(ξ) e^{-ξ^2/2}
    # with the convention (1/√(2π))∫ f(x) e^{-ixξ} dx
    
    # So G(ξ) = (c0 * i^0 * H0(ξ) + c4 * i^4 * H4(ξ) + c8 * i^8 * H8(ξ) + c12 * i^12 * H12(ξ)) * e^{-ξ^2/2}
    # i^4 = 1, i^8 = 1, i^12 = i^4 * i^8 = 1
    # So G(ξ) = (c0 * H0(ξ) + c4 * H4(ξ) + c8 * H8(ξ) + c12 * H12(ξ)) * e^{-ξ^2/2}
    
    # Wait, that's the same as g! That can't be right for all n.
    # Let me double-check: i^0=1, i^4=1, i^8=1, i^12=i^(12 mod 4)=i^0=1
    # So yes, for n divisible by 4, i^n = 1.
    
    # Therefore G(ξ) = g(ξ) in this case!
    # That means A(F) = A(f) and the product is A(f)^2.
    
    # Let me verify with n=4: H_4(x) = x^4 - 6x^2 + 3
    # F(H_4(x) e^{-x^2/2}) = i^4 H_4(ξ) e^{-ξ^2/2} = H_4(ξ) e^{-ξ^2/2}
    # Yes, this is correct.
    
    # So we just need to evaluate g on a grid and find the max absolute value.
    
    # G(ξ) has the same form as g(x), so:
    A_F = A_f  # Since G(ξ) = g(ξ) for our specific combination
    
    # But wait, I should double-check the normalization.
    # With F(ξ) = (1/√(2π)) ∫ g(x) e^{-ixξ} dx, Parseval gives ||F||_2 = ||g||_2
    # So the normalization is consistent.
    
    # The product A(f) * A(F) = A_f * A_F = A_f^2
    
    product = A_f * A_F
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 20:
        # Return early if we're running out of time
        pass
    
    return product

def construct():
    """
    Returns three coefficients (c0, c4, c8) for the Hermite combination.
    The c12 coefficient is derived to impose g(0) = 0.
    
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    g(0) = (c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0)) * 1 = 0
    
    H0(0) = 1
    H4(0) = -3 (probabilists' Hermite: He_4(x) = x^4 - 6x^2 + 3, so He_4(0) = 3)
    
    Wait, I need to be careful about which Hermite polynomials.
    Probabilists': He_0=1, He_1=x, He_2=x^2-1, He_3=x^3-3x, He_4=x^4-6x^2+3
    So He_4(0) = 3, He_8(0) = ?, He_12(0) = ?
    
    He_n(0) = 0 for odd n.
    He_2(0) = -1
    He_4(0) = 3
    He_6(0) = -15
    He_8(0) = 105
    He_10(0) = -945
    He_12(0) = 10395
    
    These follow the pattern: He_{2k}(0) = (-1)^k (2k-1)!!
    He_2(0) = -1 = (-1)^1 * 1!!
    He_4(0) = 3 = (-1)^2 * 3!! = 3
    He_6(0) = -15 = (-1)^3 * 5!! = -15
    He_8(0) = 105 = (-1)^4 * 7!! = 105
    He_10(0) = -945 = (-1)^5 * 9!! = -945
    He_12(0) = 10395 = (-1)^6 * 11!! = 10395
    
    So g(0) = c0*1 + c4*3 + c8*105 + c12*10395 = 0
    => c12 = -(c0 + 3*c4 + 105*c8) / 10395
    """
    
    start_time = time.time()
    
    # Initial guess from the seed program
    c0_init = -113 / 100
    c4_init = 1 / 25
    c8_init = 1 / 3240
    
    def objective(params):
        c0, c4, c8 = params
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        try:
            val = _compute_upper_bound(c0, c4, c8, c12, num_points=10000, x_range=12.0)
            return val
        except Exception:
            return 1e10
    
    # Check initial value
    val_init = objective([c0_init, c4_init, c8_init])
    best_params = [c0_init, c4_init, c8_init]
    best_val = val_init
    
    # Try some optimization
    try:
        result = minimize(
            objective,
            [c0_init, c4_init, c8_init],
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )
        
        if result.fun < best_val:
            best_val = result.fun
            best_params = list(result.x)
        
    except Exception as e:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed < DEADLINE_SECONDS - 30:
        # Try another optimization method with different initial points
        try:
            # Try L-BFGS-B with gradient-free finite differences
            result2 = minimize(
                objective,
                [c0_init, c4_init, c8_init],
                method='Powell',
                options={'maxiter': 3000, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            if result2.fun < best_val:
                best_val = result2.fun
                best_params = list(result2.x)
        except Exception:
            pass
    
    # Final check on time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 10:
        # Return best result found
        c0, c4, c8 = best_params
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        return np.array([c0, c4, c8])
    
    # Polish with high-precision evaluation
    try:
        def objective_high_prec(params):
            c0, c4, c8 = params
            c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
            val = _compute_upper_bound(c0, c4, c8, c12, num_points=50000, x_range=15.0)
            return val
        
        result3 = minimize(
            objective_high_prec,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
        )
        if result3.fun < best_val:
            best_params = list(result3.x)
            best_val = result3.fun
    except Exception:
        pass
    
    c0, c4, c8 = best_params
    return np.array([c0, c4, c8])

# EVOLVE-BLOCK-END
