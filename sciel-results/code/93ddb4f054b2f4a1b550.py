"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Returns three coefficients (c0, c4, c8) for the Hermite combination.
    The degree-12 weight is fixed so g(0) = 0.
    The goal is to minimize the upper bound on C4 = A(f)*A(F).
    """
    start_time = time.time()
    deadline = 210.0 - 5.0  # 5s safety margin

    # Hermite polynomials H0, H4, H8, H12
    # We work with the probabilists' Hermite polynomials He_n(x) scaled by Gaussian weight.
    # The test function is g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    # where c12 is determined by g(0) = 0.
    
    # H0(x) = 1
    # H4(x) = x^4 - 6x^2 + 3
    # H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12(x) = x^12 - 60x^10 + 1500x^8 - 15120x^6 + 69300x^4 - 103950x^2 + 10395
    
    # At x=0:
    # H0(0) = 1
    # H4(0) = 3
    # H8(0) = 105
    # H12(0) = 10395
    
    # g(0) = (c0*1 + c4*3 + c8*105 + c12*10395) * exp(0) = 0
    # => c12 = -(c0 + 3*c4 + 105*c8) / 10395
    
    # The uncertainty bound involves computing A(f) and A(F).
    # For a Gaussian-weighted Hermite function, we can compute the L1 norm and the L1 norm of the Fourier transform.
    
    # Let's use numerical integration to compute the bound.
    # The Fourier transform of H_n(x) * exp(-x^2/2) is i^n * H_n(x) * exp(-x^2/2) (up to normalization).
    # Actually, for the standard Hermite functions, the Fourier transform is related.
    
    # Let's define g(x) = P(x) * exp(-x^2/2) where P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # A(f) = integral |f(x)| dx
    # A(F) = integral |F(f)(k)| dk
    
    # For the Fourier transform, we need to be careful with conventions.
    # Let's use the convention F(f)(k) = integral f(x) exp(-2*pi*i*k*x) dx
    # Then F of exp(-x^2/2) is sqrt(2*pi) * exp(-2*pi^2*k^2)
    # And F of H_n(x) * exp(-x^2/2) involves a rotated/scaled Hermite function.
    
    # Actually, it's easier to use the fact that the Hermite functions form an eigenbasis of the Fourier transform.
    # The n-th Hermite function psi_n(x) = (2^n n! sqrt(pi))^{-1/2} H_n(x) exp(-x^2/2)
    # has Fourier transform i^n psi_n(x) (with appropriate normalization).
    
    # Let's work directly with the polynomial times Gaussian.
    # g(x) = P(x) exp(-x^2/2)
    
    # A(g) = integral |P(x) exp(-x^2/2)| dx
    # A(Fg) = integral |Fg(k)| dk
    
    # For the Fourier transform of P(x) exp(-x^2/2):
    # If P(x) = sum c_n H_n(x), then F[P exp(-x^2/2)] = sum c_n F[H_n exp(-x^2/2)]
    
    # F[H_n(x) exp(-x^2/2)] = i^n sqrt(2*pi) H_n(x) exp(-x^2/2) ... this is not quite right.
    
    # Let's use numerical integration with a grid.
    
    # Grid for integration
    N = 2000
    x = np.linspace(-10, 10, N)
    dx = x[1] - x[0]
    
    def hermite_poly(n, x):
        """Compute the n-th probabilists' Hermite polynomial."""
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return x.copy()
        else:
            H0 = np.ones_like(x)
            H1 = x.copy()
            for k in range(2, n+1):
                H2 = k * x * H1 - (k-1) * H0
                H0, H1 = H1, H2
            return H1
    
    # Precompute Hermite polynomials on the grid
    H0 = hermite_poly(0, x)
    H4 = hermite_poly(4, x)
    H8 = hermite_poly(8, x)
    H12 = hermite_poly(12, x)
    
    # Gaussian weight
    gauss = np.exp(-x**2 / 2)
    
    # c12 as function of c0, c4, c8
    def get_c12(c0, c4, c8):
        return -(c0 + 3*c4 + 105*c8) / 10395.0
    
    def compute_bound(c0, c4, c8):
        """Compute the upper bound A(g)*A(Fg)."""
        c12 = get_c12(c0, c4, c8)
        
        # g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
        P = c0*H0 + c4*H4 + c8*H8 + c12*H12
        g = P * gauss
        
        # A(g) = integral |g(x)| dx
        A_g = np.sum(np.abs(g)) * dx
        
        # For the Fourier transform, we need Fg(k).
        # Fg(k) = integral g(x) exp(-2*pi*i*k*x) dx
        # We can compute this numerically using FFT or direct integration.
        
        # Let's use the property that for Hermite functions, the Fourier transform is related.
        # Actually, let's just compute it numerically on a frequency grid.
        
        # Frequency grid
        k = np.linspace(-5, 5, N)
        dk = k[1] - k[0]
        
        # Compute Fg(k) = integral g(x) exp(-2*pi*i*k*x) dx
        # This is a matrix-vector product, which is expensive.
        # Alternative: use the fact that g(x) is a polynomial times Gaussian.
        # The Fourier transform of x^n exp(-x^2/2) can be computed analytically.
        
        # Let's expand P(x) in powers of x and use known Fourier transforms.
        # P(x) = c0*(1) + c4*(x^4 - 6x^2 + 3) + c8*(x^8 - 28x^6 + 210x^4 - 420x^2 + 105) + c12*(x^12 - 60x^10 + 1500x^8 - 15120x^6 + 69300x^4 - 103950x^2 + 10395)
        
        # Let's compute the coefficients of x^m in P(x)
        # P(x) = a0 + a1*x + a2*x^2 + ... + a12*x^12
        
        a = np.zeros(13)
        a[0] = c0 + 3*c4 + 105*c8 + 10395*c12
        a[2] = -6*c4 - 420*c8 - 103950*c12
        a[4] = c4 + 210*c8 + 69300*c12
        a[6] = -28*c8 - 15120*c12
        a[8] = c8 + 1500*c12
        a[10] = -60*c12
        a[12] = c12
        
        # The Fourier transform of x^n exp(-x^2/2) is related to derivatives of the Gaussian.
        # F[x^n exp(-x^2/2)] = i^n * d^n/dk^n [sqrt(2*pi) exp(-2*pi^2 k^2)]
        
        # This gets complicated. Let's just do numerical integration with FFT.
        # Actually, for a 210s budget, we can afford some computation.
        
        # Let's use a smaller grid for the frequency domain to keep it fast.
        Nk = 1000
        k = np.linspace(-5, 5, Nk)
        dk = k[1] - k[0]
        
        # Compute Fg(k) using the trapezoidal rule
        # Fg(k) = sum_j g(x_j) exp(-2*pi*i*k*x_j) dx
        # This is O(N*Nk) which is 2000*1000 = 2e6, acceptable.
        
        # Build the Fourier matrix
        # Actually, let's use FFT if the grid is uniform.
        # x is uniform, so we can use FFT.
        
        # But the frequency grid from FFT might not be what we want.
        # Let's just do the direct computation for now, or use a smarter approach.
        
        # Alternative: use the fact that the L1 norm of the Fourier transform can be bounded.
        # For this task, let's just compute it numerically.
        
        # To save time, let's use a coarser grid.
        N_x = 1000
        x_fine = np.linspace(-10, 10, N_x)
        dx_fine = x_fine[1] - x_fine[0]
        
        H0_f = hermite_poly(0, x_fine)
        H4_f = hermite_poly(4, x_fine)
        H8_f = hermite_poly(8, x_fine)
        H12_f = hermite_poly(12, x_fine)
        gauss_f = np.exp(-x_fine**2 / 2)
        
        P_f = c0*H0_f + c4*H4_f + c8*H8_f + c12*H12_f
        g_f = P_f * gauss_f
        
        A_g_f = np.sum(np.abs(g_f)) * dx_fine
        
        # Compute Fourier transform numerically
        # Use FFT for efficiency
        # The FFT computes the DFT, which approximates the continuous FT.
        # Fg(k) ≈ dx * sum g(x_j) exp(-2*pi*i*k*x_j)
        
        # For FFT, we need the frequency grid to match.
        # Let's just use a direct sum with a smaller grid.
        Nk_small = 500
        k_small = np.linspace(-5, 5, Nk_small)
        dk_small = k_small[1] - k_small[0]
        
        # Fg(k) for each k
        # This is a matrix product: (Nk x N_x) matrix times g_f vector
        # exp(-2*pi*i * outer(k_small, x_fine)) @ g_f * dx_fine
        
        # To avoid memory issues, let's do it in chunks or use FFT.
        # Let's use FFT.
        
        # Pad g_f to power of 2 for efficient FFT
        N_fft = 2048
        g_fft = np.zeros(N_fft, dtype=complex)
        g_fft[:N_x] = g_f
        
        # FFT
        G_fft = np.fft.fft(g_fft) * dx_fine
        
        # The FFT output is at frequencies k = 2*pi*j/(N_fft*dx_fine) for j=0,...,N_fft-1
        # We want k in [-5, 5].
        # The frequency spacing is dk_fft = 2*pi/(N_fft*dx_fine)
        # We need to find which FFT bins correspond to k in [-5, 5].
        
        k_fft = 2*np.pi*np.fft.fftfreq(N_fft, d=dx_fine)
        
        # Find indices where |k_fft| <= 5
        mask = np.abs(k_fft) <= 5
        G_selected = G_fft[mask]
        k_selected = k_fft[mask]
        
        # Sort by k
        sort_idx = np.argsort(k_selected)
        k_sorted = k_selected[sort_idx]
        G_sorted = G_selected[sort_idx]
        
        # A(Fg) = integral |Fg(k)| dk
        # Use trapezoidal rule on the sorted grid
        if len(k_sorted) > 1:
            A_Fg = np.trapz(np.abs(G_sorted), k_sorted)
        else:
            A_Fg = 0
        
        return A_g_f * A_Fg
    
    # Initial guess
    c0_init = -113/100
    c4_init = 1/25
    c8_init = 1/3240
    
    x0 = np.array([c0_init, c4_init, c8_init])
    
    best_x = x0.copy()
    best_bound = compute_bound(c0_init, c4_init, c8_init)
    
    # Optimization
    def objective(x):
        return compute_bound(x[0], x[1], x[2])
    
    # Try a few optimization methods
    methods = ['Nelder-Mead', 'Powell']
    
    for method in methods:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(objective, x0, method=method, options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if result.fun < best_bound:
                best_bound = result.fun
                best_x = result.x
        except Exception:
            pass
    
    # Check time
    if time.time() > deadline:
        return np.array(best_x)
    
    # Return the best coefficients
    return np.array(best_x)
# EVOLVE-BLOCK-END
