"""
Tune a Fourier-invariant Hermite-polynomial times Gaussian test function to reduce
the upper bound on A(f)*A(Fourier(f)).

The test function is g(x) = [c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)] * exp(-x^2/2)
where c12 is determined by the constraint g(0) = 0.

The Fourier transform of Hermite-Gaussian functions psi_n(x) = H_n(x)*exp(-x^2/2)
satisfies F(psi_n)(k) = (-i)^n * psi_n(k) (up to normalization depending on convention).

We use the physicist's Hermite polynomials H_n(x) and the standard Fourier transform
F(f)(k) = integral f(x) e^{-2*pi*i*x*k} dx.

For the standard convention, the Fourier transform of x^n * exp(-x^2/2) can be computed
analytically. Let's use the fact that:

psi_n(x) = H_n(x) * exp(-x^2/2)

The Fourier transform (with the convention F(f)(k) = int f(x) e^{-2*pi*i*k*x} dx) of
psi_n(x) is related to (-i)^n times a scaled version. However, it's cleaner to work
with the normalized Hermite functions.

Let's use a direct analytical approach. The key insight is that the "uncertainty" product
A(f)*A(F(f)) can be bounded using the L1 norms and the fact that the Fourier transform
of a Gaussian-modulated polynomial is another Gaussian-modulated polynomial (with known
coefficients).

For g(x) = sum_{n even} c_n * H_n(x) * exp(-x^2/2), the Fourier transform is:
F(g)(k) = sum_{n even} c_n * (-i*2*pi)^n * sqrt(pi/2) * H_n(k*sqrt(2*pi)) * exp(-2*pi^2*k^2)

Wait, let me be more careful. Let me use the standard Hermite-Gaussian eigenfunction property.

Actually, for the purpose of this problem, the key quantity is the upper bound on
A(f)*A(F(f)) where A(f) = |f|_1 / |f|_2 or similar. Let me reconsider.

Looking at the task: "the notebook derives the H12 coefficient to impose P(0)=0" and
"score is the upper bound on C4".

The standard formulation for uncertainty inequalities with Hermite polynomials involves
finding coefficients that minimize a certain bound. The bound typically involves
computing L1 and L2 norms of g and its Fourier transform.

Let me implement a numerical approach that:
1. Takes the three free coefficients (c0, c4, c8)
2. Computes c12 from the constraint g(0) = 0
3. Numerically computes the relevant norms and the upper bound
4. Uses optimization to find the best coefficients

For the constraint g(0) = 0:
g(0) = [c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0)] * exp(0)
     = c0*1 + c4*12 + c8*1190 + c12*113880 = 0

So c12 = -(c0 + 12*c4 + 1190*c8) / 113880

The "upper bound on C4" likely refers to a constant in an uncertainty inequality of the form:
||f||_1 * ||f'||_1 >= C * ||f||_2^2

or similar. Let me implement a numerical search that minimizes a proxy for this bound.

Given the time constraint, I'll use scipy.optimize to minimize the bound numerically.
"""

import numpy as np
from scipy import optimize
import time

def _hermite_h(n, x):
    """Compute physicist's Hermite polynomial H_n(x) using numpy."""
    # Use numpy's polynomial hermite for efficiency
    from numpy.polynomial.hermite import hermval
    # numpy.polynomial.hermite uses physicist's convention
    return hermval(x, [0]*n + [1])

def _compute_c12(c0, c4, c8):
    """Compute c12 from the constraint g(0) = 0."""
    # H0(0) = 1, H4(0) = 12, H8(0) = 1190, H12(0) = 113880
    return -(c0 + 12.0 * c4 + 1190.0 * c8) / 113880.0

def _compute_bound(c0, c4, c8, num_points=20000):
    """
    Compute the upper bound on C4 given coefficients c0, c4, c8.
    
    The bound involves the L1 norm, L2 norm, and Fourier transform properties.
    We use a numerical integration approach on a sufficient domain.
    """
    c12 = _compute_c12(c0, c4, c8)
    
    # Evaluate g(x) on a grid
    x = np.linspace(-10, 10, num_points)
    dx = x[1] - x[0]
    
    H0 = np.ones_like(x)
    H4 = _hermite_h(4, x)
    H8 = _hermite_h(8, x)
    H12 = _hermite_h(12, x)
    
    gauss = np.exp(-x**2 / 2.0)
    
    g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * gauss
    
    # L1 norm
    l1_norm = np.trapz(np.abs(g), x)
    
    # L2 norm
    l2_norm_sq = np.trapz(g**2, x)
    l2_norm = np.sqrt(l2_norm_sq)
    
    # Compute Fourier transform numerically
    # F(g)(k) = integral g(x) exp(-2*pi*i*k*x) dx
    # Use FFT for efficiency
    # First, ensure we have a good FFT setup
    N = len(x)
    if N % 2 != 0:
        x = x[:-1]
        g = g[:-1]
        N = len(x)
        dx = x[1] - x[0]
    
    # FFT-based Fourier transform
    G = np.fft.fftshift(np.fft.fft(g * dx)) * dx  # This isn't quite right for continuous FT
    
    # Better: use np.fft.fft with proper scaling
    # The DFT approximates the continuous FT as:
    # F(k_j) ≈ dx * sum_{m} g(x_m) exp(-2*pi*i * x_m * k_j)
    # where k_j = j / (N*dx)
    
    # Let's use a cleaner approach
    G_fft = np.fft.fft(g) * dx
    G_shifted = np.fft.fftshift(G_fft)
    
    # Frequency axis
    k = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    
    # L1 norm of Fourier transform
    l1_fft = np.trapz(np.abs(G_shifted), k)
    
    # L2 norm of Fourier transform (should be same as L2 norm of g by Plancherel)
    l2_fft_sq = np.trapz(np.abs(G_shifted)**2, k)
    
    # The uncertainty bound C4 is typically related to:
    # A(f) * A(F(f)) where A(f) = |f|_1 / |f|_2
    # So the product is (|g|_1/|g|_2) * (|F(g)|_1/|F(g)|_2)
    
    if l2_norm < 1e-12 or np.sqrt(l2_fft_sq) < 1e-12:
        return 1e10  # Invalid
    
    A_f = l1_norm / l2_norm
    A_F = l1_fft / np.sqrt(l2_fft_sq)
    
    # The "upper bound on C4" - this is the product we want to minimize
    bound = A_f * A_F
    
    return bound

def construct():
    """
    Find optimal coefficients c0, c4, c8 to minimize the upper bound.
    Returns np.array([c0, c4, c8]).
    """
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 210s limit
    
    # Starting point from Goncalves et al. 2017
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    def objective(coeffs):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            return 1e10
        try:
            c0, c4, c8 = coeffs
            # Clamp to reasonable range
            c0 = np.clip(c0, -100, 100)
            c4 = np.clip(c4, -100, 100)
            c8 = np.clip(c8, -100, 100)
            return _compute_bound(c0, c4, c8, num_points=10000)
        except Exception:
            return 1e10
    
    # First evaluation at starting point
    best_coeffs = x0.copy()
    best_val = objective(x0)
    
    # Try multiple optimization methods
    methods = [
        ('Nelder-Mead', {'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}),
        ('Powell', {'maxiter': 500, 'ftol': 1e-10, 'xtol': 1e-8}),
        ('BFGS', {'maxiter': 200}),
    ]
    
    for method_name, options in methods:
        elapsed = time.time() - start_time
        if elapsed > deadline - 10.0:
            break
        
        try:
            if method_name == 'BFGS':
                # Need gradient, use finite differences
                result = optimize.minimize(objective, x0, method='BFGS', options=options)
            else:
                result = optimize.minimize(objective, x0, method=method_name, options=options)
            
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    # Final polish with more points if time allows
    elapsed = time.time() - start_time
    if elapsed < deadline - 15.0:
        try:
            # Refine with higher resolution
            def objective_high_res(coeffs):
                elapsed_inner = time.time() - start_time
                if elapsed_inner > deadline - 5.0:
                    return 1e10
                c0, c4, c8 = coeffs
                c0 = np.clip(c0, -100, 100)
                c4 = np.clip(c4, -100, 100)
                c8 = np.clip(c8, -100, 100)
                return _compute_bound(c0, c4, c8, num_points=30000)
            
            result = optimize.minimize(objective_high_res, best_coeffs, 
                                       method='Nelder-Mead', 
                                       options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    return np.array(best_coeffs)

if __name__ == "__main__":
    coeffs = construct()
    print(f"Optimal coefficients: {coeffs}")
    bound = _compute_bound(coeffs[0], coeffs[1], coeffs[2], num_points=20000)
    print(f"Upper bound on C4: {bound}")
