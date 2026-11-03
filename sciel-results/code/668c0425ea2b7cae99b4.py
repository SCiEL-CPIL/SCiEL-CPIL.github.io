"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time


def _hermite_poly(n, x):
    """Compute physicist's Hermite polynomial Hn(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    h0 = np.ones_like(x)
    h1 = 2 * x
    for k in range(2, n + 1):
        h2 = 2 * x * h1 - 2 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def construct():
    start_time = time.time()
    deadline = 180  # seconds, keep margin for evaluator kill at 210s
    
    # Setup grid
    # Use a grid large enough to capture tails but small enough for speed
    # 4096 points is a good balance. Range [-20, 20] is sufficient for Gaussian decay.
    N = 4096
    x = np.linspace(-20, 20, N)
    dx = x[1] - x[0]
    freqs = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    
    # Precompute Hermite polynomials for the grid to avoid repeated calculation
    H0 = _hermite_poly(0, x)
    H4 = _hermite_poly(4, x)
    H8 = _hermite_poly(8, x)
    H12 = _hermite_poly(12, x)
    gauss = np.exp(-x**2 / 2)
    
    # Precompute weighted basis vectors for fast linear combination
    # g = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * gauss
    # where c12 = -(c0*1 + c4*12 + c8*1680) / 40320
    # So g = c0*(H0 - 1/40320*H12)*gauss + c4*(H4 - 12/40320*H12)*gauss + c8*(H8 - 1680/40320*H12)*gauss
    # Let's define effective basis vectors:
    # v0 = (H0 - H12/40320) * gauss
    # v4 = (H4 - 12*H12/40320) * gauss
    # v8 = (H8 - 1680*H12/40320) * gauss
    # Then g = c0*v0 + c4*v4 + c8*v8
    
    # Actually, let's just precompute the 3x3 Gram matrix for the effective basis
    # and the 3 vectors for the x^2 and |F|^2 terms.
    
    # Let's precompute the basis vectors v0, v4, v8
    v0 = (H0 - H12 / 40320.0) * gauss
    v4 = (H4 - 12.0 * H12 / 40320.0) * gauss
    v8 = (H8 - 1680.0 * H12 / 40320.0) * gauss
    
    # Precompute norms and cross terms for the Gram matrix of g
    # norm2 = sum(g^2) dx = sum((c0*v0 + c4*v4 + c8*v8)^2) dx
    # Let M be the 3x3 matrix where M_ij = sum(vi * vj) dx
    # Then norm2 = c^T M c
    
    # Similarly for x2_norm = sum(x^2 * g^2) dx
    # Let Mx be the 3x3 matrix where Mx_ij = sum(x^2 * vi * vj) dx
    # Then x2_norm = c^T Mx c
    
    # For the Fourier side, we need sum(freqs^2 * |G|^2) dx
    # G = FFT(g) * dx
    # |G|^2 is quadratic in c.
    # Let G0 = FFT(v0)*dx, G4 = FFT(v4)*dx, G8 = FFT(v8)*dx
    # Then G = c0*G0 + c4*G4 + c8*G8
    # freq2_norm = sum(freqs^2 * |G|^2) dx
    # This is also quadratic in c.
    # Let Mf be the 3x3 matrix where Mf_ij = sum(freqs^2 * (Re(Gi*conj(Gj)) + ... ))?
    # Actually, |G|^2 = G * conj(G).
    # sum(freqs^2 * |c0*G0 + c4*G4 + c8*G8|^2) dx
    # = sum(freqs^2 * (c0*G0 + c4*G4 + c8*G8) * (c0*conj(G0) + c4*conj(G4) + c8*conj(G8))) dx
    # = c0^2 sum(freqs^2*|G0|^2)dx + c4^2 sum(freqs^2*|G4|^2)dx + c8^2 sum(freqs^2*|G8|^2)dx
    #   + 2*c0*c4 sum(freqs^2*Re(G0*conj(G4)))dx
    #   + 2*c0*c8 sum(freqs^2*Re(G0*conj(G8)))dx
    #   + 2*c4*c8 sum(freqs^2*Re(G4*conj(G8)))dx
    
    # Let's precompute these vectors and matrices.
    G0 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v0))) * dx
    G4 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v4))) * dx
    G8 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v8))) * dx
    
    # Precompute the quadratic forms
    # M (norm)
    M = np.zeros((3, 3))
    M[0, 0] = np.sum(v0**2) * dx
    M[1, 1] = np.sum(v4**2) * dx
    M[2, 2] = np.sum(v8**2) * dx
    M[0, 1] = M[1, 0] = np.sum(v0 * v4) * dx
    M[0, 2] = M[2, 0] = np.sum(v0 * v8) * dx
    M[1, 2] = M[2, 1] = np.sum(v4 * v8) * dx
    
    # Mx (x^2 norm)
    Mx = np.zeros((3, 3))
    Mx[0, 0] = np.sum(x**2 * v0**2) * dx
    Mx[1, 1] = np.sum(x**2 * v4**2) * dx
    Mx[2, 2] = np.sum(x**2 * v8**2) * dx
    Mx[0, 1] = Mx[1, 0] = np.sum(x**2 * v0 * v4) * dx
    Mx[0, 2] = Mx[2, 0] = np.sum(x**2 * v0 * v8) * dx
    Mx[1, 2] = Mx[2, 1] = np.sum(x**2 * v4 * v8) * dx
    
    # Mf (freq^2 norm)
    # We need sum(freqs^2 * Re(Gi * conj(Gj))) * dx
    Mf = np.zeros((3, 3))
    Mf[0, 0] = np.sum(freqs**2 * np.abs(G0)**2) * dx
    Mf[1, 1] = np.sum(freqs**2 * np.abs(G4)**2) * dx
    Mf[2, 2] = np.sum(freqs**2 * np.abs(G8)**2) * dx
    Mf[0, 1] = Mf[1, 0] = np.sum(freqs**2 * np.real(G0 * np.conj(G4))) * dx
    Mf[0, 2] = Mf[2, 0] = np.sum(freqs**2 * np.real(G0 * np.conj(G8))) * dx
    Mf[1, 2] = Mf[2, 1] = np.sum(freqs**2 * np.real(G4 * np.conj(G8))) * dx
    
    def compute_bound_fast(coeffs):
        c = np.array(coeffs)
        norm2 = c @ M @ c
        if norm2 <= 0 or not np.isfinite(norm2):
            return np.inf
        x2_norm = c @ Mx @ c
        freq2_norm = c @ Mf @ c
        
        if not np.isfinite(freq2_norm) or freq2_norm < 0:
            return np.inf
            
        delta_x = np.sqrt(x2_norm / norm2)
        delta_xi = np.sqrt(freq2_norm / norm2)
        
        return delta_x * delta_xi

    # Initial guess from prior work
    c0_init = -113 / 100
    c4_init = 1 / 25
    c8_init = 1 / 3240
    
    best_coeffs = np.array([c0_init, c4_init, c8_init])
    best_bound = compute_bound_fast(best_coeffs)
    
    # Objective function for Differential Evolution
    # DE expects a function that takes a vector and returns a scalar
    def objective(coeffs):
        return compute_bound_fast(coeffs)

    # Define bounds for the coefficients
    # Based on the initial guess and typical scales
    # c0 ~ -1, c4 ~ 0.04, c8 ~ 0.0003
    # We can use wide bounds to allow DE to explore, but not too wide to waste time
    bounds = [(-10, 10), (-1, 1), (-0.1, 0.1)]
    
    # Run Differential Evolution
    # We use a reasonable number of generations and population size
    # maxiter=1000 might be too slow if each eval is heavy, but our eval is O(1) after precomp
    # However, DE calls objective many times.
    # Let's estimate: 3D problem. Pop size 30. 1000 iters = 30000 evals.
    # Each eval is a few dot products of size 3. Very fast.
    # 30000 evals * ~100ns = 3ms. This is extremely fast.
    # We can increase iterations or use tighter tolerances.
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=2000, 
            tol=1e-12, 
            popsize=30,
            polish=True, # Use L-BFGS-B for final polish
            init='sobol' # Better initialization
        )
        
        if result.success or np.isfinite(result.fun):
            if result.fun < best_bound:
                best_bound = result.fun
                best_coeffs = result.x
    except Exception:
        pass

    # Final check and return
    # Ensure we return a valid numpy array
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
