"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
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


def _hermite_poly_scalar(n, x):
    """Compute physicist's Hermite polynomial Hn(x) for scalar x."""
    if n == 0:
        return 1.0
    if n == 1:
        return 2 * x
    h0 = 1.0
    h1 = 2 * x
    for k in range(2, n + 1):
        h2 = 2 * x * h1 - 2 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def _basis_func(i, x):
    """Compute the i-th basis function (with g(0)=0 constraint baked in).
    i=0: H0 - H12/40320
    i=1: H4 - 12*H12/40320
    i=2: H8 - 1680*H12/40320
    """
    gauss = np.exp(-x**2 / 2)
    if i == 0:
        return (_hermite_poly_scalar(0, x) - _hermite_poly_scalar(12, x) / 40320.0) * gauss
    elif i == 1:
        return (_hermite_poly_scalar(4, x) - 12.0 * _hermite_poly_scalar(12, x) / 40320.0) * gauss
    elif i == 2:
        return (_hermite_poly_scalar(8, x) - 1680.0 * _hermite_poly_scalar(12, x) / 40320.0) * gauss
    else:
        raise ValueError("Invalid basis index")


def _basis_func_vec(i, x):
    """Vectorized version of basis function for array x."""
    gauss = np.exp(-x**2 / 2)
    if i == 0:
        return (_hermite_poly(0, x) - _hermite_poly(12, x) / 40320.0) * gauss
    elif i == 1:
        return (_hermite_poly(4, x) - 12.0 * _hermite_poly(12, x) / 40320.0) * gauss
    elif i == 2:
        return (_hermite_poly(8, x) - 1680.0 * _hermite_poly(12, x) / 40320.0) * gauss
    else:
        raise ValueError("Invalid basis index")


def construct():
    start_time = time.time()
    deadline = 170  # seconds, keep margin for evaluator kill at 210s

    # Precompute Gram matrices using adaptive quadrature for accuracy
    # M[i,j] = integral of basis_i(x) * basis_j(x) dx
    # Mx[i,j] = integral of x^2 * basis_i(x) * basis_j(x) dx
    # Mf[i,j] = integral of xi^2 * |Fourier(basis_i)(xi)| * conj(Fourier(basis_j)(xi)) dxi
    #          = integral of basis_i(x) * (-d^2/dx^2) basis_j(x) dx  (by Parseval)
    
    M = np.zeros((3, 3))
    Mx = np.zeros((3, 3))
    Mf = np.zeros((3, 3))
    
    # Compute M and Mx using quad
    for i in range(3):
        for j in range(i, 3):
            # M[i,j] = ∫ basis_i(x) * basis_j(x) dx
            val_M, _ = quad(lambda x: _basis_func(i, x) * _basis_func(j, x), -np.inf, np.inf, limit=200)
            M[i, j] = val_M
            M[j, i] = val_M
            
            # Mx[i,j] = ∫ x^2 * basis_i(x) * basis_j(x) dx
            val_Mx, _ = quad(lambda x: x**2 * _basis_func(i, x) * _basis_func(j, x), -np.inf, np.inf, limit=200)
            Mx[i, j] = val_Mx
            Mx[j, i] = val_Mx
            
            # Mf[i,j] = ∫ basis_i''(x) * basis_j(x) dx (negative sign from Fourier transform)
            # Actually, Mf[i,j] = ∫ xi^2 * F_i(xi) * conj(F_j(xi)) dxi
            # By Parseval: ∫ xi^2 F_i(xi) conj(F_j(xi)) dxi = ∫ (-d^2/dx^2) basis_i(x) * basis_j(x) dx
            # But we need the second derivative. Let's compute it numerically or analytically.
            # Alternative: use the fact that for Hermite-Gaussians, we can compute this.
            # For now, let's use a numerical approach with quad on a finite interval
            # and verify with FFT later.
            pass
    
    # For Mf, we need the second derivative. Let's compute it using the relation:
    # d^2/dx^2 [H_n(x) e^{-x^2/2}] = (2n - 2x^2) H_n(x) e^{-x^2/2} + 4n H_{n-2}(x) e^{-x^2/2}
    # This is getting complex. Let's use FFT-based computation for Mf since it's more straightforward.
    
    # Actually, let's use a hybrid approach: quad for M and Mx (which are simple),
    # and FFT for Mf (which involves the Fourier transform).
    
    # Recompute M and Mx properly
    for i in range(3):
        for j in range(i, 3):
            val_M, _ = quad(lambda x: _basis_func(i, x) * _basis_func(j, x), -np.inf, np.inf, limit=200)
            M[i, j] = val_M
            M[j, i] = val_M
            
            val_Mx, _ = quad(lambda x: x**2 * _basis_func(i, x) * _basis_func(j, x), -np.inf, np.inf, limit=200)
            Mx[i, j] = val_Mx
            Mx[j, i] = val_Mx
    
    # Compute Mf using FFT on a fine grid
    N = 16384
    x_grid = np.linspace(-30, 30, N)
    dx = x_grid[1] - x_grid[0]
    freqs = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    
    G = np.zeros((3, N), dtype=complex)
    for i in range(3):
        v = _basis_func_vec(i, x_grid)
        G[i] = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v))) * dx
    
    for i in range(3):
        for j in range(i, 3):
            val_Mf = np.sum(freqs**2 * np.real(G[i] * np.conj(G[j]))) * dx
            Mf[i, j] = val_Mf
            Mf[j, i] = val_Mf
    
    def compute_bound_from_coeffs(c):
        """Compute bound for a specific coefficient vector c."""
        c = np.array(c)
        norm2 = c @ M @ c
        if norm2 <= 0 or not np.isfinite(norm2):
            return np.inf
        x2_norm = c @ Mx @ c
        freq2_norm = c @ Mf @ c
        
        if not np.isfinite(freq2_norm) or freq2_norm < 0 or not np.isfinite(x2_norm) or x2_norm < 0:
            return np.inf
            
        delta_x = np.sqrt(x2_norm / norm2)
        delta_xi = np.sqrt(freq2_norm / norm2)
        return delta_x * delta_xi

    def compute_bound_from_spherical(theta, phi):
        """Compute bound for spherical parameters."""
        c0 = np.sin(theta) * np.cos(phi)
        c4 = np.sin(theta) * np.sin(phi)
        c8 = np.cos(theta)
        return compute_bound_from_coeffs([c0, c4, c8])

    # Initial best guess from seed
    c_seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    c_seed_norm = c_seed / np.linalg.norm(c_seed)
    c0, c4, c8 = c_seed_norm
    theta_init = np.arccos(c8)
    phi_init = np.arctan2(c4, c0)
    if phi_init < 0:
        phi_init += 2 * np.pi

    best_bound = compute_bound_from_spherical(theta_init, phi_init)
    best_theta = theta_init
    best_phi = phi_init

    # Use generalized eigenvalue approach as an alternative initialization
    try:
        eigvals_x, eigvecs_x = np.linalg.eigh(Mx, M)
        eigvals_f, eigvecs_f = np.linalg.eigh(Mf, M)
        
        for i in range(3):
            for j in range(3):
                for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    c_trial = alpha * eigvecs_x[:, i] + (1 - alpha) * eigvecs_f[:, j]
                    c_trial = c_trial / np.linalg.norm(c_trial)
                    c0_t = c_trial[0]
                    c4_t = c_trial[1]
                    c8_t = c_trial[2]
                    if c8_t < -0.9999 or c8_t > 0.9999:
                        continue
                    theta_t = np.arccos(np.clip(c8_t, -0.999999, 0.999999))
                    phi_t = np.arctan2(c4_t, c0_t)
                    if phi_t < 0:
                        phi_t += 2 * np.pi
                    b = compute_bound_from_spherical(theta_t, phi_t)
                    if b < best_bound:
                        best_bound = b
                        best_theta, best_phi = theta_t, phi_t
    except Exception:
        pass

    # Generate multiple random starting points for multi-start optimization
    rng = np.random.RandomState(42)
    random_starts = []
    for _ in range(30):
        t = rng.uniform(1e-6, np.pi - 1e-6)
        p = rng.uniform(1e-6, 2 * np.pi - 1e-6)
        random_starts.append((t, p))
    
    random_starts.append((theta_init, phi_init))
    
    for t in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.5, 2.8]:
        for p in [0.3, 0.8, 1.3, 1.8, 2.3, 2.8, 3.3, 3.8, 4.3, 4.8, 5.3, 5.8]:
            random_starts.append((t, p))

    for t, p in random_starts:
        b = compute_bound_from_spherical(t, p)
        if b < best_bound:
            best_bound = b
            best_theta, best_phi = t, p

    # Local refinement with multiple local optimizations
    local_starts = [
        (best_theta, best_phi),
        (theta_init, phi_init),
        (0.4, 0.8),
        (0.7, 1.5),
        (1.0, 2.5),
        (1.3, 3.5),
        (1.6, 0.8),
        (2.0, 2.0),
        (2.3, 4.5),
        (2.6, 1.2),
    ]
    
    for t, p in local_starts:
        if time.time() - start_time >= deadline - 20:
            break
        try:
            res = minimize(
                compute_bound_from_spherical, 
                x0=[t, p], 
                method='Nelder-Mead',
                options={'xatol': 1e-14, 'fatol': 1e-16, 'maxiter': 50000, 'adaptive': True}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_theta, best_phi = res.x
        except Exception:
            pass

    # Final fine-tuning
    if time.time() - start_time < deadline - 5:
        try:
            res = minimize(
                compute_bound_from_spherical, 
                x0=[best_theta, best_phi], 
                method='Nelder-Mead',
                options={'xatol': 1e-15, 'fatol': 1e-17, 'maxiter': 100000, 'adaptive': True}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_theta, best_phi = res.x
        except Exception:
            pass

    # Convert back to Cartesian coefficients
    c0 = np.sin(best_theta) * np.cos(best_phi)
    c4 = np.sin(best_theta) * np.sin(best_phi)
    c8 = np.cos(best_theta)
    
    return np.array([c0, c4, c8])

# EVOLVE-BLOCK-END
