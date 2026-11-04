"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
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
    deadline = 175  # seconds, keep margin for evaluator kill at 210s

    # Setup grid
    N = 8192
    x = np.linspace(-25, 25, N)
    dx = x[1] - x[0]
    freqs = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))

    # Precompute Hermite polynomials
    H0 = _hermite_poly(0, x)
    H4 = _hermite_poly(4, x)
    H8 = _hermite_poly(8, x)
    H12 = _hermite_poly(12, x)
    gauss = np.exp(-x**2 / 2)

    # Effective basis vectors incorporating the g(0)=0 constraint
    # c12 = -(c0 + 12c4 + 1680c8) / 40320
    v0 = (H0 - H12 / 40320.0) * gauss
    v4 = (H4 - 12.0 * H12 / 40320.0) * gauss
    v8 = (H8 - 1680.0 * H12 / 40320.0) * gauss

    # Precompute Gram matrices
    M = np.zeros((3, 3))
    M[0, 0] = np.sum(v0**2) * dx
    M[1, 1] = np.sum(v4**2) * dx
    M[2, 2] = np.sum(v8**2) * dx
    M[0, 1] = M[1, 0] = np.sum(v0 * v4) * dx
    M[0, 2] = M[2, 0] = np.sum(v0 * v8) * dx
    M[1, 2] = M[2, 1] = np.sum(v4 * v8) * dx

    Mx = np.zeros((3, 3))
    Mx[0, 0] = np.sum(x**2 * v0**2) * dx
    Mx[1, 1] = np.sum(x**2 * v4**2) * dx
    Mx[2, 2] = np.sum(x**2 * v8**2) * dx
    Mx[0, 1] = Mx[1, 0] = np.sum(x**2 * v0 * v4) * dx
    Mx[0, 2] = Mx[2, 0] = np.sum(x**2 * v0 * v8) * dx
    Mx[1, 2] = Mx[2, 1] = np.sum(x**2 * v4 * v8) * dx

    G0 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v0))) * dx
    G4 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v4))) * dx
    G8 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v8))) * dx

    Mf = np.zeros((3, 3))
    Mf[0, 0] = np.sum(freqs**2 * np.abs(G0)**2) * dx
    Mf[1, 1] = np.sum(freqs**2 * np.abs(G4)**2) * dx
    Mf[2, 2] = np.sum(freqs**2 * np.abs(G8)**2) * dx
    Mf[0, 1] = Mf[1, 0] = np.sum(freqs**2 * np.real(G0 * np.conj(G4))) * dx
    Mf[0, 2] = Mf[2, 0] = np.sum(freqs**2 * np.real(G0 * np.conj(G8))) * dx
    Mf[1, 2] = Mf[2, 1] = np.sum(freqs**2 * np.real(G4 * np.conj(G8))) * dx

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

    # Generate multiple random starting points for multi-start optimization
    rng = np.random.RandomState(42)
    random_starts = []
    for _ in range(20):
        t = rng.uniform(1e-6, np.pi - 1e-6)
        p = rng.uniform(1e-6, 2 * np.pi - 1e-6)
        random_starts.append((t, p))
    
    # Add the seed point
    random_starts.append((theta_init, phi_init))
    
    # Also add some structured points
    for t in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.5, 2.8]:
        for p in [0.3, 0.8, 1.3, 1.8, 2.3, 2.8, 3.3, 3.8, 4.3, 4.8, 5.3, 5.8]:
            random_starts.append((t, p))

    # Evaluate all starting points
    for t, p in random_starts:
        b = compute_bound_from_spherical(t, p)
        if b < best_bound:
            best_bound = b
            best_theta, best_phi = t, p

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

    # Global optimization using Differential Evolution
    if time.time() - start_time < deadline - 40:
        try:
            bounds = [(1e-6, np.pi - 1e-6), (1e-6, 2 * np.pi - 1e-6)]
            de_result = differential_evolution(
                compute_bound_from_spherical, 
                bounds, 
                seed=42, 
                maxiter=100, 
                tol=1e-8,
                popsize=15,
                polish=False
            )
            if de_result.fun < best_bound:
                best_bound = de_result.fun
                best_theta, best_phi = de_result.x
        except Exception:
            pass

    # Local refinement with multiple local optimizations from different starting points
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
        if time.time() - start_time >= deadline - 15:
            break
        try:
            res = minimize(
                compute_bound_from_spherical, 
                x0=[t, p], 
                method='Nelder-Mead',
                options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 5000, 'adaptive': True}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_theta, best_phi = res.x
        except Exception:
            pass

    # Final fine-tuning with very tight tolerance
    if time.time() - start_time < deadline - 5:
        try:
            res = minimize(
                compute_bound_from_spherical, 
                x0=[best_theta, best_phi], 
                method='Nelder-Mead',
                options={'xatol': 1e-15, 'fatol': 1e-17, 'maxiter': 50000, 'adaptive': True}
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
