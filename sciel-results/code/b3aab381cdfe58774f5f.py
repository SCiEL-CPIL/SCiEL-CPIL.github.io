"""Combined approach: quad-based exact M/Mx + spherical multi-start Nelder-Mead with eigenvalue init."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad


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
        return 2.0 * x
    h0 = 1.0
    h1 = 2.0 * x
    for k in range(2, n + 1):
        h2 = 2.0 * x * h1 - 2.0 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def _basis_func_scalar(i, x):
    """Scalar basis function with g(0)=0 constraint baked in."""
    gauss = np.exp(-x**2 / 2.0)
    h12 = _hermite_poly_scalar(12, x)
    if i == 0:
        return (1.0 - h12 / 40320.0) * gauss
    elif i == 1:
        h4 = _hermite_poly_scalar(4, x)
        return (h4 - 12.0 * h12 / 40320.0) * gauss
    elif i == 2:
        h8 = _hermite_poly_scalar(8, x)
        return (h8 - 1680.0 * h12 / 40320.0) * gauss
    else:
        raise ValueError("Invalid basis index")


def _basis_func_vec(i, x):
    """Vectorized basis function for array x."""
    gauss = np.exp(-x**2 / 2.0)
    H12 = _hermite_poly(12, x)
    if i == 0:
        return (np.ones_like(x) - H12 / 40320.0) * gauss
    elif i == 1:
        H4 = _hermite_poly(4, x)
        return (H4 - 12.0 * H12 / 40320.0) * gauss
    elif i == 2:
        H8 = _hermite_poly(8, x)
        return (H8 - 1680.0 * H12 / 40320.0) * gauss
    else:
        raise ValueError("Invalid basis index")


def construct():
    start_time = time.time()
    deadline = 175  # safety margin before 210s kill

    # --- Step 1: Compute M and Mx via adaptive quadrature (exact) ---
    M = np.zeros((3, 3))
    Mx = np.zeros((3, 3))

    for i in range(3):
        for j in range(i, 3):
            fi = lambda x, i=i: _basis_func_scalar(i, x)
            fj = lambda x, j=j: _basis_func_scalar(j, x)
            val_M, _ = quad(lambda x: fi(x) * fj(x), -np.inf, np.inf, limit=200, epsabs=1e-14, epsrel=1e-14)
            M[i, j] = val_M
            M[j, i] = val_M

            val_Mx, _ = quad(lambda x: x**2 * fi(x) * fj(x), -np.inf, np.inf, limit=200, epsabs=1e-14, epsrel=1e-14)
            Mx[i, j] = val_Mx
            Mx[j, i] = val_Mx

    # --- Step 2: Compute Mf via FFT on a fine grid ---
    N = 16384
    x_grid = np.linspace(-30, 30, N)
    dx = x_grid[1] - x_grid[0]
    freqs = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))

    G = np.zeros((3, N), dtype=complex)
    for i in range(3):
        v = _basis_func_vec(i, x_grid)
        G[i] = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v))) * dx

    Mf = np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):
            val_Mf = np.sum(freqs**2 * np.real(G[i] * np.conj(G[j]))) * dx
            Mf[i, j] = val_Mf
            Mf[j, i] = val_Mf

    # --- Step 3: Define objective in spherical coordinates ---
    def compute_bound_from_coeffs(c):
        c = np.array(c, dtype=float)
        norm2 = c @ M @ c
        if norm2 <= 1e-30 or not np.isfinite(norm2):
            return np.inf
        x2_val = c @ Mx @ c
        f2_val = c @ Mf @ c
        if not np.isfinite(x2_val) or not np.isfinite(f2_val) or x2_val < -1e-12 or f2_val < -1e-12:
            return np.inf
        x2_val = max(x2_val, 0.0)
        f2_val = max(f2_val, 0.0)
        delta_x = np.sqrt(x2_val / norm2)
        delta_xi = np.sqrt(f2_val / norm2)
        return delta_x * delta_xi

    def objective_sphere(theta, phi):
        c0 = np.sin(theta) * np.cos(phi)
        c4 = np.sin(theta) * np.sin(phi)
        c8 = np.cos(theta)
        return compute_bound_from_coeffs([c0, c4, c8])

    # --- Step 4: Initial points ---
    # Seed point
    c_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    c_seed_norm = c_seed / np.linalg.norm(c_seed)
    c0s, c4s, c8s = c_seed_norm
    theta_seed = np.arccos(np.clip(c8s, -1.0 + 1e-12, 1.0 - 1e-12))
    phi_seed = np.arctan2(c4s, c0s)
    if phi_seed < 0:
        phi_seed += 2 * np.pi

    best_bound = objective_sphere(theta_seed, phi_seed)
    best_theta = theta_seed
    best_phi = phi_seed

    # Eigenvalue-based initialization
    try:
        eigvals_x, eigvecs_x = np.linalg.eigh(Mx, M)
        eigvals_f, eigvecs_f = np.linalg.eigh(Mf, M)

        candidates = []
        # Pure eigenvectors
        for i in range(3):
            candidates.append(eigvecs_x[:, i].copy())
            candidates.append(eigvecs_f[:, i].copy())
        # Mixtures
        for i in range(3):
            for j in range(3):
                for alpha in [0.25, 0.5, 0.75]:
                    c_trial = alpha * eigvecs_x[:, i] + (1 - alpha) * eigvecs_f[:, j]
                    candidates.append(c_trial.copy())

        for c_trial in candidates:
            nrm = np.linalg.norm(c_trial)
            if nrm < 1e-12:
                continue
            c_trial = c_trial / nrm
            c0_t, c4_t, c8_t = c_trial
            if abs(c8_t) > 0.99999:
                continue
            theta_t = np.arccos(np.clip(c8_t, -1.0 + 1e-12, 1.0 - 1e-12))
            phi_t = np.arctan2(c4_t, c0_t)
            if phi_t < 0:
                phi_t += 2 * np.pi
            b = objective_sphere(theta_t, phi_t)
            if np.isfinite(b) and b < best_bound:
                best_bound = b
                best_theta = theta_t
                best_phi = phi_t
    except Exception:
        pass

    # --- Step 5: Multi-start Nelder-Mead ---
    if time.time() - start_time < deadline - 25:
        rng = np.random.RandomState(42)
        n_starts = 25

        for i in range(n_starts):
            if time.time() - start_time >= deadline - 18:
                break

            theta_rand = rng.uniform(0.05, np.pi - 0.05)
            phi_rand = rng.uniform(0.05, 2 * np.pi - 0.05)

            try:
                res = minimize(
                    objective_sphere,
                    x0=[theta_rand, phi_rand],
                    method='Nelder-Mead',
                    options={'xatol': 1e-11, 'fatol': 1e-13, 'maxiter': 4000, 'adaptive': True}
                )
                if res.fun < best_bound and np.isfinite(res.fun):
                    best_bound = res.fun
                    best_theta, best_phi = res.x
            except Exception:
                pass

    # --- Step 6: High-precision polish from best point ---
    if time.time() - start_time < deadline - 10:
        try:
            res = minimize(
                objective_sphere,
                x0=[best_theta, best_phi],
                method='Nelder-Mead',
                options={'xatol': 1e-14, 'fatol': 1e-16, 'maxiter': 20000, 'adaptive': True}
            )
            if res.fun < best_bound and np.isfinite(res.fun):
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
