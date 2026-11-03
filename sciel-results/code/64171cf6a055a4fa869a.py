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

    # Higher-resolution grid for accurate quadrature
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

    # Structural change: use a 2D parameterization where we optimize over
    # two angles that directly map to the 3 coefficients via a smooth,
    # non-periodic parameterization. We use the "tangent half-angle" trick
    # to map the sphere to a plane, avoiding periodic boundary issues.
    # Alternatively, we optimize directly in 2D with a constraint-free
    # parameterization: let c = (a, b, c) where we parameterize by
    # (u, v) in R^2 and normalize: c = (u, v, 1) / norm, but this misses
    # the hemisphere where c8 < 0. Instead, use a 2D map to the sphere
    # via the stereographic projection from the south pole.
    
    def from_stereographic(u, v):
        """Map (u, v) in R^2 to unit vector on sphere via stereographic projection from south pole."""
        r2 = u**2 + v**2
        c0 = 2 * u / (1 + r2)
        c4 = 2 * v / (1 + r2)
        c8 = (1 - r2) / (1 + r2)
        return np.array([c0, c4, c8])

    def to_stereographic(c):
        """Convert unit vector to stereographic coordinates."""
        c = np.array(c)
        # Avoid division by zero when c8 = -1 (south pole)
        if c[2] < -0.999999:
            return np.array([1e6, 0.0])
        u = c[0] / (1 + c[2])
        v = c[1] / (1 + c[2])
        return np.array([u, v])

    def compute_bound_from_stereo(u, v):
        """Compute bound from stereographic coordinates."""
        c = from_stereographic(u, v)
        return compute_bound_from_coeffs(c)

    # Initial best guess from seed
    c_seed = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    c_seed_norm = c_seed / np.linalg.norm(c_seed)
    uv_init = to_stereographic(c_seed_norm)
    
    best_bound = compute_bound_from_stereo(uv_init[0], uv_init[1])
    best_uv = uv_init.copy()

    # Generate diverse starting points in stereographic coordinates
    rng = np.random.RandomState(42)
    starts = [uv_init.copy()]
    
    # Random points in a reasonable range
    for _ in range(20):
        u = rng.uniform(-3, 3)
        v = rng.uniform(-3, 3)
        starts.append(np.array([u, v]))
    
    # Structured grid of starting points
    for u in [-2, -1, -0.5, 0, 0.5, 1, 2]:
        for v in [-2, -1, -0.5, 0, 0.5, 1, 2]:
            starts.append(np.array([u, v]))

    # Evaluate all starting points
    for uv in starts:
        b = compute_bound_from_stereo(uv[0], uv[1])
        if b < best_bound:
            best_bound = b
            best_uv = uv.copy()

    # Local refinement with Nelder-Mead from multiple starting points
    local_starts = [
        best_uv,
        uv_init,
        np.array([-0.5, 0.5]),
        np.array([0.5, -0.5]),
        np.array([1.0, 1.0]),
        np.array([-1.0, -1.0]),
        np.array([0.0, 1.5]),
        np.array([-1.5, 0.0]),
    ]
    
    for uv0 in local_starts:
        if time.time() - start_time >= deadline - 40:
            break
        try:
            res = minimize(
                compute_bound_from_stereo, 
                x0=uv0, 
                method='Nelder-Mead',
                options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 15000, 'adaptive': True}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_uv = res.x
        except Exception:
            pass

    # Global optimization using Differential Evolution
    if time.time() - start_time < deadline - 30:
        try:
            de_result = differential_evolution(
                compute_bound_from_stereo, 
                bounds=[(-4, 4), (-4, 4)],
                seed=42, 
                maxiter=150, 
                tol=1e-9,
                popsize=15,
                polish=True
            )
            if de_result.fun < best_bound:
                best_bound = de_result.fun
                best_uv = de_result.x
        except Exception:
            pass

    # Final fine-tuning with very tight tolerance
    if time.time() - start_time < deadline - 10:
        try:
            res = minimize(
                compute_bound_from_stereo, 
                x0=best_uv, 
                method='Nelder-Mead',
                options={'xatol': 1e-14, 'fatol': 1e-16, 'maxiter': 30000, 'adaptive': True}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_uv = res.x
        except Exception:
            pass

    # Convert back to Cartesian coefficients
    c = from_stereographic(best_uv[0], best_uv[1])
    
    return c

# EVOLVE-BLOCK-END
