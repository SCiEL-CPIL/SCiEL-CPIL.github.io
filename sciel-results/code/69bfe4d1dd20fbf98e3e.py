"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


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
    x = np.linspace(-30, 30, N)
    dx = x[1] - x[0]
    freqs = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))

    # Precompute Hermite polynomials
    H0 = _hermite_poly(0, x)
    H4 = _hermite_poly(4, x)
    H8 = _hermite_poly(8, x)
    H12 = _hermite_poly(12, x)
    gauss = np.exp(-x**2 / 2)

    # Effective basis vectors incorporating the g(0)=0 constraint
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

    def compute_bound(c):
        """Compute bound for a coefficient vector c (scale-invariant)."""
        c = np.asarray(c, dtype=float)
        norm2 = c @ M @ c
        if norm2 <= 0 or not np.isfinite(norm2):
            return np.inf
        x2_norm = c @ Mx @ c
        freq2_norm = c @ Mf @ c
        if (not np.isfinite(freq2_norm) or freq2_norm < 0 or
                not np.isfinite(x2_norm) or x2_norm < 0):
            return np.inf
        delta_x = np.sqrt(x2_norm / norm2)
        delta_xi = np.sqrt(freq2_norm / norm2)
        return delta_x * delta_xi

    def compute_bound_and_grad(c):
        """Compute bound and gradient w.r.t. c (unprojected)."""
        c = np.asarray(c, dtype=float)
        norm2 = c @ M @ c
        if norm2 <= 0 or not np.isfinite(norm2):
            return np.inf, np.zeros(3)
        x2_norm = c @ Mx @ c
        freq2_norm = c @ Mf @ c
        if (not np.isfinite(freq2_norm) or freq2_norm < 0 or
                not np.isfinite(x2_norm) or x2_norm < 0):
            return np.inf, np.zeros(3)

        delta_x = np.sqrt(x2_norm / norm2)
        delta_xi = np.sqrt(freq2_norm / norm2)
        val = delta_x * delta_xi

        # Gradient of f w.r.t. c:
        # f = sqrt(x2/norm) * sqrt(freq2/norm)
        # grad f = f * (0.5*grad(x2)/x2 - grad(norm)/norm + 0.5*grad(freq2)/freq2)
        x2_grad = 2.0 * (Mx @ c)
        norm_grad = 2.0 * (M @ c)
        freq2_grad = 2.0 * (Mf @ c)

        dlogf_dc = (0.5 * x2_grad / x2_norm
                    - norm_grad / norm2
                    + 0.5 * freq2_grad / freq2_norm)
        grad = val * dlogf_dc
        return val, grad

    def sphere_project(c):
        """Project c onto the unit sphere."""
        n = np.linalg.norm(c)
        if n < 1e-15:
            return np.array([1.0, 0.0, 0.0])
        return c / n

    def sphere_obj_and_grad(c):
        """Objective and projected gradient on the unit sphere."""
        c = sphere_project(c)
        val, grad = compute_bound_and_grad(c)
        # Project gradient onto tangent space of sphere at c
        grad_tan = grad - np.dot(grad, c) * c
        return val, grad_tan

    # --- Build a deterministic set of starting points on the sphere ---
    # Fibonacci sphere for uniform coverage
    def fibonacci_sphere(n):
        pts = []
        golden = (1 + np.sqrt(5)) / 2
        for i in range(n):
            z = 1 - 2 * (i + 0.5) / n
            r = np.sqrt(max(0.0, 1 - z * z))
            phi = 2 * np.pi * i / golden
            pts.append([r * np.cos(phi), r * np.sin(phi), z])
        return np.array(pts)

    # Start with Fibonacci sphere points
    fib_pts = fibonacci_sphere(200)
    starts = [sphere_project(p) for p in fib_pts]

    # Add the seed point
    c_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    starts.append(sphere_project(c_seed))

    # Add some random points
    rng = np.random.RandomState(12345)
    for _ in range(100):
        v = rng.randn(3)
        starts.append(sphere_project(v))

    # --- Phase 1: Quick evaluation of all starts to find promising regions ---
    best_bound = np.inf
    best_c = starts[0]
    for c in starts:
        b = compute_bound(c)
        if b < best_bound:
            best_bound = b
            best_c = c

    # --- Phase 2: Local optimization from top-K starts using projected gradient ---
    # Evaluate all starts and sort
    evals = []
    for c in starts:
        b = compute_bound(c)
        evals.append((b, c))
    evals.sort(key=lambda t: t[0])

    # Take top 30 starts for local optimization
    top_k = 30
    local_starts = [c for _, c in evals[:top_k]]

    def sphere_minimize(c0, maxiter=300, tol=1e-14):
        """Minimize on the sphere using projected gradient descent with line search."""
        c = sphere_project(np.array(c0, dtype=float))
        for it in range(maxiter):
            val, grad_tan = sphere_obj_and_grad(c)
            if not np.isfinite(val):
                break
            gnorm = np.linalg.norm(grad_tan)
            if gnorm < tol:
                break
            # Gradient direction
            direction = -grad_tan / gnorm
            # Backtracking line search
            step = 0.1
            c_new = sphere_project(c + step * direction)
            val_new = compute_bound(c_new)
            backtracks = 0
            while (not np.isfinite(val_new) or val_new > val) and backtracks < 20:
                step *= 0.5
                c_new = sphere_project(c + step * direction)
                val_new = compute_bound(c_new)
                backtracks += 1
            if not np.isfinite(val_new):
                break
            if val_new < val:
                c = c_new
                val = val_new
            else:
                break
        return c, compute_bound(c)

    for c0 in local_starts:
        if time.time() - start_time > deadline - 20:
            break
        try:
            c_opt, b_opt = sphere_minimize(c0)
            if b_opt < best_bound:
                best_bound = b_opt
                best_c = c_opt
        except Exception:
            pass

    # --- Phase 3: Refine the best point with scipy L-BFGS-B on sphere ---
    if time.time() - start_time < deadline - 10:
        try:
            res = minimize(
                lambda v: sphere_obj_and_grad(v)[0],
                x0=best_c,
                jac=lambda v: sphere_obj_and_grad(v)[1],
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 100}
            )
            if np.isfinite(res.fun) and res.fun < best_bound:
                best_bound = res.fun
                best_c = sphere_project(res.x)
        except Exception:
            pass

    # --- Phase 4: Additional local optimization from perturbed best ---
    if time.time() - start_time < deadline - 5:
        rng2 = np.random.RandomState(999)
        for _ in range(10):
            if time.time() - start_time > deadline - 2:
                break
            pert = best_c + 0.1 * rng2.randn(3)
            pert = sphere_project(pert)
            try:
                c_opt, b_opt = sphere_minimize(pert, maxiter=200, tol=1e-14)
                if b_opt < best_bound:
                    best_bound = b_opt
                    best_c = c_opt
            except Exception:
                pass

    # Return the three coefficients (c0, c4, c8)
    return np.array([best_c[0], best_c[1], best_c[2]])

# EVOLVE-BLOCK-END
