"""Optimized Hermite coefficient search for uncertainty inequality bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize, differential_evolution, basinhopping
from scipy.integrate import quad


def construct():
    """Return three coefficients for H0, H4, H8 Hermite combination."""
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin for 210s wall-clock limit

    # Seed coefficients
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    # Gauss-Hermite quadrature nodes (physicists' convention: weight e^{-x^2})
    n_quad = 1000
    t_nodes, t_weights = hermgauss(n_quad)
    x_nodes = np.sqrt(2.0) * t_nodes
    x_weights = np.sqrt(2.0) * t_weights

    # Precompute powers
    x2 = x_nodes**2
    x4 = x2**2
    x6 = x4 * x2
    x8 = x4**2
    x10 = x8 * x2
    x12 = x6**2

    # Physicists' Hermite polynomials at quadrature nodes
    H0 = np.ones_like(x_nodes)
    H4 = 16.0 * x4 - 48.0 * x2 + 12.0
    H8 = 256.0 * x8 - 2304.0 * x6 + 5760.0 * x4 - 4320.0 * x2 + 504.0
    H12 = (4096.0 * x12 - 55296.0 * x10 + 303104.0 * x8
           - 589824.0 * x6 + 408192.0 * x4 - 77760.0 * x2 + 10395.0)

    def get_a12(a0, a4, a8):
        """Compute a12 so that g(0) = 0. H12(0) = 10395."""
        return (a0 - 3.0 * a4 + 105.0 * a8) / 10395.0

    def compute_bound(coeffs):
        """Compute the C4 upper bound.
        
        For physicists' Hermite functions, Fourier transform gives i^n factor.
        i^12 = (i^4)^3 = 1, so FP uses +a12*H12.
        """
        a0, a4, a8 = coeffs
        a12 = get_a12(a0, a4, a8)

        P = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
        FP = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12  # i^12 = +1

        norm2_g_sq = np.sum(x_weights * P**2)
        norm2_fg_sq = np.sum(x_weights * FP**2)

        if norm2_g_sq < 1e-15 or norm2_fg_sq < 1e-15:
            return 1e10

        norm2_g = np.sqrt(norm2_g_sq)
        norm2_fg = np.sqrt(norm2_fg_sq)

        norm4_g_4 = np.sum(x_weights * P**4)
        norm4_fg_4 = np.sum(x_weights * FP**4)

        if norm4_g_4 < 1e-15 or norm4_fg_4 < 1e-15:
            return 1e10

        norm4_g = norm4_g_4 ** 0.25
        norm4_fg = norm4_fg_4 ** 0.25

        return (norm4_g * norm4_fg) / (norm2_g * norm2_fg)

    def compute_gradient(coeffs):
        """Compute gradient of the C4 bound analytically."""
        a0, a4, a8 = coeffs
        a12 = get_a12(a0, a4, a8)

        P = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
        FP = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12  # i^12 = +1

        norm2_g_sq = np.sum(x_weights * P**2)
        norm2_fg_sq = np.sum(x_weights * FP**2)

        if norm2_g_sq < 1e-15 or norm2_fg_sq < 1e-15:
            return np.zeros(3)

        norm2_g = np.sqrt(norm2_g_sq)
        norm2_fg = np.sqrt(norm2_fg_sq)

        norm4_g_4 = np.sum(x_weights * P**4)
        norm4_fg_4 = np.sum(x_weights * FP**4)

        if norm4_g_4 < 1e-15 or norm4_fg_4 < 1e-15:
            return np.zeros(3)

        norm4_g = norm4_g_4 ** 0.25
        norm4_fg = norm4_fg_4 ** 0.25

        bound = (norm4_g * norm4_fg) / (norm2_g * norm2_fg)

        da12_da = np.array([1.0, -3.0, 105.0]) / 10395.0

        H = np.array([H0, H4, H8])
        dP_da = H + da12_da[:, None] * H12[None, :]
        dFP_da = H + da12_da[:, None] * H12[None, :]  # Same sign as P

        d_norm2_g_sq_da = 2.0 * (x_weights[None, :] * P[None, :] * dP_da).sum(axis=1)
        d_norm2_fg_sq_da = 2.0 * (x_weights[None, :] * FP[None, :] * dFP_da).sum(axis=1)
        d_norm4_g_4_da = 4.0 * (x_weights[None, :] * P[None, :]**3 * dP_da).sum(axis=1)
        d_norm4_fg_4_da = 4.0 * (x_weights[None, :] * FP[None, :]**3 * dFP_da).sum(axis=1)

        d_norm2_g_da = d_norm2_g_sq_da / (2.0 * norm2_g)
        d_norm2_fg_da = d_norm2_fg_sq_da / (2.0 * norm2_fg)
        d_norm4_g_da = d_norm4_g_4_da / (4.0 * norm4_g)
        d_norm4_fg_da = d_norm4_fg_4_da / (4.0 * norm4_fg)

        grad = bound * (d_norm4_g_da / norm4_g + d_norm4_fg_da / norm4_fg
                        - d_norm2_g_da / norm2_g - d_norm2_fg_da / norm2_fg)

        return grad

    def objective(coeffs):
        return compute_bound(coeffs)

    def grad_func(coeffs):
        return compute_gradient(coeffs)

    best_coeffs = seed.copy()
    best_val = objective(seed)

    # Spherical Fibonacci lattice on S^2 for global exploration
    # The bound is scale-invariant (homogeneous degree 0), so we work on the sphere.
    # Map unit sphere directions to coefficient space.
    
    def fibonacci_sphere(n):
        """Generate n points on S^2 using Fibonacci lattice."""
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        points = np.zeros((n, 3))
        for i in range(n):
            z = 1.0 - 2.0 * (i + 0.5) / n
            r = np.sqrt(max(0.0, 1.0 - z**2))
            theta = golden_angle * i
            points[i] = [r * np.cos(theta), r * np.sin(theta), z]
        return points

    def direction_to_coeffs(direction):
        """Convert unit direction to coefficient vector (scale-invariant)."""
        # The bound is homogeneous of degree 0, so any scalar multiple gives same bound.
        # We use the direction directly as coefficients.
        return direction.copy()

    # Phase 1: Global search using Differential Evolution on spherical angles
    # The objective is scale-invariant, so we optimize over the sphere (theta, phi).
    # theta in [0, pi], phi in [0, 2*pi]
    if time.time() < deadline - 80.0:
        try:
            bounds_angle = [(0.0, np.pi), (0.0, 2.0 * np.pi)]
            
            def angle_obj(angles):
                t, p = angles
                # Handle poles to avoid singularity in sin(t)
                if np.sin(t) < 1e-10:
                    # At poles, phi is irrelevant, but we need a valid direction
                    if t < 1e-10: # North pole
                        return compute_bound(np.array([0.0, 0.0, 1.0]))
                    else: # South pole
                        return compute_bound(np.array([0.0, 0.0, -1.0]))
                
                x = np.sin(t) * np.cos(p)
                y = np.sin(t) * np.sin(p)
                z = np.cos(t)
                return compute_bound(np.array([x, y, z]))
            
            de_result = differential_evolution(
                angle_obj,
                bounds=bounds_angle,
                seed=42,
                maxiter=50,
                popsize=20,
                tol=1e-8,
                mut=(0.5, 1.0),
                recomb=0.7,
                polish=False,
                init='latinhypercube'
            )
            
            if de_result.fun < best_val:
                best_val = de_result.fun
                t, p = de_result.x
                if np.sin(t) < 1e-10:
                    if t < 1e-10:
                        best_coeffs = np.array([0.0, 0.0, 1.0])
                    else:
                        best_coeffs = np.array([0.0, 0.0, -1.0])
                else:
                    best_coeffs = np.array([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)])
        except Exception:
            pass

    # Phase 2: Local refinement using gradient-based method on sphere
    if time.time() < deadline - 50.0:
        try:
            # Project best_coeffs to unit sphere
            norm_bc = np.linalg.norm(best_coeffs)
            if norm_bc > 1e-15:
                init_dir = best_coeffs / norm_bc
            else:
                init_dir = seed / (np.linalg.norm(seed) + 1e-30)
            
            # Use L-BFGS-B on the 3D space with constraint that we normalize
            # Actually, since bound is scale-invariant, we can just optimize in 3D
            # and the scale doesn't matter. But to avoid numerical issues,
            # let's optimize on the sphere using a change of variables.
            
            # Use spherical coordinates: direction = [sin(t1)*cos(t2), sin(t1)*sin(t2), cos(t1)]
            x, y, z = init_dir
            r_xy = np.sqrt(x**2 + y**2)
            if r_xy > 1e-15:
                init_theta1 = np.arcsin(np.clip(r_xy, -1, 1))
                init_theta2 = np.arctan2(y, x)
            else:
                init_theta1 = 0.0
                init_theta2 = 0.0
            
            def angular_objective(angles):
                t1, t2 = angles
                sin_t1 = np.sin(t1)
                cos_t1 = np.cos(t1)
                sin_t2 = np.sin(t2)
                cos_t2 = np.cos(t2)
                direction = np.array([sin_t1 * cos_t2, sin_t1 * sin_t2, cos_t1])
                return compute_bound(direction)
            
            ang_result = minimize(
                angular_objective,
                [init_theta1, init_theta2],
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-18, 'adaptive': True}
            )
            
            if ang_result.fun < best_val:
                best_val = ang_result.fun
                t1, t2 = ang_result.x
                sin_t1 = np.sin(t1)
                cos_t1 = np.cos(t1)
                sin_t2 = np.sin(t2)
                cos_t2 = np.cos(t2)
                best_coeffs = np.array([sin_t1 * cos_t2, sin_t1 * sin_t2, cos_t1])
        except Exception:
            pass

    # Phase 3: Multi-start local optimization from random perturbations of best
    if time.time() < deadline - 30.0:
        try:
            # Generate multiple starting points near the current best and random points
            start_points = [best_coeffs.copy()]
            
            # Add random perturbations to best
            for _ in range(5):
                perturbed = best_coeffs + np.random.normal(0, 0.1, size=3)
                start_points.append(perturbed)
            
            # Add random points on sphere
            for _ in range(10):
                u = np.random.randn(3)
                start_points.append(u / np.linalg.norm(u))
            
            for start_dir in start_points:
                if time.time() > deadline - 30.0:
                    break
                # Local optimization in Cartesian space
                res = minimize(
                    objective,
                    start_dir,
                    jac=grad_func,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'gtol': 1e-14, 'ftol': 1e-16}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_coeffs = res.x.copy()
        except Exception:
            pass

    # Phase 4: Basin hopping for robust global refinement
    if time.time() < deadline - 20.0:
        try:
            # Use basinhopping which combines local optimization with random jumps
            # We need to provide a local optimizer
            local_min = lambda x: minimize(objective, x, jac=grad_func, method='L-BFGS-B', options={'maxiter': 200, 'gtol': 1e-12})
            
            bh_result = basinhopping(
                objective,
                best_coeffs,
                minimizer=local_min,
                niter=20,
                stepsize=0.5,
                seed=42
            )
            
            if bh_result.fun < best_val:
                best_val = bh_result.fun
                best_coeffs = bh_result.x.copy()
        except Exception:
            pass

    # Phase 5: High-precision final polish with multiple restarts
    if time.time() < deadline - 5.0:
        try:
            # Try polishing from the best found and a few neighbors
            polish_starts = [best_coeffs]
            for _ in range(3):
                polish_starts.append(best_coeffs + np.random.normal(0, 1e-4, size=3))
            
            for ps in polish_starts:
                if time.time() > deadline - 2.0:
                    break
                lb_result = minimize(
                    objective,
                    ps,
                    jac=grad_func,
                    method='L-BFGS-B',
                    options={'maxiter': 5000, 'gtol': 1e-18, 'ftol': 1e-20, 'maxls': 100}
                )
                if lb_result.fun < best_val:
                    best_val = lb_result.fun
                    best_coeffs = lb_result.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
