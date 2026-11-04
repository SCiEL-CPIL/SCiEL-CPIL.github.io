import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    # Centers of the 11 hexagons in a "flower" pattern
    # 1 center + 6 at distance sqrt(3) + 4 at distance 3
    eps = 1e-12
    centers = np.array([[0.0, 0.0]])
    
    r1 = np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])
    
    r2 = 3.0 + 2 * eps
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])
    
    n_hex = len(centers)
    
    # Base vertices of unit hexagon (pointy-right orientation, angle 0)
    base_vertices = np.array([
        [np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)
    ])

    def compute_outer_side(theta_out, phis):
        """
        Vectorized computation of minimum outer hexagon side length.
        For regular hexagons, the support function is constant in all directions,
        so this depends only on centers and outer rotation.
        """
        angles = np.deg2rad(30 + np.arange(6) * 60) + theta_out
        normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)

        max_proj = 0.0
        for i in range(n_hex):
            phi = phis[i]
            c, s = np.cos(phi), np.sin(phi)
            rot = np.array([[c, -s], [s, c]])
            verts = base_vertices @ rot.T + centers[i]  # (6, 2)
            dots = verts @ normals.T  # (6, 6)
            max_proj = max(max_proj, np.max(np.abs(dots)))

        return 2.0 * max_proj / np.sqrt(3)

    def smooth_objective(params, beta=100.0):
        """
        Vectorized smooth approximation using log-sum-exp with adaptive beta.
        """
        theta_out = params[0]
        phis = params[1:]

        angles = np.deg2rad(30 + np.arange(6) * 60) + theta_out
        normals = np.column_stack([np.cos(angles), np.sin(angles)])

        all_abs_dots = []
        for i in range(n_hex):
            phi = phis[i]
            c, s = np.cos(phi), np.sin(phi)
            rot = np.array([[c, -s], [s, c]])
            verts = base_vertices @ rot.T + centers[i]
            dots = verts @ normals.T
            all_abs_dots.append(np.abs(dots).ravel())

        all_abs_dots = np.concatenate(all_abs_dots)

        max_val = np.max(all_abs_dots)
        if max_val == 0:
            return 0.0
        shifted = (all_abs_dots - max_val) * beta
        shifted = np.clip(shifted, -500, 500)
        log_sum_exp = max_val + np.log(np.sum(np.exp(shifted)) / len(all_abs_dots)) / beta

        return 2.0 * log_sum_exp / np.sqrt(3)

    def objective(params):
        """
        params: [theta_out, phi_0, ..., phi_10]
        Returns the outer hexagon side length.
        """
        theta_out = params[0]
        phis = params[1:]
        return compute_outer_side(theta_out, phis)

    # Analytical initial guess based on known optimal configuration
    # All inner hexagons at 30 degrees (flat-top), outer at 30 degrees
    x0 = np.array([np.deg2rad(30)] + [np.deg2rad(30)] * n_hex)

    best_score = objective(x0)
    best_params = x0.copy()

    # Phase 1: Multi-start Powell optimization with various starting points
    n_starts = 12
    start_variants = [
        np.array([np.deg2rad(30)] + [np.deg2rad(30)] * n_hex),  # Analytical
        np.array([0.0] + [0.0] * n_hex),  # All zero
        np.array([np.pi/6] + [0.0] * n_hex),  # Outer at 30, inner at 0
        np.array([0.0] + [np.pi/6] * n_hex),  # Outer at 0, inner at 30
        np.array([np.deg2rad(0)] + [np.deg2rad(30)] * n_hex),
        np.array([np.deg2rad(60)] + [np.deg2rad(30)] * n_hex),
        np.array([np.deg2rad(15)] + [np.deg2rad(15)] * n_hex),
        np.array([np.deg2rad(45)] + [np.deg2rad(45)] * n_hex),
        np.array([np.deg2rad(20)] + [np.deg2rad(25)] * n_hex),
        np.array([np.deg2rad(25)] + [np.deg2rad(20)] * n_hex),
    ]

    np.random.seed(42)
    for i in range(n_starts):
        if time.time() > deadline - 35:
            break

        if i < len(start_variants):
            x_start = start_variants[i].copy()
        else:
            x_start = np.random.uniform(0, np.pi/3, n_hex + 1)

        try:
            res = minimize(objective, x_start, method='Powell',
                          options={'maxiter': 400, 'disp': False, 'ftol': 1e-15, 'xtol': 1e-15})
            if res.fun < best_score - 1e-15:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: Smooth surrogate optimization with adaptive beta
    if time.time() < deadline - 25:
        for beta_val in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 20:
                break
            try:
                res = minimize(lambda p: smooth_objective(p, beta=beta_val), best_params,
                              method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
                true_score = objective(res.x)
                if true_score < best_score - 1e-15:
                    best_score = true_score
                    best_params = res.x.copy()
            except Exception:
                pass

    # Phase 3: High-precision polish with tight tolerances
    if time.time() < deadline - 15:
        try:
            res = minimize(objective, best_params, method='Powell',
                          options={'maxiter': 2000, 'disp': False, 'ftol': 1e-16, 'xtol': 1e-16})
            true_score = objective(res.x)
            if true_score < best_score - 1e-16:
                best_score = true_score
                best_params = res.x.copy()
        except Exception:
            pass

    # Extract solution
    theta_out = best_params[0]
    phis = best_params[1:]
    
    inner = []
    for i in range(n_hex):
        inner.append([centers[i][0], centers[i][1], np.rad2deg(phis[i])])
    
    side = best_score + 1e-9  # Tiny margin for floating point safety
    
    return inner, (0.0, 0.0), side, np.rad2deg(theta_out)

def _fallback_solution():
    """
    Analytical solution: all inner hexagons at 30 degrees, outer at 30 degrees.
    This is a known good configuration.
    """
    eps = 1e-10
    centers = []
    centers.append((0.0, 0.0))
    
    r1 = np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers.append((r1 * np.cos(angle), r1 * np.sin(angle)))
    
    r2 = 3.0 + 2 * eps
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers.append((r2 * np.cos(angle), r2 * np.sin(angle)))
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
    
    # The side length for the optimal flower packing is exactly 4.0
    # Add a tiny margin for floating point safety
    side = 4.0 + 1e-9
    
    return inner, (0.0, 0.0), side, 30.0

if __name__ == "__main__":
    construct()
