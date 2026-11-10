# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min / d_max)^2.
    Uses Coulomb repulsion pre-stage + differential evolution + NM polish.
    """
    n = 14
    deadline = time.time() + 580.0

    def score(pts):
        if pts.shape != (14, 3):
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        d = pdist(pts)
        dmin = np.min(d)
        dmax = np.max(d)
        if dmax < 1e-12:
            return 0.0
        return (dmin / dmax) ** 2

    def neg_score(pts):
        return -score(pts)

    # Stage 1: Coulomb repulsion to get well-spread initial points
    def coulomb_energy(pts):
        d = pdist(pts)
        # Avoid division by zero
        d = np.maximum(d, 1e-10)
        return -np.sum(1.0 / d)

    def neg_coulomb(x):
        pts = x.reshape(n, 3)
        return -coulomb_energy(pts)

    # Multiple Coulomb-repulsion starts
    best_score = 0.0
    best_pts = None
    coulomb_starts = []

    n_coulomb = 8
    for i in range(n_coulomb):
        if time.time() > deadline - 120:
            break
        # Random initial points on unit sphere
        rng = np.random.default_rng(seed=42 + i * 1000)
        x0 = rng.standard_normal((n, 3))
        norms = np.linalg.norm(x0, axis=1, keepdims=True)
        x0 = x0 / norms

        try:
            res = minimize(neg_coulomb, x0.flatten(), method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-10})
            pts = res.x.reshape(n, 3)
            if np.all(np.isfinite(pts)):
                s = score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
                coulomb_starts.append(pts.copy())
        except Exception:
            pass

    # Fallback if Coulomb didn't work
    if best_pts is None or len(coulomb_starts) == 0:
        phi_i = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
        theta_i = np.pi * (1 + 5**0.5) * np.arange(n)
        best_pts = np.column_stack((
            np.sin(phi_i) * np.cos(theta_i),
            np.sin(phi_i) * np.sin(theta_i),
            np.cos(phi_i)
        ))
        best_score = score(best_pts)
        coulomb_starts.append(best_pts.copy())

    # Stage 2: Differential Evolution for global optimization
    # DE works on the 42-D space but we need bounds
    # Normalize to unit sphere region
    bounds = [(-1.5, 1.5)] * (n * 3)

    def de_objective(x):
        pts = x.reshape(n, 3)
        # Penalize if points go too far or coincide
        d = pdist(pts)
        dmin = np.min(d)
        dmax = np.max(d)
        if dmax < 1e-12 or dmin < 1e-12:
            return 1e6
        return -(dmin / dmax) ** 2

    # Use best Coulomb start as a hint by seeding the population
    # DE with multiple restarts
    n_de_restarts = 3
    for restart in range(n_de_restarts):
        if time.time() > deadline - 60:
            break

        try:
            # Use different strategy for each restart
            if restart == 0:
                # Seed from best Coulomb result
                pop_size = 40
                strategy = 'best1bin'
            elif restart == 1:
                pop_size = 30
                strategy = 'rand1bin'
            else:
                pop_size = 25
                strategy = 'best2bin'

            result = differential_evolution(
                de_objective,
                bounds,
                strategy=strategy,
                popsize=pop_size,
                maxiter=200,
                tol=1e-8,
                seed=42 + restart * 100,
                polish=True,
                init='latinhypercube'
            )

            if result.success or result.fun < 1e5:
                new_pts = result.x.reshape(n, 3)
                if np.all(np.isfinite(new_pts)):
                    new_score = score(new_pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_pts = new_pts.copy()
        except Exception:
            pass

    # Stage 3: Final NM polish from best point
    if time.time() < deadline - 10 and best_pts is not None:
        try:
            x0 = best_pts.flatten()
            res = minimize(neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            new_pts = res.x.reshape(n, 3)
            if np.all(np.isfinite(new_pts)):
                new_score = score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts
        except Exception:
            pass

    # Final validation
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        phi_i = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
        theta_i = np.pi * (1 + 5**0.5) * np.arange(n)
        best_pts = np.column_stack((
            np.sin(phi_i) * np.cos(theta_i),
            np.sin(phi_i) * np.sin(theta_i),
            np.cos(phi_i)
        ))

    return best_pts

# EVOLVE-BLOCK-END
