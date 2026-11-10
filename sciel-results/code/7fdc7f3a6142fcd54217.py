import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution

def min_max_dist_dim3_14() -> np.ndarray:
    start = time.time()
    DEADLINE = start + 620  # Safety margin before 630s kill

    n = 14
    d = 3

    def score(pts):
        dd = pdist(pts)
        if dd.max() <= 0:
            return 0.0
        return (dd.min() / dd.max()) ** 2

    def neg_metric(x):
        pts = x.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        return -score(pts)

    def make_nm(x0, maxiter=20000, xatol=1e-12, fatol=1e-14):
        r = minimize(neg_metric, x0, method='Nelder-Mead',
                     options={'maxiter': maxiter, 'xatol': xatol,
                              'fatol': fatol, 'adaptive': True})
        return r.x

    def make_lbfgs(x0, maxiter=1000):
        r = minimize(neg_metric, x0, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-14})
        return r.x

    # Surrogate: Coulomb repulsion
    def coulomb_obj(x):
        pts = x.reshape((n, d))
        dd = pdist(pts)
        dd = np.maximum(dd, 1e-8)
        return np.sum(1.0 / dd**2)

    def neg_coulomb(x):
        return -coulomb_obj(x)

    best_pts = None
    best_score = -1.0
    rng = np.random.RandomState(42)

    def update_best(pts):
        nonlocal best_pts, best_score
        pts = pts - np.mean(pts, axis=0)
        s = score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
        return s

    # Strategy 1: Differential Evolution on Coulomb Surrogate
    # This is a smooth function, so DE works well.
    # We run multiple DE instances with different seeds and populations.
    
    def run_de_surrogate(seed, popsize, maxiter):
        bounds = [(-2, 2)] * (n * d)
        try:
            r = differential_evolution(coulomb_obj, bounds,
                                        maxiter=maxiter, popsize=popsize,
                                        tol=1e-10, seed=seed, polish=False)
            return r.x
        except Exception:
            return None

    # Run several DE instances
    de_configs = [
        (42, 15, 50),
        (123, 20, 40),
        (999, 25, 30),
        (777, 15, 60),
    ]
    
    for seed, pop, iter_ in de_configs:
        if time.time() > DEADLINE - 100:
            break
        x_de = run_de_surrogate(seed, pop, iter_)
        if x_de is not None:
            # Polish with NM on true metric
            try:
                x_nm = make_nm(x_de, maxiter=15000)
                pts = x_nm.reshape((n, d))
                update_best(pts)
            except Exception:
                pts = x_de.reshape((n, d))
                update_best(pts)

    # Strategy 2: Direct DE on True Metric (with polish)
    # The true metric is non-smooth, but DE can handle it.
    # We use a smaller population and fewer iterations to fit in time.
    if time.time() < DEADLINE - 80:
        try:
            bounds = [(-2, 2)] * (n * d)
            r_de = differential_evolution(neg_metric, bounds,
                                           maxiter=40, popsize=15,
                                           tol=1e-8, seed=1, polish=False)
            pts = r_de.x.reshape((n, d))
            update_best(pts)
            
            # Polish the result
            if time.time() < DEADLINE - 30:
                try:
                    x_final = make_nm(r_de.x, maxiter=20000)
                    pts_final = x_final.reshape((n, d))
                    update_best(pts_final)
                except Exception:
                    pass
        except Exception:
            pass

    # Strategy 3: Icosahedral Symmetry Search
    # 12 vertices of icosahedron + 2 poles.
    # This is a strong prior for N=14.
    phi = (1 + np.sqrt(5)) / 2
    I12 = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    I12 = I12 / np.linalg.norm(I12, axis=1, keepdims=True)
    R0 = 1.0 / np.sqrt(1 + phi**2)

    def make_points_ico(p):
        r = p[0]
        pts = np.vstack([r * I12, [[0, 0, p[1]], [0, 0, p[2]]]])
        return pts

    def neg_metric_ico(p):
        pts = make_points_ico(p)
        dd = pdist(pts)
        if dd.max() <= 0:
            return 0.0
        return -(dd.min() / dd.max()) ** 2

    def make_nm_ico(p0, maxiter=30000):
        r = minimize(neg_metric_ico, p0, method='Nelder-Mead',
                     options={'maxiter': maxiter, 'xatol': 1e-14,
                              'fatol': 1e-16, 'adaptive': True})
        return r.x

    # Grid search + NM polish for icosahedral symmetry
    ico_candidates = []
    for r in np.linspace(0.8, 1.2, 9):
        for z1 in np.linspace(-0.8, 0.8, 9):
            for z2 in np.linspace(-0.8, 0.8, 9):
                if abs(z1 - z2) > 0.1: # Avoid coincident poles
                    ico_candidates.append(np.array([r, z1, z2]))

    # Add random candidates
    rng_ico = np.random.RandomState(7)
    for _ in range(50):
        r = R0 * (0.9 + 0.2 * rng_ico.rand())
        z1 = rng_ico.rand() * 1.6 - 0.8
        z2 = rng_ico.rand() * 1.6 - 0.8
        ico_candidates.append(np.array([r, z1, z2]))

    for p in ico_candidates:
        if time.time() > DEADLINE - 50:
            break
        try:
            p1 = make_nm_ico(p)
            pts = make_points_ico(p1)
            update_best(pts)
        except Exception:
            pass

    # Strategy 4: Final Polish on Best Result
    if best_pts is not None and time.time() < DEADLINE - 10:
        x_best = best_pts.flatten()
        try:
            x_final = make_nm(x_best, maxiter=50000, xatol=1e-13, fatol=1e-15)
            pts_final = x_final.reshape((n, d))
            update_best(pts_final)
        except Exception:
            pass

    if best_pts is None:
        # Fallback: Random points
        best_pts = rng.randn(n, d)
        best_pts = best_pts - np.mean(best_pts, axis=0)

    if not np.all(np.isfinite(best_pts)):
        best_pts = np.vstack([I12, [[0, 0, -0.25], [0, 0, 0.25]]])
        best_pts = best_pts - np.mean(best_pts, axis=0)

    return best_pts
