import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def construct():
    def get_vertices_from_edges(e):
        """Given 8 positive edge lengths (sum=1), return vertices of cyclic polygon."""
        def f(R):
            vals = e / (2.0 * R)
            if np.any(vals > 1.0):
                return -np.pi
            return np.sum(np.arcsin(vals)) - np.pi

        e_max = np.max(e)
        lo = e_max / 2.0 * 1.001
        hi = 100.0 * e_max

        if f(lo) < 0:
            return None
        while f(hi) > 0:
            hi *= 2
            if hi > 1e8:
                return None

        for _ in range(80):
            mid = (lo + hi) / 2.0
            if f(mid) > 0:
                lo = mid
            else:
                hi = mid

        R = (lo + hi) / 2.0
        central_angles = 2.0 * np.arcsin(e / (2.0 * R))
        cum_angles = np.concatenate(([0.0], np.cumsum(central_angles)))[:8]

        x = R * np.cos(cum_angles)
        y = R * np.sin(cum_angles)
        return np.column_stack((x, y))

    def objective(t):
        e = np.exp(t - np.max(t))
        e = e / np.sum(e)

        vertices = get_vertices_from_edges(e)
        if vertices is None:
            return 1e10

        dists = pdist(vertices)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 1e10

        dists_scaled = dists * (2.0 / d_max)
        log_prod = np.sum(np.log(dists_scaled + 1e-12))
        return -log_prod

    best_score = np.inf
    best_t = None

    np.random.seed(42)
    for restart in range(30):
        if restart == 0:
            t0 = np.zeros(8)
        else:
            t0 = np.random.randn(8) * 0.5

        result = minimize(objective, t0, method='Nelder-Mead',
                         options={'maxiter': 20000, 'xatol': 1e-9, 'fatol': 1e-9})

        if result.fun < best_score:
            best_score = result.fun
            best_t = result.x.copy()

    e = np.exp(best_t - np.max(best_t))
    e = e / np.sum(e)
    opt_vertices = get_vertices_from_edges(e)

    dists = pdist(opt_vertices)
    d_max = np.max(dists)
    if d_max > 1e-12:
        opt_vertices = opt_vertices * (2.0 / d_max)

    return opt_vertices.tolist()
