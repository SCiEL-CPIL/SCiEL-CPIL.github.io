"""
Third autocorrelation inequality: minimize 2*N*max(conv(h,h))/sum(h)^2 over h in R^N, N=400.

Strategy:
- Represent h as a sum of a small number of Gaussian bumps (low-dimensional parameterization).
- Global search with differential evolution on the parameter space.
- Local refinement with Nelder-Mead on the same parameter space.
- Final polish: direct coordinate descent on h itself for a few iterations.
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 400


def compute_value(h):
    """Compute the task value: 2*N*max(conv(h,h))/sum(h)^2"""
    h = np.asarray(h, dtype=float)
    s = np.sum(h)
    if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s ** 2))


def make_h_from_gaussians(params):
    """
    Convert parameter vector to step heights h of length N.
    params layout for each Gaussian j (0..K-1):
      [center_j, width_j, amp_j]
    Plus a final DC offset.
    Total params: 3*K + 1
    """
    K = (len(params) - 1) // 3
    t = np.arange(N, dtype=float) / N  # normalized time in [0, 1)
    h = np.zeros(N)
    for j in range(K):
        center = params[3 * j]
        width = params[3 * j + 1]
        amp = params[3 * j + 2]
        # Gaussian bump: amp * exp(-((t - center)^2) / (2 * width^2))
        h += amp * np.exp(-((t - center) ** 2) / (2.0 * width ** 2))
    h += params[-1]  # DC offset
    return h


def objective_from_gaussians(params):
    """Objective function operating on Gaussian parameter space."""
    h = make_h_from_gaussians(params)
    return compute_value(h)


def construct():
    t_start = time.time()
    DEADLINE = t_start + 140.0  # safety margin before 150s kill

    def remaining():
        return DEADLINE - time.time()

    # Number of Gaussians
    K = 8

    # Initial best: seed
    best_h = np.ones(N)
    best_val = compute_value(best_h)  # 2.0

    # --- Phase 1: Global search with differential evolution ---
    bounds = []
    for j in range(K):
        bounds.append((0.0, 1.0))    # center in [0, 1)
        bounds.append((0.01, 0.5))   # width
        bounds.append((-2.0, 2.0))   # amplitude
    bounds.append((-3.0, 3.0))       # DC offset

    if remaining() > 50.0:
        try:
            res_de = differential_evolution(
                objective_from_gaussians,
                bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-7,
                polish=False,
                init='sobol',
                updating='immediate',
                workers=1,
            )
            if res_de.fun < best_val:
                best_val = res_de.fun
                best_h = make_h_from_gaussians(res_de.x)
        except Exception:
            pass

    # --- Phase 2: Multi-restart Nelder-Mead on Gaussian parameter space ---
    rng = np.random.default_rng(123)
    n_restarts = 15
    for i in range(n_restarts):
        if remaining() < 15.0:
            break

        # Generate random parameters within bounds
        params_rand = []
        for j in range(K):
            params_rand.append(rng.uniform(0.0, 1.0))
            params_rand.append(rng.uniform(0.01, 0.5))
            params_rand.append(rng.uniform(-2.0, 2.0))
        params_rand.append(rng.uniform(-3.0, 3.0))
        params_rand = np.array(params_rand)

        try:
            res_nm = minimize(
                objective_from_gaussians,
                params_rand,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
            )
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_h = make_h_from_gaussians(res_nm.x)
        except Exception:
            pass

    # --- Phase 3: Coordinate descent polish directly on h ---
    if remaining() > 10.0:
        h = best_h.copy()
        for iteration in range(50):
            if remaining() < 3.0:
                break
            improved = False
            # Randomize coordinate order
            indices = rng.permutation(N)
            for i in indices:
                if remaining() < 3.0:
                    break
                base = h[i]
                # Try a few step sizes
                for step in [0.001, -0.001, 0.005, -0.005, 0.0001, -0.0001]:
                    h[i] = base + step
                    v = compute_value(h)
                    if v < best_val:
                        best_val = v
                        improved = True
                        break
                else:
                    h[i] = base
            if not improved:
                break
        best_h = h

    # Ensure validity
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)

    return best_h


if __name__ == "__main__":
    h = construct()
    print(f"Value: {compute_value(h)}")
    print(f"Score: {2.0 / compute_value(h)}")
