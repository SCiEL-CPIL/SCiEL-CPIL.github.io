import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    Uses differential evolution for global search followed by Nelder-Mead refinement.
    """
    n = 64
    a = np.arange(1, n + 1) / n  # shape (64,)

    def union_area_exact(x, num_h=2000):
        """
        Exact union-area over a y-grid.
        x: shape (64,)
        Returns scalar area estimate (mean union length over y).
        """
        ys = (np.arange(num_h) + 0.5) / num_h
        ws = (1.0 - ys) / n
        Ls = x[None, :] + a[None, :] * ys[:, None]
        sorted_Ls = np.sort(Ls, axis=1)
        diffs = np.diff(sorted_Ls, axis=1)
        extensions = np.minimum(ws[:, None], diffs)
        union_lens = ws + extensions.sum(axis=1)
        return union_lens.mean()

    def objective(x):
        return union_area_exact(x, num_h=500)

    # Time guard
    start_time = time.time()
    max_time = 620.0

    # Keich seed
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    seed_positions = np.zeros(n, dtype=np.float64)
    for i in range(n):
        bits = np.array(list(str(bin(i))[2:]), dtype=np.int8)
        bits = np.pad(bits, (k - len(bits), 0), constant_values=0)
        seed_positions[i] = np.sum((1.0 - j) * bits * (2.0 ** -j) / k)

    best_x = seed_positions.copy()
    best_area = objective(best_x)

    # Differential evolution for global search
    if time.time() - start_time < max_time - 30:
        bounds = [(-0.5, 1.5)] * n
        try:
            de_result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                polish=False,
                init='latinhypercube'
            )
            if de_result.fun < best_area:
                best_area = de_result.fun
                best_x = de_result.x
        except Exception:
            pass

    # Nelder-Mead refinement
    if time.time() - start_time < max_time - 10:
        try:
            nm_result = minimize(
                objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if nm_result.fun < best_area:
                best_area = nm_result.fun
                best_x = nm_result.x
        except Exception:
            pass

    # Final coordinate descent polish
    if time.time() - start_time < max_time - 5:
        x = best_x.copy()
        step = 0.005
        for _ in range(50):
            if time.time() - start_time > max_time - 2:
                break
            improved = False
            for i in range(n):
                if time.time() - start_time > max_time - 2:
                    break
                best_val = objective(x)
                best_delta = 0.0
                for delta in (step, -step, step/2, -step/2):
                    x[i] += delta
                    val = objective(x)
                    x[i] -= delta
                    if val < best_val - 1e-14:
                        best_val = val
                        best_delta = delta
                if best_delta != 0.0:
                    x[i] += best_delta
                    improved = True
            if not improved:
                break
            step *= 0.7
        final_area = objective(x)
        if final_area < best_area:
            best_area = final_area
            best_x = x

    return np.array(best_x, dtype=np.float64)
