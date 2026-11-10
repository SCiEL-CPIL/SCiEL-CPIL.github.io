# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    W = 1.0 / N
    start = time.time()
    deadline = start + 595.0  # safety margin before 630s kill

    # shear rates
    shear = np.arange(N, dtype=np.float64) / N

    # precompute index pairs (i != j) and 1/(s_i - s_j)
    idx = np.arange(N)
    I, J = np.meshgrid(idx, idx, indexing='ij')
    M = (I != J)
    dS = (shear[I] - shear[J])
    safe = np.where(np.abs(dS) < 1e-12, 1.0, dS)
    inv_dS = np.where(np.abs(dS) < 1e-12, 0.0, 1.0 / safe)
    # flatten to only i != j
    mask = M.ravel()
    If = I.ravel()[mask]
    Jf = J.ravel()[mask]
    invf = inv_dS.ravel()[mask]

    def _eval(x, need_grad):
        """Return V (and optionally dV/dx). x has shape (N,)."""
        # breakpoints
        num1 = x[Jf] - x[If] - W
        num2 = x[Jf] - x[If]
        c1 = num1 * invf
        c2 = num2 * invf
        in1 = (c1 > 0.0) & (c1 < 1.0)
        in2 = (c2 > 0.0) & (c2 < 1.0)
        pts = np.unique(np.concatenate(([0.0, 1.0], c1[in1], c2[in2])))
        pts = pts[(pts > 0.0) & (pts < 1.0)]
        pts = np.concatenate(([0.0], pts, [1.0]))

        V = 0.0
        grad = np.zeros(N) if need_grad else None

        for m in range(len(pts) - 1):
            c0, c1v = pts[m], pts[m + 1]
            dc = c1v - c0
            if dc < 1e-13:
                continue
            cm = 0.5 * (c0 + c1v)

            L = x + cm * shear
            R = L + W
            order = np.argsort(L, kind='stable')
            Ls = L[order]
            Rs = R[order]

            # union segments with original indices
            seg2 = []
            lidx = order[0]
            ridx = order[0]
            ceo = Rs[0]
            for k in range(1, N):
                if Ls[k] > ceo + 1e-15:
                    seg2.append((lidx, ridx))
                    lidx = order[k]
                    ridx = order[k]
                    ceo = Rs[k]
                else:
                    if Rs[k] > ceo:
                        ceo = Rs[k]
                        ridx = order[k]
            seg2.append((lidx, ridx))

            U = sum((x[ri] + cm*shear[ri] + W) - (x[li] + cm*shear[li]) for li, ri in seg2)
            # slope S of U within this interval (piecewise-constant, valid at midpoint)
            S = sum(shear[ri] - shear[li] for li, ri in seg2)

            a = c0 - cm
            b = c1v - cm
            V += U * U * dc + U * S * (b*b - a*a) + (S*S/3.0) * (b*b*b - a*a*a)

            if need_grad:
                dU = np.zeros(N)
                for li, ri in seg2:
                    dU[ri] += 1.0
                    dU[li] -= 1.0
                coeff = 2.0 * U * dc + S * (b*b - a*a)
                grad += coeff * dU

        return V, grad

    def objective(x):
        V, _ = _eval(x, False)
        return V

    def objective_grad(x):
        V, g = _eval(x, True)
        return V, g

    # initial candidates
    k = np.arange(N, dtype=np.float64)
    init_candidates = [
        -k / (2.0 * N),
        -k / (2.0 * N) * 0.5,
        -k / (2.0 * N) * 1.5,
        np.linspace(-0.5, 0.5, N),
        -k * W * 0.9,
        -k * W * 1.1,
    ]

    best_x = init_candidates[0].copy()
    best_val = objective(best_x)

    bounds = [(-3.0, 3.0)] * N

    # Stage 1: differential evolution (global)
    if time.time() < deadline - 120:
        try:
            res = differential_evolution(
                objective, bounds, seed=42, maxiter=300,
                tol=1e-9, popsize=18, polish=False, updating='immediate'
            )
            if np.all(np.isfinite(res.x)) and res.fun < best_val:
                best_x = res.x.copy()
                best_val = res.fun
        except Exception:
            pass

    # Stage 2: L-BFGS-B with analytic gradient
    if time.time() < deadline - 60:
        try:
            res2 = minimize(objective_grad, best_x, method='L-BFGS-B',
                            jac=True, bounds=bounds,
                            options={'maxiter': 2000, 'gtol': 1e-12, 'ftol': 1e-15,
                                     'maxls': 50})
            if np.all(np.isfinite(res2.x)) and res2.fun < best_val:
                best_x = res2.x.copy()
                best_val = res2.fun
        except Exception:
            pass

    # Stage 3: Nelder-Mead fine polish
    if time.time() < deadline - 20:
        try:
            res3 = minimize(objective, best_x, method='Nelder-Mead',
                            options={'maxiter': 10000, 'xatol': 1e-13, 'fatol': 1e-16})
            if np.all(np.isfinite(res3.x)) and res3.fun < best_val:
                best_x = res3.x.copy()
                best_val = res3.fun
        except Exception:
            pass

    # Stage 4: final L-BFGS-B ultra-fine
    if time.time() < deadline - 5:
        try:
            res4 = minimize(objective_grad, best_x, method='L-BFGS-B',
                            jac=True, bounds=bounds,
                            options={'maxiter': 5000, 'gtol': 1e-14, 'ftol': 1e-18,
                                     'maxls': 100})
            if np.all(np.isfinite(res4.x)) and res4.fun < best_val:
                best_x = res4.x.copy()
                best_val = res4.fun
        except Exception:
            pass

    # Build 64x2 positions
    positions = np.zeros((64, 2), dtype=np.float64)
    for k_idx in range(64):
        i = k_idx // N
        j = k_idx % N
        positions[k_idx, 0] = best_x[i]
        positions[k_idx, 1] = best_x[j]

    return positions

# EVOLVE-BLOCK-END
