"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start + Nelder-Mead + adaptive (1+1)-ES with restarts for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 295.0
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        calls = 0
        best_x = None
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or not time_ok():
                return None
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                fx = best_f
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Dense multi-start exploration for early AOCC
        n_initial = 30
        candidates = []
        for i in range(n_initial):
            if calls >= budget or not time_ok():
                break
            if i == 0:
                x = np.zeros(n)
            elif i == 1:
                x = rng.uniform(-3, 3, n)
            elif i == 2:
                x = rng.uniform(-4, 4, n)
            elif i == 3:
                x = rng.uniform(-5, 5, n)
            elif i == 4:
                x = rng.uniform(-2, 2, n)
            elif i == 5:
                x = rng.uniform(-1, 1, n)
            else:
                x = rng.uniform(-5, 5, n)
            fx = evaluate(x)
            if fx is not None:
                candidates.append((fx, x.copy()))

        if not candidates:
            x = rng.uniform(-5, 5, n)
            evaluate(x)
            return best_x, best_f

        candidates.sort(key=lambda t: t[0])

        # Phase 2: Nelder-Mead on top candidates
        top_k = min(3, len(candidates))
        
        for rank in range(top_k):
            if calls >= budget or not time_ok():
                break

            f0, x0 = candidates[rank]
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5

            simplex = [x0.copy()]
            for j in range(n):
                point = x0.copy()
                delta = 0.5 if x0[j] < 4.5 else -0.5
                point[j] += delta
                point = np.clip(point, -5, 5)
                simplex.append(point)

            fvals = [f0]
            for pt in simplex[1:]:
                if calls >= budget or not time_ok():
                    break
                fx = evaluate(pt)
                if fx is None:
                    break
                fvals.append(fx)

            if len(fvals) != len(simplex):
                continue

            nm_budget = (budget - calls) // (top_k - rank)
            for _ in range(nm_budget):
                if calls >= budget or not time_ok():
                    break

                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                if abs(fvals[-1] - fvals[0]) < 1e-12:
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
                fr = evaluate(xr)
                if fr is None:
                    break

                if fr < fvals[0]:
                    xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                    fe = evaluate(xe)
                    if fe is None:
                        simplex[-1], fvals[-1] = xr, fr
                        break
                    if fe < fr:
                        simplex[-1], fvals[-1] = xe, fe
                    else:
                        simplex[-1], fvals[-1] = xr, fr
                elif fr < fvals[-2]:
                    simplex[-1], fvals[-1] = xr, fr
                else:
                    if fr < fvals[-1]:
                        xc = np.clip(centroid + rho * (xr - centroid), -5, 5)
                    else:
                        xc = np.clip(centroid - rho * (centroid - simplex[-1]), -5, 5)
                    fc = evaluate(xc)
                    if fc is None:
                        break
                    if fc < min(fr, fvals[-1]):
                        simplex[-1], fvals[-1] = xc, fc
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = np.clip(simplex[0] + sigma_shrink * (simplex[i] - simplex[0]), -5, 5)
                        fvals_new = [fvals[0]]
                        for pt in simplex[1:]:
                            if calls >= budget or not time_ok():
                                break
                            fx = evaluate(pt)
                            if fx is None:
                                break
                            fvals_new.append(fx)
                        if len(fvals_new) != len(simplex):
                            break
                        fvals = fvals_new

        # Phase 3: Adaptive (1+1)-ES with restarts on best point
        if best_x is not None and calls < budget and time_ok():
            x = best_x.copy()
            f = best_f
            sigma = 0.5
            min_sigma = 1e-10
            max_sigma = 5.0
            no_improve = 0
            patience = 50

            for _ in range(budget - calls):
                if not time_ok() or calls >= budget:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = evaluate(y)
                if fy is None:
                    break
                
                if fy < f:
                    x, f = y, fy
                    sigma = min(sigma * 1.1, max_sigma)
                    no_improve = 0
                else:
                    sigma = max(sigma * 0.9, min_sigma)
                    no_improve += 1
                    
                    if no_improve >= patience and sigma < min_sigma * 10:
                        # Restart from best with fresh sigma
                        x = best_x.copy()
                        f = best_f
                        sigma = 1.0
                        no_improve = 0
                        patience = 30

        # Phase 4: Fine coordinate descent polish
        if best_x is not None and calls < budget and time_ok():
            x = best_x.copy()
            f = best_f
            step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
            
            for step in step_sizes:
                if calls >= budget or not time_ok():
                    break
                improved = True
                while improved and calls < budget and time_ok():
                    improved = False
                    for j in range(n):
                        if calls >= budget or not time_ok():
                            break
                        for direction in [1, -1]:
                            xp = x.copy()
                            xp[j] += direction * step
                            xp = np.clip(xp, -5, 5)
                            fp = evaluate(xp)
                            if fp is None:
                                break
                            if fp < f:
                                x, f = xp, fp
                                improved = True
                                break

        return best_x, best_f
# EVOLVE-BLOCK-END
