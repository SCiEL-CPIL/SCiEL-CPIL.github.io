# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Sequential trust-region optimizer with ridge-aware surrogate model."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(12345)

        def clip(v):
            return np.clip(np.asarray(v, dtype=float), -5.0, 5.0)

        def time_ok():
            return time.time() < deadline

        best_x = None
        best_f = np.inf
        calls_used = 0

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Coarse random sampling to find a good starting region
        n_coarse = min(30, max(10, budget // 15))
        coarse_points = []
        for _ in range(n_coarse):
            if calls_used >= budget or not time_ok():
                break
            x = rng.uniform(-4.5, 4.5, size=dim)
            x = clip(x)
            f = func(x)
            calls_used += 1
            record(x, f)
            coarse_points.append((x, f))

        if best_x is None:
            x = np.zeros(dim)
            f = func(x)
            calls_used += 1
            record(x, f)

        # Phase 2: Sequential trust-region with ridge-aware surrogate
        # Build a local quadratic model and use eigenvalue analysis to identify ridge direction
        
        def run_trust_region(center, delta_init, max_iter=200):
            nonlocal calls_used, best_x, best_f

            if not time_ok() or calls_used >= budget - 5:
                return

            x = clip(center.copy())
            f = func(x)
            calls_used += 1
            record(x, f)

            delta = delta_init
            H = np.eye(dim) * 1.0  # Hessian approximation
            g = np.zeros(dim)  # Gradient approximation
            H_inv = np.eye(dim) * 1.0
            delta_min = 1e-6
            delta_max = 8.0
            stall_count = 0
            f_prev = f

            for it in range(max_iter):
                if not time_ok() or calls_used >= budget - 10:
                    break

                # Estimate gradient using central differences along each axis
                # But to save calls, use a sparse approach: only estimate in promising directions
                # First, try to estimate gradient with 2*dim calls if budget allows
                if calls_used + 2 * dim <= budget - 10:
                    h = max(delta * 0.1, 1e-4)
                    g_est = np.zeros(dim)
                    for i in range(dim):
                        xp = x.copy()
                        xp[i] += h
                        xp = clip(xp)
                        fp = func(xp)
                        calls_used += 1
                        record(xp, fp)

                        xm = x.copy()
                        xm[i] -= h
                        xm = clip(xm)
                        fm = func(xm)
                        calls_used += 1
                        record(xm, fm)

                        g_est[i] = (fp - fm) / (2 * h)

                    g = g_est
                else:
                    # Use previous gradient estimate
                    pass

                # Estimate Hessian sparsely - use finite differences on gradient
                # This is expensive, so we do it adaptively
                if calls_used + dim <= budget - 20 and it % 5 == 0:
                    h_h = max(delta * 0.05, 1e-3)
                    H_est = np.zeros((dim, dim))
                    for i in range(dim):
                        if calls_used + 2 * (dim - i) > budget - 20:
                            break
                        # Second derivative along i
                        xpp = x.copy()
                        xpp[i] += h_h
                        xpp = clip(xpp)
                        fpp = func(xpp)
                        calls_used += 1
                        record(xpp, fpp)

                        xmm = x.copy()
                        xmm[i] -= h_h
                        xmm = clip(xmm)
                        fmm = func(xmm)
                        calls_used += 1
                        record(xmm, fmm)

                        H_est[i, i] = (fpp - 2 * f + fmm) / (h_h ** 2)

                        # Cross derivatives
                        for j in range(i + 1, dim):
                            if calls_used + 2 > budget - 20:
                                break
                            xij = x.copy()
                            xij[i] += h_h
                            xij[j] += h_h
                            xij = clip(xij)
                            fij = func(xij)
                            calls_used += 1
                            record(xij, fij)

                            xji = x.copy()
                            xji[i] -= h_h
                            xji[j] -= h_h
                            xji = clip(xji)
                            fji = func(xji)
                            calls_used += 1
                            record(xji, fji)

                            xji2 = x.copy()
                            xji2[i] += h_h
                            xji2[j] -= h_h
                            xji2 = clip(xji2)
                            fji2 = func(xji2)
                            calls_used += 1
                            record(xji2, fji2)

                            xij2 = x.copy()
                            xij2[i] -= h_h
                            xij2[j] += h_h
                            xij2 = clip(xij2)
                            fij2 = func(xij2)
                            calls_used += 1
                            record(fij2, fij2)

                            H_est[i, j] = (fij - fij2 - fji + fji2) / (4 * h_h ** 2)
                            H_est[j, i] = H_est[i, j]

                    # Regularize Hessian
                    H = H_est + 1e-4 * np.eye(dim)
                    try:
                        H_inv = np.linalg.inv(H)
                    except:
                        H_inv = np.eye(dim) * 1.0

                # Compute trust-region step
                try:
                    step = -np.dot(H_inv, g)
                except:
                    step = -g

                step_norm = np.linalg.norm(step)
                if step_norm > delta:
                    step = step * (delta / step_norm)
                elif step_norm < 1e-10:
                    break

                # Ridge-aware: if Hessian has a very small eigenvalue, project step
                # to avoid moving along the ridge (flat direction)
                try:
                    eigvals, eigvecs = np.linalg.eigh(H)
                    # Find the smallest eigenvalue
                    min_idx = np.argmin(eigvals)
                    if eigvals[min_idx] < 1e-6:
                        # Project out the flat direction
                        flat_dir = eigvecs[:, min_idx]
                        proj = np.dot(step, flat_dir) * flat_dir
                        step = step - proj
                        step_norm = np.linalg.norm(step)
                        if step_norm > delta:
                            step = step * (delta / step_norm)
                except:
                    pass

                x_new = clip(x + step)
                f_new = func(x_new)
                calls_used += 1
                record(x_new, f_new)

                # Compute actual improvement
                actual = f - f_new
                # Predicted improvement (approximate)
                predicted = -np.dot(g, step) - 0.5 * np.dot(step, np.dot(H, step))
                rho_ratio = actual / max(abs(predicted), 1e-12)

                if rho_ratio > 0.75:
                    # Good step, increase trust region
                    delta = min(delta * 2.0, delta_max)
                    x = x_new
                    f = f_new
                    stall_count = 0
                elif rho_ratio > 0.25:
                    # Acceptable step, keep trust region
                    x = x_new
                    f = f_new
                    stall_count = 0
                else:
                    # Bad step, decrease trust region
                    delta = max(delta * 0.5, delta_min)
                    stall_count += 1

                if stall_count > 10:
                    break

                if f < f_prev - 1e-10:
                    f_prev = f
                else:
                    stall_count += 1

                if delta < delta_min * 1.5:
                    break

        # Run trust region from best point found
        if best_x is not None and calls_used < budget - 50 and time_ok():
            run_trust_region(best_x, 2.0)

        # Phase 3: Fine polishing with smaller trust region
        if best_x is not None and calls_used < budget - 20 and time_ok():
            run_trust_region(best_x, 0.5)

        # Phase 4: Final local search with coordinate descent
        if best_x is not None and calls_used < budget - 10 and time_ok():
            x = best_x.copy()
            f = best_f
            step_size = 0.3
            for _ in range(50):
                if not time_ok() or calls_used >= budget - 5:
                    break
                improved = False
                for i in range(dim):
                    if not time_ok() or calls_used >= budget - 5:
                        break
                    for direction in [1, -1]:
                        x_try = x.copy()
                        x_try[i] += direction * step_size
                        x_try = clip(x_try)
                        f_try = func(x_try)
                        calls_used += 1
                        record(x_try, f_try)
                        if f_try < f - 1e-10:
                            x = x_try
                            f = f_try
                            improved = True
                            break
                if not improved:
                    step_size *= 0.5
                    if step_size < 1e-4:
                        break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
