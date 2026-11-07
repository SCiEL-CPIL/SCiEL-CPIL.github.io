"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Dense LHS basin identification + adaptive Nelder-Mead with re-seeding."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Phase 1: Two-tier Latin Hypercube for global basin identification
        # Coarse LHS: 40% of budget
        # Fine LHS around top-k: remaining 40%
        # NM local: last 20%
        n_coarse = max(25, int(budget * 0.40))
        n_fine = max(20, int(budget * 0.40))
        n_nm_budget = budget - n_coarse - n_fine
        if n_nm_budget < dim + 1:
            n_coarse = max(15, int(budget * 0.35))
            n_fine = max(15, int(budget * 0.35))
            n_nm_budget = budget - n_coarse - n_fine

        rng = np.random.default_rng(12345)

        def generate_lhs(n, seed_offset=0):
            r = np.random.default_rng(seed_offset)
            pts = np.zeros((n, dim))
            for d in range(dim):
                perm = r.permutation(n)
                pts[:, d] = (perm + 0.5) / n
            return pts * 10.0 - 5.0

        calls = 0
        best_x = None
        best_f = np.inf
        all_evals = []  # list of (f_val, point)

        def eval_pt(pt):
            nonlocal calls, best_f, best_x
            pt = np.clip(pt, -5.0, 5.0)
            fv = float(func(pt))
            calls += 1
            all_evals.append((fv, pt.copy()))
            if fv < best_f:
                best_f = fv
                best_x = pt.copy()
            return fv

        # Coarse LHS
        coarse_pts = generate_lhs(n_coarse, 1000)
        for pt in coarse_pts:
            eval_pt(pt)

        # Fine LHS: sample in annulus around best coarse point
        if best_x is not None and n_fine > 0:
            # Determine local scale based on how far best is from boundary
            # Use a box of half-width 1.5 centered on best, clipped to [-5,5]
            half_w = 1.5
            local_min = np.clip(best_x - half_w, -5.0, 5.0)
            local_max = np.clip(best_x + half_w, -5.0, 5.0)
            # If best is near boundary, extend the box
            for d in range(dim):
                if best_x[d] < -4.0:
                    local_max[d] = min(best_x[d] + half_w, 5.0)
                    local_min[d] = max(-5.0, best_x[d] - half_w * 0.5)
                elif best_x[d] > 4.0:
                    local_min[d] = max(-5.0, best_x[d] - half_w * 0.5)
                    local_max[d] = min(5.0, best_x[d] + half_w)

            # Also add a medium-scale ring: half_width 3.0
            med_half = 3.0
            med_min = np.clip(best_x - med_half, -5.0, 5.0)
            med_max = np.clip(best_x + med_half, -5.0, 5.0)

            # Split fine budget: 60% local, 40% medium
            n_local = max(5, int(n_fine * 0.60))
            n_med = max(5, n_fine - n_local)

            # Local LHS
            r2 = np.random.default_rng(2000)
            for i in range(n_local):
                if calls >= budget - n_nm_budget:
                    break
                pt = np.zeros(dim)
                for d in range(dim):
                    lo, hi = local_min[d], local_max[d]
                    if hi <= lo:
                        hi = lo + 1e-3
                    pt[d] = lo + (hi - lo) * r2.random()
                eval_pt(pt)

            # Medium LHS
            r3 = np.random.default_rng(3000)
            for i in range(n_med):
                if calls >= budget - n_nm_budget:
                    break
                pt = np.zeros(dim)
                for d in range(dim):
                    lo, hi = med_min[d], med_max[d]
                    if hi <= lo:
                        hi = lo + 1e-3
                    pt[d] = lo + (hi - lo) * r3.random()
                eval_pt(pt)

        # Phase 2: Adaptive Nelder-Mead from best point
        nm_start_calls = calls
        nm_budget = budget - nm_start_calls
        if nm_budget < dim + 2:
            return best_x, best_f

        def nm_search(x0, f0, budget_for_nm, dim, func, call_counter):
            """Nelder-Mead with adaptive step and re-seeding on stagnation."""
            # call_counter is a mutable list [count]
            x_best = x0.copy()
            f_best = f0

            # Initial simplex
            step = 0.5
            simplex = [x0.copy()]
            simplex_f = [f0]
            for d in range(dim):
                v = x0.copy()
                v[d] += step
                v = np.clip(v, -5.0, 5.0)
                fv = float(func(v))
                call_counter[0] += 1
                simplex.append(v.copy())
                simplex_f.append(fv)
                if fv < f_best:
                    f_best = fv
                    x_best = v.copy()

            # NM parameters
            alpha_ref = 1.0
            gamma_exp = 2.0
            rho_con = 0.5
            sigma_shrink = 0.5

            stagnation = 0
            max_stagnation = 10

            while call_counter[0] < budget_for_nm:
                # Sort
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                worst_f = simplex_f[-1]
                best_f_simplex = simplex_f[0]

                # Reflection
                centroid = np.mean(simplex[:-1], axis=0)
                x_r = centroid + alpha_ref * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = float(func(x_r))
                call_counter[0] += 1
                if f_r < f_best:
                    f_best = f_r
                    x_best = x_r.copy()

                if call_counter[0] >= budget_for_nm:
                    break

                if f_r < best_f_simplex:
                    # Try expansion
                    x_e = centroid + gamma_exp * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = float(func(x_e))
                    call_counter[0] += 1
                    if f_e < f_best:
                        f_best = f_e
                        x_best = x_e.copy()
                    if call_counter[0] >= budget_for_nm:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    if f_r < worst_f:
                        x_c = centroid + rho_con * (x_r - centroid)
                    else:
                        x_c = centroid - rho_con * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = float(func(x_c))
                    call_counter[0] += 1
                    if f_c < f_best:
                        f_best = f_c
                        x_best = x_c.copy()
                    if call_counter[0] >= budget_for_nm:
                        break
                    if f_c < min(f_r, worst_f):
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_sh = float(func(simplex[i]))
                            call_counter[0] += 1
                            simplex_f[i] = f_sh
                            if f_sh < f_best:
                                f_best = f_sh
                                x_best = simplex[i].copy()
                            if call_counter[0] >= budget_for_nm:
                                break
                        if call_counter[0] >= budget_for_nm:
                            break

                # Check stagnation: if simplex range is very small
                f_range = simplex_f[-1] - simplex_f[0]
                if f_range < 1e-6:
                    stagnation += 1
                else:
                    stagnation = 0

                if stagnation >= max_stagnation:
                    # Re-seed: perturb best point with a small random step
                    r_rs = np.random.default_rng(call_counter[0] + 7)
                    x_res = x_best.copy()
                    for d in range(dim):
                        x_res[d] += r_rs.normal(0, 0.3)
                    x_res = np.clip(x_res, -5.0, 5.0)
                    f_res = float(func(x_res))
                    call_counter[0] += 1
                    if f_res < f_best:
                        f_best = f_res
                        x_best = x_res.copy()
                    # Rebuild simplex around new best
                    simplex = [x_best.copy()]
                    simplex_f = [f_best]
                    for d in range(dim):
                        v = x_best.copy()
                        v[d] += 0.3
                        v = np.clip(v, -5.0, 5.0)
                        fv = float(func(v))
                        call_counter[0] += 1
                        simplex.append(v.copy())
                        simplex_f.append(fv)
                        if fv < f_best:
                            f_best = fv
                            x_best = v.copy()
                    stagnation = 0

                if call_counter[0] >= budget_for_nm:
                    break

            return x_best, f_best

        # Run NM from best point
        counter = [0]
        nm_best_x, nm_best_f = nm_search(best_x, best_f, nm_budget, dim, func, counter)

        # Update global best
        if nm_best_f < best_f:
            best_f = nm_best_f
            best_x = nm_best_x

        return best_x, best_f
# EVOLVE-BLOCK-END
