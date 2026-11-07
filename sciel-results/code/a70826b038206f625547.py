"""Multi-fidelity population-based ES with diverse archive and adaptive step size."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Population-based (mu+lambda)-ES with a diverse archive and a
    coarse-to-fine adaptive step size driven by the recent improvement rate."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        lo, hi = -5.0, 5.0

        # --- population / archive sizes -------------------------------------
        lam = 16          # offspring per generation
        mu = 6            # elites carried forward
        archive_size = 12 # diverse archive size (includes elites)

        # --- initial diverse population -------------------------------------
        # Seed with a spread of points, including a few near the origin region
        # where the rotated Rosenbrock optimum is likely to sit.
        pop = np.empty((lam, dim))
        pop[0] = np.zeros(dim)
        for i in range(1, lam):
            pop[i] = rng.uniform(-4.0, 4.0, dim)
        # ensure a couple of wide points to guarantee global coverage
        pop[1] = rng.uniform(lo, hi, dim)
        pop[2] = rng.uniform(lo, hi, dim)

        fit = np.empty(lam)
        for i in range(lam):
            fit[i] = func(np.clip(pop[i], lo, hi))

        order = np.argsort(fit)
        # archive: best mu elites + diverse (far from elites) members
        archive = [pop[order[k]].copy() for k in range(mu)]
        archive_fit = [fit[order[k]] for k in range(mu)]

        best_x = pop[order[0]].copy()
        best_f = float(fit[order[0]])

        # --- adaptive step size ---------------------------------------------
        sigma = 2.5
        sigma_min = 1e-4
        sigma_max = 6.0
        # improvement-rate feedback
        recent_improve = 0.0
        improve_ema = 0.0

        calls = lam
        budget = self.budget

        while calls < budget:
            # --- compute global step magnitude from improvement rate --------
            # scale sigma: grow when improving fast (explore wide basin),
            # shrink when stalled (fine exploitation). Coarse-to-fine.
            improve_ema = 0.6 * improve_ema + 0.4 * max(recent_improve, 0.0)
            # map improvement (log-scale) to a multiplicative factor in [0.5, 1.6]
            # improvement of 0 -> factor ~1 (neutral), large -> explore, tiny -> exploit
            log_imp = np.log10(improve_ema + 1e-8)
            # center around log10(1e-3) = -3 as "neutral fine"
            factor = 1.0 + 0.35 * np.clip((log_imp + 3.0) / 3.0, -1.0, 1.0)
            sigma = float(np.clip(sigma * factor, sigma_min, sigma_max))

            # --- build next generation: elites + offspring -------------------
            # elites (mu)
            gen_x = np.empty((lam, dim))
            gen_f = np.empty(lam)
            for k in range(mu):
                gen_x[k] = archive[k]
                gen_f[k] = archive_fit[k]

            # offspring: sample parents from archive with diversity weighting
            for k in range(mu, lam):
                # pick a parent; bias toward good but allow diversity
                idx = int(rng.integers(0, len(archive)))
                p = archive[idx]
                # per-dimension noise with occasional large jump for exploration
                step = sigma * rng.standard_normal(dim)
                # occasional global kick to escape local minima
                if rng.random() < 0.05:
                    step = sigma * 3.0 * rng.standard_normal(dim)
                child = p + step
                child = np.clip(child, lo, hi)
                gen_x[k] = child
                gen_f[k] = func(child)
                calls += 1

            # --- update archive ---------------------------------------------
            # combine current archive + offspring, keep best mu + diverse fill
            cand_x = np.vstack([np.array(archive), gen_x[mu:]])
            cand_f = np.array(archive_fit + list(gen_f[mu:]))
            cand_order = np.argsort(cand_f)

            new_archive = []
            new_archive_fit = []
            for idx in cand_order:
                if len(new_archive) >= mu:
                    break
                new_archive.append(cand_x[idx].copy())
                new_archive_fit.append(float(cand_f[idx]))

            # fill up to archive_size with diverse members (max-min distance)
            while len(new_archive) < archive_size:
                best_div = -1
                best_idx = -1
                for idx in cand_order:
                    if any(np.array_equal(cand_x[idx], new_archive[j]) for j in range(len(new_archive))):
                        continue
                    d = min(np.linalg.norm(cand_x[idx] - new_archive[j])
                            for j in range(len(new_archive)))
                    if d > best_div:
                        best_div = d
                        best_idx = idx
                if best_idx < 0:
                    break
                new_archive.append(cand_x[best_idx].copy())
                new_archive_fit.append(float(cand_f[best_idx]))

            archive = new_archive
            archive_fit = new_archive_fit

            # --- track best and improvement ----------------------------------
            gen_best_idx = int(np.argmin(gen_f))
            gen_best_f = float(gen_f[gen_best_idx])
            recent_improve = max(0.0, best_f - gen_best_f)
            if gen_best_f < best_f:
                best_f = gen_best_f
                best_x = gen_x[gen_best_idx].copy()

            # early stopping if convergence is essentially complete
            if best_f < 1e-9:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
