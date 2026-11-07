"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using grid-based decomposition with local simplex refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        range_ = hi - lo

        # Local RNG
        seed = np.random.get_state()[1][0]
        rng = np.random.RandomState(seed)

        # Archive management
        archive_x = []
        archive_f = []
        evals = 0

        def time_ok():
            return time.time() < deadline

        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def is_dominated(f_new, archive_f_list):
            if len(archive_f_list) == 0:
                return False
            for f in archive_f_list:
                if dominates(f, f_new):
                    return True
            return False

        def add_to_archive(x, f):
            nonlocal evals
            # Check if dominated by existing archive
            if is_dominated(f, archive_f):
                return False

            # Remove points dominated by new point
            new_archive_x = []
            new_archive_f = []
            for xi, fi in zip(archive_x, archive_f):
                if dominates(f, fi):
                    continue
                new_archive_x.append(xi)
                new_archive_f.append(fi)

            new_archive_x.append(x)
            new_archive_f.append(f)

            archive_x[:] = new_archive_x
            archive_f[:] = new_archive_f
            return True

        def evaluate_and_add(x):
            nonlocal evals
            if not time_ok() or evals >= budget:
                return
            x_clipped = np.clip(x, lo, hi)
            f = problem(x_clipped)
            evals += 1
            add_to_archive(x_clipped, f)

        # Halton sequence for initial grid coverage
        def halton_sequence(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_point(n, dim):
            p = np.zeros(dim)
            primes = [2, 3, 5, 7, 11, 13]
            for d in range(dim):
                p[d] = halton_sequence(n, primes[d])
            return p

        # Simplex local search (Nelder-Mead style)
        def local_search(x0, f0, max_iter=20):
            """Perform local simplex search starting from x0."""
            nonlocal evals
            if not time_ok() or evals >= budget:
                return

            # Initialize simplex
            simplex_x = [x0.copy()]
            simplex_f = [f0.copy()]

            # Add perturbed points
            for d in range(dim):
                x_pert = x0.copy()
                # Perturb along dimension d
                step = 0.05 * range_[d]
                x_pert[d] += step
                x_pert = np.clip(x_pert, lo, hi)
                f_pert = problem(x_pert)
                evals += 1
                if evals >= budget:
                    return
                simplex_x.append(x_pert)
                simplex_f.append(f_pert)

            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink

            for _ in range(max_iter):
                if not time_ok() or evals >= budget:
                    break

                # Sort by first objective (proxy for quality)
                order = np.argsort([f[0] for f in simplex_f])
                simplex_x = [simplex_x[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex_x[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex_x[-1])
                x_ref = np.clip(x_ref, lo, hi)
                f_ref = problem(x_ref)
                evals += 1
                if evals >= budget:
                    return

                if f_ref[0] < simplex_f[0][0]:
                    # Expansion
                    x_exp = centroid + gamma * (centroid - simplex_x[-1])
                    x_exp = np.clip(x_exp, lo, hi)
                    f_exp = problem(x_exp)
                    evals += 1
                    if evals >= budget:
                        return
                    if f_exp[0] < f_ref[0]:
                        simplex_x[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex_x[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref[0] < simplex_f[-2][0]:
                    # Accept reflection
                    simplex_x[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + rho * (simplex_x[-1] - centroid)
                    x_con = np.clip(x_con, lo, hi)
                    f_con = problem(x_con)
                    evals += 1
                    if evals >= budget:
                        return
                    if f_con[0] < simplex_f[-1][0]:
                        simplex_x[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex_x)):
                            simplex_x[i] = (simplex_x[0] + sigma * (simplex_x[i] - simplex_x[0]))
                            simplex_x[i] = np.clip(simplex_x[i], lo, hi)
                            f_shr = problem(simplex_x[i])
                            evals += 1
                            if evals >= budget:
                                return
                            simplex_f[i] = f_shr

            # Add best point from local search to archive
            best_idx = np.argmin([f[0] for f in simplex_f])
            add_to_archive(simplex_x[best_idx], simplex_f[best_idx])

        # Phase 1: Grid-based initialization with Halton sequence
        # Use 40 points for good initial coverage
        init_size = 40
        for i in range(1, init_size + 1):
            if not time_ok() or evals >= budget:
                break
            p = halton_point(i, dim)
            x = lo + range_ * p
            f = problem(x)
            evals += 1
            add_to_archive(x, f)

        # Phase 2: Local simplex search on promising archive points
        # Sort archive by a scalarized objective to prioritize good points
        if archive_f:
            # Use weighted sum with equal weights
            weights = np.ones(n_obj) / n_obj
            scores = np.array([np.dot(f, weights) for f in archive_f])
            order = np.argsort(scores)

            # Perform local search on top points
            # Limit number of local searches based on remaining budget
            remaining_budget = budget - evals
            max_local_searches = min(len(order), max(5, remaining_budget // 30))

            for idx in order[:max_local_searches]:
                if not time_ok() or evals >= budget:
                    break
                x0 = archive_x[idx]
                f0 = archive_f[idx]
                local_search(x0, f0, max_iter=15)

        # Phase 3: Fill remaining budget with adaptive random + local refinement
        while evals < budget and time_ok():
            # Decide: exploit or explore
            if archive_x and rng.rand() < 0.8:
                # Exploit: pick random archive point, perturb
                idx = rng.randint(len(archive_x))
                parent = archive_x[idx]
                # Adaptive step size
                progress = evals / budget
                step_scale = 0.1 * (1.0 - 0.5 * progress)
                step = step_scale * range_
                x = parent + step * rng.randn(dim)
                x = np.clip(x, lo, hi)
            else:
                # Explore: random point
                x = lo + range_ * rng.rand(dim)

            f_child = problem(x)
            evals += 1
            was_added = add_to_archive(x, f_child)

            # If added, try a short local refinement
            if was_added and evals < budget and time_ok() and rng.rand() < 0.3:
                local_search(x, f_child, max_iter=5)

        return archive_x
# EVOLVE-BLOCK-END
