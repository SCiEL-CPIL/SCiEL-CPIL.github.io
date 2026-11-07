import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref_point = np.array([3.0, 5.0, 7.0])

        # Archive of non-dominated points
        archive_F = []
        archive_X = []

        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def add_to_archive(x, f):
            # Check if f is dominated by any in archive
            for f_arch in archive_F:
                if dominates(f_arch, f):
                    return False
            # Remove points dominated by f
            new_archive_F = []
            new_archive_X = []
            for i, f_arch in enumerate(archive_F):
                if not dominates(f, f_arch):
                    new_archive_F.append(f_arch)
                    new_archive_X.append(archive_X[i])
            new_archive_F.append(f)
            new_archive_X.append(x)
            archive_F[:] = new_archive_F
            archive_X[:] = new_archive_X
            return True

        def hypervolume_contribution(f, ref):
            """Compute the hypervolume contribution of point f given the current archive."""
            # For 3 objectives, we can compute the contribution by finding
            # the box that f uniquely contributes
            # This is a simplified version - for exact computation we'd need more work
            # For now, use a simpler proxy: distance from reference point
            if np.any(f >= ref):
                return 0.0
            # Simple proxy: volume of box from f to ref
            return np.prod(ref - f)

        evals = 0

        # 1. Initialization with LHS-like sampling
        pop_size = min(20, budget // 10)
        X_init = lo + (hi - lo) * np.random.random((pop_size, dim))
        for x in X_init:
            f = problem(x)
            evals += 1
            add_to_archive(x, f)

        # 2. Main loop with adaptive search
        # Use a combination of:
        # - Random exploration
        # - Local refinement around good points
        # - Hypervolume-guided selection

        sigma_base = 0.05 * (hi - lo)
        stagnation = 0
        best_hv = 0.0

        while evals < budget:
            # Check time
            if time.time() > deadline:
                break

            # Determine search strategy
            if stagnation > 30:
                # Global restart: random point
                x_child = lo + (hi - lo) * np.random.random(dim)
            elif len(archive_F) > 5:
                # Hypervolume-guided selection: pick parent with high HV contribution
                # and explore nearby
                # Compute approximate HV contributions
                contribs = []
                for f in archive_F:
                    if np.all(f < ref_point):
                        contribs.append(np.prod(ref_point - f))
                    else:
                        contribs.append(0.0)
                contribs = np.array(contribs)
                total = contribs.sum()
                if total > 0:
                    probs = contribs / total
                else:
                    probs = np.ones(len(archive_F)) / len(archive_F)

                idx = np.random.choice(len(archive_F), p=probs)
                x_parent = archive_X[idx]

                # Adaptive sigma: smaller when close to convergence
                sigma = sigma_base * (0.5 + 0.5 * (1 - min(stagnation / 50.0, 1.0)))
                x_child = x_parent + sigma * np.random.standard_normal(dim)
                x_child = np.clip(x_child, lo, hi)
            else:
                # Early stage: mix of random and local
                if np.random.random() < 0.5:
                    x_child = lo + (hi - lo) * np.random.random(dim)
                else:
                    idx = np.random.randint(len(archive_F))
                    x_parent = archive_X[idx]
                    sigma = sigma_base * 2.0
                    x_child = x_parent + sigma * np.random.standard_normal(dim)
                    x_child = np.clip(x_child, lo, hi)

            f_child = problem(x_child)
            evals += 1

            if add_to_archive(x_child, f_child):
                stagnation = 0
            else:
                stagnation += 1

        # 3. Final local refinement of archive points
        # Use a few evaluations to refine the best points
        if evals < budget and archive_F:
            # Sort archive by hypervolume proxy
            scores = []
            for f in archive_F:
                if np.all(f < ref_point):
                    scores.append(np.prod(ref_point - f))
                else:
                    scores.append(0.0)
            scores = np.array(scores)
            idx_sorted = np.argsort(scores)[::-1]

            # Refine top points
            n_refine = min(5, len(archive_F), (budget - evals) // 2)
            for i in range(n_refine):
                if evals >= budget:
                    break
                if time.time() > deadline:
                    break
                idx = idx_sorted[i]
                x_parent = archive_X[idx].copy()
                f_parent = archive_F[idx]

                # Small local search
                step = 0.01 * (hi - lo)
                improved = False
                for trial in range(3):
                    if evals >= budget:
                        break
                    if time.time() > deadline:
                        break
                    direction = np.random.standard_normal(dim)
                    direction /= np.linalg.norm(direction)
                    x_trial = x_parent + step * direction
                    x_trial = np.clip(x_trial, lo, hi)
                    f_trial = problem(x_trial)
                    evals += 1

                    # Check if it's non-dominated and better in some sense
                    if not any(dominates(f_arch, f_trial) for f_arch in archive_F):
                        if add_to_archive(x_trial, f_trial):
                            improved = True
                            # Update parent to the new point if it's better
                            if np.all(f_trial <= f_parent):
                                x_parent = x_trial.copy()
                                f_parent = f_trial.copy()

        return archive_X
