# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Tchebycheff decomposition + per-subproblem DE multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Generate reference points on the (n_obj-1)-simplex
        # For 3 objectives, use a grid with 4 points per edge -> 10 points
        # But to get more coverage, use 6 points per edge -> 21 points
        # Let's use a balanced number
        n_ref = 4  # points per edge in simplex
        ref_points = self._generate_reference_points(n_obj, n_ref)
        n_sub = len(ref_points)

        # Per-subproblem population size
        # We have budget evaluations. Initial: n_sub points.
        # Then iterations of DE: each iteration evaluates n_sub offspring.
        # Total = n_sub (init) + n_iter * n_sub
        # We want n_sub * (1 + n_iter) <= budget
        # Let's use n_sub = 10, so 10 * (1 + n_iter) <= 200 -> n_iter <= 19
        # Or n_sub = 15 -> 15*(1+n_iter) <= 200 -> n_iter <= 12
        # Let's pick n_sub based on budget
        if budget >= 200:
            n_sub = 10
        elif budget >= 100:
            n_sub = 8
        else:
            n_sub = max(3, min(5, budget // 10))

        # Regenerate reference points to match n_sub
        if n_sub != len(ref_points):
            # Adjust n_ref to get approximately n_sub points
            # For 3 obj, n_ref=k gives k*(k+1)/2 points
            # k=4 -> 10, k=5 -> 15, k=3 -> 6
            for k in range(2, 20):
                pts = self._generate_reference_points(n_obj, k)
                if len(pts) >= n_sub:
                    ref_points = pts[:n_sub]
                    n_sub = len(ref_points)
                    break
            else:
                ref_points = ref_points[:n_sub]

        n_sub = len(ref_points)
        ref_points = ref_points[:n_sub]

        # Initialize: one random point per subproblem
        X = np.empty((n_sub, dim))
        F = np.empty((n_sub, n_obj))
        for i in range(n_sub):
            X[i] = lo + span * np.random.random(dim)
            F[i] = np.array(problem(X[i]))

        used = n_sub

        # Track best per subproblem
        best_X = X.copy()
        best_F = F.copy()

        # DE parameters
        F_mut = 0.8
        CR = 0.9

        # Iterative DE
        while used < budget:
            if time.time() > deadline:
                break

            # Check if we can afford another full generation
            if used + n_sub > budget:
                # Do partial generation
                remaining = budget - used
                # Pick 'remaining' subproblems to evolve
                indices = np.random.choice(n_sub, size=remaining, replace=False)
            else:
                indices = np.arange(n_sub)

            for idx in indices:
                if time.time() > deadline:
                    break
                if used >= budget:
                    break

                # DE mutation: pick 3 distinct other individuals
                others = [j for j in range(n_sub) if j != idx]
                if len(others) < 3:
                    # Not enough others, just random mutation
                    v = best_X[idx] + 0.5 * span * (np.random.random(dim) - 0.5)
                else:
                    a, b, c = np.random.choice(others, size=3, replace=False)
                    v = best_X[a] + F_mut * (best_X[b] - best_X[c])

                # Crossover
                cross_mask = np.random.random(dim) < CR
                if not np.any(cross_mask):
                    cross_mask[np.random.randint(dim)] = True
                trial = np.where(cross_mask, v, best_X[idx])
                trial = np.clip(trial, lo, hi)

                # Evaluate
                trial_F = np.array(problem(trial))
                used += 1

                # Selection: compare using Tchebycheff scalarization
                # g(x, w) = max_j |w_j * (f_j(x) - z_j^*)|
                # Use current best F as ideal point approximation
                z_star = np.min(best_F, axis=0)
                # But z_star might not be the true ideal; use min over all seen
                # For simplicity, use min over current best_F
                w = ref_points[idx]
                # Avoid zero weights
                w_safe = np.maximum(w, 1e-10)
                w_norm = w_safe / w_safe.sum()

                g_trial = np.max(w_norm * np.abs(trial_F - z_star))
                g_best = np.max(w_norm * np.abs(best_F[idx] - z_star))

                if g_trial < g_best:
                    best_X[idx] = trial
                    best_F[idx] = trial_F

        # Final polish: local search around best points
        # Use remaining budget for fine-tuning
        # Actually, we've used all budget. Return the best points found.

        return best_X

    def _generate_reference_points(self, n_obj, n_ref):
        """Generate reference points on the (n_obj-1)-simplex."""
        if n_obj == 2:
            points = []
            for i in range(n_ref + 1):
                points.append([i / n_ref, 1 - i / n_ref])
            return np.array(points)
        elif n_obj == 3:
            points = []
            for i in range(n_ref + 1):
                for j in range(n_ref + 1 - i):
                    k = n_ref - i - j
                    points.append([i / n_ref, j / n_ref, k / n_ref])
            return np.array(points)
        else:
            # General case: use Dirichlet sampling
            rng = np.random.default_rng(42)
            points = rng.dirichlet(np.ones(n_obj), size=n_ref * n_ref)
            return points
# EVOLVE-BLOCK-END
