# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using DE-style exploration with hypervolume-guided selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Initial population: diverse random points
        n_init = min(20, self.budget // 5)
        rng = np.random.default_rng(42)

        X = lo + span * rng.random((n_init, self.dim))
        F = np.array([problem(x) for x in X])
        used = n_init

        # Track all evaluated points for hypervolume
        all_X = list(X)
        all_F = list(F)

        # Main loop: differential evolution style
        # We maintain a population and use DE mutation
        pop_size = min(15, self.budget // 8)
        # Select initial population for DE
        # Use non-dominated points if possible
        non_dom = self._get_non_dominated(F)
        if len(non_dom) >= pop_size:
            pop_idx = non_dom[:pop_size]
        else:
            # Pad with random indices
            remaining = np.setdiff1d(np.arange(len(F)), non_dom)
            extra = rng.choice(remaining, size=min(pop_size - len(non_dom), len(remaining)), replace=False) if len(remaining) > 0 else np.array([], dtype=int)
            pop_idx = np.concatenate([non_dom, extra])[:pop_size]

        pop_X = X[pop_idx].copy()
        pop_F = F[pop_idx].copy()

        # DE parameters
        CR = 0.7  # crossover rate
        F_de = 0.8  # differential weight

        while used < self.budget:
            if time.time() > deadline:
                break

            # Generate trial vector using DE/rand/1/bin
            if len(pop_X) < 3:
                # Not enough parents, use random mutation
                base = pop_X[rng.integers(len(pop_X))]
                trial = base + 0.3 * span * rng.standard_normal(self.dim)
            else:
                # Select 3 distinct parents
                idxs = rng.choice(len(pop_X), size=3, replace=False)
                base = pop_X[idxs[0]]
                v = base + F_de * (pop_X[idxs[1]] - pop_X[idxs[2]])

                # Binomial crossover
                j_rand = rng.integers(self.dim)
                mask = rng.random(self.dim) < CR
                mask[j_rand] = True
                trial = np.where(mask, v, base)

            trial = np.clip(trial, lo, hi)
            f_trial = problem(trial)
            used += 1

            # Record for hypervolume
            all_X.append(trial)
            all_F.append(f_trial)

            # Selection: compare with base
            base_idx = idxs[0] if len(pop_X) >= 3 else rng.integers(len(pop_X))
            if self._dominates(f_trial, pop_F[base_idx]):
                pop_X[base_idx] = trial
                pop_F[base_idx] = f_trial
            elif not self._dominates(pop_F[base_idx], f_trial):
                # Non-dominated: replace worst in population based on hypervolume contribution
                # Simple: replace the point with highest sum of objectives (worst)
                worst_in_pop = int(np.argmax(np.sum(pop_F, axis=1)))
                pop_X[worst_in_pop] = trial
                pop_F[worst_in_pop] = f_trial

            # Occasionally inject new random points to maintain diversity
            if used % 30 == 0 and used < self.budget - 10:
                new_x = lo + span * rng.random(self.dim)
                new_f = problem(new_x)
                used += 1
                all_X.append(new_x)
                all_F.append(new_f)
                # Replace worst in population
                worst_in_pop = int(np.argmax(np.sum(pop_F, axis=1)))
                pop_X[worst_in_pop] = new_x
                pop_F[worst_in_pop] = new_f

            # Check time
            if time.time() > deadline:
                break

        # Final local polish: for each non-dominated point, try small perturbations
        if time.time() < deadline - 10:
            all_F_arr = np.array(all_F)
            non_dom_final = self._get_non_dominated(all_F_arr)
            n_polish = min(10, self.budget - used)
            for i in range(n_polish):
                if used >= self.budget or time.time() > deadline:
                    break
                nd_idx = non_dom_final[rng.integers(len(non_dom_final))]
                base_x = all_X[nd_idx]
                base_f = all_F_arr[nd_idx]

                # Try several small perturbations
                best_child = None
                best_child_f = None
                for _ in range(3):
                    child = base_x + 0.05 * span * rng.standard_normal(self.dim)
                    child = np.clip(child, lo, hi)
                    child_f = problem(child)
                    used += 1
                    all_X.append(child)
                    all_F_arr = np.array(all_F)
                    if self._dominates(child_f, base_f):
                        best_child = child
                        best_child_f = child_f
                        break
                    if not self._dominates(base_f, child_f):
                        if best_child is None:
                            best_child = child
                            best_child_f = child_f

                if best_child is not None:
                    # Update non-dominated list
                    pass

        return all_X
# EVOLVE-BLOCK-END
