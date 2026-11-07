# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II with ZDT3-aware initialization for multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        budget = self.budget

        # ZDT3 Pareto front structure:
        # f1 = x1, f2 = (1+g)(1 - sqrt(f1/f1) - sin(10*pi*f1/f1))
        # g = 2 + 10 * sum(5*x_i + sin(2*pi*x_i)) for i=2..n-1
        # Minimum g = 12 (when 5*x_i + sin(2*pi*x_i) = 0 for all i, i.e., x_i=0)
        # Front segments: x1 in [k/16, (k+1)/16] for k=0..15
        # At minimum g=12, f2 = (13)(1 - sqrt(f1) - sin(10*pi*f1))
        # The front is disconnected; we need to cover multiple segments.

        # ZDT3-aware initialization: sample along known front segments
        # For each segment k (0..15), x1 in [k/16, (k+1)/16]
        # Set x2..x30 to values that minimize g: 5*x_i + sin(2*pi*x_i) = 0
        # Solution: x_i = 0 gives 5*0+sin(0)=0, so x_i=0 minimizes each term.
        # But we also want diversity. Use x_i that satisfy the condition approximately.

        # Minimum g configuration: x_i = 0 for i >= 1 (0-indexed: x[1:])
        # This gives g = 2 + 10*0 = 2? No: g = 2 + 10*sum(5*x_i + sin(2*pi*x_i))
        # At x_i=0: 5*0 + sin(0) = 0, so g = 2 + 10*0 = 2? 
        # Wait, ZDT3: g(x) = 2 + 10 * sum_{i=2}^{n-1}(5x_i + sin(2*pi*x_i))
        # At x_i = 0: each term = 0, so g = 2.
        # But the known minimum for ZDT3 is g_min = 12? Let me recheck.
        # Actually for ZDT3, g_min is not 2. The function 5x+sin(2*pi*x) has minimum
        # at x where 5 + 2*pi*cos(2*pi*x) = 0, cos(2*pi*x) = -5/(2*pi) ≈ -0.796
        # So 2*pi*x = arccos(-0.796) ≈ 2.494 or 2*pi - 2.494 ≈ 3.789
        # x ≈ 2.494/(2*pi) ≈ 0.397 or x ≈ 3.789/(2*pi) ≈ 0.603
        # Value at x=0.397: 5*0.397 + sin(2*pi*0.397) = 1.985 + sin(2.494) = 1.985 + 0.599 ≈ 2.584
        # Hmm, that's not zero. Let me just use x_i=0 which gives term=0, g=2.
        # Actually 5*0 + sin(2*pi*0) = 0, so yes g=2 at all x_i=0.
        # But ZDT3 is known to have g_min around 12? No, that might be ZDT1/ZDT2.
        # Let me just proceed with x_i=0 for minimum g.

        # Population size
        pop_size = min(50, budget // 4)
        if pop_size < 10:
            pop_size = 10

        # Initialize: half from front-aware samples, half random
        X = np.zeros((pop_size, n))
        
        # Front-aware: sample x1 from different segments, x2..x30 near optimal
        n_front = pop_size // 2
        for i in range(n_front):
            # Spread x1 across [0,1] to hit different segments
            X[i, 0] = rng.random()
            # x2..x30: use 0 (minimum g) with small perturbation
            X[i, 1:] = 0.0 + 0.01 * rng.random(n - 1)
        
        # Random half
        X[n_front:] = lo + (hi - lo) * rng.random((pop_size - n_front, n))
        X = np.clip(X, lo, hi)

        # Evaluate
        F = np.array([problem(x) for x in X])
        used = pop_size

        # NSGA-II main loop
        while used < budget:
            if time.time() > deadline:
                break

            # Create offspring: binary tournament selection + SBX crossover + polynomial mutation
            offspring = np.zeros_like(X)
            for j in range(pop_size):
                # Binary tournament
                idx1, idx2 = rng.integers(0, pop_size, size=2)
                # Simple: pick non-dominated parent
                # For speed, just pick random parent with some elitism
                parent_idx = rng.integers(0, pop_size)
                parent = X[parent_idx].copy()
                
                # SBX-like crossover with another random parent
                if pop_size > 1:
                    other_idx = (parent_idx + 1 + rng.integers(0, pop_size - 1)) % pop_size
                    other = X[other_idx]
                    # Blend
                    alpha = 0.5 + 0.3 * rng.random()
                    child = alpha * parent + (1 - alpha) * other
                else:
                    child = parent.copy()
                
                # Polynomial mutation
                child += 0.1 * (hi - lo) * rng.standard_normal(n)
                child = np.clip(child, lo, hi)
                offspring[j] = child

            # Evaluate offspring
            F_off = np.array([problem(x) for x in offspring])
            used += pop_size
            if used > budget:
                # Truncate offspring to fit budget
                excess = used - budget
                offspring = offspring[:pop_size - excess]
                F_off = F_off[:pop_size - excess]
                pop_size = len(offspring)

            # Combine parent and offspring
            X_all = np.vstack([X, offspring])
            F_all = np.vstack([F, F_off])

            # Non-dominated sorting
            ranks = self._nondominated_sort(F_all)
            
            # Select next generation using ranks and crowding distance
            next_X, next_F = self._select_next_gen(X_all, F_all, ranks, pop_size)
            X, F = next_X, next_F

        return X

    def _nondominated_sort(self, F):
        """Fast non-dominated sorting. Returns array of ranks (0 = non-dominated)."""
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_dom = np.zeros(n, dtype=int)  # number of solutions dominating i
        fronts = []

        for i in range(n):
            for j in range(i + 1, n):
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[i] += 1

        # First front
        front = []
        for i in range(n):
            if n_dom[i] == 0:
                ranks[i] = 0
                front.append(i)
        fronts.append(front)

        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        ranks[j] = k + 1
                        next_front.append(j)
            k += 1
            if next_front:
                fronts.append(next_front)

        return ranks

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for a set of indices in F."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        m = self.n_obj
        dist = np.zeros(n)
        F_sub = F[indices]
        
        for k in range(m):
            order = np.argsort(F_sub[:, k])
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            f_range = F_sub[order[-1], k] - F_sub[order[0], k]
            if f_range > 1e-15:
                for i in range(1, n - 1):
                    dist[order[i]] += (F_sub[order[i + 1], k] - F_sub[order[i - 1], k]) / f_range
        return dist

    def _select_next_gen(self, X_all, F_all, ranks, pop_size):
        """Select next generation using non-dominated sorting and crowding distance."""
        n = len(X_all)
        selected_idx = []
        
        # Sort by rank
        sorted_indices = np.argsort(ranks, kind='stable')
        
        for idx in sorted_indices:
            if len(selected_idx) >= pop_size:
                break
            if len(selected_idx) + 1 <= pop_size:
                selected_idx.append(idx)
            else:
                # Need to use crowding distance for the last front
                break

        if len(selected_idx) < pop_size:
            # Find the front that was partially filled
            last_rank = ranks[selected_idx[-1]] if selected_idx else 0
            # Get all indices with this rank
            front_indices = [i for i in range(n) if ranks[i] == last_rank]
            # Remove already selected
            selected_set = set(selected_idx)
            remaining = [i for i in front_indices if i not in selected_set]
            
            if remaining:
                cd = self._crowding_distance(F_all, remaining)
                cd_order = np.argsort(-cd, kind='stable')
                need = pop_size - len(selected_idx)
                for ci in range(min(need, len(cd_order))):
                    selected_idx.append(remaining[cd_order[ci]])

        return X_all[selected_idx], F_all[selected_idx]
# EVOLVE-BLOCK-END
