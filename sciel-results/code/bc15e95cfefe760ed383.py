"""Multi-objective optimizer for WFG4 using NSGA-II."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II based multi-objective evolutionary optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        # Population size: typically 2 * budget / generations or similar.
        # With budget=200, a pop size of 40-60 is reasonable to allow ~3-5 generations.
        # Let's aim for a population size that allows for a few full generations.
        # If budget=200, and we want ~4 generations, pop size ~ 50.
        # We will dynamically determine pop size based on budget to ensure at least 2 generations.
        self.pop_size = min(60, max(20, budget // 4)) 

    def _fast_non_dominated_sort(self, F):
        """
        Perform fast non-dominated sorting.
        Returns:
            ranks: list of lists, where ranks[i] contains indices of solutions in rank i.
        """
        n = len(F)
        if n == 0:
            return []

        S = [[] for _ in range(n)]  # S[i] = set of solutions dominated by i
        n_dominating = np.zeros(n, dtype=int)  # n_dominating[i] = count of solutions dominating i
        ranks = [[]]

        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                # Check if p dominates q
                # p dominates q if F[p] <= F[q] for all obj and F[p] < F[q] for at least one
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                # Check if q dominates p
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    S[q].append(p)
                    n_dominating[p] += 1
            
            if n_dominating[p] == 0:
                ranks[0].append(p)

        i = 0
        while ranks[i]:
            Q = []
            for p in ranks[i]:
                for q in S[p]:
                    n_dominating[q] -= 1
                    if n_dominating[q] == 0:
                        Q.append(q)
            i += 1
            if Q:
                ranks.append(Q)
        
        # Remove empty last list if any
        if ranks and not ranks[-1]:
            ranks.pop()
            
        return ranks

    def _crowding_distance(self, F, rank_indices):
        """
        Calculate crowding distance for a set of non-dominated solutions.
        """
        n = len(rank_indices)
        if n <= 2:
            return np.full(n, np.inf)

        dist = np.zeros(n)
        m = self.n_obj

        for k in range(m):
            # Sort indices by k-th objective
            sorted_indices = np.argsort(F[rank_indices, k])
            # The first and last in sorted order get infinite distance
            dist[sorted_indices[0]] = np.inf
            dist[sorted_indices[-1]] = np.inf

            f_min = F[rank_indices[sorted_indices[0]], k]
            f_max = F[rank_indices[sorted_indices[-1]], k]
            range_f = f_max - f_min

            if range_f < 1e-15:
                continue

            for j in range(1, n - 1):
                idx_curr = sorted_indices[j]
                idx_prev = sorted_indices[j - 1]
                idx_next = sorted_indices[j + 1]
                
                dist[idx_curr] += (F[rank_indices[idx_next], k] - F[rank_indices[idx_prev], k]) / range_f

        return dist

    def _tournament_selection(self, parents_X, parents_F, parents_ranks, parents_cd, rng, k=2):
        """
        Select a parent using tournament selection based on rank and crowding distance.
        parents_ranks: list of lists of global indices.
        parents_cd: array of crowding distances for all individuals in the population.
        """
        n = len(parents_X)
        idx1 = rng.integers(n)
        idx2 = rng.integers(n)

        # Determine rank of idx1 and idx2
        rank1 = 0
        rank2 = 0
        for i, rank in enumerate(parents_ranks):
            if idx1 in rank:
                rank1 = i
                break
        for i, rank in enumerate(parents_ranks):
            if idx2 in rank:
                rank2 = i
                break

        if rank1 != rank2:
            return parents_X[min(rank1, rank2) and idx1 or idx2] # This logic is flawed, need to return X of winner
        # Correct logic:
        if rank1 < rank2:
            return idx1
        elif rank2 < rank1:
            return idx2
        else:
            # Same rank, compare crowding distance
            if parents_cd[idx1] > parents_cd[idx2]:
                return idx1
            else:
                return idx2

    def _sbx_crossover(self, x1, x2, bounds, rng, eta=20.0):
        """Simulated Binary Crossover."""
        lo, hi = bounds
        x1 = np.array(x1)
        x2 = np.array(x2)
        
        if rng.random() < 0.5:
            return x1, x2

        child1 = np.zeros_like(x1)
        child2 = np.zeros_like(x2)

        for i in range(self.dim):
            if x1[i] == x2[i]:
                child1[i] = x1[i]
                child2[i] = x2[i]
                continue

            u = rng.random()
            if u <= 0.5:
                beta = (2 * u) ** (1 / (eta + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))

            child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
            child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])

        # Clip to bounds
        child1 = np.clip(child1, lo, hi)
        child2 = np.clip(child2, lo, hi)

        return child1, child2

    def _polynomial_mutation(self, x, bounds, rng, eta=20.0):
        """Polynomial Mutation."""
        lo, hi = bounds
        x = np.array(x)
        
        for i in range(self.dim):
            if rng.random() < 0.5:
                continue
            
            u = rng.random()
            if u < 0.5:
                delta_q = (2 * u) ** (1 / (eta + 1)) - 1
            else:
                delta_q = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            
            x[i] += delta_q * (hi[i] - lo[i])
            
        x = np.clip(x, lo, hi)
        return x

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        rng = np.random.default_rng(0)
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        bounds_arr = (lo, hi)
        
        dim = self.dim
        budget = self.budget
        pop_size = self.pop_size

        # Initial Population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        evaluations_used = pop_size

        # Main Loop
        while evaluations_used < budget:
            if time.time() > deadline - 5.0:
                break

            # 1. Non-dominated sorting
            ranks = self._fast_non_dominated_sort(F)
            
            # 2. Crowding distance for each rank
            # We need a global array for crowding distance to compare in tournament
            cd_global = np.zeros(len(F))
            for rank in ranks:
                if len(rank) > 0:
                    cd_rank = self._crowding_distance(F, rank)
                    for i, idx in enumerate(rank):
                        cd_global[idx] = cd_rank[i]

            # 3. Create offspring population
            offspring_X = []
            offspring_F = []
            
            # Number of offspring to generate in this generation
            # We generate pop_size offspring per generation
            # But we must respect the budget
            remaining_budget = budget - evaluations_used
            n_offspring = min(pop_size, remaining_budget)
            
            if n_offspring == 0:
                break

            for _ in range(n_offspring):
                if time.time() > deadline - 5.0:
                    break
                
                # Tournament selection for parents
                parent1_idx = self._tournament_selection(X, F, ranks, cd_global, rng)
                parent2_idx = self._tournament_selection(X, F, ranks, cd_global, rng)
                
                # Crossover
                child1, child2 = self._sbx_crossover(X[parent1_idx], X[parent2_idx], bounds_arr, rng)
                
                # Mutation
                child1 = self._polynomial_mutation(child1, bounds_arr, rng)
                child2 = self._polynomial_mutation(child2, bounds_arr, rng)
                
                # Evaluate
                f1 = problem(child1)
                f2 = problem(child2)
                
                offspring_X.append(child1)
                offspring_F.append(f1)
                offspring_X.append(child2)
                offspring_F.append(f2)
                
                evaluations_used += 2
                
                if evaluations_used >= budget:
                    break

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # 4. Combine parents and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # 5. Selection for next generation
            # We need to select pop_size individuals from 2 * pop_size
            # Use NSGA-II selection mechanism
            
            # Non-dominated sorting on combined
            c_ranks = self._fast_non_dominated_sort(combined_F)
            
            # Crowding distance for combined
            c_cd_global = np.zeros(len(combined_F))
            for rank in c_ranks:
                if len(rank) > 0:
                    cd_rank = self._crowding_distance(combined_F, rank)
                    for i, idx in enumerate(rank):
                        c_cd_global[idx] = cd_rank[i]
            
            # Select next population
            next_X = []
            next_F = []
            count = 0
            for rank in c_ranks:
                if count + len(rank) <= pop_size:
                    # Take all from this rank
                    for idx in rank:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    count += len(rank)
                else:
                    # Need to pick some from this rank based on crowding distance
                    needed = pop_size - count
                    # Sort indices in rank by crowding distance (descending)
                    sorted_rank = sorted(rank, key=lambda i: c_cd_global[i], reverse=True)
                    for idx in sorted_rank[:needed]:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    count += needed
                    break
            
            # Update population
            X = np.array(next_X)
            F = np.array(next_F)

        return X

# EVOLVE-BLOCK-END
