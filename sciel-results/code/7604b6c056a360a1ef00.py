"""Improved program using NSGA-II for multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """Multi-objective optimizer using NSGA-II."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.evaluations = 0

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Determine population size
        # We want to use the full budget.
        # Let N be pop size.
        # Total evals = N (init) + N * (budget // N - 1) ?
        # Actually, standard NSGA-II loop:
        # 1. Init P0 (N evals)
        # 2. While evals < budget:
        #    a. Create offspring Q (N evals)
        #    b. R = P U Q
        #    c. Select P' from R (N best)
        #    d. P = P'
        # If budget is not divisible by N, we might run out.
        # Let's pick N such that budget is a multiple of N or close.
        # Budget 200. N=40 -> 5 stages (Init + 4 offspring). 40*5=200. Perfect.
        # N=50 -> 4 stages. 50*4=200. Perfect.
        # N=25 -> 8 stages.
        # Larger N is better for diversity in DTLZ2? Usually 100+ is recommended, but we are constrained.
        # Let's try N=40.
        
        N = 40
        if self.budget % N != 0:
            # Adjust N to be a divisor or just use floor division and stop when budget runs out
            # For robustness, let's just set N and stop if we exceed budget.
            pass
            
        # Ensure N is reasonable
        N = min(N, self.budget)
        
        # Initialize population
        X = lo + (hi - lo) * np.random.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        self.evaluations += N
        
        # NSGA-II parameters
        # SBX parameters
        eta_c = 20.0
        # Poly Mut parameters
        eta_m = 20.0
        prob_mut = 1.0 / self.dim
        
        # Main loop
        while self.evaluations + N <= self.budget:
            # 1. Selection and Crossover/Mutation to create offspring Q
            Q = np.empty_like(X)
            F_Q = np.empty_like(F)
            
            for i in range(N):
                # Select parents using binary tournament
                p1_idx = self._tournament_select(F, N)
                p2_idx = self._tournament_select(F, N)
                
                # Crossover (SBX)
                if np.random.random() < 0.9: # Crossover probability
                    x1, x2 = X[p1_idx], X[p2_idx]
                    c1, c2 = self._sbx_crossover(x1, x2, lo, hi, eta_c)
                else:
                    c1, c2 = X[p1_idx].copy(), X[p2_idx].copy()
                
                # Mutation (Poly Mut)
                c1 = self._poly_mutate(c1, lo, hi, eta_m, prob_mut)
                c2 = self._poly_mutate(c2, lo, hi, eta_m, prob_mut)
                
                Q[2*i] = c1
                Q[2*i+1] = c2
                
            # Evaluate offspring
            for i in range(N):
                F_Q[i] = problem(Q[i])
                self.evaluations += 1
            
            # 2. Combine P and Q
            R = np.vstack([X, Q])
            F_R = np.vstack([F, F_Q])
            
            # 3. Non-dominated sorting and crowding distance
            # Sort R by rank and crowding distance
            R_sorted, F_sorted = self._nsga2_select(R, F_R, N)
            
            # 4. Update P
            X = R_sorted
            F = F_sorted

        return X

    def _non_dominated_sort(self, F):
        """
        Perform non-dominated sorting.
        Returns ranks array of shape (N,)
        """
        N = len(F)
        ranks = np.zeros(N, dtype=int)
        S = [[] for _ in range(N)] # S[i] = indices dominated by i
        n_d = np.zeros(N, dtype=int) # Number of solutions dominating i
        
        # Calculate domination relations
        for i in range(N):
            for j in range(i+1, N):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_d[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_d[i] += 1
        
        current_front = []
        for i in range(N):
            if n_d[i] == 0:
                ranks[i] = 0
                current_front.append(i)
        
        k = 1
        while current_front:
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_d[j] -= 1
                    if n_d[j] == 0:
                        ranks[j] = k
                        next_front.append(j)
            current_front = next_front
            k += 1
            
        return ranks

    def _crowding_distance(self, F, ranks):
        """
        Calculate crowding distance for solutions in the same rank.
        Returns distance array of shape (N,)
        """
        N = len(F)
        distances = np.zeros(N)
        
        # Group by rank
        unique_ranks = np.unique(ranks)
        for r in unique_ranks:
            indices = np.where(ranks == r)[0]
            if len(indices) <= 2:
                # Boundary points or very small front
                distances[indices] = np.inf
                continue
                
            # Sort indices by each objective
            for m in range(self.n_obj):
                sorted_indices = indices[np.argsort(F[indices, m])]
                # Boundary points
                distances[sorted_indices[0]] = np.inf
                distances[sorted_indices[-1]] = np.inf
                
                # Calculate range
                f_min = F[sorted_indices[0], m]
                f_max = F[sorted_indices[-1], m]
                if f_max == f_min:
                    continue
                    
                for i in range(1, len(sorted_indices) - 1):
                    idx = sorted_indices[i]
                    idx_prev = sorted_indices[i-1]
                    idx_next = sorted_indices[i+1]
                    distances[idx] += (F[idx_next, m] - F[idx_prev, m]) / (f_max - f_min)
                    
        return distances

    def _nsga2_select(self, R, F_R, N_keep):
        """
        Select top N_keep individuals from R based on non-dominated sorting and crowding distance.
        """
        ranks = self._non_dominated_sort(F_R)
        distances = self._crowding_distance(F_R, ranks)
        
        # Sort by rank (ascending) and distance (descending)
        # We need to handle ties in rank by sorting by distance
        # Create a sort key: (rank, -distance)
        # But we need to select exactly N_keep.
        
        # Efficient way:
        # 1. Sort indices by rank, then by distance (descending)
        # Since distance is inf for boundaries, they come first.
        
        # We can't simply sort by tuple because we want to group by rank.
        # Let's iterate through ranks in order.
        
        selected_indices = []
        unique_ranks = np.unique(ranks)
        
        for r in unique_ranks:
            if len(selected_indices) >= N_keep:
                break
                
            # Get indices with this rank
            rank_indices = np.where(ranks == r)[0]
            
            if len(selected_indices) + len(rank_indices) <= N_keep:
                # All fit
                selected_indices.extend(rank_indices.tolist())
            else:
                # Need to sort this front by crowding distance
                # Sort rank_indices by distance descending
                sorted_rank_indices = rank_indices[np.argsort(-distances[rank_indices])]
                needed = N_keep - len(selected_indices)
                selected_indices.extend(sorted_rank_indices[:needed].tolist())
                
        # Reorder R and F_R to match selection
        # Note: The order of selected individuals in the next population doesn't strictly matter 
        # for NSGA-II logic, but it's good to keep them consistent.
        R_new = R[selected_indices]
        F_new = F_R[selected_indices]
        
        return R_new, F_new

    def _tournament_select(self, F, N):
        """
        Binary tournament selection based on NSGA-II criteria.
        Returns index of winner.
        """
        # Calculate ranks and distances for the whole population
        # This is expensive if done every time. 
        # In standard NSGA-II, ranks/distances are calculated once per generation for the combined set.
        # Here, we are selecting from the current parent population P.
        # We should precompute ranks and distances for P at the start of the generation.
        # However, my current structure calculates them in _nsga2_select for the combined set.
        # For selection, I need ranks/distances for P.
        
        # Optimization: Store ranks and distances from previous step?
        # In the main loop:
        # 1. We have X, F.
        # 2. We select parents from X. We need ranks/distances for X.
        # 3. We create Q.
        # 4. We combine X, Q into R.
        # 5. We select from R to form new X.
        
        # So, at the start of the loop, I should compute ranks/distances for X.
        # Let's modify the main loop logic to pass these or compute them here.
        # To avoid redundant calculation, I'll compute them in the main loop and pass them to a helper 
        # or just compute them here for simplicity, assuming N is small (40).
        # O(N^2) is 1600 ops, which is trivial.
        
        ranks = self._non_dominated_sort(F)
        distances = self._crowding_distance(F, ranks)
        
        i = np.random.randint(N)
        j = np.random.randint(N)
        
        # Compare i and j
        if ranks[i] < ranks[j]:
            return i
        elif ranks[j] < ranks[i]:
            return j
        else:
            # Same rank, compare crowding distance
            if distances[i] > distances[j]:
                return i
            else:
                return j

    def _sbx_crossover(self, x1, x2, lo, hi, eta):
        """
        Simulated Binary Crossover for real vectors.
        """
        c1 = np.zeros_like(x1)
        c2 = np.zeros_like(x2)
        
        for i in range(self.dim):
            if np.random.random() < 0.5:
                # Swap x1, x2 for this dimension if needed? 
                # Standard SBX: if x1[i] > x2[i], swap.
                if x1[i] > x2[i]:
                    x1[i], x2[i] = x2[i], x1[i]
                
                if abs(x2[i] - x1[i]) > 1e-14:
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                    
                    c1[i] = 0.5 * ((x1[i] + x2[i]) - beta * (x2[i] - x1[i]))
                    c2[i] = 0.5 * ((x1[i] + x2[i]) + beta * (x2[i] - x1[i]))
                else:
                    c1[i] = x1[i]
                    c2[i] = x2[i]
            else:
                c1[i] = x1[i]
                c2[i] = x2[i]
                
        # Clip to bounds
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        
        return c1, c2

    def _poly_mutate(self, x, lo, hi, eta, prob_mut):
        """
        Polynomial Mutation for real vectors.
        """
        for i in range(self.dim):
            if np.random.random() < prob_mut:
                y = x[i]
                yl = lo[i]
                yh = hi[i]
                
                delta1 = (y - yl) / (yh - yl)
                delta2 = (yh - y) / (yh - yl)
                
                r = np.random.random()
                
                if r < 0.5:
                    xy = 2.0 * r + (1.0 - 2.0 * r) * (delta1 ** (eta + 1.0))
                    delta_q = (xy ** (1.0 / (eta + 1.0))) - 1.0
                else:
                    xy = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (delta2 ** (eta + 1.0))
                    delta_q = 1.0 - (xy ** (1.0 / (eta + 1.0)))
                    
                y = y + delta_q * (yh - yl)
                x[i] = np.clip(y, yl, yh)
                
        return x
# EVOLVE-BLOCK-END
