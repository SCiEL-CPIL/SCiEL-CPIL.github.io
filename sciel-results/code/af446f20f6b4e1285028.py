import numpy as np
import time

class Optimizer:
    """NSGA-II based multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 300s limit, stop at 280s
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        # Determine population size
        # For small budgets, keep N moderate to allow multiple generations
        # N=40 allows 4 generations after init for B=200
        N = 40
        if self.budget < 2 * N:
            N = max(2, self.budget // 2)
            
        # 1. Initialization using Latin Hypercube Sampling
        X = self._lhs_init(N, self.dim, lo, hi)
        F = np.array([problem(x) for x in X])
        used = N
        
        # 2. Main NSGA-II Loop
        while used < self.budget and time.time() < deadline:
            # Generate offspring candidates
            offspring_X = []
            for _ in range(N):
                # Tournament selection
                p1 = self._tournament_select(X, F, N)
                p2 = self._tournament_select(X, F, N)
                
                # SBX Crossover
                c1, c2 = self._sbx_crossover(p1, p2, lo, hi)
                
                # Polynomial Mutation
                c1 = self._poly_mutation(c1, lo, hi)
                
                offspring_X.append(c1)
            
            # Evaluate offspring
            offspring_F = []
            for x in offspring_X:
                if used >= self.budget or time.time() > deadline:
                    break
                f = problem(x)
                offspring_F.append(f)
                used += 1
            
            if not offspring_F:
                break
                
            # Combine parents and offspring
            X_pop = np.vstack([X, np.array(offspring_X[:len(offspring_F)])])
            F_pop = np.vstack([F, np.array(offspring_F)])
            
            # Non-dominated sorting and selection
            X, F = self._nsga2_select(X_pop, F_pop, N)
            
        return X

    def _lhs_init(self, N, dim, lo, hi):
        """Latin Hypercube Sampling."""
        # Create N intervals for each dimension
        indices = np.arange(N)
        # Shuffle indices for each dimension
        shuffled = np.column_stack([np.random.permutation(indices) for _ in range(dim)])
        # Add random offset within the interval
        random_offset = np.random.rand(N, dim)
        # Normalize to [0, 1]
        lhs_norm = (shuffled + random_offset) / N
        # Scale to bounds
        return lo + lhs_norm * (hi - lo)

    def _tournament_select(self, X, F, N, k=2):
        """Tournament selection based on NSGA-II ranking and crowding distance."""
        # Pre-compute ranks and crowding distances if not passed, 
        # but for efficiency in loop, we should pass them or compute locally.
        # For simplicity and correctness in this small N, we compute locally.
        # Optimization: Pass ranks/crowding from previous step? 
        # NSGA-II selection uses the *current* population's ranks.
        
        ranks, crowding = self._non_dominated_sort(F)
        
        # Select k random individuals
        selected_indices = np.random.choice(N, k, replace=False)
        
        best_idx = selected_indices[0]
        best_rank = ranks[best_idx]
        best_crowd = crowding[best_idx]
        
        for idx in selected_indices[1:]:
            if ranks[idx] < best_rank:
                best_idx = idx
                best_rank = ranks[idx]
                best_crowd = crowding[idx]
            elif ranks[idx] == best_rank and crowding[idx] > best_crowd:
                best_idx = idx
                best_crowd = crowding[idx]
                
        return X[best_idx]

    def _sbx_crossover(self, p1, p2, lo, hi, eta_c=15.0):
        """Simulated Binary Crossover."""
        c1 = np.zeros_like(p1)
        c2 = np.zeros_like(p2)
        
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    x1, x2 = p1[i], p2[i]
                    if x1 > x2:
                        x1, x2 = x2, x1
                        
                    beta1 = 1.0
                    beta2 = 1.0
                    alpha1 = (x1 - lo[i]) / (x1 - x2 + 1e-14)
                    alpha2 = (hi[i] - x2) / (x2 - x1 + 1e-14)
                    
                    if np.random.rand() <= 1.0 / (eta_c + 1.0) * (1.0 + alpha1):
                        beta1 = (2.0 * np.random.rand() + (1.0 - np.random.rand()) * (1.0 - alpha1) ** (eta_c + 1.0)) ** (1.0 / (eta_c + 1.0))
                    else:
                        beta1 = (0.5 / (np.random.rand() + (1.0 - alpha1) * (0.5 * (eta_c + 1.0) - 1.0))) ** (1.0 / (eta_c + 1.0))
                        
                    # Correction for beta calculation based on standard SBX
                    # Standard formula:
                    # if rand <= 1/(eta+1) * (1+alpha): beta = (2*rand + (1-alpha)^(eta+1))^(1/(eta+1))
                    # else: beta = (1/(2*rand + (1-(1-alpha)^(eta+1))))^(1/(eta+1)) -- Wait, standard is:
                    # else: beta = (0.5 / (rand + (1 - (1-alpha)^(eta+1))))^(1/(eta+1)) is not quite right.
                    # Let's use the standard SciPy/DEAP implementation logic.
                    
                    # Re-implementing standard SBX logic clearly
                    if np.random.rand() < 0.5:
                        beta1 = (2.0 * np.random.rand()) ** (1.0 / (eta_c + 1.0))
                    else:
                        beta1 = (1.0 / (2.0 * np.random.rand())) ** (1.0 / (eta_c + 1.0))
                        
                    # Actually, the standard SBX uses alpha to adjust probability.
                    # Let's stick to a simpler, robust SBX variant or just use uniform crossover if SBX is complex.
                    # Given the small dimension, uniform crossover is less effective but robust.
                    # Let's try a simplified SBX:
                    
                    # Reset
                    beta1 = 1.0
                    beta2 = 1.0
                    alpha1 = (x1 - lo[i]) / (x1 - x2 + 1e-14)
                    alpha2 = (hi[i] - x2) / (x2 - x1 + 1e-14)
                    
                    if np.random.rand() <= 1.0 / (eta_c + 1.0) * (1.0 + alpha1):
                        beta1 = (2.0 * np.random.rand() + (1.0 - alpha1) ** (eta_c + 1.0)) ** (1.0 / (eta_c + 1.0))
                    else:
                        beta1 = (0.5 / (np.random.rand() + (1.0 - (1.0 - alpha1) ** (eta_c + 1.0)))) ** (1.0 / (eta_c + 1.0))
                        
                    if np.random.rand() <= 1.0 / (eta_c + 1.0) * (1.0 + alpha2):
                        beta2 = (2.0 * np.random.rand() + (1.0 - alpha2) ** (eta_c + 1.0)) ** (1.0 / (eta_c + 1.0))
                    else:
                        beta2 = (0.5 / (np.random.rand() + (1.0 - (1.0 - alpha2) ** (eta_c + 1.0)))) ** (1.0 / (eta_c + 1.0))
                        
                    c1[i] = 0.5 * ((1.0 + beta1) * x1 + (1.0 - beta1) * x2)
                    c2[i] = 0.5 * ((1.0 + beta2) * x2 + (1.0 - beta2) * x1)
                else:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
            else:
                c1[i] = p2[i]
                c2[i] = p1[i]
                
        # Clip to bounds
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        
        return c1, c2

    def _poly_mutation(self, x, lo, hi, eta_m=20.0):
        """Polynomial Mutation."""
        x_mut = np.zeros_like(x)
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                delta1 = (x[i] - lo[i]) / (hi[i] - lo[i] + 1e-14)
                delta2 = (hi[i] - x[i]) / (hi[i] - lo[i] + 1e-14)
                
                if np.random.rand() <= 2.0 / (eta_m + 1.0) * (1.0 + delta1):
                    xk = 1.0 - (1.0 - np.random.rand()) ** (eta_m + 1.0)
                else:
                    xk = 1.0 - (1.0 / (2.0 * np.random.rand() + (1.0 - (1.0 - delta1) ** (eta_m + 1.0)))) ** (1.0 / (eta_m + 1.0))
                    
                # Correction: Standard Poly Mutation formula
                # if rand <= 2/(eta+1)*(1+delta1):
                #   xk = 1 - (1-rand)^(eta+1)
                # else:
                #   xk = 1 - (1/(2*rand + (1-(1-delta1)^(eta+1))))^(1/(eta+1))
                
                # Re-evaluating the else branch logic from standard literature:
                # delta = (x - lo) / (hi - lo)
                # if rand < 0.5:
                #   if rand <= 1/(eta+1)*(1+delta): beta = (2*rand + (1-delta)^(eta+1))^(1/(eta+1))
                #   else: beta = (1/(2*rand + (1-(1-delta)^(eta+1))))^(1/(eta+1)) -- No, this is for SBX.
                
                # For Poly Mutation:
                # delta1 = (x - lo) / (hi - lo)
                # delta2 = (hi - x) / (hi - lo)
                # if rand <= 2/(eta+1) * (1 + delta1):
                #   xk = 1 - (1 - rand)^(eta+1)
                # else:
                #   xk = 1 - (1 / (2*rand + (1 - (1-delta1)^(eta+1))))^(1/(eta+1)) -- This looks wrong.
                
                # Correct Poly Mutation (Deb 2001):
                # delta = (x - lo) / (hi - lo)
                # if rand < 0.5:
                #   if rand <= 1/(eta+1) * (1 + delta):
                #      beta = (2*rand + (1-delta)^(eta+1))^(1/(eta+1))
                #   else:
                #      beta = (1 / (2*rand + (1 - (1-delta)^(eta+1))))^(1/(eta+1))
                #   x_new = x + (hi-lo) * (beta - 1)
                # else:
                #   # symmetric for upper bound
                
                # Let's use a simpler, effective mutation: Gaussian with adaptive sigma.
                # Given the small budget, robustness is key.
                pass
            
        # Fallback to simple Gaussian mutation if Poly is too complex to get right quickly
        # Actually, let's implement a correct simple polynomial mutation or just use Gaussian.
        # Gaussian is safer to implement correctly.
        sigma = 0.1 * (hi - lo)
        x_mut = x + sigma * np.random.standard_normal(self.dim)
        return np.clip(x_mut, lo, hi)

    def _non_dominated_sort(self, F):
        """
        Returns ranks (lower is better) and crowding distances (higher is better).
        F: (N, n_obj) array.
        """
        N = F.shape[0]
        n_obj = F.shape[1]
        
        # 1. Non-dominated sorting
        ranks = np.zeros(N, dtype=int)
        crowding = np.zeros(N)
        
        S = [set() for _ in range(N)] # S[i] = set of indices dominated by i
        n = np.zeros(N, dtype=int) # n[i] = number of individuals dominating i
        rank = []
        current_rank = 0
        
        for i in range(N):
            for j in range(N):
                if i == j: continue
                # Does i dominate j?
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].add(j)
                    n[j] += 1
                # Does j dominate i?
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n[i] += 1
                    
        # Front 0
        front = [i for i in range(N) if n[i] == 0]
        for i in front:
            ranks[i] = 0
            
        k = 0
        while front:
            next_front = []
            for i in front:
                # Calculate crowding distance for i
                # Crowding distance is sum of normalized differences in objective space
                # For boundary points, distance is inf (or large)
                
                # Sort front by each objective
                # This is O(N^2 * n_obj) which is fine for N=40
                
                # We will calculate crowding for the *current* front later, 
                # but we need it for selection. 
                # Standard NSGA-II: Calculate crowding for the front being added.
                
                for j in S[i]:
                    n[j] -= 1
                    if n[j] == 0:
                        next_front.append(j)
                        ranks[j] = k + 1
                        
            # Calculate crowding distance for individuals in 'front'
            self._calculate_crowding_distance(front, F, crowding)
            
            front = next_front
            k += 1
            
        return ranks, crowding

    def _calculate_crowding_distance(self, front_indices, F, crowding):
        """
        Calculate crowding distance for individuals in front_indices.
        """
        if len(front_indices) <= 2:
            for i in front_indices:
                crowding[i] = float('inf')
            return
            
        n_obj = F.shape[1]
        for m in range(n_obj):
            # Sort front by objective m
            sorted_indices = sorted(front_indices, key=lambda idx: F[idx, m])
            
            # Boundary points
            crowding[sorted_indices[0]] = float('inf')
            crowding[sorted_indices[-1]] = float('inf')
            
            f_min = F[sorted_indices[0], m]
            f_max = F[sorted_indices[-1], m]
            f_range = f_max - f_min + 1e-14
            
            for k in range(1, len(sorted_indices) - 1):
                idx_prev = sorted_indices[k-1]
                idx_curr = sorted_indices[k]
                idx_next = sorted_indices[k+1]
                
                delta = (F[idx_next, m] - F[idx_prev, m]) / f_range
                crowding[idx_curr] += delta

    def _nsga2_select(self, X_pop, F_pop, N):
        """
        Select best N individuals from X_pop, F_pop using NSGA-II criteria.
        """
        ranks, crowding = self._non_dominated_sort(F_pop)
        
        # Sort by rank (asc), then crowding (desc)
        # Create list of indices
        indices = np.arange(len(X_pop))
        
        # Stable sort by rank, then crowding
        # We want lowest rank first. If rank tie, highest crowding first.
        # np.lexsort works with keys in reverse order of priority.
        # Priority 1: Rank (asc). Priority 2: Crowding (desc).
        # So sort by -Crowding, then Rank.
        
        sort_keys = (-crowding, ranks)
        sorted_indices = np.lexsort(sort_keys)
        
        # Take first N
        selected_indices = sorted_indices[:N]
        
        return X_pop[selected_indices], F_pop[selected_indices]
