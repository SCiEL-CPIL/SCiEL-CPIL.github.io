"""Rewritten optimizer for RE21 using Sobol initialization and NSGA-II refinement."""
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = None
        self.eval_count = 0
        self.problem = None
        self.bounds = None
        self.lo = None
        self.hi = None
        self.results = []

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        deadline = self.start_time + 290  # Safety margin for 330s limit
        self.eval_count = 0
        self.problem = problem
        self.bounds = bounds
        self.lo = np.array([b[0] for b in bounds])
        self.hi = np.array([b[1] for b in bounds])
        self.results = []
        
        # 1. Global Exploration with Sobol Sequence
        # Population size for NSGA-II
        pop_size = 20
        
        # Ensure we don't exceed budget in initialization
        if pop_size > self.budget:
            pop_size = max(1, self.budget)
            
        # Generate Sobol points for initial population
        try:
            sobol = qmc.Sobol(d=self.dim, scramble=True)
            points = sobol.random(pop_size)
        except Exception:
            points = np.random.rand(pop_size, self.dim)
            
        X_init = self.lo + points * (self.hi - self.lo)
        
        # Evaluate initial population
        F_init = np.array([self._evaluate(x) for x in X_init])
        
        # Store all evaluated points
        for i in range(pop_size):
            self.results.append((X_init[i], F_init[i]))
            
        if self.eval_count >= self.budget:
            return np.array([r[0] for r in self.results])

        # 2. NSGA-II Optimization
        # Current population
        X_pop = X_init.copy()
        F_pop = F_init.copy()
        
        # Remaining budget
        remaining = self.budget - self.eval_count
        
        # Determine number of generations
        # Each generation produces pop_size offspring
        n_gen = remaining // pop_size
        
        for gen in range(n_gen):
            if time.time() > deadline:
                break
            if self.eval_count >= self.budget:
                break
            
            # Create offspring
            X_offspring = np.zeros((pop_size, self.dim))
            for i in range(pop_size):
                # Binary Tournament Selection
                idx1, idx2 = self._tournament_select(F_pop, 2)
                parent1 = X_pop[idx1]
                parent2 = X_pop[idx2]
                
                # Simulated Binary Crossover (SBX) or Simple Blend
                # Using simple blend crossover for simplicity and speed
                alpha = np.random.rand()
                child = alpha * parent1 + (1 - alpha) * parent2
                
                # Mutation
                # Gaussian mutation with adaptive sigma
                sigma = 0.1 * (self.hi - self.lo)
                child = child + np.random.randn(self.dim) * sigma
                
                # Clip to bounds
                child = np.clip(child, self.lo, self.hi)
                
                X_offspring[i] = child
            
            # Evaluate offspring
            F_offspring = np.array([self._evaluate(x) for x in X_offspring])
            
            # Store offspring
            for i in range(pop_size):
                self.results.append((X_offspring[i], F_offspring[i]))
                
            # Combine parents and offspring
            X_combined = np.vstack([X_pop, X_offspring])
            F_combined = np.vstack([F_pop, F_offspring])
            
            # Non-dominated sort and crowding distance
            # We need to select the best pop_size individuals from 2*pop_size
            
            # Fast Non-Dominated Sort
            ranks, fronts = self._fast_non_dominated_sort(F_combined)
            
            # Calculate Crowding Distance
            cd = self._crowding_distance(F_combined)
            
            # Selection
            new_indices = []
            for front in fronts:
                if len(new_indices) + len(front) <= pop_size:
                    new_indices.extend(front)
                else:
                    # Sort by crowding distance in descending order
                    front_sorted = sorted(front, key=lambda i: cd[i], reverse=True)
                    needed = pop_size - len(new_indices)
                    new_indices.extend(front_sorted[:needed])
                    break
            
            # Update population
            X_pop = X_combined[new_indices]
            F_pop = F_combined[new_indices]
            
            if self.eval_count >= self.budget:
                break

        # Return all evaluated points
        X_all = np.array([r[0] for r in self.results])
        return X_all

    def _evaluate(self, x):
        """Evaluate with budget tracking and time check."""
        if time.time() > self.start_time + 295:
            raise Exception("Time limit exceeded")
        
        if self.eval_count >= self.budget:
            raise Exception("Budget exceeded")
        
        self.eval_count += 1
        f = self.problem(x)
        return np.array(f)

    def _tournament_select(self, F, k):
        """Binary tournament selection."""
        idxs = np.random.choice(len(F), k, replace=False)
        # Find non-dominated among the selected
        # For simplicity, we just pick the one with better rank if we had ranks, 
        # but here we do a simple pairwise comparison.
        # Actually, standard tournament needs ranks. 
        # Since we don't have ranks here, we'll do a simple comparison.
        # If f1 dominates f2, pick 1. Else pick 2.
        i1, i2 = idxs
        if self._dominates(F[i1], F[i2]):
            return i1, i2
        elif self._dominates(F[i2], F[i1]):
            return i2, i1
        else:
            # Random tie-breaker
            return i1, i2

    def _dominates(self, f1, f2):
        """Check if f1 dominates f2 (minimization)."""
        return np.all(f1 <= f2) and np.any(f1 < f2)

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sort."""
        n = len(F)
        S = [[] for _ in range(n)]  # Set of solutions that dominate i
        n_count = [0] * n  # Number of solutions that dominate i
        rank = [0] * n
        fronts = [[]]
        
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if self._dominates(F[p], F[q]):
                    S[p].append(q)
                elif self._dominates(F[q], F[p]):
                    n_count[p] += 1
        
        for p in range(n):
            if n_count[p] == 0:
                rank[p] = 0
                fronts[0].append(p)
        
        i = 0
        while len(fronts[i]) > 0:
            Q = []
            for p in fronts[i]:
                for q in S[p]:
                    n_count[q] -= 1
                    if n_count[q] == 0:
                        rank[q] = i + 1
                        Q.append(q)
            i += 1
            if Q:
                fronts.append(Q)
            else:
                break
                
        # Remove empty last front if any
        if fronts and len(fronts[-1]) == 0:
            fronts.pop()
            
        return rank, fronts

    def _crowding_distance(self, F):
        """Calculate crowding distance."""
        n = len(F)
        cd = [0.0] * n
        if n == 0:
            return cd
        
        for m in range(self.n_obj):
            # Sort by objective m
            indices = np.argsort(F[:, m])
            cd[indices[0]] = float('inf')
            cd[indices[-1]] = float('inf')
            
            f_min = F[indices[0], m]
            f_max = F[indices[-1], m]
            f_range = f_max - f_min
            if f_range == 0:
                continue
                
            for i in range(1, n - 1):
                idx = indices[i]
                cd[idx] += (F[indices[i+1], m] - F[indices[i-1], m]) / f_range
                
        return cd

# EVOLVE-BLOCK-END
