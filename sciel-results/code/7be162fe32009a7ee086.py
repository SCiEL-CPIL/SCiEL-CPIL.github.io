"""
Multi-Island CMA-ES Optimizer for BBOB f11 (Discus).

Strategy:
1. Structure: 5 independent islands, each running a small CMA-ES-like evolution strategy.
2. Initialization: Islands start from diverse points (origin, random, and coordinate extremes) to cover the search space and handle the hidden shift/rotation.
3. Mechanism: Each island maintains a population of size 4. It samples candidates using a multivariate normal distribution (mean, covariance, sigma).
4. Cross Operator: To answer the open question "Does operator 'cross' change the chance that a child improves its island?", we implement a specific crossover mechanism.
   - Every 10 generations, the two best islands (lowest best_f) exchange their best individuals.
   - We perform a "cross" (linear recombination) between the best individual of Island A and the best individual of Island B.
   - The resulting child is injected into the worst-performing island to see if it improves that island's best.
   - This contrasts with the baseline where islands only rely on local mutation.
5. Budget Management: Strict time and call guards ensure we stop before 330s and 500 calls.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        dim = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = None
        best_f = float('inf')
        
        def can_call():
            nonlocal calls_made
            if calls_made >= budget:
                return False
            if time.time() > deadline:
                return False
            return True

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if not can_call():
                return float('inf')
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initialize 5 Islands
        num_islands = 5
        pop_size = 4
        sigma_init = 2.0
        
        # Diverse initial centers
        centers = []
        centers.append(np.zeros(dim))
        for i in range(num_islands - 1):
            rng = np.random.RandomState(100 + i)
            c = rng.randn(dim) * 2.0
            centers.append(c)
            
        # Evaluate initial centers to find starting points
        for c in centers:
            evaluate(c)

        islands = []
        for i in range(num_islands):
            # Each island has: mean, sigma, cov, population, best_x, best_f
            mean = centers[i].copy()
            sigma = sigma_init
            cov = np.eye(dim)
            
            # Initial population: sample from the island's distribution
            pop = []
            for j in range(pop_size):
                z = np.random.randn(dim)
                x = mean + sigma * np.linalg.cholesky(cov) @ z
                x = np.clip(x, -5, 5)
                f = evaluate(x)
                pop.append((x, f))
            
            # Sort to find best
            pop.sort(key=lambda p: p[1])
            best_x_island = pop[0][0]
            best_f_island = pop[0][1]
            
            islands.append({
                'mean': mean,
                'sigma': sigma,
                'cov': cov,
                'pop': pop,
                'best_x': best_x_island,
                'best_f': best_f_island
            })

        gen = 0
        cross_counter = 0
        cross_improvements = 0
        cross_attempts = 0

        while can_call():
            gen += 1
            # Update each island
            for i in range(num_islands):
                if not can_call():
                    break
                
                isl = islands[i]
                mean = isl['mean']
                sigma = isl['sigma']
                cov = isl['cov']
                pop = isl['pop']
                
                # 1. Selection: Keep best 2 (lambda/2)
                pop.sort(key=lambda p: p[1])
                parents = pop[:pop_size//2]
                
                # 2. Update Mean (Weighted average of parents)
                # Simple weights: 1.0 for best, 0.5 for second best (normalized)
                weights = np.array([1.0, 0.5])
                weights = weights / weights.sum()
                new_mean = np.zeros(dim)
                for w, (px, pf) in zip(weights, parents):
                    new_mean += w * px
                new_mean = np.clip(new_mean, -5, 5)
                
                # 3. Update Covariance (Rank-Mu update simplified)
                # We use a simplified rank-mu update to maintain stability
                c1 = 2 / ((dim + 1)**2 + 10)
                cmu = min(1 - c1, 2 * (sqrt_2 - 1) / (dim + 2))
                
                # Calculate difference vectors
                diff_vecs = []
                for px, pf in parents:
                    diff_vecs.append(px - mean)
                
                # Rank-mu update
                cov_new = (1 - c1 - cmu) * cov
                for w, dv in zip(weights, diff_vecs):
                    cov_new += cmu * w * np.outer(dv, dv)
                    
                # Ensure positive semi-definite (simple check)
                # If eigenvalues are too small, reset or shrink
                eigvals = np.linalg.eigvalsh(cov_new)
                if np.min(eigvals) < 1e-10:
                    cov_new = cov_new + 1e-10 * np.eye(dim)
                else:
                    # Regularize slightly
                    cov_new = cov_new + 1e-12 * np.eye(dim)
                    
                isl['cov'] = cov_new
                isl['mean'] = new_mean
                
                # 4. Update Sigma (CSA simplified: use step size adaptation based on acceptance rate)
                # For simplicity and robustness in 5D, we adapt sigma based on the improvement
                # If best improved, increase sigma slightly, else decrease
                if isl['best_f'] > float('inf'): # First time
                    pass
                else:
                    # Heuristic adaptation
                    # If we made progress in last gen, keep sigma or increase
                    # If not, decrease
                    # We track 'last_best_f' implicitly by comparing with previous best
                    # Here we just do a fixed small adaptation to avoid complex state
                    # A better way: use the standard CMA-ES sigma adaptation
                    # But for speed and simplicity, let's use a simple rule:
                    # If the new mean is closer to the best point, increase sigma?
                    # Actually, let's stick to a robust simple adaptation:
                    # If the best individual in the new population (not yet generated) is better, we'll increase.
                    # For now, we adapt sigma based on the change in mean distance to best?
                    # Let's use a standard heuristic: if success rate is high, increase.
                    # Since we don't track history, let's just keep sigma constant or decay it very slowly.
                    # Decay sigma to ensure convergence.
                    sigma *= 0.99
                    if sigma < 1e-4:
                        sigma = 1e-4
                isl['sigma'] = sigma

                # 5. Generate new population
                new_pop = []
                chol = np.linalg.cholesky(cov)
                for j in range(pop_size):
                    z = np.random.randn(dim)
                    x = new_mean + sigma * chol @ z
                    x = np.clip(x, -5, 5)
                    f = evaluate(x)
                    new_pop.append((x, f))
                
                # 6. Update Island Best
                new_pop.sort(key=lambda p: p[1])
                if new_pop[0][1] < isl['best_f']:
                    isl['best_f'] = new_pop[0][1]
                    isl['best_x'] = new_pop[0][0]
                
                isl['pop'] = new_pop

            # 7. Cross Operator (Inter-Island Recombination)
            # To answer the question: Does 'cross' improve an island?
            # We pick the two best islands.
            if gen % 10 == 0 and len(islands) >= 2:
                # Sort islands by best_f
                sorted_islands = sorted(islands, key=lambda x: x['best_f'])
                best_island = sorted_islands[0]
                second_best_island = sorted_islands[1]
                
                # Perform Cross: Linear recombination of best individuals
                x1 = best_island['best_x']
                x2 = second_best_island['best_x']
                
                # Create a child by mixing
                alpha = 0.5
                child_x = alpha * x1 + (1 - alpha) * x2
                child_x = np.clip(child_x, -5, 5)
                
                # Inject into the WORST island to test if it improves
                worst_island = sorted_islands[-1]
                cross_attempts += 1
                
                f_child = evaluate(child_x)
                
                if f_child < worst_island['best_f']:
                    cross_improvements += 1
                    worst_island['best_f'] = f_child
                    worst_island['best_x'] = child_x
                    # Also update the island's mean to pull it towards the improvement
                    worst_island['mean'] = worst_island['mean'] * 0.5 + child_x * 0.5
                    worst_island['mean'] = np.clip(worst_island['mean'], -5, 5)

        return best_x

# EVOLVE-BLOCK-END
