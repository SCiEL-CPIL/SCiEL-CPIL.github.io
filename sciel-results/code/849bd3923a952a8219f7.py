"""Multi-objective optimizer for DTLZ7 using constrained NSGA-II with repair."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Constrained NSGA-II for DTLZ7.
    
    Strategy:
    - Treat g(x) as a constraint to be minimized (or a 4th objective).
    - Use a repair mechanism: if g > threshold, project x towards the feasible region (x[2:]=0)
      or mutate x[2:] towards 0 aggressively.
    - Since we know the structure, we can initialize with a mix of feasible (x[2:]=0) and infeasible points,
      but let the selection pressure favor low g.
    - However, the current program already exploits x[2:]=0 directly. 
    - To be "materially different", we will use a standard EA approach but with a 
      specific mutation strategy that targets the g(x) reduction, rather than just sampling the front.
      Actually, the "different strategy" directive suggests moving away from direct front sampling.
      Let's try a Random Search with adaptive step size towards the known optimum structure, 
      but implemented as a stochastic local search from multiple random starts, 
      specifically optimizing the g(x) component while maintaining diversity in x[0:2].
      
      Wait, the most different strategy from "sample the front" is "evolve towards the front".
      Let's implement a simple genetic algorithm where:
      1. Fitness is a weighted sum of objectives + penalty for g(x).
      2. Mutation specifically targets x[2:] to drive them to 0.
      3. Crossover maintains diversity in x[0:2].
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        used = 0
        max_evals = self.budget
        pop_size = 50
        n_gen = max(1, (max_evals - pop_size) // pop_size)

        def time_ok():
            return time.time() < deadline

        def eval_point(x):
            nonlocal used
            if used >= max_evals or not time_ok():
                return None
            f = problem(x)
            used += 1
            return f

        # Initialize population
        # Mix of random and "near-front" points to give the EA a head start
        # but let the EA do the work of refining.
        pop_x = []
        pop_f = []
        
        for i in range(pop_size):
            x = np.random.rand(self.dim)
            # 50% chance to start close to the known structure (x[2:] small)
            if np.random.rand() < 0.5:
                x[2:] = np.random.rand(self.dim - 2) * 0.1
            x = np.clip(x, lo, hi)
            
            f = eval_point(x)
            if f is None:
                break
            pop_x.append(x)
            pop_f.append(f)

        if not pop_x:
            return []

        # Convert to arrays for vectorized operations if possible, but loop is fine for 50
        pop_x = np.array(pop_x)
        pop_f = np.array(pop_f)

        # Main EA loop
        for gen in range(n_gen):
            if used >= max_evals or not time_ok():
                break

            # Calculate g(x) for constraint handling / repair
            # For DTLZ7, g(x) = 1 + 10*(n-2) + sum_{i=2}^{n-1} ((x_i-0.5)^2 - cos(20*pi*(x_i-0.5)))
            # But we can just use the objective values if we assume the standard formulation.
            # However, we don't have access to g directly, only f.
            # In DTLZ7, f1..f_{M-1} depend on x[0:2] and g. f_M depends on g.
            # If g > 0, the point is dominated by any point with g=0 and similar x[0:2].
            # So standard non-dominated sorting will naturally push towards g=0.
            
            # Non-dominated sorting
            # Simple O(N^2) front sorting for small pop
            fronts = self._fast_non_dominated_sort(pop_f)
            
            # Crowding distance
            crowd_dist = np.zeros(len(pop_f))
            for front in fronts:
                if len(front) > 1:
                    front_indices = np.array(front)
                    for m in range(self.n_obj):
                        obj_vals = pop_f[front_indices, m]
                        min_val = np.min(obj_vals)
                        max_val = np.max(obj_vals)
                        if max_val > min_val:
                            sorted_idx = np.argsort(obj_vals)
                            crowd_dist[front_indices[sorted_idx[0]]] += np.inf
                            crowd_dist[front_indices[sorted_idx[-1]]] += np.inf
                            for k in range(1, len(front_indices) - 1):
                                idx_curr = front_indices[sorted_idx[k]]
                                idx_prev = front_indices[sorted_idx[k-1]]
                                idx_next = front_indices[sorted_idx[k+1]]
                                crowd_dist[idx_curr] += (pop_f[idx_next, m] - pop_f[idx_prev, m]) / (max_val - min_val)
            
            # Selection (Tournament)
            next_pop_x = []
            next_pop_f = []
            
            # We need to generate new offspring to replace the worst part of the population
            # Standard NSGA-II: create offspring of size pop_size, combine, select best pop_size.
            
            # To save evaluations, we will only generate enough offspring to fill the gap
            # or replace the worst individuals.
            # Let's generate pop_size offspring.
            
            for _ in range(pop_size):
                if used >= max_evals or not time_ok():
                    break
                
                # Select parents
                p1 = self._tournament(pop_f, fronts, crowd_dist, 2)
                p2 = self._tournament(pop_f, fronts, crowd_dist, 2)
                
                # Crossover (SBX or simple blend)
                # Let's use simple uniform crossover with mutation
                if np.random.rand() < 0.9:
                    x_child = np.random.choice([pop_x[p1], pop_x[p2]], size=self.dim, replace=True)
                else:
                    x_child = pop_x[p1].copy()
                
                # Mutation
                # Key: Aggressive mutation on x[2:] to drive towards 0 (feasible region)
                # and random mutation on x[0:2] to explore the front.
                
                # 1. Explore x[0:2]
                if np.random.rand() < 0.5:
                    idx = np.random.randint(0, 2)
                    step = np.random.normal(0, 0.1)
                    x_child[idx] += step
                
                # 2. Repair/Drive x[2:] towards 0
                # This is the "constraint handling" part.
                # If we assume the current point is infeasible (g>0), we should push x[2:] to 0.
                # We don't know g, but we can always apply a contraction towards 0 for x[2:].
                # This ensures that over generations, x[2:] converges to 0.
                
                # Contraction factor
                contraction = 0.5
                x_child[2:] = x_child[2:] * contraction + np.random.normal(0, 0.01, self.dim - 2)
                
                x_child = np.clip(x_child, lo, hi)
                
                f = eval_point(x_child)
                if f is None:
                    break
                next_pop_x.append(x_child)
                next_pop_f.append(f)
            
            if not next_pop_x:
                break
            
            # Combine and Select
            all_x = np.vstack([pop_x, np.array(next_pop_x)])
            all_f = np.vstack([pop_f, np.array(next_pop_f)])
            
            fronts = self._fast_non_dominated_sort(all_f)
            crowd_dist = np.zeros(len(all_f))
            for front in fronts:
                if len(front) > 1:
                    front_indices = np.array(front)
                    for m in range(self.n_obj):
                        obj_vals = all_f[front_indices, m]
                        min_val = np.min(obj_vals)
                        max_val = np.max(obj_vals)
                        if max_val > min_val:
                            sorted_idx = np.argsort(obj_vals)
                            crowd_dist[front_indices[sorted_idx[0]]] += np.inf
                            crowd_dist[front_indices[sorted_idx[-1]]] += np.inf
                            for k in range(1, len(front_indices) - 1):
                                idx_curr = front_indices[sorted_idx[k]]
                                idx_prev = front_indices[sorted_idx[k-1]]
                                idx_next = front_indices[sorted_idx[k+1]]
                                crowd_dist[idx_curr] += (all_f[idx_next, m] - all_f[idx_prev, m]) / (max_val - min_val)
            
            # Select best pop_size
            selected_indices = []
            for front in fronts:
                if len(selected_indices) + len(front) <= pop_size:
                    selected_indices.extend(front)
                else:
                    # Sort by crowding distance
                    front_sorted = sorted(front, key=lambda i: crowd_dist[i], reverse=True)
                    needed = pop_size - len(selected_indices)
                    selected_indices.extend(front_sorted[:needed])
                    break
            
            pop_x = all_x[selected_indices]
            pop_f = all_f[selected_indices]

        # Return all evaluated points? Or just the final population?
        # The task says "Every returned objective vector is recorded".
        # The return value of __call__ is ignored.
        # So we should just return something, but the evaluations are what matter.
        # The current program returns X (list of x's). We should probably return the final population or all points.
        # Since the return value is ignored, we can return an empty list or the final pop.
        return pop_x

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                    n_dom[i] += 1
                elif self._dominates(F[j], F[i]):
                    S[j].append(i)
                    n_dom[j] += 1
                # Note: Standard NSGA-II logic:
                # if i dominates j, j is added to S[i] and n_dom[j] increases?
                # Let's stick to a simpler definition:
                # A dominates B if A is better in all and at least one strictly better.
                # If i dominates j, then i is "better".
                # In standard sorting:
                # n_dom[i] = number of individuals that dominate i.
                # S[i] = set of individuals dominated by i.
                
                # Let's re-implement carefully.
        
        # Re-doing sort logic correctly
        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                    n_dom[j] += 1 # i dominates j, so j's domination count increases
                # elif self._dominates(F[j], F[i]):
                #    S[j].append(i)
                #    n_dom[i] += 1
                # This is redundant if we loop all pairs.
        
        # Initial front
        for i in range(n):
            if n_dom[i] == 0:
                fronts[0].append(i)
        
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            k += 1
        
        return fronts

    def _dominates(self, a, b):
        # a dominates b if a is <= b in all objectives and < in at least one
        return np.all(a <= b) and np.any(a < b)

    def _tournament(self, F, fronts, crowd_dist, k):
        # Assign rank
        rank = np.zeros(len(F), dtype=int)
        for i, front in enumerate(fronts):
            for idx in front:
                rank[idx] = i
        
        candidates = np.random.randint(0, len(F), size=k)
        best_idx = candidates[0]
        for c in candidates:
            if rank[c] < rank[best_idx]:
                best_idx = c
            elif rank[c] == rank[best_idx]:
                if crowd_dist[c] > crowd_dist[best_idx]:
                    best_idx = c
        return best_idx
# EVOLVE-BLOCK-END
