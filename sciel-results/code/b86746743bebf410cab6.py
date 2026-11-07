"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II style approach with WFG9-specific insights."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # 300s limit, keep 15s margin
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # NSGA-II style population
        pop_size = min(60, budget // 2)
        if pop_size < 20:
            pop_size = 20
            
        # WFG9: The Pareto front is determined by the last 2 variables (x[10], x[11])
        # The first 10 variables affect the distance to the front
        # Strategy: Sample the last 2 variables systematically, optimize first 10 to be near 0.5
        
        # Create initial population with focus on the decision variables that matter
        # For WFG9, the Pareto front is parameterized by x[10] and x[11]
        # Let's create a structured initial population
        
        X = np.zeros((pop_size, dim))
        # Initialize first 10 dims around 0.5 (where they should be for optimal front)
        X[:, :10] = 0.5 + (hi[:10] - lo[:10]) * 0.1 * rng.standard_normal((pop_size, 10))
        # Initialize last 2 dims spread across their range
        X[:, 10] = lo[10] + (hi[10] - lo[10]) * rng.random(pop_size)
        X[:, 11] = lo[11] + (hi[11] - lo[11]) * rng.random(pop_size)
        X = np.clip(X, lo, hi)
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Keep track of all evaluated points for hypervolume
        all_points_X = [X.copy()]
        all_points_F = [F.copy()]
        
        def fast_non_dominated_sort(F):
            """Fast non-dominated sorting for NSGA-II"""
            n = len(F)
            if n == 0:
                return []
            
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int)  # n_dom[i] = number of individuals dominating i
            rank = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1
            
            current_front = np.where(n_dom == 0)[0].tolist()
            fronts = []
            current_rank = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = current_rank + 1
                            next_front.append(j)
                current_front = next_front
                current_rank += 1
            
            return fronts
        
        def crowding_distance(F, front_indices):
            """Compute crowding distance for a front"""
            if len(front_indices) <= 2:
                return {i: float('inf') for i in front_indices}
            
            cd = {i: 0.0 for i in front_indices}
            n_obj = F.shape[1]
            
            for m in range(n_obj):
                sorted_front = sorted(front_indices, key=lambda i: F[i, m])
                cd[sorted_front[0]] = float('inf')
                cd[sorted_front[-1]] = float('inf')
                
                f_min = F[sorted_front[0], m]
                f_max = F[sorted_front[-1], m]
                f_range = f_max - f_min
                if f_range > 1e-10:
                    for k in range(1, len(sorted_front) - 1):
                        cd[sorted_front[k]] += (F[sorted_front[k+1], m] - F[sorted_front[k-1], m]) / f_range
            
            return cd
        
        def tournament_select(fronts, cd_dict, X, k=2):
            """Tournament selection based on rank and crowding distance"""
            candidates = [rng.integers(len(X)) for _ in range(k)]
            best = candidates[0]
            for c in candidates[1:]:
                # Compare rank first
                rank_best = next((r for r, front in enumerate(fronts) if best in front), len(fronts))
                rank_c = next((r for r, front in enumerate(fronts) if c in front), len(fronts))
                
                if rank_best < rank_c:
                    continue
                elif rank_c < rank_best:
                    best = c
                else:
                    # Same rank, compare crowding distance
                    cd_best = cd_dict.get(best, 0)
                    cd_c = cd_dict.get(c, 0)
                    if cd_c > cd_best:
                        best = c
            
            return X[best]
        
        # Main evolutionary loop
        mutation_scale = 0.05
        crossover_prob = 0.8
        
        while used < budget:
            if time.time() > deadline:
                break
            
            # Sort into non-dominated fronts
            fronts = fast_non_dominated_sort(F)
            
            # Compute crowding distance for each front
            cd_dict = {}
            for front in fronts:
                front_cd = crowding_distance(F, front)
                cd_dict.update(front_cd)
            
            # Create offspring
            offspring_count = min(pop_size, budget - used)
            
            for _ in range(offspring_count):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Select two parents
                p1 = tournament_select(fronts, cd_dict, X)
                p2 = tournament_select(fronts, cd_dict, X)
                
                # Crossover (SBX-like)
                if rng.random() < crossover_prob:
                    alpha = rng.random()
                    child = alpha * p1 + (1 - alpha) * p2
                else:
                    child = p1.copy()
                
                # Mutation
                # WFG9: First 10 variables should be near 0.5, last 2 determine the front position
                # Use different mutation strategies for different variables
                mut_child = child.copy()
                
                # For first 10 vars: small mutations to keep them near optimal
                mut_child[:10] += mutation_scale * (hi[:10] - lo[:10]) * rng.standard_normal(10)
                
                # For last 2 vars: larger mutations to explore the Pareto front
                mut_child[10] += 0.1 * (hi[10] - lo[10]) * rng.standard_normal()
                mut_child[11] += 0.1 * (hi[11] - lo[11]) * rng.standard_normal()
                
                mut_child = np.clip(mut_child, lo, hi)
                
                fc = problem(mut_child)
                used += 1
                
                # Combine parent and offspring populations
                X_combined = np.vstack([X, mut_child.reshape(1, -1)])
                F_combined = np.vstack([F, fc.reshape(1, -1)])
                
                # Select best pop_size from combined
                fronts_combined = fast_non_dominated_sort(F_combined)
                
                # Select from fronts until we have pop_size individuals
                selected_indices = []
                for front in fronts_combined:
                    if len(selected_indices) + len(front) <= pop_size:
                        selected_indices.extend(front)
                    else:
                        # Need to use crowding distance to select
                        remaining = pop_size - len(selected_indices)
                        front_cd = crowding_distance(F_combined, front)
                        front_sorted = sorted(front, key=lambda i: front_cd[i], reverse=True)
                        selected_indices.extend(front_sorted[:remaining])
                        break
                
                X = X_combined[selected_indices]
                F = F_combined[selected_indices]
            
            # Update tracking
            all_points_X.append(X.copy())
            all_points_F.append(F.copy())
            
            # Adaptive mutation: decrease as we converge
            progress = used / budget
            mutation_scale = 0.05 * (1 - 0.5 * progress)
        
        # Collect all unique non-dominated points for return
        # The evaluator records all returned objective vectors
        # So we should return a diverse set of good points
        
        # Get all evaluated points
        all_X = np.vstack(all_points_X)
        all_F = np.vstack(all_points_F)
        
        # Remove duplicates
        unique_mask = np.ones(len(all_F), dtype=bool)
        for i in range(len(all_F)):
            if not unique_mask[i]:
                continue
            for j in range(i + 1, len(all_F)):
                if not unique_mask[j]:
                    continue
                if np.allclose(all_F[i], all_F[j], atol=1e-10):
                    unique_mask[j] = False
        
        unique_F = all_F[unique_mask]
        unique_X = all_X[unique_mask]
        
        # Get non-dominated points
        if len(unique_F) > 0:
            fronts = fast_non_dominated_sort(unique_F)
            non_dom_indices = fronts[0] if fronts else []
            
            # If we have very few non-dominated points, add some more diverse points
            if len(non_dom_indices) < 20:
                # Add points from next fronts
                for front in fronts[1:]:
                    for idx in front:
                        non_dom_indices.append(idx)
                        if len(non_dom_indices) >= 50:
                            break
                    if len(non_dom_indices) >= 50:
                        break
        
        return unique_X[non_dom_indices] if len(non_dom_indices) > 0 else unique_X[:min(50, len(unique_X))]
# EVOLVE-BLOCK-END
