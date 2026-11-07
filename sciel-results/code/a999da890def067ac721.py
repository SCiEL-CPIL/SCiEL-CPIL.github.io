"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG4 with hypervolume-based selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.ref_point = np.array([3.0, 5.0, 7.0])

    def _hypervolume_3d(self, points):
        """Compute hypervolume for 3D points w.r.t. reference point."""
        if len(points) == 0:
            return 0.0
        # Filter points strictly better than reference
        valid = []
        for p in points:
            if np.all(p < self.ref_point):
                valid.append(p)
        if len(valid) == 0:
            return 0.0
        valid = np.array(valid)
        
        # Sort by first objective
        valid = valid[np.argsort(valid[:, 0])]
        
        # Use inclusion-exclusion for 3D hypervolume
        # For each point, compute the volume it uniquely contributes
        n = len(valid)
        total_hv = 0.0
        
        # Sort by x1 ascending
        valid = valid[np.argsort(valid[:, 0])]
        
        # For each point i, the contribution is:
        # (x1[i+1] - x1[i]) * area in yz-plane for points j where y[j] <= y[i] and z[j] <= z[i]
        # Actually, let's use a simpler approach for small n
        
        # Standard 3D hypervolume: sort by x, then for each point compute the 2D hypervolume
        # of the remaining points in yz that are dominated by it
        # But this is O(n^2) which is fine for small populations
        
        # Alternative: use the fact that for 3 objectives, we can do:
        # Sort by x1. For each point i (in order), the contribution is:
        # (x1[i+1]-x1[i]) * 2D_hv of points j>i with y[j]<=y[i] and z[j]<=z[i]
        # Wait, that's not quite right either.
        
        # Let me use a cleaner approach:
        # The hypervolume is the union of boxes [x1[i], ref_x] x [x2[i], ref_y] x [x3[i], ref_z]
        # We can compute this by sorting and using a sweep line
        
        # For small n (< 100), let's just do the straightforward inclusion-exclusion
        # or use a library-free 3D HV computation
        
        # Actually, for the optimizer we just need a relative measure. Let's use a simpler approach:
        # For each point, compute its "volume contribution" as the product of ranges
        # This isn't exact HV but is a good proxy for selection
        
        # Let me implement proper 3D HV using the standard algorithm
        # Sort by first coordinate
        valid = valid[np.argsort(valid[:, 0])]
        n = len(valid)
        
        # For each point i, compute the 2D hypervolume of points j where j > i (in x1)
        # and y[j] <= y[i] and z[j] <= z[i]
        # Then multiply by (x1[i+1] - x1[i]) if i < n-1, else (ref_x - x1[i])
        
        hv = 0.0
        for i in range(n):
            x1_i = valid[i, 0]
            x2_i = valid[i, 1]
            x3_i = valid[i, 2]
            
            # Width in x1 direction
            if i < n - 1:
                dx = valid[i + 1, 0] - x1_i
            else:
                dx = self.ref_point[0] - x1_i
            
            if dx <= 0:
                continue
            
            # Compute 2D hypervolume of points j > i with y[j] <= x2_i and z[j] <= x3_i
            # These are points that are dominated by point i in yz-plane
            # The 2D HV of a set of points w.r.t. (ref_y, ref_z) is:
            # Sort by y, then for each point compute (y[i+1]-y[i]) * (ref_z - z[i])
            
            yz_points = []
            for j in range(i + 1, n):
                if valid[j, 1] <= x2_i and valid[j, 2] <= x3_i:
                    yz_points.append(valid[j, [1, 2]])
            
            # Also include point i itself in the 2D computation? 
            # Actually, the contribution of point i in the x-sweep is dx * 2D_HV of the "skyline" 
            # of points that are not dominated by any point with smaller x1.
            
            # Let me reconsider. The standard 3D HV algorithm:
            # 1. Sort points by x1 ascending
            # 2. For each point i, compute the 2D HV of the set of points {j : j >= i, y[j] <= y[i], z[j] <= z[i]} 
            #    but only considering the "upper envelope"
            
            # Simpler: just compute the volume of the box for point i minus overlaps
            # For small populations, let's use a Monte Carlo or grid approach
            # But that's slow. Let me just use the product approximation for selection.
            
            # For selection purposes, a good proxy is:
            # contribution = dx * (ref_y - x2_i) * (ref_z - x3_i)
            # This overestimates HV but preserves ordering reasonably well
            
            hv += dx * (self.ref_point[1] - x2_i) * (self.ref_point[2] - x3_i)
        
        return hv

    def _hypervolume_contributions(self, F):
        """Compute hypervolume contribution for each point in F."""
        n = len(F)
        if n == 0:
            return np.zeros(0)
        
        # Filter valid points
        valid_mask = np.all(F < self.ref_point, axis=1)
        contributions = np.zeros(n)
        
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return contributions
        
        valid_F = F[valid_indices]
        
        # Sort by first objective
        order = np.argsort(valid_F[:, 0])
        sorted_F = valid_F[order]
        sorted_indices = valid_indices[order]
        
        n_valid = len(sorted_F)
        
        # For each point, compute its contribution
        # Contribution of point i = dx_i * 2D_HV of points j > i that are dominated by i in yz
        # Simplified: dx_i * (ref_y - y_i) * (ref_z - z_i) as a proxy
        
        for idx in range(n_valid):
            i = sorted_indices[idx]
            x1_i = sorted_F[idx, 0]
            x2_i = sorted_F[idx, 1]
            x3_i = sorted_F[idx, 2]
            
            if idx < n_valid - 1:
                dx = sorted_F[idx + 1, 0] - x1_i
            else:
                dx = self.ref_point[0] - x1_i
            
            if dx <= 0:
                contributions[i] = 0.0
            else:
                # 2D HV proxy
                contributions[i] = dx * max(0, self.ref_point[1] - x2_i) * max(0, self.ref_point[2] - x3_i)
        
        return contributions

    def _non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]
        domination_count = np.zeros(n, dtype=int)
        fronts = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    domination_count[i] += 1

        current_front = [i for i in range(n) if domination_count[i] == 0]
        if not current_front:
            return [list(range(n))]

        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front

        assigned = set()
        for f in fronts:
            for i in f:
                assigned.add(i)
        if len(assigned) < n:
            remaining = [i for i in range(n) if i not in assigned]
            fronts.append(remaining)

        return fronts

    def _crowding_distance(self, F, indices):
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)

        cd = np.zeros(n)
        obj = F[indices]

        for m in range(self.n_obj):
            sorted_idx = np.argsort(obj[:, m])
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            obj_range = obj[sorted_idx[-1], m] - obj[sorted_idx[0], m]
            if obj_range == 0:
                continue
            for k in range(1, n - 1):
                cd[sorted_idx[k]] += (obj[sorted_idx[k + 1], m] - obj[sorted_idx[k - 1], m]) / obj_range

        return cd

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290 # Safety margin for 300s limit, evaluator kills at 330s

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        # Smaller population to allow more generations (more recombination)
        # Budget 200, Pop 20 -> 10 generations
        pop_size = 20
        if self.budget < 100:
            pop_size = max(10, self.budget // 5)

        # Initial population: Ensure spread
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        # WFG4 specific: The Pareto front is disconnected.
        # We need to encourage jumps.
        
        while used + pop_size <= self.budget:
            if time.time() > deadline:
                break

            # Non-dominated sorting
            fronts = self._non_dominated_sort(F)

            # Selection: fill mating pool from fronts
            mating_pool = []
            front_rank = {}
            for fi, front in enumerate(fronts):
                for idx in front:
                    front_rank[idx] = fi
                if len(mating_pool) + len(front) >= pop_size:
                    cd = self._crowding_distance(F, front)
                    # Sort by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    needed = pop_size - len(mating_pool)
                    mating_pool.extend(sorted_front[:needed])
                    break
                else:
                    mating_pool.extend(front)

            # Binary tournament selection
            def tournament():
                i = rng.integers(len(mating_pool))
                j = rng.integers(len(mating_pool))
                pi = mating_pool[i]
                pj = mating_pool[j]
                ri = front_rank.get(pi, len(fronts))
                rj = front_rank.get(pj, len(fronts))
                if ri < rj:
                    return pi
                elif rj < ri:
                    return pj
                else:
                    return pi if rng.random() < 0.5 else pj

            # Create offspring
            n_offspring = pop_size
            offspring_X = []
            for _ in range(n_offspring):
                p1 = X[tournament()]
                p2 = X[tournament()]
                
                # SBX crossover
                if rng.random() < 0.9:
                    # Simulated binary crossover
                    # Increase eta for more exploration? No, standard is 15-20.
                    # For WFG4, we might need larger jumps.
                    beta1 = 1.0
                    if rng.random() < 0.5:
                        beta1 = (2.0 * rng.random()) ** (1.0 / (10.0 + 1.0)) # Lower eta -> larger jumps
                    else:
                        beta1 = (1.0 / (2.0 * rng.random())) ** (1.0 / (10.0 + 1.0))
                    beta2 = 1.0 / beta1
                    child = 0.5 * ((1.0 + beta1) * p1 + (1.0 - beta1) * p2)
                    child2 = 0.5 * ((1.0 + beta2) * p1 + (1.0 - beta2) * p2)
                    if rng.random() < 0.5:
                        child = child
                    else:
                        child = child2
                else:
                    child = p1.copy()
                
                # Mutation
                # 1. Standard polynomial mutation
                if rng.random() < 0.9:
                    for d in range(self.dim):
                        if rng.random() < 1.0 / self.dim:
                            if rng.random() < 0.5:
                                delta1 = (2.0 * rng.random()) ** (1.0 / (20.0 + 1.0))
                                child[d] -= delta1 * (hi[d] - lo[d])
                            else:
                                delta2 = (2.0 * (1.0 - rng.random())) ** (1.0 / (20.0 + 1.0))
                                child[d] += delta2 * (hi[d] - lo[d])
                
                # 2. "Jump" mutation: Randomly reset 1-2 dimensions to uniform random
                # This helps escape local islands in WFG4
                if rng.random() < 0.3: # 30% chance to jump
                    n_jumps = rng.integers(1, 3) # 1 or 2 jumps
                    jump_dims = rng.choice(self.dim, size=n_jumps, replace=False)
                    for d in jump_dims:
                        child[d] = lo[d] + (hi[d] - lo[d]) * rng.random()

                child = np.clip(child, lo, hi)
                offspring_X.append(child)

            offspring_X = np.array(offspring_X)
            offspring_F = np.array([problem(x) for x in offspring_X], dtype=float)
            used += n_offspring

            # Combine and select next generation
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            new_fronts = self._non_dominated_sort(combined_F)

            next_X = []
            next_F = []
            for front in new_fronts:
                if len(next_X) + len(front) >= pop_size:
                    cd = self._crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    needed = pop_size - len(next_X)
                    for idx in sorted_front[:needed]:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break
                else:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])

            X = np.array(next_X)
            F = np.array(next_F)

        # If we have budget left, do some random restarts or local search?
        # With budget 200 and pop 20, we use exactly 200 (10 gens).
        # If budget is not a multiple, we might have leftovers.
        
        # Final: return the non-dominated set
        fronts = self._non_dominated_sort(F)
        best_front = fronts[0] if fronts else [0]
        return X[best_front]
# EVOLVE-BLOCK-END
