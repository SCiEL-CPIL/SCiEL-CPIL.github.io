"""Multi-objective optimizer on RE37 using NSGA-II style approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Objective Optimizer using NSGA-II with hypervolume-based selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin of 20s before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size - keep it moderate for 200 budget
        pop_size = min(60, budget)
        n_gen_max = (budget - pop_size) // pop_size + 1  # approximate generations

        # Random initialization
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Reference point for hypervolume (used for selection)
        ref_point = np.array([1.1, 1.3, 1.1])

        # Helper: check if f1 dominates f2
        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        # Helper: compute hypervolume contribution of a single point
        # Given a set of points, compute the hypervolume dominated by a single point
        # relative to the reference point, assuming other points are not considered
        def hv_contribution_single(f, ref):
            """Compute hypervolume of box [f, ref] if f is strictly better than ref in all objectives."""
            if np.any(f >= ref):
                return 0.0
            return np.prod(ref - f)

        # Helper: non-dominated sorting
        def non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            if n == 0:
                return []
            
            domination_count = np.zeros(n, dtype=int)  # number of points that dominate i
            dominated_by = [[] for _ in range(n)]  # list of indices dominated by i
            
            for i in range(n):
                for j in range(i + 1, n):
                    if dominates(F[i], F[j]):
                        domination_count[j] += 1
                        dominated_by[i].append(j)
                    elif dominates(F[j], F[i]):
                        domination_count[i] += 1
                        dominated_by[j].append(i)
            
            fronts = []
            current_front = [i for i in range(n) if domination_count[i] == 0]
            if not current_front:
                # All points dominate each other? Shouldn't happen with proper dominance
                current_front = [0]
            
            front_idx = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in dominated_by[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                current_front = next_front
            
            return fronts

        # Helper: crowding distance assignment
        def crowding_distance(F, front_indices):
            """Assign crowding distances for points in a front."""
            n = len(front_indices)
            if n <= 2:
                return {idx: float('inf') for idx in front_indices}
            
            distances = {idx: 0.0 for idx in front_indices}
            
            for m in range(n_obj):
                # Sort by objective m
                sorted_idx = sorted(front_indices, key=lambda i: F[i][m])
                f_min = F[sorted_idx[0]][m]
                f_max = F[sorted_idx[-1]][m]
                f_range = f_max - f_min
                
                if f_range < 1e-15:
                    continue
                    
                # End points get infinity
                distances[sorted_idx[0]] = float('inf')
                distances[sorted_idx[-1]] = float('inf')
                
                for i in range(1, n - 1):
                    distances[sorted_idx[i]] += (F[sorted_idx[i + 1]][m] - F[sorted_idx[i - 1]][m]) / f_range
            
            return distances

        # Helper: binary tournament selection based on non-dominated rank and crowding distance
        def tournament_select(pop_indices, ranks, crowding):
            """Select one index using binary tournament."""
            i1 = np.random.randint(len(pop_indices))
            i2 = np.random.randint(len(pop_indices))
            idx1 = pop_indices[i1]
            idx2 = pop_indices[i2]
            
            rank1 = ranks[idx1]
            rank2 = ranks[idx2]
            
            if rank1 < rank2:
                return idx1
            elif rank2 < rank1:
                return idx2
            else:
                # Same rank, use crowding distance
                if crowding[idx1] > crowding[idx2]:
                    return idx1
                elif crowding[idx2] > crowding[idx1]:
                    return idx2
                else:
                    return idx1 if np.random.random() < 0.5 else idx2

        # Helper: compute hypervolume of a set of points (for potential use)
        def compute_hypervolume(F_set, ref):
            """Compute exact hypervolume for 3D using inclusion-exclusion or sorting."""
            # Filter points strictly better than reference
            valid = [f for f in F_set if np.all(f < ref)]
            if not valid:
                return 0.0
            
            n = len(valid)
            if n == 1:
                return hv_contribution_single(valid[0], ref)
            
            # For 3D, we can use a sorting-based approach
            # Sort by first objective
            valid_sorted = sorted(valid, key=lambda x: x[0])
            
            # Use the standard 3D hypervolume algorithm
            # This is O(n^2) but n is small (archive size)
            hv = 0.0
            for i in range(n):
                f_i = valid_sorted[i]
                # Compute contribution: points dominated by f_i but not by any point with lower f[0]
                # Actually, simpler: for each point, compute the volume it uniquely contributes
                # A point f_i contributes the volume of the box [f_i, ref] minus volumes already counted
                # This is complex. Let's use a different approach.
                
                # For 3D, we can use the following:
                # Sort by x, then for each point, the contribution is:
                # (ref[0] - f_i[0]) * (area in yz-plane of region dominated by f_i but not by previous points)
                
                # This is getting complex. Let's just use a simple O(n^2) approach:
                # For each pair of points, check dominance and compute contributions
                pass
            
            # Simpler approach: use the fact that for small n, we can enumerate
            # Actually, let's just use a straightforward computation
            # For 3 objectives, hypervolume can be computed as:
            # Sort points by first coordinate
            # For each point i (in sorted order), compute the "new" volume it adds
            # The new volume is the box [f_i, ref] intersected with the complement of union of boxes of previous points
            # This is still complex. Let's use a different strategy.
            
            # For our purposes, we don't need exact HV for selection.
            # We'll use NSGA-II's non-dominated sorting + crowding distance instead.
            # This function is provided but not used in the main loop.
            return 0.0  # Placeholder

        # Main optimization loop using NSGA-II
        while used < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Create offspring population
            offspring_X = []
            offspring_F = []
            
            for _ in range(pop_size):
                # Binary tournament selection for two parents
                # First, compute ranks and crowding for current population
                fronts = non_dominated_sort(F)
                ranks = np.zeros(pop_size, dtype=int)
                for rank, front in enumerate(fronts):
                    for idx in front:
                        ranks[idx] = rank
                
                # Compute crowding distance for the first front (most common case)
                # For simplicity, compute for all points
                crowding = {i: 0.0 for i in range(pop_size)}
                for front in fronts:
                    cd = crowding_distance(F, front)
                    for idx, dist in cd.items():
                        crowding[idx] = dist
                
                p1 = tournament_select(range(pop_size), ranks, crowding)
                p2 = tournament_select(range(pop_size), ranks, crowding)
                
                # SBX (Simulated Binary Crossover)
                if np.random.random() < 0.9:  # Crossover probability
                    # SBX with eta_c = 20
                    eta_c = 20.0
                    child1 = np.zeros(dim)
                    child2 = np.zeros(dim)
                    for j in range(dim):
                        if np.random.random() < 0.5:
                            if abs(p1[j] - p2[j]) < 1e-14:
                                child1[j] = p1[j]
                                child2[j] = p2[j]
                            else:
                                u = np.random.random()
                                if u <= 0.5:
                                    beta = (2 * u) ** (1.0 / (eta_c + 1))
                                else:
                                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                                child1[j] = 0.5 * ((1 + beta) * p1[j] + (1 - beta) * p2[j])
                                child2[j] = 0.5 * ((1 - beta) * p1[j] + (1 + beta) * p2[j])
                        else:
                            if abs(p1[j] - p2[j]) < 1e-14:
                                child1[j] = p1[j]
                                child2[j] = p2[j]
                            else:
                                u = np.random.random()
                                if u <= 0.5:
                                    beta = (2 * u) ** (1.0 / (eta_c + 1))
                                else:
                                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                                child1[j] = 0.5 * ((1 - beta) * p1[j] + (1 + beta) * p2[j])
                                child2[j] = 0.5 * ((1 + beta) * p1[j] + (1 - beta) * p2[j])
                    
                    # Polynomial mutation
                    eta_m = 20.0
                    pm = 1.0 / dim
                    for j in range(dim):
                        if np.random.random() < pm:
                            u = np.random.random()
                            if u < 0.5:
                                delta_q = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                            else:
                                delta_q = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                            child1[j] += delta_q * (hi[j] - lo[j])
                        if np.random.random() < pm:
                            u = np.random.random()
                            if u < 0.5:
                                delta_q = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                            else:
                                delta_q = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                            child2[j] += delta_q * (hi[j] - lo[j])
                    
                    # Clip to bounds
                    child1 = np.clip(child1, lo, hi)
                    child2 = np.clip(child2, lo, hi)
                    
                    offspring_X.append(child1)
                    offspring_X.append(child2)
                else:
                    # No crossover, just copy parents with mutation
                    child1 = p1.copy()
                    child2 = p2.copy()
                    
                    eta_m = 20.0
                    pm = 1.0 / dim
                    for j in range(dim):
                        if np.random.random() < pm:
                            u = np.random.random()
                            if u < 0.5:
                                delta_q = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                            else:
                                delta_q = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                            child1[j] += delta_q * (hi[j] - lo[j])
                        if np.random.random() < pm:
                            u = np.random.random()
                            if u < 0.5:
                                delta_q = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                            else:
                                delta_q = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                            child2[j] += delta_q * (hi[j] - lo[j])
                    
                    child1 = np.clip(child1, lo, hi)
                    child2 = np.clip(child2, lo, hi)
                    
                    offspring_X.append(child1)
                    offspring_X.append(child2)
                
                # Check time and budget
                if time.time() > deadline or used + len(offspring_X) > budget:
                    break
            
            if not offspring_X:
                break
            
            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += len(offspring_X)
            
            if used > budget:
                break
            
            # Combine parent and offspring populations
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, offspring_F])
            n_combined = len(combined_F)
            
            # Non-dominated sorting on combined population
            fronts = non_dominated_sort(combined_F)
            
            # Select next generation
            next_X = []
            next_F = []
            next_indices = []
            
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    # Take all from this front
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                        next_indices.append(idx)
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(combined_F, front)
                    # Sort by crowding distance (descending)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    for idx in sorted_front[:remaining]:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                        next_indices.append(idx)
                    break
            
            # Update population
            X = np.array(next_X)
            F = np.array(next_F)
            
            if len(X) < pop_size:
                # Pad with random points if needed
                n_pad = pop_size - len(X)
                pad_X = lo + (hi - lo) * np.random.random((n_pad, dim))
                pad_F = np.array([problem(x) for x in pad_X])
                used += n_pad
                X = np.vstack([X, pad_X])
                F = np.vstack([F, pad_F])

        return X
# EVOLVE-BLOCK-END
