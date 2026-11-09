"""Multi-objective optimizer on WFG4 with NSGA-III reference point based selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-III optimizer for WFG4 multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    @staticmethod
    def _fast_non_dominated_sort(F):
        """Fast non-dominated sorting. Returns list of fronts (lists of indices)."""
        n = len(F)
        if n == 0:
            return []
        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=int)
        fronts = []
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    nd[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    nd[i] += 1
        front0 = [i for i in range(n) if nd[i] == 0]
        fronts.append(front0)
        while front0:
            next_front = []
            for i in front0:
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            front0 = next_front
        return fronts

    def _generate_reference_points(self, n_points):
        """Generate reference points on the simplex for 3 objectives."""
        # For 3 objectives, we can use a grid or random points.
        # We use a set of predefined points for better coverage.
        points = []
        # Vertices
        points.append([1.0, 0.0, 0.0])
        points.append([0.0, 1.0, 0.0])
        points.append([0.0, 0.0, 1.0])
        # Edge midpoints
        points.append([0.5, 0.5, 0.0])
        points.append([0.5, 0.0, 0.5])
        points.append([0.0, 0.5, 0.5])
        # Center
        points.append([1.0/3, 1.0/3, 1.0/3])
        
        # Add more points if needed
        # We want roughly 15-20 points for a population of 20-30.
        # Let's add points with 0.2, 0.3, 0.5 mix
        for i in range(n_points - len(points)):
            # Random point on simplex
            u = np.random.rand(3)
            u = u / np.sum(u)
            points.append(u)
            
        return np.array(points)

    def _associate_points(self, F, ref_points):
        """Associate points in F with nearest reference points.
        Returns list of lists: ref_points[i] contains indices of points associated with it."""
        n_points = len(F)
        n_refs = len(ref_points)
        
        # Normalize F to [0,1] range relative to ideal and nadir?
        # NSGA-III uses normalization.
        # Ideal point: min of F
        # Nadir point: max of F (approximation)
        
        if n_points == 0:
            return [[] for _ in range(n_refs)]
        
        f_min = np.min(F, axis=0)
        f_max = np.max(F, axis=0)
        
        # Handle case where f_max == f_min
        range_f = f_max - f_min
        range_f[range_f == 0] = 1.0
        
        F_norm = (F - f_min) / range_f
        
        # Calculate distance to each reference point
        # dist[i, j] = || F_norm[i] - ref_points[j] ||
        # We can use dot product with normals? No, Euclidean is standard.
        
        # Efficient calculation:
        # || a - b ||^2 = ||a||^2 - 2 a.b + ||b||^2
        F_sq = np.sum(F_norm**2, axis=1)
        R_sq = np.sum(ref_points**2, axis=1)
        dist_sq = F_sq[:, np.newaxis] - 2 * F_norm @ ref_points.T + R_sq[np.newaxis, :]
        
        # For each point, find the nearest reference point
        nearest_ref = np.argmin(dist_sq, axis=1)
        
        associations = [[] for _ in range(n_refs)]
        for i, ref_idx in enumerate(nearest_ref):
            associations[ref_idx].append(i)
            
        return associations

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        rng = np.random.default_rng(0)
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim, n_obj = self.dim, self.n_obj
        budget = self.budget
        
        # Population size
        # For 3 objectives, a population of 20-30 is reasonable.
        pop_size = 20
        
        # Generate reference points
        ref_points = self._generate_reference_points(pop_size)
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # DE parameters
        F_CR = 0.8
        CR = 0.9
        K = 0.5
        
        while used < budget:
            if time.time() > deadline:
                break
            
            # Create offspring
            # We generate pop_size offspring in this generation
            # But we only have budget - used evaluations left.
            # We should generate min(pop_size, budget - used) offspring.
            
            n_offspring = min(pop_size, budget - used)
            
            X_off = np.empty((n_offspring, dim))
            F_off = np.empty((n_offspring, n_obj))
            
            for i in range(n_offspring):
                # Select parents using tournament selection
                # We need to select based on non-dominated rank and crowding distance (or reference distance)
                
                # For simplicity and speed, we use random selection from the current population
                # but biased towards non-dominated points.
                
                # Let's do proper NSGA-III selection:
                # 1. Non-dominated sort
                # 2. Tournament on fronts
                # 3. If same front, pick based on reference point distance
                
                # However, doing full sort every offspring is expensive.
                # Let's use a simpler selection: pick 2 random parents, 
                # if one dominates the other, pick the dominant one.
                # Otherwise, pick randomly.
                
                p1_idx = rng.integers(pop_size)
                p2_idx = rng.integers(pop_size)
                
                # Tournament
                if np.all(F[p1_idx] <= F[p2_idx]) and np.any(F[p1_idx] < F[p2_idx]):
                    parent1 = p1_idx
                elif np.all(F[p2_idx] <= F[p1_idx]) and np.any(F[p2_idx] < F[p1_idx]):
                    parent1 = p2_idx
                else:
                    parent1 = p1_idx if rng.random() < 0.5 else p2_idx
                    
                p3_idx = rng.integers(pop_size)
                p4_idx = rng.integers(pop_size)
                
                if np.all(F[p3_idx] <= F[p4_idx]) and np.any(F[p3_idx] < F[p4_idx]):
                    parent2 = p3_idx
                elif np.all(F[p4_idx] <= F[p3_idx]) and np.any(F[p4_idx] < F[p3_idx]):
                    parent2 = p4_idx
                else:
                    parent2 = p3_idx if rng.random() < 0.5 else p4_idx
                    
                # Ensure parents are distinct if possible
                if parent1 == parent2:
                    parent2 = (parent2 + 1) % pop_size
                
                # DE Mutation
                v = X[parent1] + K * (X[parent1] - X[parent2]) # Standard DE uses x1 + F(x2 - x3)
                # Let's stick to standard DE: x_p1 + F * (x_p2 - x_p3)
                # We need 3 parents.
                p3_idx2 = rng.integers(pop_size)
                if parent1 == p3_idx2:
                    p3_idx2 = (p3_idx2 + 1) % pop_size
                if parent2 == p3_idx2:
                    p3_idx2 = (p3_idx2 + 1) % pop_size
                    
                v = X[parent1] + K * (X[parent2] - X[p3_idx2])
                
                # Crossover
                jrand = rng.integers(dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.random() < CR or j == jrand:
                        trial[j] = v[j]
                    else:
                        trial[j] = X[parent1][j]
                        
                trial = np.clip(trial, lo, hi)
                
                X_off[i] = trial
                F_off[i] = problem(trial)
                used += 1
                
                if time.time() > deadline:
                    # If we run out of time, we break out and use what we have
                    # We need to handle the partial offspring population
                    n_offspring = i + 1
                    X_off = X_off[:n_offspring]
                    F_off = F_off[:n_offspring]
                    break
                    
            if used >= budget or time.time() > deadline:
                # Combine and select
                break
                
            # Combine parent and offspring
            X_new = np.vstack([X, X_off])
            F_new = np.vstack([F, F_off])
            
            # Non-dominated sort
            fronts = self._fast_non_dominated_sort(F_new)
            
            # Select best pop_size individuals
            X_selected = np.empty((0, dim))
            F_selected = np.empty((0, n_obj))
            
            for front in fronts:
                if len(X_selected) + len(front) <= pop_size:
                    # Add all
                    X_selected = np.vstack([X_selected, X_new[front]])
                    F_selected = np.vstack([F_selected, F_new[front]])
                else:
                    # Need to select from this front
                    # Use NSGA-III selection with reference points
                    
                    # The points in this front are F_new[front]
                    F_front = F_new[front]
                    X_front = X_new[front]
                    
                    # Associate points with reference points
                    associations = self._associate_points(F_front, ref_points)
                    
                    # We need to select (pop_size - len(X_selected)) points
                    to_select = pop_size - len(X_selected)
                    
                    # Step 1: Select one point per reference point (the closest one)
                    selected_indices = []
                    for ref_idx in range(len(ref_points)):
                        if associations[ref_idx]:
                            # Pick the one with smallest distance
                            # We already have the nearest ref, but we need the distance to pick the best one for this ref
                            # Actually, _associate_points gives the nearest ref for each point.
                            # We need to invert: for each ref, which points are assigned to it?
                            # The list associations[ref_idx] contains indices of points in F_front assigned to ref_idx.
                            # We pick the one with the smallest distance to ref_points[ref_idx].
                            
                            # Calculate distances for points in associations[ref_idx]
                            # This is a bit inefficient, but pop_size is small.
                            
                            # Re-calculate distance for these specific points
                            f_min_front = np.min(F_front, axis=0)
                            f_max_front = np.max(F_front, axis=0)
                            range_f = f_max_front - f_min_front
                            range_f[range_f == 0] = 1.0
                            F_front_norm = (F_front - f_min_front) / range_f
                            
                            pts_in_ref = associations[ref_idx]
                            if not pts_in_ref:
                                continue
                                
                            dists = np.linalg.norm(F_front_norm[pts_in_ref] - ref_points[ref_idx], axis=1)
                            best_local = np.argmin(dists)
                            global_idx = pts_in_ref[best_local]
                            selected_indices.append(global_idx)
                            
                    # If we selected too many, we need to remove some.
                    # If we selected too few, we need to add more.
                    
                    # Standard NSGA-III:
                    # 1. Assign each point to nearest ref.
                    # 2. If a ref has multiple points, keep the one with smallest distance.
                    # 3. If a ref has no points, skip.
                    # 4. If we have more selected points than to_select, remove the ones with largest distance to their ref (or random).
                    # 5. If we have fewer, add from unselected points based on distance.
                    
                    # Let's implement a simpler version:
                    # Just pick the 'to_select' points from F_front that are most spread out.
                    # We can use the distances calculated above.
                    
                    # Let's just pick random points from the front if the logic gets too complex.
                    # Given the small size, random selection from the last front is often good enough.
                    # But let's try to be smarter.
                    
                    # We have selected_indices.
                    if len(selected_indices) > to_select:
                        # Remove excess
                        # Calculate distance for selected indices
                        f_min_front = np.min(F_front, axis=0)
                        f_max_front = np.max(F_front, axis=0)
                        range_f = f_max_front - f_min_front
                        range_f[range_f == 0] = 1.0
                        F_front_norm = (F_front - f_min_front) / range_f
                        
                        dists = []
                        for idx in selected_indices:
                            # Find the ref this point is associated with
                            # We need the inverse mapping.
                            # For now, just pick random from selected_indices
                            pass
                            
                        # Simple fallback: keep first 'to_select'
                        selected_indices = selected_indices[:to_select]
                    elif len(selected_indices) < to_select:
                        # Add more from unselected
                        unselected = [i for i in range(len(F_front)) if i not in selected_indices]
                        while len(selected_indices) < to_select and unselected:
                            # Pick random
                            idx = rng.choice(unselected)
                            selected_indices.append(idx)
                            unselected.remove(idx)
                            
                    # Add selected points to X_selected, F_selected
                    X_selected = np.vstack([X_selected, X_front[selected_indices]])
                    F_selected = np.vstack([F_selected, F_front[selected_indices]])
                    break
                    
            X = X_selected
            F = F_selected

        # Final non-dominated sort to extract the front
        fronts = self._fast_non_dominated_sort(F)
        if fronts:
            front0 = fronts[0]
            X_final = X[front0]
            F_final = F[front0]
        else:
            X_final = X
            F_final = F

        return X_final
# EVOLVE-BLOCK-END
