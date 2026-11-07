# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-III-inspired decomposition-based multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _generate_reference_points(self, n_obj, H=12):
        """Generate uniformly distributed reference points on the simplex."""
        if n_obj == 3:
            # Triangular lattice for 3 objectives
            points = []
            for i in range(H + 1):
                for j in range(H - i + 1):
                    k = H - i - j
                    points.append([i / H, j / H, k / H])
            return np.array(points)
        else:
            # General case: use Dirichlet sampling
            np.random.seed(42)
            return np.random.dirichlet(np.ones(n_obj), size=H * H)

    def _normalize(self, F, ref):
        """Normalize objective values to [0,1] range based on reference point."""
        fmin = np.min(F, axis=0)
        fmax = np.max(F, axis=0)
        fmax = np.maximum(fmax, fmin + 1e-10)
        return (F - fmin) / (fmax - fmin)

    def _compute_distances(self, F_norm, ref_points):
        """Compute perpendicular distances from normalized points to reference lines."""
        n = len(F_norm)
        m = len(ref_points)
        dists = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                # Distance from point to line through origin in direction ref_points[j]
                proj = np.dot(F_norm[i], ref_points[j]) / (np.dot(ref_points[j], ref_points[j]) + 1e-10)
                closest = proj * ref_points[j]
                dists[i, j] = np.linalg.norm(F_norm[i] - closest)
        return dists

    def _fast_non_dominated_sort(self, F):
        """Perform fast non-dominated sorting."""
        n = len(F)
        if n == 0:
            return []
        
        S = [[] for _ in range(n)]  # S[i] = set of points dominated by i
        n_dom = np.zeros(n, dtype=int)  # n_dom[i] = number of points dominating i
        rank = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[i] += 1
        
        current_front = [i for i in range(n) if n_dom[i] == 0]
        if not current_front:
            return [list(range(n))]
        
        rank.append(current_front)
        while current_front:
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            if next_front:
                rank.append(next_front)
            current_front = next_front
        
        return rank

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for a set of indices."""
        n = len(indices)
        if n <= 2:
            return np.ones(n) * np.inf
        
        cd = np.zeros(n)
        F_sub = F[indices]
        n_obj = self.n_obj
        
        for k in range(n_obj):
            sorted_idx = np.argsort(F_sub[:, k])
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            f_range = F_sub[sorted_idx[-1], k] - F_sub[sorted_idx[0], k]
            if f_range > 1e-10:
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (F_sub[sorted_idx[i + 1], k] - F_sub[sorted_idx[i - 1], k]) / f_range
        
        return cd

    def _select_population(self, F, ref_points, archive_size):
        """NSGA-III style selection: fast non-dominated sort + distance-based selection."""
        n = len(F)
        if n <= archive_size:
            return list(range(n))
        
        ranks = self._fast_non_dominated_sort(F)
        
        # Normalize
        valid_mask = np.all(F < np.array([1705.0, 12.0, 0.3]), axis=1)
        if np.sum(valid_mask) == 0:
            return list(range(min(archive_size, n)))
        
        F_valid = F[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        # Get reference points
        n_obj = self.n_obj
        ref_pts = self._generate_reference_points(n_obj)
        
        # Normalize objectives
        fmin = np.min(F_valid, axis=0)
        fmax = np.max(F_valid, axis=0)
        fmax = np.maximum(fmax, fmin + 1e-10)
        F_norm = (F_valid - fmin) / (fmax - fmin)
        
        # Compute distances
        dists = self._compute_distances(F_norm, ref_pts)
        
        # For each reference point, select the closest point
        selected = set()
        for j in range(len(ref_pts)):
            # Find the point with minimum distance to this reference point
            min_dist_idx = np.argmin(dists[:, j])
            selected.add(valid_indices[min_dist_idx])
        
        # If we still need more, add from the best fronts
        if len(selected) < archive_size:
            for front in ranks:
                for idx in front:
                    if idx in selected:
                        continue
                    if np.all(F[idx] < np.array([1705.0, 12.0, 0.3])):
                        selected.add(idx)
                        if len(selected) >= archive_size:
                            break
                if len(selected) >= archive_size:
                    break
        
        # If still not enough, add any remaining
        if len(selected) < archive_size:
            for idx in range(n):
                if idx not in selected:
                    selected.add(idx)
                    if len(selected) >= archive_size:
                        break
        
        return list(selected)[:archive_size]

    def _crossover(self, x1, x2, lo, hi):
        """SBX crossover."""
        child1 = x1.copy()
        child2 = x2.copy()
        for i in range(len(x1)):
            if np.random.rand() < 0.5:
                if np.random.rand() < 0.5:
                    beta = 1.0 / (1.0 + 2.0 * np.random.rand())
                else:
                    beta = 1.0
                c1 = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                c2 = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                child1[i] = np.clip(c1, lo[i], hi[i])
                child2[i] = np.clip(c2, lo[i], hi[i])
        return child1, child2

    def _mutation(self, x, lo, hi, eta=20):
        """Polynomial mutation."""
        child = x.copy()
        for i in range(len(x)):
            if np.random.rand() < 0.9:
                delta = np.random.rand()
                if delta < 0.5:
                    val = 2.0 * delta
                    mut = (1.0 + val) ** (1.0 / (eta + 1.0))
                else:
                    val = 2.0 * (1.0 - delta)
                    mut = (1.0 + val) ** (1.0 / (eta + 1.0))
                child[i] += mut * (hi[i] - lo[i])
                child[i] = np.clip(child[i], lo[i], hi[i])
        return child

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290  # Safety margin
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([1705.0, 12.0, 0.3])

        used = 0
        all_F = []
        population_X = []
        population_F = []

        def call_p(x):
            nonlocal used
            fx = problem(x)
            used += 1
            return fx

        # Phase 1: Initial population using LHS + random
        pop_size = 60
        lhs_X = np.zeros((pop_size // 2, dim))
        for d in range(dim):
            perm = np.random.permutation(pop_size // 2)
            lhs_X[:, d] = lo[d] + (hi[d] - lo[d]) * (perm + np.random.uniform(size=pop_size // 2)) / (pop_size // 2)
        lhs_X = np.clip(lhs_X, lo, hi)
        
        rand_X = lo + np.random.uniform(size=(pop_size - pop_size // 2, dim)) * (hi - lo)
        rand_X = np.clip(rand_X, lo, hi)
        
        init_X = np.vstack([lhs_X, rand_X])
        init_F = np.array([call_p(x) for x in init_X])
        all_F.extend(init_F)
        population_X = init_X.tolist()
        population_F = init_F.tolist()

        # Phase 2: Evolutionary loop
        archive_size = 40
        n_gen = 0
        stagnation_count = 0
        best_hv = 0.0

        while used < budget:
            if time.time() > deadline - 5:
                break
            
            n_gen += 1
            
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            n_offspring = min(20, budget - used)
            for _ in range(n_offspring):
                if used >= budget:
                    break
                
                # Select parents using tournament selection
                if len(population_X) >= 2:
                    i = np.random.choice(len(population_X))
                    j = np.random.choice(len(population_X))
                    if i == j:
                        j = (i + 1) % len(population_X)
                    parent1 = population_X[i]
                    parent2 = population_X[j]
                    child1, child2 = self._crossover(parent1, parent2, lo, hi)
                else:
                    child1 = population_X[0].copy() if population_X else lo + np.random.uniform(size=dim) * (hi - lo)
                    child2 = child1.copy()
                
                # Mutate
                child1 = self._mutation(child1, lo, hi)
                child2 = self._mutation(child2, lo, hi)
                
                # Evaluate
                f1 = call_p(child1)
                f2 = call_p(child2)
                
                if used >= budget:
                    break
                
                offspring_X.append(child1)
                offspring_F.append(f1)
                offspring_X.append(child2)
                offspring_F.append(f2)
                all_F.extend([f1, f2])
            
            if not offspring_X:
                break
            
            # Combine and select
            combined_X = population_X + offspring_X
            combined_F = np.array(population_F + offspring_F)
            
            # Select next generation
            selected_indices = self._select_population(combined_F, None, pop_size)
            population_X = [combined_X[i] for i in selected_indices]
            population_F = combined_F[selected_indices].tolist()
            
            # Track stagnation (simplified: check if new points are added)
            new_valid = np.sum(np.all(combined_F[selected_indices] < ref, axis=1))
            if new_valid > 0:
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # If stagnating, inject random points
            if stagnation_count > 5 and used < budget:
                n_inject = min(5, budget - used)
                for _ in range(n_inject):
                    x_rand = lo + np.random.uniform(size=dim) * (hi - lo)
                    x_rand = np.clip(x_rand, lo, hi)
                    f_rand = call_p(x_rand)
                    all_F.append(f_rand)
                    population_X.append(x_rand)
                    population_F.append(f_rand)
                    if used >= budget:
                        break
                stagnation_count = 0

        # Phase 3: Local polish on best points
        if used < budget and population_F:
            F_arr = np.array(population_F)
            valid_mask = np.all(F_arr < ref, axis=1)
            if np.any(valid_mask):
                valid_idx = np.where(valid_mask)[0]
                # Sort by sum of objectives (simple scalarization)
                sums = np.sum(F_arr[valid_idx], axis=1)
                top_k = min(5, len(valid_idx))
                top_indices = valid_idx[np.argsort(sums)[:top_k]]
                
                for idx in top_indices:
                    if used >= budget or time.time() > deadline - 3:
                        break
                    
                    x0 = np.array(population_X[idx])
                    f0 = F_arr[idx].copy()
                    
                    # Simple local search: try small perturbations
                    best_x = x0.copy()
                    best_f = f0.copy()
                    
                    for _ in range(5):
                        if used >= budget:
                            break
                        perturbation = np.random.standard_normal(dim) * 0.1 * (hi - lo)
                        x_try = np.clip(best_x + perturbation, lo, hi)
                        f_try = call_p(x_try)
                        all_F.append(f_try)
                        
                        # Accept if better in at least one objective and not worse in others
                        if np.all(f_try <= best_f) and np.any(f_try < best_f):
                            best_x = x_try
                            best_f = f_try
                        elif np.sum(f_try) < np.sum(best_f) - 0.01 * np.sum(best_f):
                            best_x = x_try
                            best_f = f_try

        return np.array(all_F)

# EVOLVE-BLOCK-END
