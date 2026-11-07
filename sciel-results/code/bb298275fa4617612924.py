"""Multi-objective optimizer for RE21 with budget 200, dim 4, n_obj 2."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II style approach with hypervolume-based selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Use fixed seed for reproducibility
        rng = np.random.RandomState(42)

        # Population size
        pop_size = min(40, max(20, budget // 5))

        # Initialize population
        X = lo + (hi - lo) * rng.rand(pop_size, dim)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Reference point for hypervolume
        ref = np.array([3000.0, 0.06])

        def hypervolume_2d(points, ref_pt):
            """Compute 2D hypervolume for a set of points relative to reference point."""
            # Filter points that are better than reference in all objectives
            mask = np.all(points < ref_pt, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            
            # Sort by first objective
            pts = pts[np.argsort(pts[:, 0])]
            
            hv = 0.0
            prev_y = ref_pt[1]
            for p in pts:
                if p[1] < prev_y:
                    hv += (ref_pt[0] - p[0]) * (prev_y - p[1])
                    prev_y = p[1]
            return hv

        def dominates(a, b):
            """Check if a dominates b (a is better in all objectives and at least one strictly)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_front(points):
            """Return indices of non-dominated points."""
            n = len(points)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if dominates(points[j], points[i]):
                        dominated[i] = True
                        break
            return np.where(~dominated)[0]

        def crowding_distance(points, front_indices):
            """Compute crowding distance for points in a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            dist = np.zeros(n)
            pts = points[front_indices]
            
            for obj in range(self.n_obj):
                sorted_idx = np.argsort(pts[:, obj])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                min_val = pts[sorted_idx[0], obj]
                max_val = pts[sorted_idx[-1], obj]
                if max_val - min_val < 1e-12:
                    continue
                for k in range(1, n - 1):
                    dist[sorted_idx[k]] += (pts[sorted_idx[k + 1], obj] - pts[sorted_idx[k - 1], obj]) / (max_val - min_val)
            
            return dist

        def fast_non_dominated_sort(points):
            """Perform fast non-dominated sorting. Returns list of fronts (each is list of indices)."""
            n = len(points)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            
            for i in range(n):
                for j in range(i + 1, n):
                    if dominates(points[i], points[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    elif dominates(points[j], points[i]):
                        dominated_set[j].append(i)
                        domination_count[i] += 1
            
            fronts = []
            current_front = [i for i in range(n) if domination_count[i] == 0]
            if not current_front:
                return [list(range(n))]
            
            fronts.append(current_front)
            
            while current_front:
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front
            
            return fronts

        # Main evolution loop
        while used < budget:
            if time.time() > deadline:
                break
            
            # Sort population into non-dominated fronts
            fronts = fast_non_dominated_sort(F)
            
            # Select parents from first front (or multiple fronts if needed)
            n_select = min(10, len(fronts[0]))
            if n_select < 2:
                # Fall back to random selection
                parent_indices = rng.choice(len(F), size=2, replace=False)
            else:
                # Use crowding distance for selection within first front
                cd = crowding_distance(F, fronts[0])
                # Select based on crowding distance
                parent_indices = []
                front_arr = np.array(fronts[0])
                # Pick top n_select by crowding distance
                sorted_by_cd = np.argsort(-cd)
                parent_indices = front_arr[sorted_by_cd[:n_select]]
                # Ensure we have at least 2 distinct parents
                while len(set(parent_indices.tolist())) < 2 and len(parents_indices) < len(front_arr):
                    parent_indices = np.append(parent_indices, front_arr[rng.randint(len(front_arr))])
            
            # Generate offspring using SBX-like crossover and mutation
            for _ in range(min(2, budget - used)):
                if time.time() > deadline:
                    break
                
                # Pick two parents
                p1_idx = parent_indices[rng.randint(len(parent_indices))]
                p2_idx = parent_indices[rng.randint(len(parent_indices))]
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                # Crossover: blend
                alpha = rng.rand()
                child1 = alpha * p1 + (1 - alpha) * p2
                child2 = (1 - alpha) * p1 + alpha * p2
                
                # Mutation
                mutation_rate = 0.3
                for child in [child1, child2]:
                    mask = rng.rand(dim) < mutation_rate
                    child[mask] += 0.1 * (hi - lo)[mask] * rng.standard_normal(np.sum(mask))
                    child[:] = np.clip(child, lo, hi)
                
                # Evaluate
                f1 = problem(child1)
                used += 1
                f2 = problem(child2)
                used += 1
                
                if used > budget:
                    break
                
                # Replace worst point in population
                # Use hypervolume contribution as a proxy
                # Simple approach: replace the point with worst sum of objectives
                worst_idx = np.argmax(np.sum(F, axis=1))
                
                # Decide which child to add (or both if budget allows)
                if used <= budget - 1:
                    # Add both, replacing two worst
                    worst_indices = np.argsort(np.sum(F, axis=1))[:2]
                    X[worst_indices[0]] = child1
                    F[worst_indices[0]] = f1
                    X[worst_indices[1]] = child2
                    F[worst_indices[1]] = f2
                else:
                    # Add only the better child
                    if np.sum(f1) < np.sum(f2):
                        X[worst_idx] = child1
                        F[worst_idx] = f1
                    else:
                        X[worst_idx] = child2
                        F[worst_idx] = f2

        return X
# EVOLVE-BLOCK-END
