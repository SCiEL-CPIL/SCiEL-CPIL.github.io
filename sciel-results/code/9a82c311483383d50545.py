# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using a combination of random initialization,
    local search (Nelder-Mead), and evolutionary mutation on non-dominated points."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # 280s safety margin out of 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Fixed reference point for hypervolume (as per task)
        ref_point = np.array([1.1, 1.3, 1.1], dtype=np.float64)

        # State
        X = []  # list of vectors
        F = []  # list of objective vectors

        used = 0

        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=np.float64)
            used += 1
            return x, f

        def dominates(a, b):
            """True if a dominates b (a is better in all, strictly better in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def is_non_dominated(f, F_list):
            """Check if f is non-dominated by any in F_list."""
            for g in F_list:
                if dominates(g, f):
                    return False
            return True

        def update_archive(x, f):
            """Add (x, f) to archive if it is non-dominated. Remove dominated points."""
            # Check if f is dominated by any existing point
            for i, g in enumerate(F):
                if dominates(g, f):
                    return
            # Remove points dominated by f
            new_X = []
            new_F = []
            for i in range(len(F)):
                if not dominates(f, F[i]):
                    new_X.append(X[i])
                    new_F.append(F[i])
            new_X.append(x)
            new_F.append(f)
            X[:] = new_X
            F[:] = new_F

        def hypervolume_3d(points, ref):
            """Compute hypervolume in 3D. Points are (n, 3), ref is (3,)."""
            if len(points) == 0:
                return 0.0
            # Filter points strictly better than reference
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0

            # Sort by first objective
            idx = np.argsort(pts[:, 0])
            pts = pts[idx]

            # For 3D, we can use the standard algorithm:
            # Sort by f1, then for each point, compute the 2D hypervolume of the slice
            hv = 0.0
            prev_f1 = ref[0]
            for i in range(len(pts)):
                f1 = pts[i, 0]
                # The width in f1 dimension
                width = prev_f1 - f1
                if width <= 0:
                    continue
                # The 2D hypervolume of points i..end in (f2, f3) with reference (ref[1], ref[2])
                # But actually, for the slice at f1, we need the area in (f2, f3) of the
                # non-dominated front of pts[i:] in dimensions 2 and 3.
                # This is complex. Let's use a simpler approach for 3D:
                # Sort by f1, then maintain a list of (f2, f3) for the current front.
                # Actually, let's use the standard 3D HV algorithm.
                
                # For efficiency with small n, we can use the following:
                # Sort by f1 ascending.
                # For each point i, the contribution is (f1[i-1] - f1[i]) * area_2d(front[i:])
                # where area_2d is the 2D hypervolume of the points from i to end in (f2, f3).
                
                # This is O(n^2) which is fine for n <= 200.
                # Let's precompute the 2D areas.
                
                # Actually, let's just compute it directly for each point.
                # For point i, the "slice" is from f1[i] to f1[i-1] (or ref[0] for i=0).
                # The 2D hypervolume is the area of the union of rectangles for points j>=i.
                
                # Let's implement a helper for 2D HV.
                hv += width * _hv_2d(pts[i:, 1:3], ref[1:])
                prev_f1 = f1

            return hv

        def _hv_2d(points, ref):
            """Compute 2D hypervolume. Points are (n, 2), ref is (2,)."""
            if len(points) == 0:
                return 0.0
            # Filter points strictly better than reference
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0

            # Sort by first coordinate
            idx = np.argsort(pts[:, 0])
            pts = pts[idx]

            area = 0.0
            prev_y = ref[1]
            for i in range(len(pts)):
                x = pts[i, 0]
                y = pts[i, 1]
                if y >= prev_y:
                    continue
                width = prev_y - y
                # The x-width is from x to ref[0]? No.
                # In 2D, sort by x. The contribution of point i is (x[i] - x[i-1]) * (ref[1] - y[i])?
                # No, that's not right either.
                
                # Standard 2D HV:
                # Sort by x ascending.
                # For each point, the width in x is from x[i] to x[i-1] (or ref[0] for i=0).
                # The height is ref[1] - y[i].
                # But we must ensure that y[i] is the minimum y for all points with x >= x[i].
                
                # Let's do it properly:
                # Sort by x ascending.
                # Maintain the minimum y seen so far from the right?
                # Actually, the standard algorithm is:
                # Sort by x ascending.
                # For i from 0 to n-1:
                #   if y[i] < y[i+1] (or i is last), then it's on the front.
                #   contribution = (x[i+1] - x[i]) * (ref[1] - y[i])
                # where x[n] = ref[0].
                
                # Let's re-implement.
                pass

            # Let's use a simpler, correct O(n^2) approach for 2D.
            # For each point, compute the area of the rectangle it uniquely contributes.
            # This is tricky. Let's use the standard method.
            
            # Sort by x
            pts = pts[np.argsort(pts[:, 0])]
            
            area = 0.0
            # For each point, find the next point with a smaller y.
            # If none, the width extends to ref[0].
            for i in range(len(pts)):
                x_i = pts[i, 0]
                y_i = pts[i, 1]
                # Find the minimum x of all points j with y_j < y_i and x_j > x_i?
                # No.
                
                # Let's use the "sweep" method.
                # Sort by x.
                # For each point, the height is ref[1] - y_i.
                # The width is the distance to the next point with a smaller y, or ref[0] if none.
                
                # Find next point with smaller y
                next_x = ref[0]
                for j in range(i + 1, len(pts)):
                    if pts[j, 1] < y_i:
                        next_x = pts[j, 0]
                        break
                width = next_x - x_i
                if width > 0:
                    area += width * (ref[1] - y_i)
            
            return area

        # Re-implement hypervolume_3d correctly using the sweep method.
        def hypervolume_3d_correct(points, ref):
            if len(points) == 0:
                return 0.0
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0

            # Sort by first objective
            idx = np.argsort(pts[:, 0])
            pts = pts[idx]

            hv = 0.0
            # For each point i, the width in f1 is from f1[i] to f1[i-1] (or ref[0]).
            # The 2D hypervolume is the area in (f2, f3) of the non-dominated front of pts[i:].
            
            # To compute this efficiently, we can precompute the 2D HV for each suffix.
            # But for n <= 200, O(n^2 * n) = O(n^3) might be too slow.
            # Let's just compute it on the fly.
            
            for i in range(len(pts)):
                f1 = pts[i, 0]
                if i == 0:
                    width = ref[0] - f1
                else:
                    width = pts[i-1, 0] - f1
                if width <= 0:
                    continue
                
                # Compute 2D HV of pts[i:, 1:3] with ref[1:]
                hv += width * _hv_2d_correct(pts[i:, 1:3], ref[1:])
            
            return hv

        def _hv_2d_correct(points, ref):
            if len(points) == 0:
                return 0.0
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0

            # Sort by first coordinate
            idx = np.argsort(pts[:, 0])
            pts = pts[idx]

            area = 0.0
            for i in range(len(pts)):
                x_i = pts[i, 0]
                y_i = pts[i, 1]
                # Find next point with smaller y
                next_x = ref[0]
                for j in range(i + 1, len(pts)):
                    if pts[j, 1] < y_i:
                        next_x = pts[j, 0]
                        break
                width = next_x - x_i
                if width > 0:
                    area += width * (ref[1] - y_i)
            
            return area

        # Phase 1: Random initialization
        n_init = min(50, budget)
        for _ in range(n_init):
            x = lo + (hi - lo) * np.random.rand(dim)
            x, f = evaluate(x)
            update_archive(x, f)
            if used >= budget or time.time() > deadline:
                break

        # Phase 2: Local search on non-dominated points
        # We will perform local search on the current non-dominated points.
        # Use Nelder-Mead on a scalarized objective (e.g., sum of objectives).
        
        def scalarize(f):
            return np.sum(f)

        # Get a copy of the current archive
        archive_X = X[:]
        archive_F = F[:]
        
        # Shuffle to randomize the order
        indices = np.random.permutation(len(archive_X))
        
        for idx in indices:
            if used >= budget or time.time() > deadline:
                break
            
            x0 = archive_X[idx]
            f0 = archive_F[idx]
            
            # Nelder-Mead parameters
            simplex = [x0.copy()]
            for i in range(dim):
                x_i = x0.copy()
                x_i[i] += 0.05 * (hi[i] - lo[i])
                x_i = np.clip(x_i, lo, hi)
                simplex.append(x_i)
            
            # Evaluate the simplex
            simplex_F = []
            for x_s in simplex:
                x_s, f_s = evaluate(x_s)
                simplex_F.append(f_s)
                if used >= budget or time.time() > deadline:
                    break
            
            if used >= budget or time.time() > deadline:
                break
            
            # Nelder-Mead iterations
            max_iter = 30
            for _ in range(max_iter):
                if used >= budget or time.time() > deadline:
                    break
                
                # Sort by scalarized objective
                order = np.argsort([np.sum(f) for f in simplex_F])
                simplex = [simplex[i] for i in order]
                simplex_F = [simplex_F[i] for i in order]
                
                # Check convergence
                if np.std([np.sum(f) for f in simplex_F]) < 1e-8:
                    break
                
                # Reflection
                x_worst = simplex[-1]
                f_worst = simplex_F[-1]
                
                centroid = np.mean(simplex[:-1], axis=0)
                x_ref = centroid + (centroid - x_worst)
                x_ref = np.clip(x_ref, lo, hi)
                x_ref, f_ref = evaluate(x_ref)
                
                if np.sum(f_ref) < np.sum(simplex_F[-2]):
                    # Expansion
                    x_exp = centroid + 2 * (centroid - x_worst)
                    x_exp = np.clip(x_exp, lo, hi)
                    x_exp, f_exp = evaluate(x_exp)
                    
                    if np.sum(f_exp) < np.sum(f_ref):
                        simplex[-1] = x_exp
                        simplex_F[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_F[-1] = f_ref
                elif np.sum(f_ref) < np.sum(simplex_F[0]):
                    # Contraction
                    x_con = centroid + 0.5 * (x_worst - centroid)
                    x_con = np.clip(x_con, lo, hi)
                    x_con, f_con = evaluate(x_con)
                    
                    if np.sum(f_con) < np.sum(f_worst):
                        simplex[-1] = x_con
                        simplex_F[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            x_shrunk, f_shrunk = evaluate(simplex[i])
                            simplex[i] = x_shrunk
                            simplex_F[i] = f_shrunk
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        x_shrunk, f_shrunk = evaluate(simplex[i])
                        simplex[i] = x_shrunk
                        simplex_F[i] = f_shrunk
                
                # Update archive with the best point in the simplex
                best_idx = np.argmin([np.sum(f) for f in simplex_F])
                update_archive(simplex[best_idx], simplex_F[best_idx])
            
            # Update archive with the best point found during local search
            if len(simplex_F) > 0:
                best_idx = np.argmin([np.sum(f) for f in simplex_F])
                update_archive(simplex[best_idx], simplex_F[best_idx])

        # Phase 3: Evolutionary search
        # If we still have budget, use evolutionary search.
        if used < budget and time.time() < deadline:
            # Get non-dominated points
            non_dominated_X = X[:]
            non_dominated_F = F[:]
            
            # Create a population of size 20
            pop_size = min(20, len(non_dominated_X) * 2, budget - used)
            if pop_size > 0:
                # Initialize population by sampling from non-dominated points with noise
                pop_X = []
                pop_F = []
                for _ in range(pop_size):
                    idx = np.random.randint(len(non_dominated_X))
                    x = non_dominated_X[idx] + 0.1 * (hi - lo) * np.random.randn(dim)
                    x = np.clip(x, lo, hi)
                    x, f = evaluate(x)
                    pop_X.append(x)
                    pop_F.append(f)
                    if used >= budget or time.time() > deadline:
                        break
                
                # Evolutionary loop
                while used < budget and time.time() < deadline:
                    # Select parents using tournament selection
                    parent1_idx = np.random.randint(len(pop_X))
                    parent2_idx = np.random.randint(len(pop_X))
                    
                    # Crossover
                    if np.random.rand() < 0.5:
                        # Simulated binary crossover (SBX)
                        child = np.zeros(dim)
                        for i in range(dim):
                            if np.random.rand() < 0.5:
                                child[i] = pop_X[parent1_idx][i]
                            else:
                                child[i] = pop_X[parent2_idx][i]
                    else:
                        # Blend crossover
                        alpha = np.random.rand()
                        child = alpha * pop_X[parent1_idx] + (1 - alpha) * pop_X[parent2_idx]
                    
                    # Mutation
                    child = child + 0.1 * (hi - lo) * np.random.randn(dim)
                    child = np.clip(child, lo, hi)
                    
                    child, f_child = evaluate(child)
                    
                    # Replace the worst point in the population
                    # Use hypervolume contribution to determine the worst
                    # For simplicity, use the sum of objectives
                    worst_idx = np.argmax([np.sum(f) for f in pop_F])
                    pop_X[worst_idx] = child
                    pop_F[worst_idx] = f_child
                    
                    # Update archive
                    update_archive(child, f_child)

        return X
# EVOLVE-BLOCK-END
