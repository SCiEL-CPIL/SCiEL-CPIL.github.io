"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-based multi-objective optimizer for RE21 (budget=200, dim=4, n_obj=2)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        ref = np.array([3000.0, 0.06])

        # Archive: list of (x, f) pairs that are non-dominated
        # We use a simple list-based archive with hypervolume contribution pruning
        archive_X = []
        archive_F = []
        used = 0

        def dominates(a, b):
            """Check if a dominates b (a <= b in all objectives, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def is_dominated(f, archive_F_list):
            """Check if f is dominated by any point in archive_F_list."""
            for af in archive_F_list:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def hv_contribution(f, archive_F_list):
            """Compute the hypervolume contribution of f given the current archive.
            For 2 objectives with reference point ref, this is the area of the box
            that f covers that isn't already covered by the archive."""
            # f dominates points worse than it. The contribution is the area
            # between f and the "skyline" of the archive in the region dominated by f.
            # For 2D: contribution = (ref[0]-f[0]) * (ref[1]-f[1]) - (area already covered)
            # Simpler: compute the hypervolume of the set {f} ∪ archive and subtract
            # hypervolume of archive alone. But that's expensive.
            # For 2D, we can compute directly:
            # The contribution of f is the rectangle [f[0], ref[0]] x [f[1], ref[1]]
            # minus the parts already covered by archive points that are "above-left" of f.
            
            if f[0] >= ref[0] or f[1] >= ref[1]:
                return 0.0
            
            # In 2D, the contribution is:
            # (ref[0] - f[0]) * (ref[1] - f[1]) - sum of overlaps with archive
            # Actually, for 2D hypervolume contribution:
            # f contributes the area of the rectangle it dominates that isn't dominated by archive
            # This is: (ref[0]-f[0])*(ref[1]-f[1]) minus the area covered by archive points 
            # that are in the region [f[0], ref[0]] x [f[1], ref[1]]
            
            # A simpler approach: sort archive points and compute the skyline
            # For 2D, the hypervolume contribution of a new point f is:
            # Let's find all archive points that are "relevant" (in the rectangle dominated by f)
            # and compute the uncovered area.
            
            # Actually, the cleanest way for 2D:
            # The contribution is the area of the L-shaped region or rectangle that f adds.
            # In 2D, if we have archive points, the contribution of f is:
            # (ref[0]-f[0]) * (ref[1]-f[1]) - (area of union of rectangles from archive 
            #  that are within [f[0],ref[0]] x [f[1],ref[1]])
            
            # For simplicity and speed, let's use a different strategy:
            # Just compute the full hypervolume of archive + f and subtract hypervolume of archive.
            # But that's O(n^2) per insertion. With budget 200, archive size ~50-100, this is fine.
            
            return None  # Placeholder, we'll use a different approach

        def compute_hypervolume_2d(points, ref_point):
            """Compute exact hypervolume for 2D points with given reference point.
            points: list of 2-element arrays (objective values)
            Only points strictly better than ref_point in every objective count."""
            # Filter points that are strictly better than reference in every objective
            valid = [p for p in points if p[0] < ref_point[0] and p[1] < ref_point[1]]
            if not valid:
                return 0.0
            
            # Sort by first objective (ascending)
            valid.sort(key=lambda p: p[0])
            
            # Compute hypervolume by sweeping
            hv = 0.0
            prev_y = ref_point[1]
            for p in valid:
                if p[1] < prev_y:
                    hv += (ref_point[0] - p[0]) * (prev_y - p[1])
                    prev_y = p[1]
            return hv

        def add_to_archive(x, f):
            """Add a point to the archive if it's not dominated. Remove dominated points."""
            nonlocal archive_X, archive_F
            if f[0] >= ref[0] or f[1] >= ref[1]:
                return False  # Not useful for scoring
            
            # Check if dominated by existing archive
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return False
            
            # Remove points dominated by f
            new_X = []
            new_F = []
            for i, (ax, af) in enumerate(zip(archive_X, archive_F)):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_X.append(ax)
                    new_F.append(af)
            
            new_X.append(x)
            new_F.append(f)
            archive_X = new_X
            archive_F = new_F
            return True

        def evaluate_and_add(x):
            """Evaluate x and add to archive if non-dominated."""
            nonlocal used
            f = problem(x)
            f = np.array(f, dtype=np.float64)
            used += 1
            add_to_archive(x.copy(), f)

        # Phase 1: Random initialization
        n_init = min(40, self.budget // 5)
        for i in range(n_init):
            x = lo + (hi - lo) * np.random.random(self.dim)
            evaluate_and_add(x)

        # Phase 2: Evolutionary search with hypervolume-based selection
        # Use a population-based approach where we select parents based on 
        # hypervolume contribution and use mutation/crossover
        
        # For RE21 with 4 variables and 2 objectives, we'll use:
        # - Selection: pick parents with high hypervolume contribution
        # - Crossover: uniform crossover with some probability
        # - Mutation: Gaussian mutation with adaptive step size
        
        # Track the best archive hypervolume for early stopping if needed
        best_hv = compute_hypervolume_2d(archive_F, ref)
        
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # If archive is empty, do random search
            if not archive_F:
                x = lo + (hi - lo) * np.random.random(self.dim)
                evaluate_and_add(x)
                continue
            
            # Select parents based on hypervolume contribution
            # Compute contribution of each archive point
            n_archive = len(archive_F)
            if n_archive <= 2:
                # Just pick randomly
                idx = np.random.randint(n_archive)
                parent1 = archive_X[idx]
                parent2 = parent1
            else:
                # Compute hypervolume contribution for each point
                # For 2D, the contribution of point i is the area of the rectangle
                # it uniquely covers. We approximate by computing HV without each point.
                # This is O(n^2) but n is small (~50-100)
                
                # Simpler: use a heuristic - points with lower objectives have higher contribution
                # Compute a score for each point
                scores = []
                for i, f in enumerate(archive_F):
                    # Contribution approximation: area of rectangle from f to ref,
                    # minus areas covered by other points
                    rect_area = (ref[0] - f[0]) * (ref[1] - f[1])
                    # Subtract overlaps with other points that are "better" in at least one dim
                    overlap = 0.0
                    for j, fj in enumerate(archive_F):
                        if i == j:
                            continue
                        # If fj is better than f in both dims, it doesn't overlap with f's unique region
                        # If fj is better in one dim and worse in another, they might overlap
                        # For 2D, the overlap of rectangles [f[0],ref[0]]x[f[1],ref[1]] and
                        # [fj[0],ref[0]]x[fj[1],ref[1]] is:
                        # [max(f[0],fj[0]), min(ref[0],ref[0])] x [max(f[1],fj[1]), min(ref[1],ref[1])]
                        # = [max(f[0],fj[0]), ref[0]] x [max(f[1],fj[1]), ref[1]]
                        # But this isn't quite right for "unique" contribution.
                        # Let's just use the rectangle area as a proxy, with a penalty
                        # for points that are close to other points.
                        pass
                    scores.append(rect_area)
                
                # Use softmax selection
                scores = np.array(scores)
                scores -= scores.max()
                exp_scores = np.exp(scores / (scores.std() + 1e-10))
                probs = exp_scores / exp_scores.sum()
                
                # Pick two parents (with replacement)
                idx1 = np.random.choice(n_archive, p=probs)
                idx2 = np.random.choice(n_archive, p=probs)
                parent1 = archive_X[idx1]
                parent2 = archive_X[idx2]
            
            # Crossover and mutation
            if np.random.random() < 0.5 and parent1 is not parent2:
                # Uniform crossover
                mask = np.random.random(self.dim) < 0.5
                child = np.where(mask, parent1, parent2)
            else:
                child = parent1.copy()
            
            # Gaussian mutation with adaptive step size
            # Step size decreases over time
            progress = used / self.budget
            step_size = 0.3 * (1.0 - 0.5 * progress)
            child = child + step_size * (hi - lo) * np.random.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            evaluate_and_add(child)
            
            # Occasionally add a random point to maintain diversity
            if used % 10 == 0 and used < self.budget - 1:
                x = lo + (hi - lo) * np.random.random(self.dim)
                evaluate_and_add(x)
        
        return archive_X if archive_X else [lo + (hi - lo) * 0.5]
# EVOLVE-BLOCK-END
