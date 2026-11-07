import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        d = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume calculation
        ref_point = np.array([1.1, 1.3, 1.1])

        # Generate a dense candidate lattice
        # For 4 dimensions, we need a manageable number of candidates
        # Let's use 5 points per dimension: 5^4 = 625 candidates
        # But we can be smarter: use stratified sampling
        n_per_dim = 4
        grid_points = []
        for i in range(n_per_dim):
            grid_points.append(np.linspace(lo[i], hi[i], n_per_dim + 1))
        
        # Create all combinations
        candidates = []
        for x0 in grid_points[0]:
            for x1 in grid_points[1]:
                for x2 in grid_points[2]:
                    for x3 in grid_points[3]:
                        candidates.append([x0, x1, x2, x3])
        candidates = np.array(candidates)
        
        # Add some random points for diversity
        n_random = 200
        random_candidates = lo + (hi - lo) * np.random.random((n_random, d))
        candidates = np.vstack([candidates, random_candidates])
        
        # Evaluate a small initial set to get a rough idea
        # Use stratified initialization
        init_size = min(20, budget)
        X_init = np.zeros((init_size, d))
        for i in range(d):
            edges = np.linspace(lo[i], hi[i], init_size + 1)
            centers = (edges[:-1] + edges[1:]) / 2.0
            np.random.shuffle(centers)
            X_init[:, i] = centers
        
        F_init = np.array([problem(x) for x in X_init])
        used = init_size
        
        # Archive of non-dominated points
        archive_X = X_init.copy()
        archive_F = F_init.copy()
        
        # Filter to only points that are strictly better than reference point
        mask = np.all(archive_F < ref_point, axis=1)
        archive_X = archive_X[mask]
        archive_F = archive_F[mask]
        
        # Pre-compute candidate objective values lazily
        # We'll evaluate candidates on-demand
        
        # Greedy hypervolume maximization
        # At each step, select the candidate that maximizes marginal hypervolume
        
        def compute_hypervolume(F, ref):
            """Compute hypervolume of a set of points with reference point."""
            if len(F) == 0:
                return 0.0
            
            # Filter to points strictly better than reference
            mask = np.all(F < ref, axis=1)
            F = F[mask]
            if len(F) == 0:
                return 0.0
            
            # For 3D, we can use a simple approach
            # Sort by first objective
            order = np.argsort(F[:, 0])
            F_sorted = F[order]
            
            # Use inclusion-exclusion for 3D hypervolume
            # This is a simplified approach - for accuracy, we'd need a proper algorithm
            # For now, use a Monte Carlo-like estimate with deterministic grid
            
            # Actually, let's use a proper 3D hypervolume calculation
            # Sort by objective 0
            F_sorted = F[np.argsort(F[:, 0])]
            
            hv = 0.0
            for i in range(len(F_sorted)):
                # The box from F_sorted[i] to ref, minus boxes already counted
                # This is O(n^2) but n is small (archive size)
                x_min = F_sorted[i]
                x_max = ref
                vol = np.prod(x_max - x_min)
                
                # Subtract overlaps with previous points
                for j in range(i):
                    # Intersection of box i and box j
                    inter_min = np.maximum(x_min, F_sorted[j])
                    inter_max = np.minimum(x_max, ref)
                    if np.all(inter_min < inter_max):
                        vol -= np.prod(inter_max - inter_min)
                
                hv += vol
            
            return hv
        
        def marginal_hv(f_new, archive_F, ref):
            """Compute marginal hypervolume contribution of f_new to archive."""
            # New hypervolume with f_new
            F_new = np.vstack([archive_F, f_new.reshape(1, -1)])
            hv_new = compute_hypervolume(F_new, ref)
            
            # Old hypervolume
            hv_old = compute_hypervolume(archive_F, ref)
            
            return hv_new - hv_old
        
        # Main loop: greedy selection
        while used < budget:
            if time.time() > deadline:
                break
            
            # Find the best candidate by marginal hypervolume
            best_candidate = None
            best_marginal = -np.inf
            best_f = None
            
            # Sample a subset of candidates to keep it fast
            n_samples = min(len(candidates), 50)
            sample_idx = np.random.choice(len(candidates), n_samples, replace=False)
            
            for idx in sample_idx:
                x = candidates[idx]
                f = problem(x)
                used += 1
                
                # Check if strictly better than reference
                if not np.all(f < ref_point):
                    continue
                
                # Check if dominated by archive
                dominated = False
                for af in archive_F:
                    if np.all(af <= f) and np.any(af < f):
                        dominated = True
                        break
                if dominated:
                    continue
                
                # Compute marginal hypervolume
                marginal = marginal_hv(f, archive_F, ref_point)
                
                if marginal > best_marginal:
                    best_marginal = marginal
                    best_candidate = x
                    best_f = f
            
            if best_candidate is None:
                # No good candidate found, add a random point
                x = lo + (hi - lo) * np.random.random(d)
                f = problem(x)
                used += 1
                
                if np.all(f < ref_point):
                    # Update archive
                    dominated = False
                    for af in archive_F:
                        if np.all(af <= f) and np.any(af < f):
                            dominated = True
                            break
                    if not dominated:
                        # Remove dominated archive points
                        keep = []
                        for i, af in enumerate(archive_F):
                            if not (np.all(f <= af) and np.any(f < af)):
                                keep.append(i)
                        if keep:
                            archive_X = np.vstack([archive_X[keep], x])
                            archive_F = np.vstack([archive_F[keep], f.reshape(1, -1)])
                        else:
                            archive_X = x.reshape(1, -1)
                            archive_F = f.reshape(1, -1)
                break
            
            # Update archive with best candidate
            # Remove dominated archive points
            keep = []
            for i, af in enumerate(archive_F):
                if not (np.all(best_f <= af) and np.any(best_f < af)):
                    keep.append(i)
            if keep:
                archive_X = np.vstack([archive_X[keep], best_candidate])
                archive_F = np.vstack([archive_F[keep], best_f.reshape(1, -1)])
            else:
                archive_X = best_candidate.reshape(1, -1)
                archive_F = best_f.reshape(1, -1)
            
            # Remove the used candidate from the pool
            candidates = np.delete(candidates, sample_idx[list(sample_idx).index(np.where(candidates == best_candidate)[0][0])], axis=0)
        
        return archive_X
