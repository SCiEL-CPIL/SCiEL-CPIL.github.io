import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.ref_point = np.array([1.1, 1.3, 1.1])

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        
        archive_X = []
        archive_F = []
        evals = 0
        
        def compute_hv(points, ref):
            """
            Compute hypervolume in 3D using inclusion-exclusion or simple union.
            For small sets, we can use a grid-based approximation or exact calculation.
            Here we use a recursive decomposition for exact HV in 3D.
            """
            if not points:
                return 0.0
            
            # Filter points that are strictly better than ref in all objectives
            valid = [p for p in points if all(p[i] < ref[i] for i in range(3))]
            if not valid:
                return 0.0
            
            # Remove dominated points
            nd_points = []
            for p in valid:
                dominated = False
                for q in valid:
                    if p is q:
                        continue
                    if all(q[i] <= p[i] for i in range(3)) and any(q[i] < p[i] for i in range(3)):
                        dominated = True
                        break
                if not dominated:
                    nd_points.append(p)
            
            if not nd_points:
                return 0.0
            
            # Sort by first objective descending
            nd_points.sort(key=lambda x: -x[0])
            
            def hv_2d(pts_2d, ref_2d):
                if not pts_2d:
                    return 0.0
                # Filter valid
                valid_2d = [p for p in pts_2d if p[0] < ref_2d[0] and p[1] < ref_2d[1]]
                if not valid_2d:
                    return 0.0
                # Remove dominated
                nd_2d = []
                for p in valid_2d:
                    dominated = False
                    for q in valid_2d:
                        if p is q:
                            continue
                        if q[0] <= p[0] and q[1] <= p[1] and (q[0] < p[0] or q[1] < p[1]):
                            dominated = True
                            break
                    if not dominated:
                        nd_2d.append(p)
                
                if not nd_2d:
                    return 0.0
                
                # Sort by first coord descending
                nd_2d.sort(key=lambda x: -x[0])
                
                hv = 0.0
                for i in range(len(nd_2d)):
                    f1, f2 = nd_2d[i]
                    next_f1 = nd_2d[i+1][0] if i < len(nd_2d) - 1 else 0.0
                    w1 = f1 - next_f1
                    h2 = ref_2d[1] - f2
                    hv += w1 * h2
                return hv
            
            # 3D HV: decompose by first objective
            total_hv = 0.0
            for i in range(len(nd_points)):
                f1, f2, f3 = nd_points[i]
                next_f1 = nd_points[i+1][0] if i < len(nd_points) - 1 else 0.0
                w1 = f1 - next_f1
                
                # Consider points 0..i
                subset = nd_points[:i+1]
                pts_2d = [(p[1], p[2]) for p in subset]
                hv2 = hv_2d(pts_2d, (ref[1], ref[2]))
                total_hv += w1 * hv2
            
            return total_hv
        
        # Initialize with a few random points
        init_count = 10
        for _ in range(init_count):
            x = lo + (hi - lo) * rng.random(self.dim)
            f = problem(x)
            archive_X.append(x)
            archive_F.append(f)
            evals += 1
        
        # Main loop
        while evals < self.budget:
            # Get non-dominated set
            F = np.array(archive_F)
            nd_mask = np.ones(len(F), dtype=bool)
            for i in range(len(F)):
                if not nd_mask[i]:
                    continue
                for j in range(len(F)):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd_mask[i] = False
                        break
            
            nd_indices = np.where(nd_mask)[0]
            
            if len(nd_indices) == 0:
                x = lo + (hi - lo) * rng.random(self.dim)
                f = problem(x)
                archive_X.append(x)
                archive_F.append(f)
                evals += 1
                continue
            
            nd_points = [tuple(archive_F[i]) for i in nd_indices]
            
            # Calculate marginal contributions
            total_hv = compute_hv(nd_points, self.ref_point)
            
            marginals = []
            for k in range(len(nd_points)):
                reduced = nd_points[:k] + nd_points[k+1:]
                hv_minus = compute_hv(reduced, self.ref_point)
                marginals.append(total_hv - hv_minus)
            
            # Select the point with highest marginal contribution
            # But to avoid getting stuck, we use a softmax-like selection or random among top-k
            # Let's pick the one with highest marginal, but add some noise for exploration
            max_marginal = max(marginals)
            if max_marginal < 1e-10:
                # All marginals are tiny, just pick randomly
                chosen_local = rng.integers(len(marginals))
            else:
                # Normalize marginals to probabilities
                # Use a temperature to control exploration
                temp = 0.1
                exp_marginals = np.exp(np.array(marginals) / temp)
                probs = exp_marginals / np.sum(exp_marginals)
                chosen_local = rng.choice(len(marginals), p=probs)
            
            chosen_global = nd_indices[chosen_local]
            parent_x = archive_X[chosen_global]
            
            # Generate child using local search around the parent
            # Adaptive sigma: start small, increase if no improvement
            sigma = 0.1
            child_x = parent_x + sigma * rng.standard_normal(self.dim)
            child_x = np.clip(child_x, lo, hi)
            
            # Evaluate
            child_f = problem(child_x)
            archive_X.append(child_x)
            archive_F.append(child_f)
            evals += 1
            
        return archive_X
