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
        
        def compute_nd_indices(F):
            n = len(F)
            if n == 0:
                return []
            nd_mask = np.ones(n, dtype=bool)
            F_arr = np.array(F)
            for i in range(n):
                if not nd_mask[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        nd_mask[i] = False
                        break
            return np.where(nd_mask)[0].tolist()
        
        def hv_2d(pts, ref):
            if not pts:
                return 0.0
            pts = sorted(pts, key=lambda x: -x[0])
            nd = []
            max_y = -np.inf
            for p in pts:
                if p[1] > max_y:
                    nd.append(p)
                    max_y = p[1]
            if not nd:
                return 0.0
            hv = 0.0
            r2, r3 = ref
            n = len(nd)
            for i in range(n):
                f2, f3 = nd[i]
                next_f2 = nd[i+1][0] if i < n - 1 else 0.0
                w2 = f2 - next_f2
                h3 = r3 - f3
                hv += w2 * h3
            return hv
        
        def hv_3d(pts, ref):
            if not pts:
                return 0.0
            pts = sorted(pts, key=lambda x: -x[0])
            n = len(pts)
            hv = 0.0
            r1, r2, r3 = ref
            for i in range(n):
                f1 = pts[i][0]
                next_f1 = pts[i+1][0] if i < n - 1 else 0.0
                w1 = f1 - next_f1
                subset = pts[:i+1]
                pts_2d = [(p[1], p[2]) for p in subset]
                hv2 = hv_2d(pts_2d, (r2, r3))
                hv += w1 * hv2
            return hv
        
        def marginal_hv(pts, ref, idx):
            total = hv_3d(pts, ref)
            reduced = pts[:idx] + pts[idx+1:]
            minus = hv_3d(reduced, ref)
            return total - minus
        
        # Initial stratified sampling
        n_init = 30
        for i in range(n_init):
            x = (lo + (hi - lo) * (np.arange(self.dim) + 1) / (self.dim + 1)) + rng.random(self.dim) * 0.1
            x = np.clip(x, lo, hi)
            f = problem(x)
            archive_X.append(x)
            archive_F.append(f)
            evals += 1
        
        # Additional random init for diversity
        for i in range(10):
            x = lo + (hi - lo) * rng.random(self.dim)
            f = problem(x)
            archive_X.append(x)
            archive_F.append(f)
            evals += 1
        
        while evals < self.budget:
            nd_idx = compute_nd_indices(archive_F)
            if len(nd_idx) == 0:
                x = lo + (hi - lo) * rng.random(self.dim)
                f = problem(x)
                archive_X.append(x)
                archive_F.append(f)
                evals += 1
                continue
            
            nd_points = [tuple(archive_F[i]) for i in nd_idx]
            
            # Compute marginal HV for each ND point
            marginals = []
            for k in range(len(nd_idx)):
                m = marginal_hv(nd_points, self.ref_point, k)
                marginals.append(m)
            
            # Find the top candidates by marginal HV
            top_k = min(3, len(marginals))
            top_indices = np.argsort(marginals)[::-1][:top_k]
            
            # For each top candidate, generate multiple children with different perturbations
            best_child_x = None
            best_child_score = -np.inf
            
            for local_idx in top_indices:
                global_idx = nd_idx[local_idx]
                parent_x = archive_X[global_idx]
                
                # Try multiple perturbation scales
                for sigma in [0.05, 0.1, 0.2, 0.3]:
                    # Generate several children
                    for _ in range(5):
                        child_x = parent_x + sigma * rng.standard_normal(self.dim)
                        child_x = np.clip(child_x, lo, hi)
                        
                        # Estimate potential: closer to parent in decision space,
                        # but we want to fill gaps in objective space
                        # Use a simple heuristic: prefer points that are not too close to existing ND points
                        min_dist = np.inf
                        for nd_g in nd_idx:
                            dist = np.linalg.norm(child_x - archive_X[nd_g])
                            min_dist = min(min_dist, dist)
                        
                        # Score: balance between proximity to high-marginal parent and diversity
                        score = marginals[local_idx] * 0.7 + min_dist * 0.3
                        if score > best_child_score:
                            best_child_score = score
                            best_child_x = child_x
            
            if best_child_x is None:
                x = lo + (hi - lo) * rng.random(self.dim)
            else:
                x = best_child_x
            
            f = problem(x)
            archive_X.append(x)
            archive_F.append(f)
            evals += 1
            
        return archive_X
