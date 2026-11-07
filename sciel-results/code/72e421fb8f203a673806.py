import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # Deadline guard
        import time
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s margin

        # Center of the sphere: The lower bound (lo) is a good guess for WFG9
        # as f1 = sum(x) is minimized near 0.
        center = lo.copy()
        
        # Max radius: distance from center to the furthest corner (hi)
        max_radius = np.linalg.norm(hi - lo)
        
        # Non-dominated set
        nd_x = []
        nd_f = []
        
        def is_dominated(f_new, nd_f_list):
            for f in nd_f_list:
                if np.all(f <= f_new) and np.any(f < f_new):
                    return True
            return False

        def dominates(f_new, f_old):
            return np.all(f_new <= f_old) and np.any(f_new < f_old)

        # Phase 1: Global exploration with shrinking sphere (60% of budget)
        # Phase 2: Local refinement around non-dominated points (40% of budget)
        n_phase1 = int(0.6 * n)
        
        for t in range(n):
            # Check time
            if time.time() > deadline:
                break
                
            if t < n_phase1:
                # Phase 1: Shrinking sphere centered at lo
                progress = t / max(1, n_phase1 - 1)
                # Use a slower shrink: keep exploring with larger radius longer
                current_radius = max_radius * (1.0 - 0.5 * progress) + 0.05 * max_radius
                
                # Sample point in hyper-sphere
                direction = np.random.randn(dim)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                else:
                    direction = np.ones(dim) / np.sqrt(dim)
                    
                # Random distance (uniform in volume)
                u = np.random.random()
                r = current_radius * (u ** (1.0 / dim))
                
                x = center + direction * r
            else:
                # Phase 2: Local refinement around non-dominated points
                if len(nd_x) == 0:
                    # Fallback to sphere sampling if no non-dominated points
                    direction = np.random.randn(dim)
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction = direction / norm
                    else:
                        direction = np.ones(dim) / np.sqrt(dim)
                    u = np.random.random()
                    r = 0.1 * max_radius * (u ** (1.0 / dim))
                    x = center + direction * r
                else:
                    # Pick a random non-dominated point and perturb it
                    idx = np.random.randint(0, len(nd_x))
                    base_x = nd_x[idx]
                    # Gaussian perturbation with decreasing scale
                    progress2 = (t - n_phase1) / max(1, n - n_phase1)
                    sigma = 0.1 * max_radius * (1.0 - 0.5 * progress2)
                    x = base_x + np.random.randn(dim) * sigma

            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            # Evaluate
            f = np.array(problem(x))
            
            # Update non-dominated set
            if not is_dominated(f, nd_f):
                # Check if it dominates any existing
                new_nd_f = []
                new_nd_x = []
                for i in range(len(nd_f)):
                    if not dominates(f, nd_f[i]):
                        new_nd_f.append(nd_f[i])
                        new_nd_x.append(nd_x[i])
                
                nd_f.append(f)
                nd_x.append(x)
                
                nd_f = new_nd_f + [f]
                nd_x = new_nd_x + [x]

        return nd_x
