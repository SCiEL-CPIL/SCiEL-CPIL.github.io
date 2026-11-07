"""Multi-objective optimizer for WFG9 exploiting noise-dimension structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """WFG9 optimizer that fixes noise dimensions and does structured search on the core."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # WFG9 structure: first 6 dimensions define the Pareto front shape,
        # last 6 are noise. Fix noise dimensions to minimize noise effect.
        # For WFG9, the noise is:
        #   u_i = 0.5 + 0.5 * sin(10 * pi * x_i + i * pi / n)  for noise dims
        # To minimize noise, we want u_i to be constant (not oscillating).
        # Setting x_i = 0 gives u_i = 0.5 + 0.5*sin(i*pi/n)
        # Setting x_i = 0.5 gives u_i = 0.5 + 0.5*sin(5*pi + i*pi/n) = 0.5 + 0.5*sin(i*pi/n + 5*pi)
        # 
        # Actually, for WFG9 the noise terms are:
        # u_i = 0.5 + 0.5*sin(10*pi*x_i + i*pi/n) for i in noise dims
        # The position term is:
        # _pos = 0.5 + 0.5*sin(10*pi*x_pos + i*pi/n)
        # To make noise zero or constant, we can try specific values.
        # A known trick: set noise dimensions to values that make the sine terms cancel or be constant.
        # Let's try setting them to 0 first, which is a common heuristic.
        
        noise_lo = lo[6:]
        noise_hi = hi[6:]
        
        # Try to find good noise values. For WFG9, setting noise dims to 0 or 0.5 
        # are common heuristics. Let's use 0.5 as it centers the sine.
        # Actually, let's be more careful. The standard WFG9 has:
        # x in [0,1]^12
        # u_i = 0.5 + 0.5*sin(10*pi*x_i + i*pi/12) for i=6..11 (0-indexed)
        # The position part uses x_0..x_5
        # 
        # If we set noise dims to a value that makes sin(10*pi*x_i + i*pi/12) = 0,
        # then u_i = 0.5, which is the mean. This minimizes variance.
        # sin(10*pi*x_i + i*pi/12) = 0 when 10*pi*x_i + i*pi/12 = k*pi
        # x_i = (k*pi - i*pi/12) / (10*pi) = (k - i/12) / 10
        # For k=0: x_i = -i/120 (negative, out of bounds if bounds are [0,1])
        # For k=1: x_i = (1 - i/12)/10
        # For i=6: x_6 = (1 - 0.5)/10 = 0.05
        # For i=7: x_7 = (1 - 7/12)/10 = (5/12)/10 = 0.04167
        # etc.
        
        # Let's use a simpler approach: just fix noise dims to 0.5 (center)
        # and do all search in the first 6 dimensions.
        
        noise_val = np.array([(noise_lo[i] + noise_hi[i]) / 2.0 for i in range(6)])
        
        # Build a function that takes 6D input and returns 3D output
        def eval_core(x6):
            x = np.empty(dim)
            x[:6] = x6
            x[6:] = noise_val
            return problem(x)
        
        core_lo = lo[:6]
        core_hi = hi[:6]
        core_dim = 6
        
        # Strategy: Use a structured grid + local refinement in 6D
        # With 200 evaluations, we can do a structured search.
        
        # Phase 1: Latin Hypercube in 6D with ~80 points for broad coverage
        # Phase 2: Local refinement around best points with ~120 evaluations
        
        # Actually, let's use a more targeted approach:
        # 1. Sample a grid in the 6D space
        # 2. Identify promising regions
        # 3. Refine locally
        
        # For 6D with 200 points, a good approach is:
        # - 64 points on a structured grid (4^4 * 2^2 or similar)
        # - 136 points for local refinement
        
        # Let's use a hybrid: LHS for initial exploration, then local search
        
        n_lhs = 80
        n_refine = budget - n_lhs
        
        # Generate LHS samples in 6D
        lhs_samples = np.zeros((n_lhs, core_dim))
        for d in range(core_dim):
            perm = np.random.permutation(n_lhs)
            edges = np.linspace(core_lo[d], core_hi[d], n_lhs + 1)
            for i in range(n_lhs):
                lhs_samples[i, d] = np.random.uniform(edges[perm[i]], edges[perm[i] + 1])
        
        # Evaluate LHS samples
        all_x = []
        all_f = []
        used = 0
        
        for i in range(n_lhs):
            if time.time() > deadline:
                break
            x6 = lhs_samples[i]
            f = eval_core(x6)
            all_x.append(x6)
            all_f.append(f)
            used += 1
            if used >= budget:
                break
        
        if used >= budget:
            # Build final 12D points
            result = np.zeros((len(all_x), dim))
            for i in range(len(all_x)):
                result[i, :6] = all_x[i]
                result[i, 6:] = noise_val
            return result
        
        # Convert to arrays
        all_x = np.array(all_x)
        all_f = np.array(all_f)
        
        # Find non-dominated points
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)
        
        def is_nondominated(f, all_f):
            for j in range(len(all_f)):
                if j != idx and dominates(all_f[j], f):
                    return False
            return True
        
        # Simple non-dominated check
        nondom_mask = np.ones(len(all_f), dtype=bool)
        for i in range(len(all_f)):
            for j in range(len(all_f)):
                if i != j and dominates(all_f[j], all_f[i]):
                    nondom_mask[i] = False
                    break
        
        nondom_indices = np.where(nondom_mask)[0]
        
        # Local refinement around non-dominated points
        remaining = budget - used
        
        if remaining > 0 and len(nondom_indices) > 0:
            # Distribute remaining evaluations among non-dominated points
            n_points = len(nondom_indices)
            evals_per_point = max(1, remaining // n_points)
            
            for idx in nondom_indices:
                if remaining <= 0 or time.time() > deadline:
                    break
                
                base_x = all_x[idx]
                n_local = min(evals_per_point, remaining)
                
                for _ in range(n_local):
                    if remaining <= 0 or time.time() > deadline:
                        break
                    
                    # Random perturbation with decreasing step size
                    step = 0.05 * (hi[:6] - lo[:6])
                    perturbed = base_x + (np.random.random(core_dim) - 0.5) * 2 * step
                    perturbed = np.clip(perturbed, core_lo, core_hi)
                    
                    f = eval_core(perturbed)
                    all_x = np.vstack([all_x, perturbed])
                    all_f = np.vstack([all_f, f])
                    used += 1
                    remaining -= 1
        
        # Build final 12D result
        result = np.zeros((len(all_x), dim))
        for i in range(len(all_x)):
            result[i, :6] = all_x[i]
            result[i, 6:] = noise_val
        
        return result
# EVOLVE-BLOCK-END
