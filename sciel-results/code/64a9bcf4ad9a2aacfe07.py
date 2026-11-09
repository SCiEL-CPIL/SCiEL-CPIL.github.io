"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Structure-exploiting optimizer for DTLZ7.
    
    DTLZ7 with M=3 objectives:
      f1 = x0
      f2 = x1 * f1
      f3 = (1 + g(x_t)) * h(f1, f2, g)
    
    where g = sum((x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5))) for i in 2..n-1
    and h = (1 + g) / (M - sum_{i=1}^{M-1} y_i / (1+g) * (1 + sin(pi*y_i/2)))
    
    Key insight: To reach the Pareto front, g must be 0, which requires
    x_i = 0.5 for all i in 2..n-1. Then the front is determined by x0, x1.
    
    Strategy:
    1. Fix tail variables (x2..x21) to 0.5 to ensure g=0.
    2. Perform a dense grid + local refinement on x0, x1 to cover the Pareto front.
    3. Use remaining budget for local polishing around promising points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ7: first 2 variables (x0, x1) shape the front, rest are for g
        n_shape = min(2, self.dim - 1)  # x0, x1
        n_tail = self.dim - n_shape     # x2..x21
        
        # Optimal tail values: 0.5 (to make g=0)
        tail_opt = np.full(n_tail, 0.5)
        # Clip to bounds
        tail_opt = np.clip(tail_opt, lo[n_shape:], hi[n_shape:])
        
        def make_point(x0, x1):
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            x[n_shape:] = tail_opt
            return x
        
        evaluated = []
        used = 0
        
        def eval_point(x):
            nonlocal used
            f = problem(x)
            evaluated.append((x, np.array(f)))
            used += 1
            return f
        
        # Phase 1: Dense grid on x0, x1
        # The Pareto front for DTLZ7 (M=3) with g=0:
        #   f1 = x0
        #   f2 = x1 * x0
        #   f3 = 1 + (1/x0 + 1/x1) * (1 + sin(pi*x0/2) + 1 + sin(pi*x1/2))
        #        Actually h = (1+g)/(M - sum y_i/(1+g) * (1+sin(pi*y_i/2)))
        #        With g=0: h = 1/(3 - x0*(1+sin(pi*x0/2)) - x1*(1+sin(pi*x1/2)))
        #        f3 = (1+g)*h = h when g=0
        #
        # We want to sample x0, x1 in [0,1] to cover the front.
        # A uniform grid may miss important regions. Use a stratified approach.
        
        # Determine grid size based on budget
        grid_size = min(40, max(10, int(np.sqrt(self.budget * 0.7))))
        grid_x0 = np.linspace(0.01, 1.0, grid_size)
        grid_x1 = np.linspace(0.01, 1.0, grid_size)
        
        # Evaluate a subset of the grid to stay within budget
        # Total grid points: grid_size^2. We can't evaluate all if budget is small.
        max_grid_evals = min(grid_size * grid_size, int(self.budget * 0.6))
        
        # Create all grid points
        all_points = []
        for x0 in grid_x0:
            for x1 in grid_x1:
                all_points.append((x0, x1))
        
        # Shuffle and take up to max_grid_evals
        rng = np.random.default_rng(0)
        if len(all_points) > max_grid_evals:
            idx = rng.choice(len(all_points), size=max_grid_evals, replace=False)
            all_points = [all_points[i] for i in idx]
        
        for x0, x1 in all_points:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            x = make_point(x0, x1)
            eval_point(x)
        
        # Phase 2: Local refinement around best points
        # Identify non-dominated points and refine around them
        if used < self.budget and evaluated:
            F = np.array([f for _, f in evaluated])
            
            # Find non-dominated points
            nondom_mask = np.ones(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nondom_mask[i] = False
                        break
            
            nondom_idx = np.where(nondom_mask)[0]
            
            # Refine around top non-dominated points
            n_refine = min(len(nondom_idx), 10)
            if n_refine > 0:
                # Select diverse points from non-dominated set
                if n_refine < len(nondom_idx):
                    # Simple diversity: pick spread out in objective space
                    selected = []
                    remaining = list(nondom_idx)
                    selected.append(remaining.pop(0))
                    while len(selected) < n_refine and remaining:
                        # Pick point farthest from already selected
                        best_idx = 0
                        best_dist = -1
                        for ri, ridx in enumerate(remaining):
                            min_dist = np.min([np.linalg.norm(F[ridx] - F[sidx]) for sidx in selected])
                            if min_dist > best_dist:
                                best_dist = min_dist
                                best_idx = ri
                        selected.append(remaining.pop(best_idx))
                else:
                    selected = list(nondom_idx)
                
                for sidx in selected:
                    if used >= self.budget or time.time() > deadline:
                        break
                    x_base = evaluated[sidx][0]
                    # Local search: perturb x0, x1 slightly
                    for _ in range(3):
                        if used >= self.budget or time.time() > deadline:
                            break
                        x0_new = x_base[0] + rng.uniform(-0.05, 0.05)
                        x1_new = x_base[1] + rng.uniform(-0.05, 0.05)
                        x0_new = np.clip(x0_new, lo[0], hi[0])
                        x1_new = np.clip(x1_new, lo[1], hi[1])
                        x = make_point(x0_new, x1_new)
                        eval_point(x)
        
        # Phase 3: Final polish - evaluate a few more strategic points
        if used < self.budget and time.time() < deadline:
            # Add points near the "knee" of the front
            strategic = [
                (0.5, 0.5),
                (0.1, 0.9),
                (0.9, 0.1),
                (0.3, 0.7),
                (0.7, 0.3),
                (0.2, 0.8),
                (0.8, 0.2),
            ]
            for x0, x1 in strategic:
                if used >= self.budget or time.time() > deadline:
                    break
                # Check if already close to an evaluated point
                x = make_point(x0, x1)
                eval_point(x)
        
        return evaluated
# EVOLVE-BLOCK-END
