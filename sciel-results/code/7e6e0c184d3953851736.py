# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Deterministic multi-start local search for BBOB-style f13 (Sharp Ridge).
    
    Strategy:
    1. Coarse grid of initial points to find promising basins (deterministic, no RNG needed).
    2. Coordinate-wise line search (Powell-like) with adaptive step sizes per axis.
       This exploits the ridge structure: large steps along flat directions, small steps
       along sharp directions, avoiding the isotropic step-size bottleneck of (1+1)-ES.
    3. Golden-section 1D refinement along each axis for precise convergence.
    4. Stall detection triggers a jump to the next grid point.
    5. Strict time and budget guards.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget

        calls_made = [0]
        best_f = np.inf
        best_x = np.zeros(n)

        def wrapped_func(x):
            nonlocal best_f, best_x
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            calls_made[0] += 1
            if calls_made[0] > budget:
                raise Exception("Budget exceeded")
            
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Coarse grid of initial points
        # Deterministic grid in [-3, 3] with spacing 2.0 -> points at -3, -1, 1, 3
        # Plus the origin and a few extra points for coverage
        grid_points = []
        
        # Center
        grid_points.append(np.zeros(n))
        
        # 1D shifts along each axis
        for d in range(n):
            for val in [-3.0, -1.5, 1.5, 3.0]:
                pt = np.zeros(n)
                pt[d] = val
                grid_points.append(pt)
        
        # 2D shifts along pairs of axes (a subset for budget reasons)
        for d1 in range(n):
            for d2 in range(d1 + 1, n):
                for v1 in [-2.0, 2.0]:
                    for v2 in [-2.0, 2.0]:
                        pt = np.zeros(n)
                        pt[d1] = v1
                        pt[d2] = v2
                        grid_points.append(pt)
        
        # Sort by distance from center to prioritize nearby points
        grid_points = np.array(grid_points)
        dists = np.linalg.norm(grid_points, axis=1)
        order = np.argsort(dists)
        grid_points = grid_points[order]

        # Evaluate initial grid points to find a good starting point
        init_budget = min(30, budget // 4)  # Spend at most 25% of budget on initialization
        for i in range(min(init_budget, len(grid_points))):
            if calls_made[0] >= budget:
                break
            if time.time() > deadline:
                break
            wrapped_func(grid_points[i])

        if best_x is None or np.isinf(best_f):
            # Fallback
            wrapped_func(np.zeros(n))

        x = best_x.copy()
        fx = best_f

        # Phase 2: Coordinate-wise line search with adaptive steps
        # Step sizes per axis, initialized based on local curvature estimate
        steps = np.ones(n) * 1.0

        # We'll cycle through grid points as restart locations if we stall
        grid_idx = min(init_budget, len(grid_points))  # Start after the initial grid points
        no_improvement = 0
        max_no_improvement = 30  # Steps without improvement before restarting

        # Golden section ratio
        gr = (np.sqrt(5) - 1) / 2  # ~0.618

        while calls_made[0] < budget:
            if time.time() > deadline:
                break

            improved_this_cycle = False

            for d in range(n):
                if calls_made[0] >= budget:
                    break
                if time.time() > deadline:
                    break

                # 1D search along axis d
                # We'll do a few iterations of golden-section-like refinement
                # First, try to find a direction of improvement
                x_plus = x.copy()
                x_plus[d] = np.clip(x[d] + steps[d], -5, 5)
                x_minus = x.copy()
                x_minus[d] = np.clip(x[d] - steps[d], -5, 5)

                fx_plus = wrapped_func(x_plus)
                fx_minus = wrapped_func(x_minus)

                if fx_plus < fx and fx_plus <= fx_minus:
                    # Move in positive direction
                    x = x_plus
                    fx = fx_plus
                    improved_this_cycle = True
                    no_improvement = 0
                    # Expand step
                    steps[d] = min(steps[d] * 1.5, 4.0)
                    # Do a quick golden-section refinement along this axis
                    # We'll do 2-3 iterations
                    a = x[d] - steps[d]
                    b = x[d] + steps[d]
                    a = max(a, -5.0)
                    b = min(b, 5.0)
                    if b - a < 1e-6:
                        continue
                    for _ in range(2):
                        if calls_made[0] >= budget:
                            break
                        if time.time() > deadline:
                            break
                        c = b - gr * (b - a)
                        d_val = a + gr * (b - a)
                        x_c = x.copy()
                        x_c[d] = c
                        x_d = x.copy()
                        x_d[d] = d_val
                        fx_c = wrapped_func(x_c)
                        fx_d = wrapped_func(x_d)
                        if fx_c < fx_d:
                            b = d_val
                        else:
                            a = c
                        # Update x to the better of the two
                        if fx_c < fx_d:
                            x[d] = c
                            fx = fx_c
                        else:
                            x[d] = d_val
                            fx = fx_d
                        if b - a < 1e-5:
                            break
                    # Shrink step for next time
                    steps[d] = max(steps[d] * 0.8, 0.1)

                elif fx_minus < fx and fx_minus <= fx_plus:
                    # Move in negative direction
                    x = x_minus
                    fx = fx_minus
                    improved_this_cycle = True
                    no_improvement = 0
                    steps[d] = min(steps[d] * 1.5, 4.0)
                    # Golden-section refinement
                    a = x[d] - steps[d]
                    b = x[d] + steps[d]
                    a = max(a, -5.0)
                    b = min(b, 5.0)
                    if b - a < 1e-6:
                        continue
                    for _ in range(2):
                        if calls_made[0] >= budget:
                            break
                        if time.time() > deadline:
                            break
                        c = b - gr * (b - a)
                        d_val = a + gr * (b - a)
                        x_c = x.copy()
                        x_c[d] = c
                        x_d = x.copy()
                        x_d[d] = d_val
                        fx_c = wrapped_func(x_c)
                        fx_d = wrapped_func(x_d)
                        if fx_c < fx_d:
                            b = d_val
                        else:
                            a = c
                        if fx_c < fx_d:
                            x[d] = c
                            fx = fx_c
                        else:
                            x[d] = d_val
                            fx = fx_d
                        if b - a < 1e-5:
                            break
                    steps[d] = max(steps[d] * 0.8, 0.1)

                else:
                    # No improvement in either direction
                    no_improvement += 1
                    steps[d] = max(steps[d] * 0.5, 0.05)

            if improved_this_cycle:
                no_improvement = 0
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # Stagnation: restart from next grid point
                if grid_idx < len(grid_points):
                    x = grid_points[grid_idx].copy()
                    fx = wrapped_func(x)
                    grid_idx += 1
                    steps = np.ones(n) * 1.0
                    no_improvement = 0
                else:
                    # No more grid points; try a random perturbation
                    d_rand = np.random.randint(n)
                    jump = np.random.uniform(-1.0, 1.0)
                    x[d_rand] = np.clip(x[d_rand] + jump, -5, 5)
                    fx = wrapped_func(x)
                    steps[d_rand] = 1.0
                    no_improvement = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
