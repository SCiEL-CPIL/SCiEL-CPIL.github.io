"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead simplex method for BBOB f20 (Schwefel).

    Strategy:
      1. Initialize a diverse set of starting points (center, random, corners).
      2. Run Nelder-Mead simplex optimization from each start point.
      3. Use a shared budget and time deadline to manage the multi-start process.
      4. Return the best result found.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin

        call_count = [0]

        def make_call(x):
            if call_count[0] >= budget:
                return None
            call_count[0] += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            return float(func(x))

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def nelder_mead(func, x0, budget_remaining, deadline, call_count, update_best, best_state):
            """
            Deterministic Nelder-Mead simplex method.
            x0: initial point (dim,)
            Returns: best point and value found during this run.
            """
            dim = len(x0)
            
            # Initialize simplex: x0 and dim neighbors offset by 0.05 in each coordinate
            simplex = []
            simplex.append(x0.copy())
            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.05
                simplex.append(xi)
            
            # Evaluate initial simplex
            fvals = []
            for i in range(dim + 1):
                if call_count[0] >= budget:
                    break
                if time.time() > deadline:
                    break
                fi = make_call(simplex[i])
                if fi is None:
                    break
                fvals.append(fi)
                update_best(simplex[i], fi)
            
            if len(fvals) < dim + 1:
                return best_state[0], best_state[1]
            
            # Main Nelder-Mead loop
            while call_count[0] < budget and time.time() < deadline:
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i].copy() for i in order]
                fvals = [fvals[i] for i in order]
                
                # If already very good, break
                if fvals[0] < -1e6:
                    break
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                f_worst = fvals[-1]
                
                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - worst)
                x_ref = np.clip(x_ref, -5.0, 5.0)
                
                if call_count[0] >= budget or time.time() > deadline:
                    break
                f_ref = make_call(x_ref)
                if f_ref is None:
                    break
                update_best(x_ref, f_ref)
                
                if f_ref < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - worst)
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    
                    if call_count[0] >= budget or time.time() > deadline:
                        break
                    f_exp = make_call(x_exp)
                    if f_exp is None:
                        break
                    update_best(x_exp, f_exp)
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    # Simple reflection
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    rho = 0.5
                    x_con = centroid + rho * (worst - centroid)
                    x_con = np.clip(x_con, -5.0, 5.0)
                    
                    if call_count[0] >= budget or time.time() > deadline:
                        break
                    f_con = make_call(x_con)
                    if f_con is None:
                        break
                    update_best(x_con, f_con)
                    
                    if f_con < f_worst:
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            if call_count[0] >= budget or time.time() > deadline:
                                break
                            f_sh = make_call(simplex[i])
                            if f_sh is None:
                                break
                            update_best(simplex[i], f_sh)
                            fvals[i] = f_sh
                        break  # Exit loop if we had to shrink significantly
            
            return best_state[0], best_state[1]

        # Phase 1: Generate diverse starting points
        start_points = []
        
        # Center
        start_points.append(np.zeros(dim))
        
        # Random points
        for _ in range(5):
            start_points.append(rng.uniform(-4.0, 4.0, dim))
        
        # Some structured points
        for d in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[d] = sign * 3.0
                start_points.append(p)
        
        # Sort by distance to center to prioritize central regions
        start_points.sort(key=lambda x: np.linalg.norm(x))
        
        # Phase 2: Run Nelder-Mead from each start point
        for sp in start_points:
            if call_count[0] >= budget - 10:
                break
            if time.time() > deadline - 10:
                break
            
            nelder_mead(func, sp, budget - call_count[0], deadline, call_count, update_best, [best_x, best_f])
        
        return best_x, best_f
# EVOLVE-BLOCK-END
