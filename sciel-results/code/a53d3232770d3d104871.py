"""Direct-structured Nelder-Mead optimizer with boundary repair for BBOB f20 (Schwefel)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Nelder-Mead simplex optimizer with boundary error repair.
    Designed for BBOB f20 (Schwefel) on [-5,5]^5 with budget 500.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s kill
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        call_count = 0

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        def safe_call(x):
            nonlocal call_count
            if call_count >= self.budget:
                return None
            if time.time() > deadline:
                return None
            # Clip to box
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            call_count += 1
            update_best(x_clipped, fx)
            return x_clipped, fx

        # Phase 1: Multi-start Nelder-Mead
        # We use 5 distinct starting points to handle the rugged landscape
        n_starts = 5
        starts = []
        # One at origin
        starts.append(np.zeros(self.dim))
        # Others random but bounded
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4.0, 4.0, self.dim))

        # Allocate budget per start
        budget_per_start = self.budget // n_starts
        # Give a bit more to the first one if remainder exists
        remainder = self.budget % n_starts
        
        for i, x0 in enumerate(starts):
            if call_count >= self.budget:
                break
            if time.time() > deadline:
                break

            # Initialize simplex
            # Standard Nelder-Mead initialization
            simplex = []
            f_simplex = []
            
            # First vertex
            res = safe_call(x0)
            if res is None:
                break
            x_v, f_v = res
            simplex.append(x_v)
            f_simplex.append(f_v)

            # Other vertices: small perturbations
            delta = 0.5
            for j in range(self.dim):
                x_j = x_v.copy()
                x_j[j] += delta
                res = safe_call(x_j)
                if res is None:
                    break
                x_v, f_v = res
                simplex.append(x_v)
                f_simplex.append(f_v)
            
            if len(simplex) < self.dim + 1:
                continue

            simplex = np.array(simplex)
            f_simplex = np.array(f_simplex)

            # Nelder-Mead Parameters
            alpha = 1.0      # Reflection
            gamma = 2.0      # Expansion
            rho = 0.5        # Contraction
            sigma = 0.5      # Shrinkage
            
            max_iters = budget_per_start - (len(simplex)) # Remaining budget for this start
            # Ensure we don't exceed total budget
            max_iters = min(max_iters, self.budget - call_count)

            for it in range(max_iters):
                if time.time() > deadline:
                    break
                if call_count >= self.budget:
                    break

                # Sort simplex by function value
                order = np.argsort(f_simplex)
                simplex = simplex[order]
                f_simplex = f_simplex[order]

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # 1. Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                res = safe_call(x_r)
                if res is None:
                    break
                x_r, f_r = res

                if f_r < f_simplex[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    res = safe_call(x_e)
                    if res is None:
                        break
                    x_e, f_e = res
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    # Outside contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    res = safe_call(x_c)
                    if res is None:
                        break
                    x_c, f_c = res
                    if f_c < f_simplex[-1]:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrinkage
                        # Shrink all points except the best towards the best
                        for k in range(1, len(simplex)):
                            simplex[k] = simplex[0] + sigma * (simplex[k] - simplex[0])
                            res = safe_call(simplex[k])
                            if res is None:
                                break
                            simplex[k], f_simplex[k] = res
                        if res is None:
                            break

                # Error Repair: If any point in the simplex is outside the box (due to numerical drift or aggressive moves),
                # we repair it by clipping. This is the 'error_repair' operator.
                # Note: safe_call already clips before evaluation, but the simplex state might hold unclipped coordinates
                # if we were to track them differently. Here, we ensure the stored simplex points are valid.
                # Since safe_call returns clipped x, the simplex is always valid.
                # However, if we detect a "stagnation" or "boundary stick", we can nudge.
                # For this strategy, the repair is implicit in the clipping of safe_call.
                # To make it explicit for the hypothesis: if a reflection/contraction leads to a point on the boundary
                # and the gradient suggests going out, we might be stuck. 
                # A simple repair: if the worst point is on the boundary, try a small random perturbation inward.
                # But standard NM with clipping usually handles this.
                
                # Check for convergence
                f_range = f_simplex[-1] - f_simplex[0]
                if f_range < 1e-8:
                    break

        # Phase 2: Final Polish with a very tight Nelder-Mead or Coordinate Descent
        if best_x is not None and call_count < self.budget:
            # Use remaining budget for a final local refinement
            # Initialize a small simplex around best_x
            x0 = best_x.copy()
            simplex = []
            f_simplex = []
            
            res = safe_call(x0)
            if res is not None:
                simplex.append(res[0])
                f_simplex.append(res[1])
            
            delta = 0.05
            for j in range(self.dim):
                x_j = x0.copy()
                x_j[j] += delta
                res = safe_call(x_j)
                if res is not None:
                    simplex.append(res[0])
                    f_simplex.append(res[1])

            if len(simplex) == self.dim + 1:
                simplex = np.array(simplex)
                f_simplex = np.array(f_simplex)
                
                max_iters = self.budget - call_count
                for it in range(max_iters):
                    if time.time() > deadline:
                        break
                    if call_count >= self.budget:
                        break

                    order = np.argsort(f_simplex)
                    simplex = simplex[order]
                    f_simplex = f_simplex[order]

                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    x_r = centroid + 1.0 * (centroid - simplex[-1])
                    res = safe_call(x_r)
                    if res is None:
                        break
                    x_r, f_r = res

                    if f_r < f_simplex[0]:
                        x_e = centroid + 2.0 * (centroid - simplex[-1])
                        res = safe_call(x_e)
                        if res is None:
                            break
                        x_e, f_e = res
                        if f_e < f_r:
                            simplex[-1] = x_e
                            f_simplex[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            f_simplex[-1] = f_r
                    elif f_r < f_simplex[-2]:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                    else:
                        x_c = centroid + 0.5 * (simplex[-1] - centroid)
                        res = safe_call(x_c)
                        if res is None:
                            break
                        x_c, f_c = res
                        if f_c < f_simplex[-1]:
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            for k in range(1, len(simplex)):
                                simplex[k] = simplex[0] + 0.5 * (simplex[k] - simplex[0])
                                res = safe_call(simplex[k])
                                if res is None:
                                    break
                                simplex[k], f_simplex[k] = res
                            if res is None:
                                break

                    f_range = f_simplex[-1] - f_simplex[0]
                    if f_range < 1e-10:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
