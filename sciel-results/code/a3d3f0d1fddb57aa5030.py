"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio of (1+1)-ES restarts with simple adaptive step size.

    Strategy:
      1. Multiple (1+1)-ES restarts with diverse random initializations.
      2. Simple, robust step-size adaptation (1.5x on success, 0.87x on failure).
      3. Equal budget allocation for restarts to ensure diversity.
      4. Final local refinement phase using the best point found.
      5. Explicit wall-clock deadline guard.
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

        # Reserve calls for final refinement
        reserve = min(80, budget // 4)
        
        # Phase 1: Portfolio of (1+1)-ES restarts
        n_restarts = 5
        alloc_per_restart = (budget - reserve) // n_restarts
        
        # Diverse initializations - include Schwefel optimum neighborhood
        schwefel_opt = np.full(dim, 4.209687)
        init_points = [
            rng.uniform(-5, 5, dim),
            rng.uniform(-5, 5, dim),
            rng.uniform(-5, 5, dim),
            schwefel_opt + 0.5 * rng.standard_normal(dim),
            rng.uniform(-3, 3, dim),
        ]
        
        # Diverse initial sigmas
        initial_sigmas = [2.0, 1.5, 3.0, 0.5, 1.0]

        for r in range(n_restarts):
            if call_count[0] >= budget - reserve:
                break
            if time.time() > deadline - 20:
                break

            x = init_points[r].copy()
            x = np.clip(x, -5.0, 5.0)
            fx = make_call(x)
            if fx is None:
                break
            update_best(x, fx)

            sigma = initial_sigmas[r]
            
            # Simple (1+1)-ES loop with 1/5 rule adaptation
            success_count = 0
            total_count = 0
            
            for step in range(alloc_per_restart):
                if call_count[0] >= budget - reserve:
                    break
                if time.time() > deadline - 20:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = make_call(y)
                if fy is None:
                    break

                total_count += 1
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    success_count += 1
                else:
                    pass

                # 1/5 rule: adjust sigma based on recent success rate
                if total_count % 10 == 0 and total_count > 0:
                    success_rate = success_count / total_count
                    if success_rate > 0.6:
                        sigma *= 1.3
                    elif success_rate < 0.4:
                        sigma *= 0.7
                    success_count = 0
                    total_count = 0

                # Keep sigma within bounds
                sigma = float(np.clip(sigma, 0.01, 10.0))

        # Phase 2: Final local refinement around the best point using Nelder-Mead style
        if best_x is not None and call_count[0] < budget - 5 and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            remaining_calls = budget - call_count[0] - 5
            
            # Use coordinate descent + random perturbation hybrid
            for _ in range(remaining_calls):
                if call_count[0] >= budget - 5:
                    break
                if time.time() > deadline - 5:
                    break

                # Alternate between coordinate search and random perturbation
                if _ % 3 == 0:
                    # Coordinate descent step
                    for d in range(dim):
                        if call_count[0] >= budget - 5:
                            break
                        step_size = sigma * (1 if _ % 2 == 0 else -1)
                        y = x.copy()
                        y[d] += step_size
                        y = np.clip(y, -5.0, 5.0)
                        fy = make_call(y)
                        if fy is None:
                            break
                        if fy < fx:
                            x, fx = y, fy
                            update_best(x, fx)
                        else:
                            # Try opposite direction
                            y2 = x.copy()
                            y2[d] -= step_size
                            y2 = np.clip(y2, -5.0, 5.0)
                            fy2 = make_call(y2)
                            if fy2 is None:
                                break
                            if fy2 < fx:
                                x, fx = y2, fy2
                                update_best(x, fx)
                else:
                    # Random perturbation
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                    fy = make_call(y)
                    if fy is None:
                        break

                    if fy < fx:
                        x, fx = y, fy
                        update_best(x, fx)
                        sigma *= 1.1
                    else:
                        sigma *= 0.85

                sigma = float(np.clip(sigma, 0.001, 1.0))

        return best_x, best_f
# EVOLVE-BLOCK-END
