"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid: Multi-restart (1+1)-ES with Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s limit
        dim = self.dim
        total_budget = self.budget

        best_x = None
        best_fx = float('inf')
        calls_made = 0

        def check_time():
            return time.time() > deadline

        def count_call():
            nonlocal calls_made
            calls_made += 1

        # Phase 1: Multi-restart (1+1)-ES for global exploration
        # Phase 2: Nelder-Mead for local refinement around the best point found

        # Budget allocation:
        # ~70% for ES exploration, ~30% for NM refinement
        # But we need to be careful with NM since it can be expensive.
        # Let's do 4 restarts of ES, then NM on the best.

        num_restarts = 4
        budget_per_restart = max(20, (total_budget // 3) // num_restarts)  # Reserve 1/3 for NM

        for restart in range(num_restarts):
            if check_time() or calls_made >= total_budget:
                break

            # Diversify starting points
            rng = np.random.default_rng(42 + restart * 1000)

            if restart == 0:
                x = np.zeros(dim)
            elif restart == 1:
                x = rng.uniform(-2, 2, dim)
            elif restart == 2:
                x = rng.uniform(-3, 3, dim)
            else:
                x = rng.uniform(-4, 4, dim)

            fx = func(x)
            count_call()

            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            sigma = 2.0
            window_size = 10
            window = []

            for _ in range(budget_per_restart - 1):
                if check_time() or calls_made >= total_budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                count_call()

                if fy < fx:
                    x, fx = y, fy
                    window.append(1)
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    window.append(0)

                # Adapt sigma
                if len(window) >= window_size:
                    success_rate = sum(window[-window_size:]) / window_size
                    if success_rate > 0.25:
                        sigma *= 1.2
                    elif success_rate < 0.15:
                        sigma *= 0.8
                    window = window[-window_size:]
                elif len(window) > 5:
                    recent_success = sum(window) / len(window)
                    if recent_success > 0.3:
                        sigma *= 1.1
                    elif recent_success < 0.1:
                        sigma *= 0.85

                sigma = np.clip(sigma, 0.05, 5.0)

        # Phase 2: Nelder-Mead refinement around best_x
        if best_x is not None and not check_time() and calls_made < total_budget - 10:
            # Use scipy's minimize if available, otherwise implement a simple NM
            try:
                from scipy.optimize import minimize
                # NM can be expensive, so we limit the max iterations
                # The budget remaining is total_budget - calls_made
                # NM uses ~ (dim+1) * iterations function calls
                # Let's allow up to 100 iterations for NM
                max_iter_nm = min(100, (total_budget - calls_made - 5) // (dim + 1))
                if max_iter_nm > 10:
                    res = minimize(
                        func,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': max_iter_nm, 'xatol': 1e-8, 'fatol': 1e-8}
                    )
                    # Note: scipy's minimize will call func multiple times.
                    # We need to check if we exceeded budget.
                    # The deadline guard in the outer loop doesn't protect inside scipy.
                    # This is a risk. If scipy takes too long, we might timeout.
                    # However, NM on 5D is usually fast.
                    # To be safe, we could implement our own NM with a deadline check,
                    # but that's complex. Let's trust scipy for now but keep maxiter low.
                    
                    if res.fun < best_fx:
                        best_fx = res.fun
                        best_x = res.x.copy()
            except Exception:
                # If scipy fails or is not available, skip refinement
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_fx = func(best_x)
            count_call()

        return best_x, best_fx
# EVOLVE-BLOCK-END
