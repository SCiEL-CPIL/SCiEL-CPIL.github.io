"""Portfolio search optimizer for BBOB f20 (Schwefel) with explicit deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-strategy portfolio optimizer for BBOB f20 (Schwefel) on [-5, 5]^5.
    Phase 1: Round-robin between multiple (1+1)-ES and a CMA-ES for global exploration.
    Phase 2: Focused (1+1)-ES polish around the best point found.
    Explicit deadline guard to stay well within the 330s wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # 20s safety margin before 330s kill
        total_budget = self.budget
        dim = self.dim

        best_x = None
        best_fx = float('inf')
        calls_used = 0

        def try_func(x):
            nonlocal calls_used, best_fx, best_x
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        def time_ok():
            return time.time() < deadline

        def budget_ok():
            return calls_used < total_budget

        # --- Phase 1: Global exploration with portfolio ---
        # Use round-robin between multiple (1+1)-ES with different seeds
        # and a CMA-ES for better global search

        n_agents = 6
        rng_seeds = [1000 + i * 7919 for i in range(n_agents)]
        agents = []
        for i in range(n_agents):
            rng = np.random.default_rng(rng_seeds[i])
            x = rng.uniform(-5, 5, dim)
            fx = try_func(x)
            sigma = 2.0
            agents.append({'x': x.copy(), 'fx': fx, 'sigma': sigma, 'rng': rng})

        # CMA-ES agent (scipy) - run in chunks
        cma_rng = np.random.default_rng(42)
        cma_x0 = cma_rng.uniform(-5, 5, dim)
        cma_fx = try_func(cma_x0)
        cma_best = {'x': cma_x0.copy(), 'fx': cma_fx}

        # Phase 1 budget: use ~40% of total for exploration
        phase1_budget = int(total_budget * 0.40)
        phase1_calls = calls_used
        max_phase1_calls = phase1_budget

        # Round-robin between agents
        agent_idx = 0
        while phase1_calls < max_phase1_calls and time_ok() and budget_ok():
            agent = agents[agent_idx % n_agents]
            rng = agent['rng']
            x = agent['x']
            sigma = agent['sigma']

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = try_func(y)
            phase1_calls += 1

            if fy < agent['fx']:
                agent['x'] = y.copy()
                agent['fx'] = fy
                agent['sigma'] = min(sigma * 1.25, 5.0)
            else:
                agent['sigma'] = max(sigma * 0.85, 1e-6)

            agent_idx += 1

            # Every 30 calls, try a CMA-ES step (cheap local optimization)
            if agent_idx % 30 == 0 and time_ok() and budget_ok():
                try:
                    res = minimize(
                        func, cma_best['x'], method='Nelder-Mead',
                        options={'maxiter': 5, 'xatol': 1e-4, 'fatol': 1e-6}
                    )
                    if res.fun < cma_best['fx']:
                        cma_best['x'] = res.x.copy()
                        cma_best['fx'] = res.fun
                except Exception:
                    pass

        # --- Phase 2: Focused exploitation around the best point ---
        if best_x is not None and time_ok() and budget_ok():
            # Start from the best point found
            x = best_x.copy()
            fx = best_fx
            rng = np.random.default_rng(777)
            sigma = 0.5  # Start with moderate sigma

            # Adaptive (1+1)-ES with restart on stagnation
            no_improve = 0
            max_no_improve = 30
            phase2_calls = 0
            phase2_max = total_budget - calls_used

            while phase2_calls < phase2_max and time_ok() and budget_ok():
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = try_func(y)
                phase2_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.2, 3.0)
                    no_improve = 0
                else:
                    sigma = max(sigma * 0.9, 1e-7)
                    no_improve += 1

                # If stuck, try a larger jump (restart around best)
                if no_improve >= max_no_improve:
                    # Try a few random restarts around the best point
                    for _ in range(3):
                        if not time_ok() or not budget_ok():
                            break
                        y = np.clip(best_x + 1.0 * rng.standard_normal(dim), -5, 5)
                        fy = try_func(y)
                        if fy < fx:
                            x, fx = y, fy
                            no_improve = 0
                            break
                    no_improve = 0
                    sigma = 1.0

        # --- Phase 3: Fine polish with very small steps ---
        if best_x is not None and time_ok() and budget_ok():
            x = best_x.copy()
            fx = best_fx
            rng = np.random.default_rng(999)
            sigma = 0.05
            polish_calls = 0
            polish_max = min(50, total_budget - calls_used)

            for _ in range(polish_max):
                if not time_ok() or not budget_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = try_func(y)
                polish_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 0.95
                else:
                    sigma = max(sigma * 0.9, 1e-8)

        # Fallback
        if best_x is None:
            x = np.zeros(dim)
            best_fx = try_func(x)
            best_x = x

        return best_x, best_fx
# EVOLVE-BLOCK-END
