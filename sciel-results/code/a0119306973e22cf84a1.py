"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with periodic best-so-far restart and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Multi-start: maintain K independent ES trajectories
        K = 5  # number of concurrent agents
        # Initialize agents at different positions
        agents = []
        for i in range(K):
            if i == 0:
                x = np.zeros(dim)  # zero-centered prior
            else:
                x = rng.uniform(-4, 4, dim)
            fx = func(x)
            sigma = 2.0 / (i + 1)  # varying initial step sizes
            agents.append({'x': x, 'fx': fx, 'sigma': sigma})

        best_x = min(agents, key=lambda a: a['fx'])['x'].copy()
        best_fx = min(agents, key=lambda a: a['fx'])['fx']
        calls_made = K

        # Periodic restart: every 100 calls, restart worst agent at best position + noise
        restart_interval = 100
        calls_since_restart = 0

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Select agent to evolve (round-robin)
            idx = calls_made % K
            agent = agents[idx]

            # Generate trial point
            y = np.clip(agent['x'] + agent['sigma'] * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1

            # Update agent
            if fy < agent['fx']:
                agent['x'] = y
                agent['fx'] = fy
                agent['sigma'] *= 1.3
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
            else:
                agent['sigma'] *= 0.85
                # Prevent sigma from becoming too small
                agent['sigma'] = max(agent['sigma'], 1e-4)

            calls_since_restart += 1

            # Periodic restart of worst agent
            if calls_since_restart >= restart_interval:
                calls_since_restart = 0
                # Find worst agent
                worst_idx = max(range(K), key=lambda i: agents[i]['fx'])
                # Restart worst agent near best position with small perturbation
                agents[worst_idx]['x'] = np.clip(best_x + 0.5 * rng.standard_normal(dim), -5, 5)
                agents[worst_idx]['fx'] = func(agents[worst_idx]['x'])
                calls_made += 1
                agents[worst_idx]['sigma'] = 1.0

            if calls_made >= budget:
                break

        # Final polish phase: Nelder-Mead from best point
        if calls_made < budget - 50 and time.time() < deadline - 10:
            try:
                from scipy.optimize import minimize
                x0 = best_x.copy()
                result = minimize(func, x0, method='Nelder-Mead',
                                  options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12,
                                           'adaptive': True})
                if result.success or result.fun < best_fx:
                    best_x = np.clip(result.x, -5, 5)
                    best_fx = result.fun
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
