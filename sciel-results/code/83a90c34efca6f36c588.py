"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex optimizer with multiple restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _nelder_mead(self, func, x0, budget, rng, deadline, initial_sigma=1.0):
        """Run Nelder-Mead simplex optimization."""
        n = self.dim
        # Initialize simplex
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0
        for i in range(n):
            simplex[i + 1] = x0.copy()
            simplex[i + 1][i] += initial_sigma

        # Evaluate initial simplex
        fx = np.array([func(np.clip(v, -5, 5)) for v in simplex])
        calls = n + 1

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        best_idx = 0
        best_x = simplex[0].copy()
        best_f = fx[0]

        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Sort simplex by function value
            order = np.argsort(fx)
            simplex = simplex[order]
            fx = fx[order]

            # Update best
            if fx[0] < best_f:
                best_f = fx[0]
                best_x = simplex[0].copy()

            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1

            if f_r < fx[0]:
                # Try expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    fx[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fx[-1] = f_r
            elif f_r < fx[-2]:
                # Accept reflection
                simplex[-1] = x_r
                fx[-1] = f_r
            else:
                # Contraction
                if f_r < fx[-1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls += 1

                if f_c <= min(f_r, fx[-1]):
                    simplex[-1] = x_c
                    fx[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fx[i] = func(simplex[i])
                        calls += 1
                        if calls >= budget:
                            break

            # Check convergence
            if np.max(fx) - np.min(fx) < 1e-10:
                break

        return best_x, best_f, calls

    def __call__(self, func):
        rng = np.random.default_rng(0)
        deadline = time.time() + 280  # Safety margin before 300s limit

        # Multiple restarts with different initial points and sigma values
        # Use budget efficiently across restarts
        n_restarts = 5
        calls_per_restart = self.budget // n_restarts

        best_x = None
        best_f = np.inf
        total_calls = 0

        for restart in range(n_restarts):
            # Check deadline
            if time.time() > deadline:
                break

            # Generate initial point
            if restart == 0:
                x0 = np.zeros(self.dim)
            else:
                x0 = rng.uniform(-4, 4, self.dim)

            # Initial sigma based on restart
            sigma = max(0.5, 2.0 / (restart + 1))

            # Run Nelder-Mead
            x_opt, f_opt, calls = self._nelder_mead(
                func, x0, calls_per_restart - total_calls, rng, deadline, sigma
            )
            total_calls += calls

            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

            # If we've used all budget, break
            if total_calls >= self.budget:
                break

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
