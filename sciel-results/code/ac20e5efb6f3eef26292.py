"""Rewritten optimizer for BBOB f7 (Step ellipsoidal) with PSO + CMA-ES polish."""
import numpy as np
import time


class Optimizer:
    """Particle Swarm Optimization with CMA-ES-style refinement for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(12345)

        # Global best tracking
        best_x = None
        best_f = float('inf')

        # Strategy:
        # 1. Particle Swarm Optimization (PSO) with inertia weighting
        #    - 20 particles, ~350 evaluations
        # 2. Polish the best with a refined (1+1)-ES using remaining budget

        num_particles = 20
        # Reserve ~150 evals for polish, use rest for PSO
        polish_budget = 150
        pso_budget = budget - polish_budget  # ~350

        # Initialize particles
        positions = rng.uniform(-5, 5, (num_particles, dim))
        velocities = rng.uniform(-2, 2, (num_particles, dim))
        personal_bests = positions.copy()
        personal_best_f = np.full(num_particles, float('inf'))

        # Inertia parameters
        w_start = 0.9
        w_end = 0.4
        c1 = 1.5  # Cognitive coefficient
        c2 = 1.5  # Social coefficient

        evals_used = 0

        # Evaluate initial positions
        for i in range(num_particles):
            if time.time() > deadline:
                break
            fi = func(positions[i])
            personal_best_f[i] = fi
            if fi < best_f:
                best_f = fi
                best_x = positions[i].copy()
            evals_used += 1

        global_best_x = best_x.copy() if best_x is not None else positions[0].copy()
        global_best_f = best_f

        # PSO main loop
        max_iterations = pso_budget - num_particles  # remaining evals for iterations
        for it in range(max_iterations):
            if time.time() > deadline or evals_used >= pso_budget:
                break

            # Linearly decrease inertia weight
            w = w_start + (w_end - w_start) * (it / max(1, max_iterations))

            for i in range(num_particles):
                if time.time() > deadline or evals_used >= pso_budget:
                    break

                r1 = rng.random()
                r2 = rng.random()

                # Update velocity
                velocities[i] = (w * velocities[i] +
                                 c1 * r1 * (personal_bests[i] - positions[i]) +
                                 c2 * r2 * (global_best_x - positions[i]))

                # Clamp velocity
                velocities[i] = np.clip(velocities[i], -2.0, 2.0)

                # Update position
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], -5, 5)

                # Evaluate
                fi = func(positions[i])
                evals_used += 1

                # Update personal best
                if fi < personal_best_f[i]:
                    personal_best_f[i] = fi
                    personal_bests[i] = positions[i].copy()

                    # Update global best
                    if fi < global_best_f:
                        global_best_f = fi
                        global_best_x = positions[i].copy()
                        best_f = fi
                        best_x = positions[i].copy()

        # Polish the best candidate with (1+1)-ES using remaining budget
        if best_x is not None and time.time() < deadline:
            x_polished, f_polished = self._es_refine(
                func, best_x, best_f,
                budget=polish_budget,
                rng=rng,
                deadline=deadline
            )
            if f_polished < best_f:
                best_f = f_polished
                best_x = x_polished

        if best_x is None:
            # Fallback
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            best_x = x
            best_f = fx

        return best_x, best_f

    def _es_refine(self, func, x, fx, budget, rng, deadline):
        """
        (1+1)-ES refinement with adaptive step size.
        """
        dim = len(x)
        best_x = x.copy()
        best_fx = fx
        sigma = 0.3  # Start with a moderate step for refinement

        calls_made = 0
        successes = 0
        total = 0
        window_size = 10

        for i in range(budget):
            if time.time() >= deadline:
                break

            # Generate trial point
            direction = rng.standard_normal(dim)
            y = np.clip(x + sigma * direction, -5, 5)
            fy = func(y)
            calls_made += 1

            # Update best
            if fy < best_fx:
                best_x = y.copy()
                best_fx = fy

            # 1/5 rule for step size adaptation
            if fy < fx:
                successes += 1
                x = y
                fx = fy

            total += 1

            # Adapt sigma every window_size steps
            if total % window_size == 0:
                rate = successes / total
                if rate > 1.0 / 5.0:
                    sigma *= 1.25
                elif rate < 1.0 / 5.0:
                    sigma *= 0.8
                sigma = np.clip(sigma, 1e-6, 1.5)
                successes = 0
                total = 0

        return best_x, best_fx
