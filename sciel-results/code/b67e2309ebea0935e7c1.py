"""Parallel Best-First Nelder-Mead with Latin Hypercube initialization for BBOB f14."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq


class Optimizer:
    """Parallel Best-First Nelder-Mead with LHS initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def budget_ok():
            return calls < budget

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def evaluate(x):
            nonlocal calls
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = func(x)
            calls += 1
            record(x, f)
            return x, f

        rng = np.random.default_rng(42)

        # Latin Hypercube Sampling for initial simplex centers
        def lhs_sample(n_samples, dim, rng):
            points = np.zeros((n_samples, dim))
            for d in range(dim):
                perm = rng.permutation(n_samples)
                for i in range(n_samples):
                    points[i, d] = (perm[i] + rng.random()) / n_samples * 10.0 - 5.0
            return points

        # Nelder-Mead simplex state
        class Simplex:
            def __init__(self, vertices, values):
                self.vertices = vertices  # list of (dim,) arrays
                self.values = values      # list of floats
                self.id = 0

            def potential(self):
                """Lower is better: how promising this simplex is for improvement."""
                # Use the gap between best and centroid as a proxy for improvement potential
                if len(self.vertices) < 2:
                    return np.inf
                centroid = np.mean(self.vertices[:-1], axis=0)
                best_val = min(self.values)
                # Potential: how much room there is to improve below the current best
                # A simplex with a low best value and high spread has more potential
                spread = max(self.values) - min(self.values)
                # We want to prioritize simplexes that might find lower values
                # Use negative best value as primary, spread as secondary
                return best_val - 0.1 * spread

            def step(self, func_eval, rng):
                """
                Perform one Nelder-Mead step on this simplex.
                Returns (new_vertex, new_value, action)
                action: 'reflect', 'expand', 'contract', 'shrink'
                """
                n = len(self.vertices)
                # Sort by value
                order = np.argsort(self.values)
                self.vertices = [self.vertices[i] for i in order]
                self.values = [self.values[i] for i in order]

                x_best = self.vertices[0]
                f_best = self.values[0]
                x_worst = self.vertices[-1]
                f_worst = self.values[-1]
                x_second_worst = self.vertices[-2]
                f_second_worst = self.values[-2]

                centroid = np.mean(self.vertices[:-1], axis=0)

                gamma = 1.0
                rho = 2.0
                sigma = 0.5
                shrink = 0.5

                x_r = centroid + gamma * (centroid - x_worst)
                x_r, f_r = func_eval(x_r)

                if f_r < f_best:
                    x_e = centroid + rho * (x_r - centroid)
                    x_e, f_e = func_eval(x_e)
                    if f_e < f_r:
                        self.vertices[-1] = x_e
                        self.values[-1] = f_e
                        return x_e, f_e, 'expand'
                    else:
                        self.vertices[-1] = x_r
                        self.values[-1] = f_r
                        return x_r, f_r, 'reflect'
                elif f_r < f_second_worst:
                    self.vertices[-1] = x_r
                    self.values[-1] = f_r
                    return x_r, f_r, 'reflect'
                else:
                    if f_r < f_worst:
                        x_c = centroid + sigma * (x_r - centroid)
                    else:
                        x_c = centroid - sigma * (x_worst - centroid)
                    x_c, f_c = func_eval(x_c)
                    if f_c < min(f_r, f_worst):
                        self.vertices[-1] = x_c
                        self.values[-1] = f_c
                        return x_c, f_c, 'contract'
                    else:
                        # Shrink towards best
                        for i in range(1, n):
                            self.vertices[i] = x_best + shrink * (self.vertices[i] - x_best)
                            self.vertices[i], self.values[i] = func_eval(self.vertices[i])
                        return self.vertices[1], self.values[1], 'shrink'

        # Initialize simplexes via Latin Hypercube
        n_simplexes = 5
        lhs_points = lhs_sample(n_simplexes, dim, rng)

        simplex_queue = []  # min-heap by potential (lower is better)
        simplex_counter = [0]

        for idx in range(n_simplexes):
            center = lhs_points[idx].copy()
            # Build initial simplex around center
            alpha = 0.25
            vertices = [center.copy()]
            values = []
            center_clipped, f_center = evaluate(center)
            vertices[0] = center_clipped
            values.append(f_center)

            for d in range(dim):
                x_i = center_clipped.copy()
                x_i[d] += alpha
                x_i_clipped, f_i = evaluate(x_i)
                vertices.append(x_i_clipped)
                values.append(f_i)

            simplex = Simplex(vertices, values)
            simplex.id = simplex_counter[0]
            simplex_counter[0] += 1
            heapq.heappush(simplex_queue, (simplex.potential(), simplex))

        # Main loop: best-first Nelder-Mead
        max_total_steps = budget - 2 * n_simplexes * (dim + 1) - 50  # reserve some budget
        step_count = 0

        while time_ok() and budget_ok() and simplex_queue:
            # Get the most promising simplex
            pot, simplex = heapq.heappop(simplex_queue)

            # Check if simplex has converged
            f_spread = max(simplex.values) - min(simplex.values)
            if f_spread < 1e-12:
                # Simplex converged, skip it
                continue

            # Perform one NM step
            new_v, new_f, action = simplex.step(evaluate, rng)

            step_count += 1

            # Re-evaluate potential and re-insert
            new_pot = simplex.potential()
            heapq.heappush(simplex_queue, (new_pot, simplex))

            # Occasionally add a new simplex if we have budget
            if step_count % 50 == 0 and len(simplex_queue) < 8 and budget_ok() and time_ok():
                # Add a new simplex at a random location or near the current best
                if best_x is not None and rng.random() < 0.5:
                    new_center = best_x + rng.normal(0, 0.5, dim)
                else:
                    new_center = rng.uniform(-4, 4, dim)

                alpha = 0.2
                vertices = []
                values = []
                center_clipped, f_center = evaluate(new_center)
                vertices.append(center_clipped)
                values.append(f_center)

                for d in range(dim):
                    x_i = center_clipped.copy()
                    x_i[d] += alpha
                    x_i_clipped, f_i = evaluate(x_i)
                    vertices.append(x_i_clipped)
                    values.append(f_i)

                if budget_ok() and time_ok():
                    simplex = Simplex(vertices, values)
                    simplex.id = simplex_counter[0]
                    simplex_counter[0] += 1
                    heapq.heappush(simplex_queue, (simplex.potential(), simplex))

        return best_x, best_f
# EVOLVE-BLOCK-END
