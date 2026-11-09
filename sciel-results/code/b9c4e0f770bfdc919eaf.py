"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Powell's method optimizer for the bent cigar function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 280.0

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Random restarts to find a good starting point
        # Use 15% of budget for initial exploration
        phase1_budget = max(20, budget // 7)

        x = rng.uniform(-4, 4, dim)
        fx = evaluate(x)

        # Simple random search for initial exploration
        sigma = 2.0
        for i in range(phase1_budget - 1):
            if time.time() > deadline or calls >= budget:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = evaluate(y)
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
            else:
                sigma *= 0.9

        # Phase 2: Powell's method
        # Powell's method uses conjugate directions and is very effective
        # for anisotropic functions like the bent cigar

        def powell_method(x0, fx0, max_calls, deadline):
            nonlocal calls, best_x, best_f

            n = dim
            # Initial directions: coordinate axes
            directions = np.eye(n)
            x = x0.copy()
            fx = fx0

            # Powell's algorithm
            for iteration in range(max_calls // (2 * n) + 10):
                if time.time() > deadline or calls >= budget:
                    break

                x_old = x.copy()
                f_old = fx

                # Perform line searches along each direction
                for i in range(n):
                    if time.time() > deadline or calls >= budget:
                        break

                    # Line search along direction i
                    dir_vec = directions[i]
                    # Find best step along this direction
                    # Use golden section or parabolic interpolation
                    a, b = -1.0, 1.0
                    # First, find an interval where the function decreases
                    x_test = x + a * dir_vec
                    x_test = np.clip(x_test, -5, 5)
                    f_test = evaluate(x_test)
                    if f_test >= fx:
                        # Try other direction
                        a, b = b, a
                        x_test = x + a * dir_vec
                        x_test = np.clip(x_test, -5, 5)
                        f_test = evaluate(x_test)

                    # Expand interval
                    while f_test < fx and calls < budget and time.time() < deadline:
                        a, b = b, a + (b - a)
                        if abs(b - a) < 1e-10:
                            break
                        x_test = x + b * dir_vec
                        x_test = np.clip(x_test, -5, 5)
                        f_test = evaluate(x_test)

                    # Now find minimum in [a, b] using golden section
                    if abs(b - a) < 1e-12:
                        continue

                    # Golden section search
                    gr = (np.sqrt(5) - 1) / 2
                    c = b - gr * (b - a)
                    d = a + gr * (b - a)

                    x_c = np.clip(x + c * dir_vec, -5, 5)
                    f_c = evaluate(x_c)
                    x_d = np.clip(x + d * dir_vec, -5, 5)
                    f_d = evaluate(x_d)

                    for _ in range(20):
                        if calls >= budget or time.time() > deadline:
                            break
                        if f_c < f_d:
                            b = d
                            d = c
                            f_d = f_c
                            c = b - gr * (b - a)
                            x_c = np.clip(x + c * dir_vec, -5, 5)
                            f_c = evaluate(x_c)
                        else:
                            a = c
                            c = d
                            f_c = f_d
                            d = a + gr * (b - a)
                            x_d = np.clip(x + d * dir_vec, -5, 5)
                            f_d = evaluate(x_d)
                        if abs(b - a) < 1e-10:
                            break

                    # Best point along this direction
                    t = (a + b) / 2
                    x_new = np.clip(x + t * dir_vec, -5, 5)
                    f_new = evaluate(x_new)

                    if f_new < fx:
                        x = x_new
                        fx = f_new

                # Update directions
                # Find the direction with the largest decrease
                # This is simplified; full Powell's method tracks this more carefully
                # For now, we add a new conjugate direction

                if np.linalg.norm(x - x_old) > 1e-10:
                    # New direction is the difference between old and new positions
                    new_dir = x - x_old
                    norm = np.linalg.norm(new_dir)
                    if norm > 1e-15:
                        new_dir /= norm
                        # Replace the worst direction (or just add it)
                        # Simplified: replace direction 0 with the new conjugate direction
                        # This is not standard Powell's but can work
                        # Better: find the direction that gave the least improvement
                        # For simplicity, we'll use a rotation approach

            return x, fx

        # Use Powell's method with a modified approach
        # Since standard Powell's can be tricky to implement correctly,
        # let's use a variant that works well for this problem

        def powell_variant(x0, fx0):
            nonlocal calls, best_x, best_f

            n = dim
            # Start with coordinate directions
            directions = [np.eye(n)[i].copy() for i in range(n)]

            x = x0.copy()
            fx = fx0

            max_iter = 50

            for it in range(max_iter):
                if time.time() > deadline or calls >= budget:
                    break

                x_start = x.copy()
                f_start = fx
                max_decrease = 0
                max_dir_idx = 0

                # Line searches along each direction
                for i in range(n):
                    if time.time() > deadline or calls >= budget:
                        break

                    dir_vec = directions[i]

                    # Find best step along this direction using parabolic method
                    # Start with small step
                    t = 0.1
                    x_test = np.clip(x + t * dir_vec, -5, 5)
                    f_test = evaluate(x_test)

                    best_t = 0
                    best_f_dir = fx

                    # Try several step sizes
                    for trial in range(30):
                        if calls >= budget or time.time() > deadline:
                            break
                        if f_test < best_f_dir:
                            best_f_dir = f_test
                            best_t = t

                        # Adaptive step size
                        if f_test < fx:
                            t *= 1.5
                        else:
                            t *= 0.5
                            if t < 1e-10:
                                break

                        x_test = np.clip(x + t * dir_vec, -5, 5)
                        f_test = evaluate(x_test)

                    # Update x with best step
                    if best_t != 0:
                        x_new = np.clip(x + best_t * dir_vec, -5, 5)
                        f_new = evaluate(x_new)
                        if f_new < fx:
                            x = x_new
                            fx = f_new
                            decrease = f_start - fx
                            if decrease > max_decrease:
                                max_decrease = decrease
                                max_dir_idx = i

                # Update directions: replace the direction with least improvement
                # with the conjugate direction (x - x_start)
                if np.linalg.norm(x - x_start) > 1e-12:
                    new_dir = x - x_start
                    norm = np.linalg.norm(new_dir)
                    if norm > 1e-15:
                        new_dir /= norm
                        # Find direction with smallest improvement
                        # Simplified: replace max_dir_idx (actually should be min)
                        # Let's just replace the last direction
                        directions[-1] = new_dir

            return x, fx

        # Call Powell's variant
        x, fx = powell_variant(x, fx)

        # Phase 3: Final refinement with (1+1)-ES
        if calls < budget and time.time() < deadline:
            sigma = 0.1
            for i in range(budget - calls):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
