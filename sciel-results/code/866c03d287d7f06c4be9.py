import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer for BBOB-style functions:
    - Phase 1: Halton-sequence global exploration for broad coverage
    - Phase 2: Adaptive (1+1)-ES with 1/5 success rule for valley exploitation
    - Phase 3: Nelder-Mead simplex refinement for final convergence
    Designed for ill-conditioned functions like Schaffers F7.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    @staticmethod
    def _halton(index, base):
        """Generate Halton sequence value."""
        result = 0.0
        f = 1.0 / base
        i = index
        while i > 0:
            result += f * (i % base)
            i //= base
            f /= base
        return result

    def _halton_point(self, index, dim):
        """Generate a dim-dimensional Halton point in [0,1]^d."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        point = []
        for d in range(dim):
            point.append(self._halton(index, primes[d]))
        return np.array(point)

    def __call__(self, func):
        deadline = time.time() + 275  # Safety margin before 330s kill
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            calls += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Halton-sequence global exploration
        # Use ~30% of budget for broad space-filling coverage
        n_halton = max(15, budget // 4)
        for i in range(1, n_halton + 1):
            if calls >= budget - 20 or time.time() > deadline:
                break
            # Map Halton [0,1]^d to [-5,5]^d
            h_point = self._halton_point(i, d)
            x = h_point * 10.0 - 5.0
            fx = evaluate(x)

        if best_x is None:
            # Fallback: single random point
            x = rng.uniform(-5, 5, d)
            fx = evaluate(x)

        # Phase 2: Adaptive (1+1)-ES with 1/5 success rule
        # Use ~50% of budget for local exploitation
        x = best_x.copy()
        fx = best_f
        sigma = 1.5  # Start with moderate step size for 5D space
        successes = 0
        attempts = 0
        steps_since_improvement = 0

        es_budget = budget // 2
        es_calls = 0

        while es_calls < es_budget and calls < budget - 20 and time.time() < deadline:
            # Generate offspring
            y = x + sigma * rng.standard_normal(d)
            fy = evaluate(y)
            es_calls += 1

            if fy < fx:
                # Accept
                x, fx = y, fy
                successes += 1
                steps_since_improvement = 0
            else:
                # Reject
                steps_since_improvement += 1

            attempts += 1

            # Adapt sigma using 1/5 success rule
            if attempts >= 10:
                success_rate = successes / attempts
                if success_rate > 1.0 / 5.0:
                    sigma *= 1.15
                elif success_rate < 1.0 / 5.0:
                    sigma *= 0.85
                # Reset counters
                successes = 0
                attempts = 0

            # Stagnation detection: restart with perturbation from best
            if sigma < 1e-4 or steps_since_improvement > 50:
                if calls >= budget - 20:
                    break
                # Small random perturbation from best known point
                x = np.clip(best_x + rng.uniform(-0.3, 0.3, d), -5, 5)
                fx = evaluate(x)
                es_calls += 1
                sigma = 0.8
                steps_since_improvement = 0
                successes = 0
                attempts = 0

        # Phase 3: Nelder-Mead local refinement on the best point
        if best_x is not None and calls < budget - 10 and time.time() < deadline:
            x_nm = best_x.copy()
            f_nm = best_f

            # Initialize simplex around best point
            simplex = [(f_nm, x_nm.copy())]
            step_size = 0.1
            for i in range(d):
                x_temp = x_nm.copy()
                x_temp[i] += step_size
                f_temp = evaluate(x_temp)
                simplex.append((f_temp, x_temp.copy()))

            # Sort by function value
            simplex.sort(key=lambda t: t[0])

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5

            nm_budget = budget - calls - 5
            nm_iter = 0
            max_nm_iter = min(nm_budget, 100)

            while nm_iter < max_nm_iter and calls < budget and time.time() < deadline:
                nm_iter += 1

                # Compute centroid of best d vertices (excluding worst)
                centroid = np.mean([s[1] for s in simplex[:-1]], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1][1])
                f_ref = evaluate(x_ref)

                if f_ref < simplex[0][0]:
                    # Try expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    f_exp = evaluate(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = (f_exp, x_exp.copy())
                    else:
                        simplex[-1] = (f_ref, x_ref.copy())
                elif f_ref < simplex[-2][0]:
                    # Accept reflection
                    simplex[-1] = (f_ref, x_ref.copy())
                else:
                    # Contraction
                    if f_ref < simplex[-1][0]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (simplex[-1][1] - centroid)
                    f_con = evaluate(x_con)

                    if f_con < min(f_ref, simplex[-1][0]):
                        simplex[-1] = (f_con, x_con.copy())
                    else:
                        # Shrink toward best vertex
                        for i in range(1, len(simplex)):
                            simplex[i][1] = simplex[0][1] + sigma_nm * (simplex[i][1] - simplex[0][1])
                            if calls < budget and time.time() < deadline:
                                simplex[i][0] = evaluate(simplex[i][1])
                        simplex.sort(key=lambda t: t[0])

                # Check convergence
                if len(simplex) > 1:
                    f_range = abs(simplex[0][0] - simplex[-1][0])
                    if f_range < 1e-10:
                        break

        return best_x, best_f
