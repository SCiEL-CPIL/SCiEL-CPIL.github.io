"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Gradient-based optimizer with central finite differences and backtracking line search."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # State tracking
        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def safe_call(x):
            nonlocal call_count
            x_clipped = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            try:
                val = float(func(x_clipped))
                call_count += 1
                if np.isfinite(val):
                    return val
                return 1e10
            except Exception:
                call_count += 1
                return 1e10

        # Estimate gradient using central finite differences
        def estimate_gradient(x, f_current, h=1e-4):
            grad = np.zeros(dim)
            for i in range(dim):
                x_plus = x.copy()
                x_plus[i] += h
                x_minus = x.copy()
                x_minus[i] -= h
                fp = safe_call(x_plus)
                fm = safe_call(x_minus)
                if call_count >= budget:
                    return None
                grad[i] = (fp - fm) / (2 * h)
            return grad

        # Backtracking line search
        def backtracking_line_search(x, f_current, direction, initial_step=1.0):
            step = initial_step
            c1 = 1e-4
            rho = 0.5
            max_iterations = 20
            for _ in range(max_iterations):
                x_new = np.clip(x + step * direction, -5.0, 5.0)
                f_new = safe_call(x_new)
                if call_count >= budget:
                    return None, None
                if f_new <= f_current + c1 * step * np.dot(direction, direction):
                    return x_new, f_new
                step *= rho
            return None, None

        # Phase 1: Initial random sampling to get a decent starting point
        n_initial = min(50, budget // 10)
        best_x = None
        best_f = np.inf

        for i in range(n_initial):
            if call_count >= budget or time.time() > deadline:
                break
            x_candidate = np.random.uniform(-5.0, 5.0, dim)
            f_candidate = safe_call(x_candidate)
            if f_candidate < best_f:
                best_f = f_candidate
                best_x = x_candidate.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)

        # Phase 2: Gradient descent with adaptive step size
        current_x = best_x.copy()
        current_f = best_f
        step_size = 1.0
        min_step = 1e-6
        max_step = 10.0
        no_improve_count = 0
        max_no_improve = 5

        while call_count < budget and time.time() < deadline:
            # Check for timeout
            if time.time() > deadline - 5.0:
                break

            # Estimate gradient
            grad = estimate_gradient(current_x, current_f)
            if grad is None:
                break

            # Skip if gradient is very small (near convergence)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-8:
                break

            # Normalize gradient direction
            direction = -grad / grad_norm

            # Try gradient descent step
            x_new, f_new = backtracking_line_search(current_x, current_f, direction, step_size)

            if x_new is not None and f_new < current_f - 1e-10:
                current_x = x_new
                current_f = f_new
                no_improve_count = 0
                # Increase step size if we made good progress
                step_size = min(step_size * 1.5, max_step)

                # Update global best
                if current_f < best_f:
                    best_f = current_f
                    best_x = current_x.copy()
            else:
                # No improvement; try random restarts or perturbations
                no_improve_count += 1

                if no_improve_count >= max_no_improve:
                    # Try a random perturbation from the best point
                    perturbation = np.random.normal(0, 0.5, dim)
                    x_perturbed = np.clip(best_x + perturbation, -5.0, 5.0)
                    f_perturbed = safe_call(x_perturbed)

                    if call_count < budget and f_perturbed < current_f:
                        current_x = x_perturbed
                        current_f = f_perturbed
                        no_improve_count = 0
                        step_size = 0.5
                    else:
                        # Shrink step size and try again
                        step_size = max(step_size * 0.5, min_step)
                        if step_size <= min_step:
                            # Try a larger random jump
                            jump = np.random.uniform(-2.0, 2.0, dim)
                            x_jump = np.clip(best_x + jump, -5.0, 5.0)
                            f_jump = safe_call(x_jump)
                            if call_count < budget and f_jump < best_f:
                                best_x = x_jump
                                best_f = f_jump
                                current_x = x_jump
                                current_f = f_jump
                                no_improve_count = 0
                                step_size = 1.0
                            elif step_size <= min_step:
                                break

        return best_x, best_f
# EVOLVE-BLOCK-END
