"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Powell's Method Optimizer for BBOB f14.
    Strategy:
    1. Start at the origin (or a few diverse points) to handle potential shifts.
    2. Use Powell's method (derivative-free coordinate-wise line search with conjugate direction updates).
       This exploits the separable structure of f14 to rapidly refine each dimension.
       It is more efficient than isotropic ES for functions with strong anisotropy.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit
        calls_used = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def check_time():
            nonlocal calls_used
            return (time.time() >= deadline) or (calls_used >= budget)

        def minimize_powell(x0, max_iter=50):
            """
            Powell's method implementation.
            Returns best point and value found.
            """
            nonlocal calls_used, func, dim, rng

            x = x0.copy()
            fx = func(x)
            calls_used += 1
            best_x = x.copy()
            best_f = fx

            # Initial directions: standard basis vectors
            directions = [np.eye(dim)[i] for i in range(dim)]

            # Initial step size
            delta = 0.5

            for _ in range(max_iter):
                if check_time():
                    break

                x_last = x.copy()
                f_last = fx
                f_max = -np.inf
                i_max = 0

                for i in range(dim):
                    if check_time():
                        break

                    # Line search along direction[i]
                    d = directions[i]
                    # Try positive and negative steps
                    # Simple 3-point search or just try delta and -delta
                    # Let's do a small local search to find the minimum along the line
                    # We'll use a simple bracketing or just fixed steps with refinement.
                    # To save calls, let's try +/- delta and pick the best, then refine if needed.
                    
                    x_trial1 = x + delta * d
                    x_trial2 = x - delta * d
                    
                    f1 = func(x_trial1)
                    calls_used += 1
                    f2 = func(x_trial2)
                    calls_used += 1
                    
                    # Determine best step in this direction
                    if f1 < fx and f1 < f2:
                        # Move in positive direction
                        # Refine: try smaller steps? Or just accept?
                        # Let's accept the best of the two and update x
                        x_new = x_trial1
                        f_new = f1
                    elif f2 < fx and f2 < f1:
                        # Move in negative direction
                        x_new = x_trial2
                        f_new = f2
                    else:
                        # No improvement, keep current
                        x_new = x
                        f_new = fx
                    
                    if f_new < f_max:
                        f_max = f_new
                        i_max = i

                    x = x_new
                    fx = f_new

                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

                # Check for convergence
                if abs(f_last - fx) < 1e-8:
                    break

                # Conjugate direction update
                # If the function decreased significantly in the last cycle,
                # add the new direction (x - x_last) to the set.
                if fx < f_last - 1e-8:
                    new_dir = (x - x_last)
                    norm_new_dir = np.linalg.norm(new_dir)
                    if norm_new_dir > 1e-10:
                        new_dir = new_dir / norm_new_dir
                        
                        # Replace the direction that had the largest function value
                        # with the new conjugate direction
                        directions[i_max] = new_dir

                # Adjust delta
                # If we made progress, increase delta slightly?
                # If not, decrease?
                # Powell's method typically uses a fixed delta or adaptive.
                # Let's keep delta fixed for simplicity but maybe shrink if stuck.
                # Actually, standard Powell uses the same delta.
                # Let's try to adapt delta based on success.
                # If fx < f_last, we are making progress.
                # Let's leave delta as is for now, or shrink it slightly to converge.
                # delta *= 0.9 might be too aggressive.
                # Let's try a simple reduction if no improvement was made in the last cycle?
                # No, we already broke if no improvement.
                # Let's just continue.

            return best_x, best_f

        # Global Search Phase
        # Start with origin
        x_origin = np.zeros(dim)
        x_best, f_best = minimize_powell(x_origin, max_iter=10)
        
        # If origin is not good, try a few other points
        # Vertices of a smaller box, e.g., [-2, 2]^5
        # Or random points
        # Let's try 3 random points if we have budget
        for _ in range(3):
            if check_time():
                break
            x_rand = rng.uniform(-4, 4, size=dim)
            x_cand, f_cand = minimize_powell(x_rand, max_iter=5)
            if f_cand < f_best:
                x_best = x_cand
                f_best = f_cand

        # Final polish with more iterations on the best point
        if not check_time():
            x_best, f_best = minimize_powell(x_best, max_iter=20)

        return x_best, f_best
# EVOLVE-BLOCK-END
