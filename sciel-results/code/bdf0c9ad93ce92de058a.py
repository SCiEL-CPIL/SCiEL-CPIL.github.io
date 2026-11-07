"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Two-phase Ridge-Aware Optimizer for Sharp Ridge (f13)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls = 0

        def time_ok():
            return time.time() < deadline

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget or not time_ok():
                return np.inf
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def ridge_aware_optimize(x0, f0, remaining_budget):
            nonlocal calls, best_x, best_f
            
            if remaining_budget < 20 or not time_ok():
                return

            n = self.dim
            # Phase 1: Identify Ridge Direction
            # Sample points in a sphere around x0 to estimate local Hessian
            # For a sharp ridge, the Hessian has one very large eigenvalue (along ridge)
            # and small eigenvalues (across ridge). Wait, actually:
            # Sharp ridge f(x) = (x_1^2 + ... ) is not sharp.
            # BBOB f13 is: f(x) = 10 * |x_1| + sum_{i=2}^n (x_i^2)
            # The "ridge" is along the x_1 axis? No, the minimum is at 0.
            # The function is very flat along x_1 near 0?
            # Let's check BBOB f13 definition: 
            # f(x) = 10 * |x_1| + sum_{i=2}^n x_i^2
            # This is NOT a ridge in the sense of a long flat valley.
            # It is a "sharp" function because of the absolute value kink at x_1=0.
            # The "ridge" terminology in BBOB f13 might refer to the shape being 
            # sharp along one dimension and quadratic in others.
            # Actually, let's look at the name: "Sharp ridge".
            # In BBOB, f13 is indeed "Sharp ridge".
            # The function is: f(x) = 10 * |x_1| + sum_{i=2}^n x_i^2
            # The term 10*|x_1| makes it non-differentiable at x_1=0.
            # The term sum x_i^2 makes it convex and smooth elsewhere.
            
            # Strategy:
            # 1. The function is separable in the rotated frame.
            # 2. We need to find the axis corresponding to the absolute value term.
            # 3. Once found, we can optimize the quadratic part using gradient descent
            #    and handle the absolute value part carefully (bisection or direct check).
            
            # Estimate the "sharp" axis.
            # Sample points along axes from x0.
            # If moving along an axis changes f linearly (slope ~ constant), it's the sharp axis.
            # If it changes quadratically, it's a smooth axis.
            
            # Let's sample 2*n points: +/- delta along each axis.
            delta = 0.1
            samples = []
            f_samples = []
            
            for i in range(n):
                for sign in [1, -1]:
                    x_probe = x0.copy()
                    x_probe[i] += sign * delta
                    x_probe = np.clip(x_probe, -5, 5)
                    fp = evaluate(x_probe)
                    if not np.isinf(fp):
                        samples.append(x_probe)
                        f_samples.append(fp)
            
            if len(samples) < n:
                return
                
            samples = np.array(samples)
            f_samples = np.array(f_samples)
            
            # We expect pairs (i, +) and (i, -).
            # Let's analyze the change in f for each axis.
            # For the sharp axis (say axis k), f(x0 + d e_k) - f(x0) ~ 10*|x0_k + d| - 10|x0_k|
            # If x0_k is far from 0, this is linear.
            # For smooth axes, f(x0 + d e_i) - f(x0) ~ (x0_i + d)^2 - x0_i^2 = 2*x0_i*d + d^2.
            
            # A better way to identify the sharp axis:
            # Check the symmetry. f(x0 + d e_k) + f(x0 - d e_k) - 2*f0
            # For sharp axis: 10|x0_k+d| + 10|x0_k-d| - 20|x0_k|
            # If x0_k > d: 10(x0_k+d) + 10(x0_k-d) - 20x0_k = 0.
            # If x0_k < d: 10(d-x0_k) + 10(d+x0_k) - 20x0_k = 20d - 40x0_k. Not 0.
            # This is getting complicated.
            
            # Alternative: Just use a robust global optimizer that handles non-differentiability.
            # Nelder-Mead is robust. The previous score 1.38 was with NM.
            # The prompt suggests a "low-rank quadratic surrogate".
            # Let's try to fit a quadratic model to the samples and find the minimum.
            
            # Build design matrix for quadratic model: [1, x, x_i*x_j]
            # Since n=5, number of terms is 1 + 5 + 15 = 21.
            # We have 10 samples. Under-determined.
            # We can assume diagonal Hessian (no cross terms) initially.
            # Model: f(x) ~ c + sum b_i x_i + sum a_ii x_i^2
            
            # Let's collect more samples if budget allows?
            # No, let's use the 10 samples to fit diagonal quadratic.
            # Variables: c, b_1..b_5, a_11..a_55. Total 11 variables.
            # 10 equations. Close to determined.
            
            # Construct A matrix
            # Row for sample x: [1, x_1, ..., x_5, x_1^2, ..., x_5^2]
            A = np.zeros((len(samples), 11))
            A[:, 0] = 1
            A[:, 1:6] = samples
            A[:, 6:11] = samples**2
            
            y = f_samples
            
            # Least squares
            try:
                coeffs, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
            except:
                return
                
            c = coeffs[0]
            b = coeffs[1:6]
            a = coeffs[6:11]
            
            # Find minimum of this quadratic model
            # f(x) = c + b^T x + x^T diag(a) x
            # Gradient: b + 2 * diag(a) * x = 0
            # x_i = -b_i / (2 * a_i) if a_i != 0
            
            x_model_min = x0.copy()
            for i in range(n):
                if abs(a[i]) > 1e-6:
                    x_model_min[i] = -b[i] / (2 * a[i])
                else:
                    # If linear, go to boundary? Or keep current?
                    # For sharp ridge, the sharp axis might appear linear locally.
                    # We should probe the sharp axis separately.
                    pass
            
            x_model_min = np.clip(x_model_min, -5, 5)
            
            # Evaluate model minimum
            f_model_min = evaluate(x_model_min)
            
            # Phase 2: Refine
            # If we found a good point, do a local Nelder-Mead or Powell.
            # Let's use a simple coordinate descent or NM for the remaining budget.
            
            if calls < self.budget - 5 and time_ok():
                # Use Nelder-Mead for final polish
                self._nelder_mead(x_model_min, f_model_min)

        def _nelder_mead(self, x0, f0):
            nonlocal calls, best_x, best_f
            
            if calls >= self.budget - 5 or not time_ok():
                return

            n = self.dim
            simplex = [x0.copy()]
            f_values = [f0]
            
            # Initialize simplex
            for i in range(n):
                x_new = x0.copy()
                x_new[i] += 0.1
                x_new = np.clip(x_new, -5, 5)
                fx = evaluate(x_new)
                if np.isinf(fx):
                    break
                simplex.append(x_new)
                f_values.append(fx)
                
            if len(simplex) < n + 1:
                return

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            # Limit iterations
            max_iter = 100
            for _ in range(max_iter):
                if calls >= self.budget - 5 or not time_ok():
                    break

                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]

                if f_values[-1] - f_values[0] < 1e-10:
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = evaluate(x_r)
                if np.isinf(f_r):
                    break

                if f_values[0] <= f_r < f_values[-2]:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                elif f_r < f_values[0]:
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = evaluate(x_e)
                    if np.isinf(f_e):
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                else:
                    if f_r < f_values[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (x_r - centroid)
                    
                    x_c = np.clip(x_c, -5, 5)
                    f_c = evaluate(x_c)
                    if np.isinf(f_c):
                        break

                    if f_c < min(f_r, f_values[-1]):
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_new = evaluate(simplex[i])
                            if np.isinf(f_new):
                                break
                            f_values[i] = f_new
                            if calls >= self.budget - 5 or not time_ok():
                                return

        # Main Logic
        # 1. Start with a few random points to get a baseline
        # 2. Use the best point to initialize the ridge-aware model
        # 3. Run ridge_aware_optimize
        
        # Initial sampling
        n_init = 10
        for _ in range(n_init):
            if calls >= self.budget or not time_ok():
                break
            x_rand = rng.uniform(-5, 5, self.dim)
            evaluate(x_rand)
            
        if best_x is not None:
            ridge_aware_optimize(best_x, best_f, self.budget - calls)
            
        return best_x, best_f

# EVOLVE-BLOCK-END
