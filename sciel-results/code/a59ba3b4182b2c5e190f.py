"""Optimizer using Derivative-Free Trust-Region (DFTR) with multi-start for BBOB f18 (Schaffers F7 ill-conditioned)."""
import numpy as np
import time

class Optimizer:
    """DFTR optimizer with quadratic model fitting and multi-start strategy."""
    
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def _clip(self, x):
        return np.clip(x, -5.0, 5.0)
    
    def _fit_quadratic_model(self, points, values):
        """Fit a quadratic model m(x) = c + g^T(x-x0) + 0.5*(x-x0)^T H (x-x0) to data.
        
        Uses least squares with the points relative to x0 = points[0].
        Returns (c, g, H) where m(x) = c + g@(x-x0) + 0.5*(x-x0)@H@(x-x0)
        """
        n = len(points)
        dim = self.dim
        if n < 2:
            return values[0], np.zeros(dim), np.eye(dim) * 1e-6
        
        x0 = points[0]
        # Build design matrix for quadratic terms
        # Features: 1, dx_i (dim), dx_i*dx_j (dim*(dim+1)/2)
        n_feat = 1 + dim + dim * (dim + 1) // 2
        A = np.zeros((n, n_feat))
        A[:, 0] = 1.0  # constant
        
        for i in range(n):
            dx = points[i] - x0
            # Linear terms
            for j in range(dim):
                A[i, 1 + j] = dx[j]
            # Quadratic terms
            k = 1 + dim
            for j in range(dim):
                for l in range(j, dim):
                    A[i, k] = dx[j] * dx[l]
                    k += 1
        
        # Solve least squares
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(A, values, rcond=None)
        except np.linalg.LinAlgError:
            coeffs = np.zeros(n_feat)
            coeffs[0] = values[0]
        
        c = coeffs[0]
        g = coeffs[1:1+dim].copy()
        
        # Reconstruct Hessian (symmetric)
        H = np.zeros((dim, dim))
        k = 1 + dim
        for j in range(dim):
            for l in range(j, dim):
                if j == l:
                    H[j, j] += coeffs[k]
                else:
                    H[j, l] += coeffs[k]
                    H[l, j] += coeffs[k]
                k += 1
        
        return c, g, H
    
    def _dftr_step(self, func, x, fx, points, values, delta, rng, deadline):
        """One DFTR iteration: fit quadratic model, find minimizer, update trust region.
        
        Returns (new_x, new_fx, points, values, delta, calls_used, improved)
        """
        dim = self.dim
        calls = 0
        
        # Fit quadratic model
        c, g, H = self._fit_quadratic_model(points, values)
        
        # Regularize Hessian for numerical stability
        H_reg = H + 1e-6 * np.eye(dim)
        
        # Find minimizer of model within trust region
        # Model: m(x) = c + g^T(x-x) + 0.5*(x-x)^T H_reg (x-x)
        # Gradient: g + H_reg*(x-x) = 0 => x_new = x - H_reg^{-1} g
        try:
            step_vec = np.linalg.solve(H_reg, -g)
        except np.linalg.LinAlgError:
            step_vec = -g / (np.linalg.norm(g) + 1e-12)
        
        # Constrain to trust region
        step_norm = np.linalg.norm(step_vec)
        if step_norm > delta:
            step_vec = step_vec * (delta / step_norm)
        
        x_trial = self._clip(x + step_vec)
        fx_trial = func(x_trial)
        calls += 1
        
        # Check if model predicted improvement
        predicted_reduction = fx - (c + g @ (x_trial - x) + 0.5 * (x_trial - x) @ H_reg @ (x_trial - x))
        actual_reduction = fx - fx_trial
        rho = actual_reduction / max(abs(predicted_reduction), 1e-12)
        
        improved = False
        if fx_trial < fx:
            # Accept step
            x = x_trial.copy()
            fx = fx_trial
            improved = True
            
            # Add new point to model data
            points.append(x_trial.copy())
            values.append(fx_trial)
            
            # Keep only the most recent 2*dim+2 points for model fitting
            max_points = 2 * dim + 2
            if len(points) > max_points:
                # Remove oldest points
                points.pop(0)
                values.pop(0)
            
            # Expand trust region
            delta = min(delta * 2.0, 10.0)
        else:
            # Reject step, shrink trust region
            delta = max(delta * 0.5, 1e-6)
            
            # Add a random point to improve model
            if len(points) < 2 * dim + 2:
                x_rand = self._clip(x + rng.normal(0, delta, size=dim))
                fx_rand = func(x_rand)
                calls += 1
                points.append(x_rand.copy())
                values.append(fx_rand)
        
        return x, fx, points, values, delta, calls, improved
    
    def _dftr_optimize(self, func, x_start, budget, rng, deadline):
        """Run DFTR optimization from x_start.
        
        Returns (best_x, best_f, calls_used).
        """
        dim = self.dim
        x = self._clip(x_start.copy())
        fx = func(x)
        calls = 1
        best_x = x.copy()
        best_f = fx
        
        # Initialize trust region
        delta = 1.0
        
        # Initialize model points with x and coordinate perturbations
        points = [x.copy()]
        values = [fx]
        
        # Add coordinate directions to build initial model
        for i in range(dim):
            for sign in [1.0, -1.0]:
                if calls >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break
                x_trial = self._clip(x + sign * 0.5 * np.eye(dim)[i])
                fx_trial = func(x_trial)
                calls += 1
                points.append(x_trial.copy())
                values.append(fx_trial)
                if fx_trial < best_f:
                    best_f = fx_trial
                    best_x = x_trial.copy()
        
        # Main DFTR loop
        stagnation_count = 0
        max_stagnation = 10
        
        while calls < budget:
            if time.time() > deadline - 5.0:
                break
            
            x, fx, points, values, delta, calls_this, improved = self._dftr_step(
                func, x, fx, points, values, delta, rng, deadline
            )
            calls += calls_this
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            if stagnation_count >= max_stagnation:
                # Try a random restart within trust region
                x_jump = self._clip(x + rng.normal(0, delta * 0.5, size=dim))
                fx_jump = func(x_jump)
                calls += 1
                if fx_jump < fx:
                    x = x_jump.copy()
                    fx = fx_jump
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    stagnation_count = 0
                    # Reset model
                    points = [x.copy()]
                    values = [fx]
                    for i in range(dim):
                        for sign in [1.0, -1.0]:
                            if calls >= budget:
                                break
                            x_trial = self._clip(x + sign * 0.5 * np.eye(dim)[i])
                            fx_trial = func(x_trial)
                            calls += 1
                            points.append(x_trial.copy())
                            values.append(fx_trial)
                else:
                    break
        
        return best_x, best_f, calls
    
    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0
        
        # Phase 1: Multi-start initialization
        # Use a grid + random sampling to find diverse starting points
        n_initial = min(40, budget // 4)
        initial_points = []
        initial_values = []
        
        # Structured points
        test_points = [
            np.zeros(dim),
            np.ones(dim) * 1.5,
            np.ones(dim) * -1.5,
            np.ones(dim) * 3.0,
            np.ones(dim) * -3.0,
        ]
        
        for xp in test_points:
            if total_calls >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            xp = self._clip(xp)
            fxp = func(xp)
            total_calls += 1
            initial_points.append(xp.copy())
            initial_values.append(fxp)
            if fxp < best_f:
                best_f = fxp
                best_x = xp.copy()
        
        # Random points
        for i in range(n_initial - len(test_points)):
            if total_calls >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            xi = rng.uniform(-4.5, 4.5, size=dim)
            fi = func(xi)
            total_calls += 1
            initial_points.append(xi.copy())
            initial_values.append(fi)
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()
        
        # Phase 2: Run DFTR from top diverse starting points
        order = np.argsort(initial_values)
        selected_starts = []
        for idx in order:
            if len(selected_starts) >= 6:
                break
            p = initial_points[idx]
            # Check diversity
            too_close = False
            for s in selected_starts:
                if np.linalg.norm(p - s) < 0.8:
                    too_close = True
                    break
            if not too_close:
                selected_starts.append(p.copy())
        
        # Allocate budget for DFTR runs
        remaining_budget = budget - total_calls
        if remaining_budget <= 0:
            return best_x, best_f
        
        n_runs = min(len(selected_starts), 4)
        calls_per_run = max(30, remaining_budget // max(1, n_runs))
        
        for start_idx in range(n_runs):
            if total_calls >= budget:
                break
            if time.time() > deadline - 10.0:
                break
            
            x0 = selected_starts[start_idx]
            max_calls_this = min(calls_per_run, budget - total_calls)
            
            if max_calls_this < 15:
                continue
            
            x_dftr, f_dftr, calls_dftr = self._dftr_optimize(func, x0, max_calls_this, rng, deadline)
            total_calls += calls_dftr
            
            if f_dftr < best_f:
                best_f = f_dftr
                best_x = x_dftr.copy()
        
        return best_x, best_f
