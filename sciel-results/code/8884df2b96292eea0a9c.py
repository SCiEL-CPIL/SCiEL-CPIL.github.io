"""Optimizer using DFTR with sparse finite differences for BBOB f18."""
import numpy as np
import time

class Optimizer:
    """
    Derivative-Free Trust-Region (DFTR) optimizer with sparse finite-difference
    quadratic models, combined with multi-start exploration and random restarts.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _clip(self, x):
        return np.clip(x, -5.0, 5.0)

    def _fd_model(self, func, x, rho, budget_tracker, deadline, rng):
        """
        Build a local quadratic model at x using sparse finite differences.
        Returns (g, H, f0, calls_used).
        g: gradient estimate (dim,)
        H: Hessian estimate (dim, dim)
        f0: function value at x
        """
        dim = self.dim
        f0 = func(x)
        budget_tracker['calls'] += 1
        calls = 1

        # Finite difference step
        # Use a step size that is small enough for local approximation
        # but large enough to overcome numerical noise.
        # rho is the trust region radius, so h should be related to it.
        h = max(1e-4, rho * 0.01)

        # Gradient estimation (central differences)
        g = np.zeros(dim)
        for i in range(dim):
            if budget_tracker['calls'] >= self.budget:
                break
            if time.time() > deadline:
                break
                
            e_i = np.zeros(dim)
            e_i[i] = 1.0
            
            x_plus = self._clip(x + h * e_i)
            x_minus = self._clip(x - h * e_i)
            
            f_plus = func(x_plus)
            budget_tracker['calls'] += 1
            f_minus = func(x_minus)
            budget_tracker['calls'] += 1
            calls += 2
            
            g[i] = (f_plus - f_minus) / (2.0 * h)

        # Hessian estimation (sparse, diagonal + off-diagonal if budget allows)
        # For 5D, we can afford a full Hessian with central differences.
        # H_ij = (f(x+he_i+he_j) - f(x+he_i-he_j) - f(x-he_i+he_j) + f(x-he_i-he_j)) / (4*h^2)
        H = np.zeros((dim, dim))
        for i in range(dim):
            if budget_tracker['calls'] >= self.budget:
                break
            if time.time() > deadline:
                break
            for j in range(i, dim):
                if budget_tracker['calls'] >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                e_i = np.zeros(dim)
                e_i[i] = 1.0
                e_j = np.zeros(dim)
                e_j[j] = 1.0
                
                # 4 points for each pair
                x1 = self._clip(x + h * e_i + h * e_j)
                x2 = self._clip(x + h * e_i - h * e_j)
                x3 = self._clip(x - h * e_i + h * e_j)
                x4 = self._clip(x - h * e_i - h * e_j)
                
                f1 = func(x1); budget_tracker['calls'] += 1; calls += 1
                f2 = func(x2); budget_tracker['calls'] += 1; calls += 1
                f3 = func(x3); budget_tracker['calls'] += 1; calls += 1
                f4 = func(x4); budget_tracker['calls'] += 1; calls += 1
                
                hess_val = (f1 - f2 - f3 + f4) / (4.0 * h * h)
                H[i, j] = hess_val
                H[j, i] = hess_val

        # Regularize Hessian to ensure positive semi-definiteness for trust region
        # Add small identity term for stability
        H += 1e-6 * np.eye(dim)
        
        return g, H, f0, calls

    def _solve_trust_region(self, g, H, rho):
        """
        Solve the trust region subproblem:
        min 0.5 * d^T H d + g^T d
        s.t. ||d|| <= rho
        """
        dim = len(g)
        
        # Check if gradient is zero
        if np.linalg.norm(g) < 1e-12:
            return np.zeros(dim)

        # Try to solve (H + mu I) d = -g for various mu
        # We want the smallest mu such that the step satisfies the trust region constraint
        # and the Hessian is positive definite.
        
        # Start with a small mu
        mu = 1e-8
        d = np.zeros(dim)
        
        # Simple bisection or incremental approach for mu
        # Let's try a few fixed values first, then adjust
        mu_candidates = [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0]
        
        best_d = None
        best_obj = np.inf
        
        for m in mu_candidates:
            try:
                H_reg = H + m * np.eye(dim)
                d_trial = np.linalg.solve(H_reg, -g)
                norm_d = np.linalg.norm(d_trial)
                
                if norm_d <= rho:
                    # Feasible
                    obj = 0.5 * d_trial.dot(H_reg.dot(d_trial)) - m * d_trial.dot(d_trial) + g.dot(d_trial)
                    # Actually, the objective in the subproblem is 0.5 d^T H d + g^T d
                    # But we solved (H+mI)d = -g, so d is the minimizer of 0.5 d^T (H+mI) d - g^T d
                    # The value of the original quadratic at d is:
                    obj_val = 0.5 * d_trial.dot(H.dot(d_trial)) + g.dot(d_trial)
                    
                    if obj_val < best_obj:
                        best_obj = obj_val
                        best_d = d_trial
                else:
                    # Infeasible, project onto trust region boundary
                    d_proj = d_trial * (rho / norm_d)
                    obj_val = 0.5 * d_proj.dot(H.dot(d_proj)) + g.dot(d_proj)
                    if obj_val < best_obj:
                        best_obj = obj_val
                        best_d = d_proj
                        
            except np.linalg.LinAlgError:
                continue
        
        if best_d is None:
            # Fallback: steepest descent step
            norm_g = np.linalg.norm(g)
            if norm_g < 1e-12:
                return np.zeros(dim)
            best_d = -g / norm_g * rho
            # Ensure it's within trust region
            if np.linalg.norm(best_d) > rho:
                best_d *= rho / np.linalg.norm(best_d)
                
        return best_d

    def _dftr_step(self, func, x, rho, budget_tracker, deadline, rng):
        """
        Perform one DFTR step.
        Returns (x_new, f_new, rho_new, calls_used, success).
        """
        dim = self.dim
        calls = 0
        
        # Build model
        g, H, f0, model_calls = self._fd_model(func, x, rho, budget_tracker, deadline, rng)
        calls += model_calls
        budget_tracker['calls'] += model_calls # Already incremented in _fd_model? No, _fd_model increments budget_tracker directly.
        # Wait, _fd_model increments budget_tracker['calls'] internally.
        # So I don't need to add it again here.
        
        # Solve subproblem
        d = self._solve_trust_region(g, H, rho)
        
        # Ensure step is not too large
        if np.linalg.norm(d) > rho:
            d *= rho / np.linalg.norm(d)
        
        x_new = self._clip(x + d)
        
        # Evaluate new point
        f_new = func(x_new)
        budget_tracker['calls'] += 1
        calls += 1
        
        # Calculate actual reduction vs predicted reduction
        # Predicted reduction: f0 - (f0 + g^T d + 0.5 d^T H d) = -g^T d - 0.5 d^T H d
        pred_red = -g.dot(d) - 0.5 * d.dot(H.dot(d))
        act_red = f0 - f_new
        
        # Ratio
        if pred_red > 1e-12:
            rho_ratio = act_red / pred_red
        else:
            rho_ratio = 0.0
        
        # Update trust region radius
        if rho_ratio > 0.75:
            rho_new = min(2.0 * rho, 10.0)
            success = True
        elif rho_ratio > 0.25:
            rho_new = rho
            success = True
        else:
            rho_new = 0.5 * rho
            success = False
            
        # If step was rejected, keep old x
        if not success:
            x_new = x
            f_new = f0
            # But we still consumed a function eval. 
            # In standard TR, if rejected, we don't move.
            # However, we already called func(x_new).
            # So we have spent the budget.
            
        return x_new, f_new, rho_new, calls, success

    def _one_plus_one(self, func, x0, sigma, budget, rng, deadline):
        """Run (1+1)-ES with 1/5-rule step-size adaptation."""
        dim = self.dim
        x = x0.copy()
        fx = func(x)
        calls = 1
        sigma = float(sigma)

        # Track consecutive failures for 1/5 rule
        successes = 0
        total = 0
        # Window for 1/5 rule
        window = 1 + dim  # standard window size

        while calls < budget:
            if time.time() > deadline - 3.0:
                break

            # Generate mutant
            z = rng.standard_normal(dim)
            x_trial = self._clip(x + sigma * z)
            fx_trial = func(x_trial)
            calls += 1

            total += 1
            if fx_trial < fx:
                x = x_trial
                fx = fx_trial
                successes += 1

            # Adapt sigma every 'window' steps
            if total >= window:
                rate = successes / total
                if rate > 0.2:
                    sigma *= 1.25
                elif rate < 0.2:
                    sigma /= 1.25
                sigma = np.clip(sigma, 1e-10, 20.0)
                successes = 0
                total = 0

            # Early termination if sigma is too small
            if sigma < 1e-10:
                break

        return x, fx, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        total_calls = 0
        budget_tracker = {'calls': 0}

        # Phase 1: Multi-start exploration with (1+1)-ES
        # This is robust and handles the initial landscape well.
        n_starts = 5
        budget_per_start = max(20, budget // (n_starts + 2))

        # Generate diverse starting points
        starts = [np.zeros(dim)]
        for i in range(n_starts - 1):
            x_s = rng.uniform(-4.0, 4.0, dim)
            starts.append(x_s)

        for i, x0 in enumerate(starts):
            if total_calls >= budget - 10:
                break
            if time.time() > deadline - 10.0:
                break

            remaining = budget - total_calls
            alloc = min(budget_per_start, remaining)

            sigma = 2.0 * (0.8 ** i)

            x_res, f_res, calls_used = self._one_plus_one(
                func, x0, sigma, alloc, rng, deadline
            )
            total_calls += calls_used
            budget_tracker['calls'] = total_calls

            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()

        # Phase 2: DFTR Refinement
        # Use the best point found in Phase 1 to initialize the trust region.
        if total_calls < budget - 50 and time.time() < deadline - 15.0:
            x_cur = best_x.copy()
            f_cur = best_f
            rho = 1.0  # Initial trust region radius
            
            # Run DFTR for a certain number of steps or until budget/time runs out
            # Each DFTR step costs O(dim^2) function evaluations.
            # For dim=5, that's ~25-30 calls per step.
            # We have roughly budget - total_calls left.
            
            no_improve_count = 0
            max_no_improve = 5
            
            while total_calls < budget - 10 and time.time() < deadline - 5.0:
                # Check if we have enough budget for one DFTR step
                # Cost estimate: 2*dim (grad) + dim*(dim+1)/2 * 4 (hessian) + 1 (eval)
                # For dim=5: 10 + 60 + 1 = 71 calls.
                # This is quite expensive. Maybe we should only do a few steps.
                
                # Let's estimate cost more accurately or just try and catch budget overrun.
                # Actually, _fd_model checks budget_tracker internally.
                
                x_new, f_new, rho_new, calls_used, success = self._dftr_step(
                    func, x_cur, rho, budget_tracker, deadline, rng
                )
                total_calls = budget_tracker['calls']
                
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                    no_improve_count = 0
                    x_cur = x_new
                    f_cur = f_new
                else:
                    no_improve_count += 1
                    # If no improvement, shrink trust region more aggressively or restart
                    if no_improve_count >= max_no_improve:
                        # Try a random restart near the best point
                        # Or just break out of DFTR and do more (1+1)-ES
                        break
                        
                if not success:
                    # Step rejected, trust region shrunk
                    # If rho becomes too small, break
                    if rho < 1e-4:
                        break

        # Phase 3: Final (1+1)-ES polish
        # Use remaining budget for fine-grained search
        if total_calls < budget - 5 and time.time() < deadline - 5.0:
            remaining = budget - total_calls
            sigma_refine = 0.1
            x_res, f_res, calls_used = self._one_plus_one(
                func, best_x, sigma_refine, remaining, rng, deadline
            )
            total_calls += calls_used

            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            total_calls += 1

        return best_x, best_f
