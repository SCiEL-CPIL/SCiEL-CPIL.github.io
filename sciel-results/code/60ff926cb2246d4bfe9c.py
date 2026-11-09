"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Model-based optimizer with quadratic fitting and Lunacek-aware basin jumping."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        n = self.dim
        total_budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def fit_quadratic(center, points, values):
            """Fit a full quadratic model and return the predicted minimizer."""
            m = len(points)
            if m < n + 1:
                return None
            
            # Build design matrix for quadratic: [1, d_i, d_i*d_j for i<=j]
            X = np.ones((m, 1 + n + n*(n+1)//2))
            y = np.array(values)
            
            for k, p in enumerate(points):
                d = p - center
                col = 1
                for i in range(n):
                    X[k, col] = d[i]
                    col += 1
                for i in range(n):
                    for j in range(i, n):
                        X[k, col] = d[i] * d[j]
                        col += 1
            
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            except np.linalg.LinAlgError:
                return None
            
            # Extract parameters
            c0 = coeffs[0]
            g = coeffs[1:n+1].copy()
            
            # Build Hessian
            H = np.zeros((n, n))
            idx = n + 1
            for i in range(n):
                for j in range(i, n):
                    if i == j:
                        H[i, i] = 2.0 * coeffs[idx]
                    else:
                        H[i, j] = coeffs[idx]
                        H[j, i] = coeffs[idx]
                    idx += 1
            
            # Check if H is positive definite; if not, regularize
            eigvals, eigvecs = np.linalg.eigh(H)
            if np.min(eigvals) <= 0:
                # Add small regularization to make it PD
                H_reg = H + (abs(np.min(eigvals)) + 0.1) * np.eye(n)
            else:
                H_reg = H
            
            try:
                dx = np.linalg.solve(H_reg, -g)
            except np.linalg.LinAlgError:
                return None
            
            # Verify the step improves the model
            model_center = c0
            model_new = c0 + g @ dx + 0.5 * dx @ H_reg @ dx
            if model_new >= model_center:
                return None
            
            new_x = center + dx
            new_x = np.clip(new_x, -5, 5)
            return new_x

        def do_es_around(x, fx, budget, sigma_init=1.0, sigma_min=1e-4, sigma_max=5.0, lam=6, max_no_improve=25):
            """Run (1+lambda)-ES around a starting point."""
            nonlocal calls_made, best_x, best_f
            
            sigma = sigma_init
            no_improve = 0
            steps = 0
            budget_used = 0
            
            while budget_used < budget and calls_made < total_budget:
                if time.time() > deadline:
                    break
                
                # Generate offspring
                offspring = []
                offspring_f = []
                for i in range(lam):
                    if calls_made >= total_budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    y = x + sigma * z
                    fy = evaluate(y)
                    offspring.append(y)
                    offspring_f.append(fy)
                    budget_used += 1
                
                if not offspring:
                    break
                
                best_idx = np.argmin(offspring_f)
                best_y = offspring[best_idx]
                best_yf = offspring_f[best_idx]
                
                if best_yf < fx:
                    x = best_y
                    fx = best_yf
                    no_improve = 0
                    sigma *= 1.15
                else:
                    no_improve += 1
                    sigma *= 0.85
                
                sigma = np.clip(sigma, sigma_min, sigma_max)
                steps += 1
                
                if no_improve >= max_no_improve:
                    break
            
            return x, fx

        # Phase 1: Multiple diverse starts with ES
        # Use 5 restarts with different strategies
        num_restarts = 5
        budget_fracs = [0.22, 0.18, 0.18, 0.22, 0.20]
        
        for restart_idx in range(num_restarts):
            if calls_made >= total_budget or time.time() > deadline:
                break
            
            budget_this = int(total_budget * budget_fracs[restart_idx])
            budget_this = min(budget_this, total_budget - calls_made)
            if budget_this < 15:
                continue
            
            # Diverse initialization strategies
            if restart_idx == 0:
                # Start at origin (common for Lunacek)
                x = np.zeros(n)
                sigma_init = 2.0
            elif restart_idx == 1:
                # Random in inner box
                x = rng.uniform(-3.5, 3.5, n)
                sigma_init = 1.5
            elif restart_idx == 2:
                # Random in full box
                x = rng.uniform(-5, 5, n)
                sigma_init = 2.5
            elif restart_idx == 3:
                # If we have a good point, perturb it
                if best_x is not None:
                    x = np.clip(best_x + rng.standard_normal(n) * 3.0, -5, 5)
                    sigma_init = 1.0
                else:
                    x = rng.uniform(-5, 5, n)
                    sigma_init = 2.0
            else:
                # Fine-tune around best
                if best_x is not None:
                    x = np.clip(best_x + rng.standard_normal(n) * 0.8, -5, 5)
                    sigma_init = 0.5
                else:
                    x = rng.uniform(-5, 5, n)
                    sigma_init = 1.5
            
            fx = evaluate(x)
            
            # Run ES
            x, fx = do_es_around(x, fx, budget_this, sigma_init=sigma_init)
            
            # After ES, try quadratic fit to jump to basin center
            if calls_made < total_budget and time.time() < deadline:
                h = 0.4
                center = x.copy()
                points = [center]
                values = [fx]
                for i in range(n):
                    for sign in [1, -1]:
                        p = center.copy()
                        p[i] += sign * h
                        p = np.clip(p, -5, 5)
                        if calls_made >= total_budget or time.time() > deadline:
                            break
                        v = evaluate(p)
                        points.append(p)
                        values.append(v)
                
                jump = fit_quadratic(center, points, values)
                if jump is not None:
                    fj = evaluate(jump)
                    if fj < fx:
                        x = jump
                        fx = fj
                        # Refine with fine ES
                        x, fx = do_es_around(x, fx, 15, sigma_init=0.2, sigma_min=1e-5, sigma_max=1.0, lam=4, max_no_improve=15)
        
        # Phase 2: Lunacek-specific basin jump
        # Lunacek has two basins; if we're in one, the other might be better
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            # Try reflecting through the center of the domain
            # The two minima of Lunacek are typically symmetric around some point
            for center_offset in [0.0, 1.0, -1.0]:
                if calls_made >= total_budget or time.time() > deadline:
                    break
                
                # Reflect best_x through a shifted center
                reflect_center = np.full(n, center_offset)
                reflected = 2 * reflect_center - best_x
                reflected = np.clip(reflected, -5, 5)
                
                fr = evaluate(reflected)
                if fr < best_f:
                    # Found a potentially better basin
                    x_ref = reflected.copy()
                    # Run ES around this new point
                    x_ref, fr = do_es_around(x_ref, fr, 20, sigma_init=0.5, sigma_min=1e-5, sigma_max=2.0, lam=5, max_no_improve=20)
                    
                    # Try quadratic fit
                    if calls_made < total_budget and time.time() < deadline:
                        h = 0.3
                        center = x_ref.copy()
                        points = [center]
                        values = [fr]
                        for i in range(n):
                            for sign in [1, -1]:
                                p = center.copy()
                                p[i] += sign * h
                                p = np.clip(p, -5, 5)
                                if calls_made >= total_budget or time.time() > deadline:
                                    break
                                v = evaluate(p)
                                points.append(p)
                                values.append(v)
                        
                        jump = fit_quadratic(center, points, values)
                        if jump is not None:
                            fj = evaluate(jump)
                            if fj < fr:
                                x_ref = jump
                                fr = fj
                                x_ref, fr = do_es_around(x_ref, fr, 10, sigma_init=0.1, sigma_min=1e-6, sigma_max=0.5, lam=4, max_no_improve=10)
        
        # Phase 3: Final polishing with very fine steps
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            # Multiple fine quadratic fits with decreasing h
            for h in [0.2, 0.1, 0.05]:
                if calls_made >= total_budget or time.time() > deadline:
                    break
                
                center = best_x.copy()
                points = [center]
                values = [best_f]
                for i in range(n):
                    for sign in [1, -1]:
                        p = center.copy()
                        p[i] += sign * h
                        p = np.clip(p, -5, 5)
                        if calls_made >= total_budget or time.time() > deadline:
                            break
                        v = evaluate(p)
                        points.append(p)
                        values.append(v)
                
                jump = fit_quadratic(center, points, values)
                if jump is not None:
                    fj = evaluate(jump)
                    if fj < best_f:
                        best_x = jump
                        best_f = fj
                        # Tiny refinement
                        do_es_around(best_x, best_f, 5, sigma_init=0.05, sigma_min=1e-7, sigma_max=0.2, lam=3, max_no_improve=5)
        
        # Ensure we return valid values
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
