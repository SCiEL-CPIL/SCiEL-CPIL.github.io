"""Rewritten optimizer using DF-TBFGS with global initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-free Trust-Region BFGS optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        num_calls = 0
        best_x = None
        best_f = np.inf

        def check_time():
            return time.time() > deadline

        def evaluate(x):
            nonlocal num_calls
            x = np.clip(x, -5, 5)
            f = func(x)
            num_calls += 1
            return f

        # Phase 1: Global Initialization
        # Use 20% of budget for global search to find a good starting point
        n_init = max(10, budget // 5)
        candidates = []
        for _ in range(n_init):
            if num_calls >= budget or check_time():
                break
            x = rng.uniform(-5, 5, dim)
            f = evaluate(x)
            candidates.append((x, f))
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if not candidates:
            return best_x, best_f

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[1])
        x = candidates[0][0].copy()
        f = candidates[0][1]

        # Phase 2: Local DF-TBFGS Refinement
        # Initialize Hessian approximation as Identity
        H = np.eye(dim)
        delta = 1.0  # Initial trust region radius
        
        # Pre-allocate for speed
        g = None
        
        while num_calls < budget and not check_time():
            # 1. Compute gradient at x using forward differences
            # Need dim calls
            if num_calls + dim >= budget or check_time():
                break
                
            g = np.zeros(dim)
            eps = 1e-4
            for i in range(dim):
                x_plus = x.copy()
                x_plus[i] += eps
                f_plus = evaluate(x_plus)
                g[i] = (f_plus - f) / eps
            
            # Check for convergence
            if np.linalg.norm(g) < 1e-8:
                break
            
            # 2. Compute step using BFGS inverse (solve H p = -g)
            try:
                p = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                p = -g  # Fallback to gradient descent
            
            # 3. Trust region constraint
            p_norm = np.linalg.norm(p)
            if p_norm > delta:
                p = p * (delta / p_norm)
            elif p_norm < 1e-10:
                # Step is too small, try expanding or breaking
                delta *= 2.0
                if delta > 10.0:
                    break
                continue

            # 4. Evaluate trial point
            if num_calls + 1 >= budget or check_time():
                break
                
            x_trial = x + p
            x_trial = np.clip(x_trial, -5, 5)
            f_trial = evaluate(x_trial)
            
            # 5. Compute predicted reduction
            # Model: f(x) + g^T p + 0.5 p^T H p
            Hp = H @ p
            pred = g @ p + 0.5 * p @ Hp
            
            # Handle case where pred is non-positive (model not convex at step)
            if pred <= 0:
                # Fall back to gradient descent direction if model is bad
                # Or just reject and shrink
                # Let's try a pure gradient step if pred is bad
                p_grad = -g
                p_grad_norm = np.linalg.norm(p_grad)
                if p_grad_norm > 1e-10:
                    p_new = p_grad * (delta / p_grad_norm)
                    x_trial_new = np.clip(x + p_new, -5, 5)
                    if num_calls + 1 < budget and not check_time():
                        f_trial_new = evaluate(x_trial_new)
                        if f_trial_new < f:
                            # Accept gradient step
                            # Update H? We don't have y easily without re-evaluating grad.
                            # For simplicity, if we take a non-BFGS step, we might reset H or keep it.
                            # Let's just accept and re-estimate gradient next time.
                            x = x_trial_new
                            f = f_trial_new
                            # Do not update H, or reset it? Resetting is safer if model was bad.
                            # H = np.eye(dim) # Maybe too aggressive
                            delta = min(2.0 * delta, 5.0)
                            continue
                        else:
                            delta *= 0.5
                            continue
                    else:
                        break
                else:
                    break

            act = f - f_trial
            rho = act / pred if abs(pred) > 1e-10 else 0.0

            # 6. Decision
            if rho > 0.25:
                # Accept step
                # Need new gradient for BFGS update
                if num_calls + dim >= budget or check_time():
                    # Not enough budget for new gradient, but we have a better point
                    x = x_trial
                    f = f_trial
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    break # Stop loop, no budget for grad update
                    
                g_new = np.zeros(dim)
                for i in range(dim):
                    x_plus = x_trial.copy()
                    x_plus[i] += eps
                    f_plus = evaluate(x_plus)
                    g_new[i] = (f_plus - f_trial) / eps
                
                s = x_trial - x
                y = g_new - g
                
                # BFGS Update
                ys = y @ s
                if ys > 1e-10:
                    Hs = H @ s
                    shs = s @ Hs
                    if shs > 1e-10:
                        # H_new = H + (y y^T)/(y^T s) - (H s s^T H)/(s^T H s)
                        H = H + np.outer(y, y) / ys - np.outer(Hs, Hs) / shs
                        # Ensure symmetry
                        H = (H + H.T) / 2
                        # Regularize if needed
                        # Check eigenvalues
                        eigvals = np.linalg.eigvalsh(H)
                        if np.min(eigvals) < 1e-6:
                            H += (1e-5 - np.min(eigvals)) * np.eye(dim)
                
                x = x_trial
                f = f_trial
                g = g_new
                
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                
                # Expand trust region
                if rho > 0.75:
                    delta = min(2.0 * delta, 5.0)
                else:
                    delta = min(1.5 * delta, 5.0)
            else:
                # Reject step
                delta *= 0.5
                if delta < 1e-6:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
