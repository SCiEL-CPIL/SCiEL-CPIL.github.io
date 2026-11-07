"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned).
    
    Strategy:
    1. Multi-start (1+1)-ES with 1/5 success rule for robust exploration.
    2. Local Derivative-Free Trust-Region (DFTR) polishing using sparse finite differences.
    3. Global random restarts triggered by stagnation or trust-region degeneration.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _es_1plus1(self, func, rng, x0, sigma0, max_calls, deadline):
        """
        (1+1)-ES with 1/5 success rule.
        Returns best_x, best_f, calls_used.
        """
        dim = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        sigma = sigma0
        calls = 1
        
        # 1/5 success rule parameters
        c_inc = 1.1
        c_dec = 0.9

        best_x = x.copy()
        best_f = fx
        
        while calls < max_calls:
            if time.time() > deadline - 1.0:
                break
                
            z = rng.standard_normal(dim)
            y = x + sigma * z
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x = y
                fx = fy
                sigma *= c_inc
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= c_dec
                
            sigma = np.clip(sigma, 1e-8, 10.0)

        return best_x, best_f, calls

    def _dftr_polish(self, func, rng, x0, max_calls, deadline):
        """
        Derivative-Free Trust-Region method using sparse finite differences.
        Constructs a local quadratic model and solves the trust-region subproblem.
        """
        dim = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        calls = 1
        
        # Trust region radius
        delta = 0.5
        min_delta = 1e-6
        max_delta = 5.0
        
        # Finite difference step
        h = 1e-4
        
        # Quadratic model coefficients: f(x) ~= f0 + g^T (x-x0) + 0.5 * (x-x0)^T H (x-x0)
        # We will maintain g and H.
        
        # Initial gradient estimate
        g = np.zeros(dim)
        for i in range(dim):
            xp = x.copy()
            xp[i] += h
            xp = np.clip(xp, -5, 5)
            fp = func(xp)
            calls += 1
            
            xm = x.copy()
            xm[i] -= h
            xm = np.clip(xm, -5, 5)
            fm = func(xm)
            calls += 1
            
            # Handle boundary clipping carefully
            # If clipped, one-sided diff
            if xp[i] == 5.0 and xm[i] == x[i]:
                g[i] = (fp - fx) / h
            elif xm[i] == -5.0 and xp[i] == x[i]:
                g[i] = (fx - fm) / h
            else:
                g[i] = (fp - fm) / (2 * h)
                
        # Initial Hessian estimate (diagonal only for stability and speed, or full if budget allows)
        # For ill-conditioned, full Hessian is better but expensive.
        # Let's try a diagonal Hessian first, then maybe update it.
        # Actually, for Schaffers F7, the Hessian is often ill-conditioned.
        # Let's use a simple BFGS-like update or just diagonal.
        # Given the budget, diagonal is safer.
        H = np.diag(np.abs(g) + 1e-4) # Ensure positive definite-ish
        
        # If gradient is zero, we might be at a minimum or stuck.
        if np.linalg.norm(g) < 1e-8:
            return x, fx, calls

        while calls < max_calls:
            if time.time() > deadline - 1.0:
                break
                
            # Solve trust region subproblem: min g^T d + 0.5 d^T H d s.t. ||d|| <= delta
            # For diagonal H, we can solve this more efficiently, but let's use a simple Newton step clipped to delta.
            # Newton step: d = -H^{-1} g
            # If H is diagonal: d_i = -g_i / H_ii
            d = -g / np.diag(H)
            
            # Clip step to trust region
            norm_d = np.linalg.norm(d)
            if norm_d > delta:
                d = d * (delta / norm_d)
            
            # Proposed step
            x_new = x + d
            x_new = np.clip(x_new, -5, 5)
            
            # Evaluate
            f_new = func(x_new)
            calls += 1
            
            # Predicted reduction
            pred_red = -g @ d - 0.5 * d @ H @ d
            # Actual reduction
            act_red = fx - f_new
            
            # Ratio
            if abs(pred_red) > 1e-12:
                rho = act_red / pred_red
            else:
                rho = 0.0
            
            if rho > 0.25:
                # Accept step
                x = x_new
                fx = f_new
                
                # Update trust region
                if rho > 0.75 and norm_d == delta:
                    delta = min(2.0 * delta, max_delta)
                else:
                    delta = min(delta * 1.5, max_delta)
                
                # Update model
                # Recompute gradient? Or update?
                # Recomputing gradient is expensive (2*dim calls).
                # Let's recompute gradient every few steps or if rho is very good.
                # For simplicity and robustness, recompute gradient every step? No, too expensive.
                # Let's update g and H using BFGS formula?
                # Or just recompute gradient if we have budget.
                # Given the "polish" phase, we want high precision.
                # Let's recompute gradient every 5 steps or if delta changes significantly.
                # Actually, let's just recompute gradient every step in this small dim=5 case if budget allows?
                # 500 calls total. ES uses ~200. Polish has ~300.
                # 2*5=10 calls per gradient. 300/10 = 30 gradients. That's fine.
                
                # Recompute gradient
                g_new = np.zeros(dim)
                for i in range(dim):
                    xp = x.copy()
                    xp[i] += h
                    xp = np.clip(xp, -5, 5)
                    fp = func(xp)
                    calls += 1
                    
                    xm = x.copy()
                    xm[i] -= h
                    xm = np.clip(xm, -5, 5)
                    fm = func(xm)
                    calls += 1
                    
                    if xp[i] == 5.0 and xm[i] == x[i]:
                        g_new[i] = (fp - fx) / h
                    elif xm[i] == -5.0 and xp[i] == x[i]:
                        g_new[i] = (fx - fm) / h
                    else:
                        g_new[i] = (fp - fm) / (2 * h)
                
                g = g_new
                
                # Update Hessian (diagonal)
                # Simple heuristic: H_ii = |g_i| + epsilon
                # Or keep previous H if it worked well.
                # Let's just update diagonal to reflect current curvature scale.
                H = np.diag(np.abs(g) + 1e-4)
                
                # Check convergence
                if np.linalg.norm(g) < 1e-8:
                    break
            else:
                # Reject step
                delta = max(0.5 * delta, min_delta)
                if delta <= min_delta + 1e-10:
                    break
                
                # Update gradient slightly? Or just keep it.
                # If step rejected, model might be bad.
                # Let's recompute gradient to ensure we have a fresh direction.
                g_new = np.zeros(dim)
                for i in range(dim):
                    xp = x.copy()
                    xp[i] += h
                    xp = np.clip(xp, -5, 5)
                    fp = func(xp)
                    calls += 1
                    
                    xm = x.copy()
                    xm[i] -= h
                    xm = np.clip(xm, -5, 5)
                    fm = func(xm)
                    calls += 1
                    
                    if xp[i] == 5.0 and xm[i] == x[i]:
                        g_new[i] = (fp - fx) / h
                    elif xm[i] == -5.0 and xp[i] == x[i]:
                        g_new[i] = (fx - fm) / h
                    else:
                        g_new[i] = (fp - fm) / (2 * h)
                
                g = g_new
                H = np.diag(np.abs(g) + 1e-4)
                
                if np.linalg.norm(g) < 1e-8:
                    break

        return x, fx, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0
        
        # Multi-start strategy
        # Phase 1: Exploration with (1+1)-ES
        # Phase 2: Polishing with DFTR
        
        # Allocate 60% of budget to ES, 40% to DFTR
        es_budget = int(budget * 0.6)
        dftr_budget = budget - es_budget
        
        num_restarts = 5
        calls_per_restart = es_budget // num_restarts
        
        starts = [np.zeros(dim)]
        for _ in range(num_restarts - 1):
            x0 = rng.uniform(-4, 4, size=dim)
            starts.append(x0)
        
        # Run ES phases
        for i, x0 in enumerate(starts):
            if total_calls >= es_budget:
                break
            
            remaining = es_budget - total_calls
            if i == len(starts) - 1:
                max_calls = remaining
            else:
                max_calls = min(calls_per_restart, remaining)
            
            if max_calls <= 1:
                break
                
            sigma0 = 2.5
            bx, bf, used = self._es_1plus1(func, rng, x0, sigma0, max_calls, deadline)
            total_calls += used
            
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()
                
            if time.time() > deadline - 5.0:
                break
        
        # Run DFTR polishing phase
        if total_calls < budget and time.time() < deadline - 5.0:
            remaining = budget - total_calls
            # Run DFTR from the best point found
            bx, bf, used = self._dftr_polish(func, rng, best_x, remaining, deadline)
            total_calls += used
            
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()
            
            # If we have more budget and time, try DFTR from other good points?
            # For now, just one polish.
            # We could also restart DFTR if it stagnates, but the function handles stagnation.
            # Let's add a second polish if we still have budget.
            if total_calls < budget and time.time() < deadline - 2.0:
                remaining = budget - total_calls
                # Perturb slightly to escape potential local minima?
                # Or just run again from the same point?
                # Let's run again from the best point.
                bx, bf, used = self._dftr_polish(func, rng, best_x, remaining, deadline)
                total_calls += used
                
                if bf < best_f:
                    best_f = bf
                    best_x = bx.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
