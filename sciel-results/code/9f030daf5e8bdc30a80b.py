import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin

        dim = self.dim
        budget = self.budget

        # Phase 1: Evaluate origin
        x0 = np.zeros(dim)
        f0 = func(x0)
        best_f = f0
        best_x = x0.copy()
        calls = 1

        # Phase 2: Robust gradient estimation using central differences
        # Use multiple epsilon values for robustness against near-zero gradients
        eps_values = [2.0, 1.0, 0.5]
        grad = np.zeros(dim)
        
        for eps in eps_values:
            if time.time() > deadline:
                break
            if calls >= budget:
                break
                
            grad_partial = np.zeros(dim)
            for i in range(dim):
                if time.time() > deadline:
                    break
                if calls >= budget:
                    break
                    
                xp = x0.copy()
                xm = x0.copy()
                xp[i] += eps
                xm[i] -= eps
                
                fp = func(xp)
                calls += 1
                if time.time() > deadline:
                    break
                fm = func(xm)
                calls += 1
                
                if fp < best_f:
                    best_f = fp
                    best_x = xp.copy()
                if fm < best_f:
                    best_f = fm
                    best_x = xm.copy()
                
                grad_partial[i] = (fp - fm) / (2.0 * eps)
            
            # Combine gradient estimates: use the one with largest magnitude per component
            # This is robust: if gradient is large, small eps gives accurate estimate;
            # if gradient is small, large eps gives more reliable sign
            for i in range(dim):
                if abs(grad_partial[i]) > abs(grad[i]):
                    grad[i] = grad_partial[i]

        # Phase 3: Determine optimal vertex
        # For linear f(x) = c + g^T x, minimum over [-5,5]^d is at vertex where
        # each component is -5 if g_i > 0, +5 if g_i < 0
        x_vertex = np.zeros(dim)
        for i in range(dim):
            if grad[i] > 0:
                x_vertex[i] = -5.0
            elif grad[i] < 0:
                x_vertex[i] = 5.0
            else:
                # Gradient near zero: component doesn't matter much, stay at 0
                x_vertex[i] = 0.0

        # Phase 4: Evaluate optimal vertex
        if time.time() < deadline and calls < budget:
            f_vertex = func(x_vertex)
            calls += 1
            if f_vertex < best_f:
                best_f = f_vertex
                best_x = x_vertex.copy()

        # Phase 5: Light refinement using remaining budget
        # Since the function is linear, the vertex IS the global minimum.
        # But to be safe against any nonlinearity or numerical issues,
        # do a few local probes around the best point.
        if time.time() < deadline and calls < budget:
            rng = np.random.default_rng(42)
            sigma = 0.5
            no_improve = 0
            max_no_improve = 30
            
            while calls < budget and time.time() < deadline:
                # Generate candidate near best_x
                y = best_x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                
                fy = func(y)
                calls += 1
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    no_improve = 0
                    sigma = min(sigma * 1.1, 2.0)
                else:
                    no_improve += 1
                    sigma *= 0.8
                    if sigma < 1e-6:
                        break
                    if no_improve >= max_no_improve:
                        # Try a random restart
                        x_restart = rng.uniform(-5.0, 5.0, size=dim)
                        fr = func(x_restart)
                        calls += 1
                        if fr < best_f:
                            best_f = fr
                            best_x = x_restart.copy()
                            sigma = 1.0
                            no_improve = 0
                        else:
                            break

        return best_x, best_f
