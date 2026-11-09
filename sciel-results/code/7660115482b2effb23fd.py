"""DF-T (Derivative-Free Trust-Region) optimizer for BBOB f18."""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 315.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        calls_used = 0
        best_fx = float('inf')
        best_x = np.zeros(dim)

        def check():
            return calls_used < budget and time.time() < deadline - 3.0

        # Phase 1: Initial Sampling
        n_init = min(30, budget // 10)
        for i in range(n_init):
            if not check(): break
            x = rng.uniform(-4, 4, dim)
            fx = func(x)
            calls_used += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
        
        # Phase 2: DF-T Optimization
        # State
        x_c = best_x.copy()
        delta = 1.0
        # Points: list of (x, f)
        points = [(x_c.copy(), best_fx)]
        
        # Generate initial points along axes
        for i in range(dim):
            for sign in [1, -1]:
                x_new = x_c.copy()
                x_new[i] += sign * delta
                x_new = np.clip(x_new, -5, 5)
                if check():
                    fx_new = func(x_new)
                    calls_used += 1
                    points.append((x_new, fx_new))
                    if fx_new < best_fx:
                        best_fx = fx_new
                        best_x = x_new.copy()

        stagnation = 0
        max_stagnation = 20
        
        while check():
            # Fit Quadratic Model
            # Design matrix: [1, x, x_i x_j]
            n_pts = len(points)
            n_params = 1 + dim + dim * (dim + 1) // 2
            if n_pts < n_params:
                # Add more random points if needed
                x_new = x_c + delta * rng.standard_normal(dim)
                x_new = np.clip(x_new, -5, 5)
                fx_new = func(x_new)
                calls_used += 1
                points.append((x_new, fx_new))
                if fx_new < best_fx:
                    best_fx = fx_new
                    best_x = x_new.copy()
                continue

            X = np.array([p[0] for p in points])
            F = np.array([p[1] for p in points])
            
            # Build design matrix
            A = np.zeros((n_pts, n_params))
            A[:, 0] = 1
            A[:, 1:1+dim] = X
            # Quadratic terms
            for i in range(dim):
                for j in range(i, dim):
                    idx = 1 + dim + i * (dim + 1) // 2 - (i * (i - 1)) // 2 + j - i
                    # Simpler indexing:
                    # 0: const
                    # 1..d: linear
                    # d+1..: quadratic
                    # Let's just build it explicitly
            # Rebuild A properly
            A = np.zeros((n_pts, n_params))
            A[:, 0] = 1.0
            A[:, 1:1+dim] = X
            col = 1 + dim
            for i in range(dim):
                for j in range(i, dim):
                    A[:, col] = X[:, i] * X[:, j]
                    col += 1
            
            # Fit model
            coeffs, residuals, rank, sv = np.linalg.lstsq(A, F, rcond=None)
            
            # Extract gradient and Hessian at x_c
            # Q(x) = c + g^T(x-x_c) + 0.5 (x-x_c)^T H (x-x_c)
            # We need to transform the global quadratic to local coordinates around x_c
            # Or just compute gradient and Hessian from coeffs
            # c = coeffs[0]
            # g_i = coeffs[1+i] + sum_j coeffs[quad_ij] * 2 * x_c_j (if i!=j) + coeffs[quad_ii] * x_c_i
            # This is getting complex. Let's evaluate Q and its derivatives numerically? No.
            
            # Alternative: Use the fitted model to propose a step.
            # Minimize Q(x) s.t. ||x - x_c|| <= delta
            # Use scipy.optimize.minimize on the quadratic function?
            # def q_model(x):
            #     # Evaluate quadratic model
            #     pass
            
            # Let's use a simpler step: Cauchy point
            # Gradient of Q at x_c
            g = np.zeros(dim)
            for i in range(dim):
                # Linear term
                g[i] = coeffs[1+i]
                # Quadratic terms
                for j in range(dim):
                    # Find index of x_i x_j in coeffs
                    # Index mapping:
                    # 1+i is linear
                    # Quadratic starts at 1+dim
                    # For i<=j: idx = 1+dim + i*(dim+1)//2 - i*(i-1)//2 + j - i ?
                    # Let's just re-evaluate the design matrix row for x_c to get values, 
                    # but we need derivatives.
                    pass
            
            # This manual extraction is error-prone. 
            # Let's use a finite difference on the MODEL to get gradient? 
            # Or just use the model to find the minimizer using scipy?
            
            # Let's use scipy.optimize.minimize on the quadratic model defined by coeffs
            def quad_model(x):
                # Evaluate A(x) @ coeffs
                row = np.zeros(n_params)
                row[0] = 1.0
                row[1:1+dim] = x
                col = 1 + dim
                for i in range(dim):
                    for j in range(i, dim):
                        row[col] = x[i] * x[j]
                        col += 1
                return row @ coeffs
            
            # Minimize quad_model within trust region
            # Use Nelder-Mead on the model? No, that's not efficient.
            # Use the fact that it's a quadratic.
            # We can find the unconstrained minimizer of the quadratic.
            # Q(x) = 0.5 x^T H x + g^T x + c
            # H is symmetric.
            # Let's extract H and g from coeffs.
            
            g = np.zeros(dim)
            H = np.zeros((dim, dim))
            
            # Linear coeffs: coeffs[1:1+dim]
            # Quadratic coeffs: coeffs[1+dim:]
            # Mapping:
            quad_idx = 1 + dim
            for i in range(dim):
                for j in range(i, dim):
                    c_ij = coeffs[quad_idx]
                    H[i, j] += c_ij
                    H[j, i] += c_ij
                    quad_idx += 1
            
            # Gradient at x_c: g(x_c) = coeffs_linear + H @ x_c
            g = coeffs[1:1+dim] + H @ x_c
            
            # Unconstrained minimizer: x_u = x_c - H^{-1} g
            try:
                # Solve H dx = -g
                dx = np.linalg.solve(H, -g)
                x_u = x_c + dx
            except np.linalg.LinAlgError:
                # H is singular, use pseudo-inverse
                dx = np.linalg.lstsq(H, -g, rcond=None)[0]
                x_u = x_c + dx
            
            # Check if x_u is within trust region
            dist = np.linalg.norm(x_u - x_c)
            if dist < delta:
                x_new = x_u
            else:
                # Project x_u onto the trust region boundary
                x_new = x_c + delta * (x_u - x_c) / dist
            
            x_new = np.clip(x_new, -5, 5)
            
            # Evaluate
            fx_new = func(x_new)
            calls_used += 1
            
            # Update
            if fx_new < best_fx:
                best_fx = fx_new
                best_x = x_new.copy()
                stagnation = 0
            else:
                stagnation += 1
            
            # Update Trust Region
            # Predicted improvement: Q(x_c) - Q(x_new)
            q_c = quad_model(x_c)
            q_new = quad_model(x_new)
            pred_imp = q_c - q_new
            act_imp = F[0] - fx_new # F[0] is f(x_c) if x_c is first point? 
            # Wait, x_c might not be in points anymore or might not be the first one.
            # We should track f(x_c) separately or find it in points.
            # Let's assume x_c is the best point we are expanding from.
            # Actually, we should update x_c to x_new if it's better.
            
            if fx_new < F[np.argmin(F)]: # If new point is better than current best in set
                # Move center to new point
                x_c = x_new
                delta *= 1.5 # Expand
                # Remove worst point from set
                # Find point with largest distance from x_c
                dists = [np.linalg.norm(p[0] - x_c) for p in points]
                worst_idx = np.argmax(dists)
                points.pop(worst_idx)
                points.append((x_new, fx_new))
            else:
                # Failure or marginal
                delta *= 0.5 # Shrink
                if delta < 1e-5:
                    break # Stuck
                # Keep points, maybe add new one if it's distinct
                if np.linalg.norm(x_new - x_c) > 1e-6:
                    points.append((x_new, fx_new))
                    # Remove worst
                    dists = [np.linalg.norm(p[0] - x_c) for p in points]
                    worst_idx = np.argmax(dists)
                    points.pop(worst_idx)
            
            if stagnation > max_stagnation:
                # Restart
                x_c = best_x.copy()
                delta = 1.0
                points = [(x_c, best_fx)]
                # Regenerate initial points
                for i in range(dim):
                    for sign in [1, -1]:
                        x_new = x_c.copy()
                        x_new[i] += sign * delta
                        x_new = np.clip(x_new, -5, 5)
                        if check():
                            fx_new = func(x_new)
                            calls_used += 1
                            points.append((x_new, fx_new))
                            if fx_new < best_fx:
                                best_fx = fx_new
                                best_x = x_new.copy()
                stagnation = 0

        return best_x, best_fx
