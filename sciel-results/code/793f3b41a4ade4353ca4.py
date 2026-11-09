"""
Derivative-Free Trust-Region (DF-T) optimizer for BBOB f18 (Schaffers F7 ill-conditioned).
Uses a local quadratic model and adaptive trust region to navigate the landscape.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 330s kill limit. Leave 10s margin for Python overhead and safety.
        deadline = start_time + 320.0
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        n_evals = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        
        def record(x, fx):
            nonlocal n_evals, best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 1: Random Initialization
        # Find a decent basin. 20% of budget.
        n_init = max(10, int(0.2 * budget))
        for _ in range(n_init):
            if n_evals >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            fx = func(x)
            n_evals += 1
            record(x, fx)
            
        # Phase 2: DF-T Optimization
        # Start from the best found so far.
        x0 = best_x.copy()
        f0 = best_f
        
        # Trust region radius
        rho = 1.0
        min_rho = 1e-8
        max_rho = 5.0
        
        # Quadratic model coefficients: f(x) ~ f0 + g^T (x-x0) + 0.5 (x-x0)^T H (x-x0)
        # We estimate H using finite differences on the model points.
        
        # Number of points for model
        # For dim=5, we need at least dim+1 points for linear, 2*dim+1 for quadratic.
        # Let's use a simplex-like or coordinate-based set.
        # To be efficient, we update the model as we go.
        
        # Initial model points
        # We already have x0. Let's generate 2*dim points around x0 to build initial H.
        # But we must be careful with budget.
        
        # Instead of building a full model upfront which might waste budget if x0 is bad,
        # let's use a sequential model update strategy.
        
        # State for DF-T
        # We will maintain a set of points P and values F.
        # Start with x0.
        P = [x0.copy()]
        F = [f0]
        
        # We need to build a local model.
        # Let's add 2*dim points in coordinate directions to estimate gradient and Hessian.
        # This costs 2*dim = 10 evaluations.
        
        h = 1e-1 # Step size for finite differences in model building
        # Actually, for DF-T, we usually sample points in the trust region.
        
        # Let's use a simpler approach: 
        # 1. Estimate gradient and Hessian at x0 using 2*dim points.
        # 2. Solve for step s minimizing model within trust region ||s|| <= rho.
        # 3. Evaluate x_new = x0 + s.
        # 4. Update trust region and model.
        
        # Build initial model
        if n_evals + 2*dim < budget and time.time() < deadline:
            # Coordinate points
            for i in range(dim):
                for sign in [1, -1]:
                    x_trial = x0.copy()
                    x_trial[i] += sign * h
                    x_trial = np.clip(x_trial, -5.0, 5.0)
                    if n_evals >= budget or time.time() > deadline:
                        break
                    fx = func(x_trial)
                    n_evals += 1
                    record(x_trial, fx)
                    P.append(x_trial)
                    F.append(fx)
        else:
            # Not enough budget for model building, fall back to random or just return
            return best_x, best_f
            
        # Ensure we have enough points
        if len(P) < 2*dim + 1:
            # Fill with random points if needed? Or just use what we have.
            # For robustness, let's just proceed with available points if > dim+1
            pass

        # Helper to build quadratic model from P, F
        def build_model(P, F, x_ref):
            # P is list of arrays, F is list of floats
            # We want to fit f(x) = c + g^T(x-x_ref) + 0.5 (x-x_ref)^T H (x-x_ref)
            # Let d = x - x_ref
            # f(x) - F_ref = g^T d + 0.5 d^T H d
            # We have N points.
            # If N > dim + dim*(dim+1)/2, we have overdetermined system.
            # For dim=5, params = 1 + 5 + 15 = 21.
            # We have ~11 points initially. Underdetermined.
            # We need to add more points or use regularization.
            # Let's assume we will add points sequentially.
            
            # For now, let's just use the points we have.
            # If underdetermined, use least squares with Tikhonov regularization.
            
            N = len(P)
            # Build design matrix A and vector b
            # A row i: [1, d_i^T, 0.5 * (d_i^T H d_i) terms?]
            # Actually, vectorizing H:
            # d^T H d = sum_{i,j} H_ij d_i d_j
            # We can form A with columns for 1, d_i, and 0.5*d_i*d_j (i<=j)
            
            n_params = 1 + dim + dim*(dim+1)//2
            
            if N < n_params:
                # Underdetermined. Use minimum norm solution?
                # Or just return a simple model?
                # Let's use a diagonal H assumption if underdetermined?
                # Or just use linear model?
                # For simplicity and speed, let's use a diagonal H approximation if N < n_params.
                # f(x) ~ f0 + g^T d + 0.5 * d^T H_diag d
                # Params: 1 + dim + dim = 11.
                # We have ~11 points. This works!
                
                # Fit linear + diagonal quadratic
                # A row: [1, d_i, 0.5 * d_i^2]
                A = np.zeros((N, 1 + dim + dim))
                b = np.array(F) - F[0] # Assuming P[0] is x_ref? No, F[0] is f(x0).
                # Wait, x_ref is x0. P[0] is x0. So F[0] is f(x0).
                # b_i = F_i - F_0
                
                for i in range(N):
                    d = P[i] - x_ref
                    A[i, 0] = 1
                    A[i, 1:1+dim] = d
                    A[i, 1+dim:1+2*dim] = 0.5 * d**2
                
                # Solve min ||A theta - b||^2 + lambda ||theta||^2
                # Use np.linalg.lstsq
                try:
                    theta, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    theta = np.zeros(1 + 2*dim)
                
                c = theta[0]
                g = theta[1:1+dim]
                h_diag = theta[1+dim:1+2*dim]
                H = np.diag(h_diag)
                
                return c, g, H

            else:
                # Full quadratic
                A = np.zeros((N, n_params))
                b = np.array(F) - F[0]
                
                for i in range(N):
                    d = P[i] - x_ref
                    A[i, 0] = 1
                    A[i, 1:1+dim] = d
                    # Quadratic terms
                    k = 1 + dim
                    for r in range(dim):
                        for c_idx in range(r, dim):
                            A[i, k] = 0.5 * d[r] * d[c_idx]
                            if r != c_idx:
                                A[i, k + 1] = 0.5 * d[c_idx] * d[r] # Wait, vectorization order
                                # Let's use a consistent vectorization.
                                # Standard: H is symmetric. 
                                # Vector h = [H00, H01, H02..., H11, H12..., ...]
                                # This is tricky to map indices.
                                # Let's stick to diagonal H for simplicity and robustness in 5D.
                                # The diagonal approximation often works well enough for CMA-ES-like tasks.
                                # Given the "ill-conditioned" nature, diagonal might be too simple, 
                                # but full H is hard to fit with few points.
                                # Let's revert to diagonal H always for this implementation.
                                break
                        break
                # Since we broke out, let's just use the diagonal path always.
                return build_model_diagonal(P, F, x_ref)

        def build_model_diagonal(P, F, x_ref):
            N = len(P)
            A = np.zeros((N, 1 + 2*dim))
            b = np.array(F) - F[0]
            
            for i in range(N):
                d = P[i] - x_ref
                A[i, 0] = 1
                A[i, 1:1+dim] = d
                A[i, 1+dim:1+2*dim] = 0.5 * d**2
                
            try:
                theta, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                theta = np.zeros(1 + 2*dim)
            
            c = theta[0]
            g = theta[1:1+dim]
            h_diag = theta[1+dim:1+2*dim]
            H = np.diag(h_diag)
            
            return c, g, H

        # Main DF-T Loop
        while n_evals < budget:
            if time.time() > deadline:
                break
                
            # Build model
            c, g, H = build_model_diagonal(P, F, x0)
            
            # Solve subproblem: min m(s) = c + g^T s + 0.5 s^T H s s.t. ||s|| <= rho
            # If H is positive definite, we can solve H s = -g, then project to sphere if needed.
            # If H is not PD, we might need a more complex solver.
            # For simplicity, assume H is PD or use a damped Newton step.
            
            # Check if H is PD
            try:
                eigvals = np.linalg.eigvalsh(H)
                if np.min(eigvals) < 1e-6:
                    # Add damping
                    H_reg = H + 1e-4 * np.eye(dim)
                else:
                    H_reg = H
            except:
                H_reg = H + 1e-4 * np.eye(dim)
                
            # Solve H_reg s = -g
            try:
                s_newton = np.linalg.solve(H_reg, -g)
            except np.linalg.LinAlgError:
                s_newton = -g / (np.linalg.norm(g) + 1e-10) * rho
                
            # Project to trust region
            norm_s = np.linalg.norm(s_newton)
            if norm_s > rho:
                s_newton = s_newton / norm_s * rho
            else:
                # If norm_s < rho, we might be able to take a full Newton step, 
                # but we are constrained by rho.
                # Actually, if the unconstrained minimizer is inside the trust region, 
                # the solution is s_newton.
                pass
                
            # Evaluate new point
            x_new = x0 + s_newton
            x_new = np.clip(x_new, -5.0, 5.0)
            
            if n_evals >= budget:
                break
                
            fx_new = func(x_new)
            n_evals += 1
            record(x_new, fx_new)
            
            # Calculate actual reduction vs predicted reduction
            # Predicted: m(0) - m(s) = f0 - (c + g^T s + 0.5 s^T H s)
            # Note: c is relative to F[0]=f0? 
            # In build_model_diagonal, b = F - F[0]. So c is the intercept relative to 0?
            # m(s) = c + g^T s + 0.5 s^T H s
            # At s=0, m(0) = c. But we fitted b = F - F[0].
            # So m(s) approximates F(x0+s) - F(x0).
            # So F(x0+s) ~ F(x0) + m(s).
            # Predicted reduction = F(x0) - F(x0+s_pred) = -m(s) ?
            # No, reduction is f(x0) - f(x0+s).
            # f(x0) = F[0].
            # f(x0+s) ~ F[0] + m(s).
            # Pred_red = F[0] - (F[0] + m(s)) = -m(s).
            
            pred_red = - (c + g @ s_newton + 0.5 * s_newton @ H @ s_newton)
            act_red = f0 - fx_new
            
            # Update trust region
            if act_red > 0.25 * pred_red and pred_red > 0:
                # Good step
                rho = min(2.0 * rho, max_rho)
                x0 = x_new
                f0 = fx_new
                P.append(x_new)
                F.append(fx_new)
                
                # Optionally remove worst points to keep model local
                if len(P) > 2*dim + 10:
                    # Keep best 2*dim+10 points? Or closest to x0?
                    # Let's keep closest to x0.
                    dists = [np.linalg.norm(p - x0) for p in P]
                    # Sort indices by distance
                    idx_sorted = np.argsort(dists)
                    # Keep first 2*dim+10
                    keep_idx = idx_sorted[:2*dim+10]
                    P = [P[i] for i in keep_idx]
                    F = [F[i] for i in keep_idx]
                    
            elif act_red > 0.125 * pred_red and pred_red > 0:
                # Decent step
                rho = rho # No change
                x0 = x_new
                f0 = fx_new
                P.append(x_new)
                F.append(fx_new)
                if len(P) > 2*dim + 10:
                    dists = [np.linalg.norm(p - x0) for p in P]
                    idx_sorted = np.argsort(dists)
                    keep_idx = idx_sorted[:2*dim+10]
                    P = [P[i] for i in keep_idx]
                    F = [F[i] for i in keep_idx]
            else:
                # Bad step
                rho = 0.5 * rho
                if rho < min_rho:
                    # Stagnation, try random restart around best
                    if n_evals < budget - 10:
                        x0 = best_x.copy()
                        f0 = best_f
                        rho = 1.0
                        # Rebuild model points?
                        # Just reset P, F to x0 and a few neighbors
                        P = [x0.copy()]
                        F = [f0]
                        # Add coordinate points again
                        for i in range(dim):
                            for sign in [1, -1]:
                                x_trial = x0.copy()
                                x_trial[i] += sign * 1e-1
                                x_trial = np.clip(x_trial, -5.0, 5.0)
                                if n_evals >= budget: break
                                fx = func(x_trial)
                                n_evals += 1
                                record(x_trial, fx)
                                P.append(x_trial)
                                F.append(fx)
                    else:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
