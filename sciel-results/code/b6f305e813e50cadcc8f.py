import numpy as np
import time


class Optimizer:
    """
    Optimizer using Quadratic Initialization followed by (1+1)-ES.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: evaluator kills at 330s, we aim for 290s
        deadline = start_time + 290
        
        d = self.dim
        calls = 0
        best_x = None
        best_f = float('inf')

        def safe_call(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            except Exception:
                return None

        # 1. Quadratic Initialization
        # We need 2d+1 points for a star pattern, but for a full quadratic fit in d=5,
        # we need 1 + 5 + 15 = 21 points.
        # Points: Origin, +/- e_i, +/- (e_i + e_j)
        
        points = []
        fvals = []
        
        # Origin
        x0 = np.zeros(d)
        f0 = safe_call(x0)
        if f0 is None:
            return best_x if best_x is not None else x0, best_f if best_f != float('inf') else 0.0
            
        points.append(x0)
        fvals.append(f0)
        
        delta = 1.0
        
        # Axis points: +/- delta * e_i
        for i in range(d):
            for sign in [1, -1]:
                x = np.zeros(d)
                x[i] = sign * delta
                fx = safe_call(x)
                if fx is None:
                    break
                points.append(x)
                fvals.append(fx)
            if fx is None:
                break
        
        # Diagonal points: +/- delta * (e_i + e_j)
        if fx is not None:
            for i in range(d):
                for j in range(i + 1, d):
                    for sign in [1, -1]:
                        x = np.zeros(d)
                        x[i] = sign * delta
                        x[j] = sign * delta
                        fx = safe_call(x)
                        if fx is None:
                            break
                        points.append(x)
                        fvals.append(fx)
                    if fx is None:
                        break
                if fx is None:
                    break

        # Fit Quadratic Model
        # Parameters: c, g (d), H (d x d symmetric) -> 1 + d + d(d+1)/2
        n_pts = len(points)
        if n_pts >= 1:
            n_params = 1 + d + d * (d + 1) // 2
            A = np.zeros((n_pts, n_params))
            y = np.array(fvals)
            
            for k, x in enumerate(points):
                A[k, 0] = 1.0
                A[k, 1:1 + d] = x
                idx = 1 + d
                for i in range(d):
                    for j in range(i, d):
                        A[k, idx] = x[i] * x[j]
                        idx += 1
                        
            try:
                # Use lstsq for stability
                theta, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
                
                c = theta[0]
                g = theta[1:1 + d]
                H_flat = theta[1 + d:]
                
                H = np.zeros((d, d))
                idx = 0
                for i in range(d):
                    for j in range(i, d):
                        val = H_flat[idx]
                        H[i, j] = val
                        if i != j:
                            H[j, i] = val
                        idx += 1
                
                # Find minimum of the quadratic model
                # x_min = -0.5 * H^{-1} * g
                try:
                    # Check if H is invertible
                    if np.linalg.matrix_rank(H) == d:
                        x_min = np.linalg.solve(H, -0.5 * g)
                        x_min = np.clip(x_min, -5, 5)
                        f_min = safe_call(x_min)
                        if f_min is not None:
                            pass # safe_call updates best_x, best_f
                except np.linalg.LinAlgError:
                    pass
            except Exception:
                pass

        # 2. Polish with (1+1)-ES using Hessian-informed initial sigma
        remaining = self.budget - calls
        
        if remaining > 0 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            
            # Estimate initial sigma from Hessian eigenvalues if available
            sigma = 0.3
            try:
                if 'H' in dir() and H is not None:
                    eigvals = np.linalg.eigvalsh(H)
                    # Use the smallest positive eigenvalue to set scale
                    pos_eigs = eigvals[eigvals > 1e-8]
                    if len(pos_eigs) > 0:
                        min_eig = np.min(pos_eigs)
                        # sigma ~ 1/sqrt(lambda_min) but scaled down
                        sigma = min(1.0, 0.5 / np.sqrt(min_eig))
            except Exception:
                sigma = 0.3
            
            rng = np.random.default_rng(42)
            
            # Adaptive sigma logic with faster convergence
            # Use 1/5 rule: 5 successes -> multiply by 1.5, 5 failures -> multiply by 0.75
            success_count = 0
            failure_count = 0
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(d)
                y = np.clip(y, -5, 5)
                fy = safe_call(y)
                if fy is None:
                    break
                
                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    failure_count = 0
                    if success_count >= 5:
                        sigma *= 1.5
                        success_count = 0
                else:
                    failure_count += 1
                    success_count = 0
                    if failure_count >= 5:
                        sigma *= 0.75
                        failure_count = 0
                
                # Floor sigma to avoid stalling
                if sigma < 1e-10:
                    sigma = 1e-10

            best_x, best_f = x, fx
            
            # 3. Final high-precision polish: very small deterministic steps
            if remaining > 10 and best_x is not None:
                x = best_x.copy()
                fx = best_f
                tiny_sigma = 1e-6
                for i in range(min(20, remaining)):
                    if time.time() > deadline:
                        break
                    # Try small perturbations in each coordinate
                    for coord in range(d):
                        if time.time() > deadline:
                            break
                        for direction in [1, -1]:
                            y = x.copy()
                            y[coord] += direction * tiny_sigma
                            y = np.clip(y, -5, 5)
                            fy = safe_call(y)
                            if fy is None:
                                break
                            if fy < fx:
                                x, fx = y, fy
                        if fy is None:
                            break
                    if fy is None:
                        break
                
                best_x, best_f = x, fx

        if best_x is None:
            best_x = np.zeros(d)
            best_f = 0.0

        return best_x, best_f
