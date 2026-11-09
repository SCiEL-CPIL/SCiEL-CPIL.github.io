import numpy as np
import time

class Optimizer:
    """DF-TR optimizer for BBOB-style functions with quadratic model fitting."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        n = self.dim
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(n)
        
        def eval_point(x):
            nonlocal calls, best_f, best_x
            if calls >= self.budget or time.time() > deadline:
                return None
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # Phase 1: Quick random search to find a good starting point
        rng = np.random.default_rng(42)
        initial_candidates = []
        
        # Evaluate origin
        f0 = eval_point(np.zeros(n))
        if f0 is None:
            return best_x, best_f
        initial_candidates.append((f0, np.zeros(n)))
        
        # Evaluate a few random points
        for _ in range(10):
            if calls >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-4, 4, size=n)
            f = eval_point(x)
            if f is None:
                break
            initial_candidates.append((f, x))
        
        # Pick the best initial point
        initial_candidates.sort(key=lambda t: t[0])
        x_center = initial_candidates[0][1].copy()
        
        if calls >= self.budget or time.time() > deadline:
            return best_x, best_f
        
        # Phase 2: DF-TR with quadratic model
        # Build interpolation set around x_center
        tr_radius = 1.0
        max_tr_radius = 4.0
        min_tr_radius = 1e-4
        max_iters = 20
        
        # Precompute the design matrix structure
        # For n=5: need 1 + 2n + n(n-1)/2 = 1 + 10 + 10 = 21 points
        # Points: center, ±e_i*r, (e_i+e_j)*r/sqrt(2) for i<j
        
        def build_points(center, radius):
            pts = [center.copy()]
            for i in range(n):
                xp = center.copy()
                xp[i] += radius
                pts.append(xp)
                xm = center.copy()
                xm[i] -= radius
                pts.append(xm)
            for i in range(n):
                for j in range(i + 1, n):
                    xp = center.copy()
                    xp[i] += radius / np.sqrt(2)
                    xp[j] += radius / np.sqrt(2)
                    pts.append(xp)
            return pts
        
        def fit_quadratic_model(center, points, fvals):
            """Fit quadratic model f(x) ≈ f0 + g^T (x-center) + 0.5 (x-center)^T H (x-center)"""
            m = len(points) - 1  # exclude center
            num_unknowns = n + n * (n + 1) // 2
            A = np.zeros((m, num_unknowns))
            y = np.array(fvals[1:]) - fvals[0]
            
            for k in range(m):
                d = points[k + 1] - center
                for i in range(n):
                    A[k, i] = d[i]
                for i in range(n):
                    for j in range(i, n):
                        if i == j:
                            coeff = 0.5 * d[i] * d[j]
                        else:
                            coeff = d[i] * d[j]
                        # Index for Hessian term (i,j) with i<=j
                        idx = n + i * n - i * (i - 1) // 2 + (j - i)
                        A[k, idx] = coeff
                        
            try:
                theta, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
            except:
                return None, None
                
            g = theta[:n]
            H = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    val = theta[n + i * n - i * (i - 1) // 2 + (j - i)]
                    H[i, j] = val
                    H[j, i] = val
            return g, H
        
        for iter_idx in range(max_iters):
            if time.time() > deadline:
                break
            remaining = self.budget - calls
            if remaining < 25:
                break
                
            # Build and evaluate points
            points = build_points(x_center, tr_radius)
            fvals = []
            valid = True
            for p in points:
                f = eval_point(p)
                if f is None:
                    valid = False
                    break
                fvals.append(f)
                
            if not valid or len(fvals) < 2:
                break
                
            # Fit quadratic model
            g, H = fit_quadratic_model(x_center, points, fvals)
            if g is None:
                break
                
            # Regularize Hessian if needed
            try:
                eigvals = np.linalg.eigvalsh(H)
                if np.min(eigvals) < -1e-4:
                    H += (abs(np.min(eigvals)) + 1e-4) * np.eye(n)
            except:
                H = np.eye(n) * 0.1
                
            # Compute Newton step
            try:
                step = np.linalg.solve(H, -g)
            except:
                step = -g * 0.1
                
            # Scale to trust region
            step_norm = np.linalg.norm(step)
            if step_norm > tr_radius:
                step = step * (tr_radius / step_norm)
            elif step_norm < 1e-10:
                # At minimum or gradient too small
                tr_radius *= 0.5
                if tr_radius < min_tr_radius:
                    break
                continue
                
            x_next = x_center + step
            x_next = np.clip(x_next, -5, 5)
            
            f_next = eval_point(x_next)
            if f_next is None:
                break
                
            # Accept/reject based on predicted vs actual reduction
            predicted_reduction = fvals[0] - (fvals[0] + g @ step + 0.5 * step @ H @ step)
            actual_reduction = fvals[0] - f_next
            
            if actual_reduction > 0.25 * predicted_reduction and f_next < fvals[0]:
                x_center = x_next
                tr_radius = min(tr_radius * 2.0, max_tr_radius)
            else:
                tr_radius *= 0.5
                
            if tr_radius < min_tr_radius:
                break
                
        # Phase 3: Final refinement with smaller trust region
        if time.time() < deadline and calls < self.budget - 20:
            tr_radius = min(tr_radius, 0.1)
            for _ in range(10):
                if time.time() > deadline or calls >= self.budget - 5:
                    break
                if tr_radius < 1e-5:
                    break
                    
                points = build_points(x_center, tr_radius)
                fvals = []
                valid = True
                for p in points:
                    f = eval_point(p)
                    if f is None:
                        valid = False
                        break
                    fvals.append(f)
                    
                if not valid or len(fvals) < 2:
                    break
                    
                g, H = fit_quadratic_model(x_center, points, fvals)
                if g is None:
                    break
                    
                try:
                    eigvals = np.linalg.eigvalsh(H)
                    if np.min(eigvals) < -1e-6:
                        H += (abs(np.min(eigvals)) + 1e-6) * np.eye(n)
                except:
                    H = np.eye(n) * 0.1
                    
                try:
                    step = np.linalg.solve(H, -g)
                except:
                    step = -g * 0.1
                    
                step_norm = np.linalg.norm(step)
                if step_norm > tr_radius:
                    step = step * (tr_radius / step_norm)
                elif step_norm < 1e-10:
                    tr_radius *= 0.5
                    continue
                    
                x_next = x_center + step
                x_next = np.clip(x_next, -5, 5)
                
                f_next = eval_point(x_next)
                if f_next is None:
                    break
                    
                predicted_reduction = fvals[0] - (fvals[0] + g @ step + 0.5 * step @ H @ step)
                actual_reduction = fvals[0] - f_next
                
                if actual_reduction > 0.25 * predicted_reduction and f_next < fvals[0]:
                    x_center = x_next
                    tr_radius = min(tr_radius * 1.5, 0.5)
                else:
                    tr_radius *= 0.5
                    
                if tr_radius < 1e-5:
                    break

        return best_x, best_f
