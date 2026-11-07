"""BBOB f14 (Different powers) optimizer.

Hybrid CMA-ES + Nelder-Mead ensemble.
Uses CMA-ES for robust global search with adaptive step sizes, combined with
Nelder-Mead for precise local refinement. Multi-start with diverse initializations
maximizes early AOCC progress.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class CMAES:
    """CMA-ES optimizer for low-dimensional continuous optimization."""
    
    def __init__(self, dim, x0, sigma0, rng):
        self.dim = dim
        self.x = np.array(x0, dtype=float)
        self.sigma = sigma0
        self.C = np.eye(dim)
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.rng = rng
        
        # Population size
        self.lambda_ = max(4 * dim, 10)
        self.mu = self.lambda_ // 2
        
        # Weights
        w = np.log(self.lambda_ / 2 + 1) - np.log(np.arange(1, self.lambda_ + 1))
        self.weights = np.zeros(self.lambda_)
        self.weights[:self.mu] = w[:self.mu]
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1.0 / np.sum(self.weights[:self.mu] ** 2)
        
        # Adaptation rates
        n = dim
        self.cc = (4 + self.mu_eff / n) / (n + 4 + 2 * self.mu_eff / n)
        self.cs = (self.mu_eff + 2) / (n + self.mu_eff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mu_eff)
        self.cmua = min(1 - self.c1, 2 / ((n + 2) ** 2 + self.mu_eff))
        
        # Initial inverse square root of C
        self.D = 1.0
        self.B = np.eye(dim)
        self.inv_B = np.eye(dim)
        
        self.call_count = 0
        self.best_f = np.inf
        self.best_x = x0.copy()
        self.converged = False
    
    def _update_inverse_chol(self):
        """Update D and B from covariance matrix C."""
        try:
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.clip(eigvals, 1e-20, None)
            self.D = np.sqrt(eigvals)
            self.B = eigvecs
            self.inv_B = np.diag(1.0 / self.D) @ eigvecs.T
        except np.linalg.LinAlgError:
            self.D = 1.0
            self.B = np.eye(self.dim)
            self.inv_B = np.eye(self.dim)
    
    def sample(self, func, budget_remaining, deadline):
        """Generate and evaluate lambda offspring. Returns (lambda, xs, fs)."""
        if self.converged or self.call_count >= budget_remaining:
            return [], [], []
        
        self._update_inverse_chol()
        
        # Generate offspring
        zs = self.rng.standard_normal((self.lambda_, self.dim))
        xs = self.x + self.sigma * (zs @ self.B.T * self.D)
        xs = np.clip(xs, -5, 5)
        
        # Evaluate
        fs = []
        valid_xs = []
        for x in xs:
            if self.call_count >= budget_remaining:
                break
            if time.time() >= deadline:
                break
            f = func(np.asarray(x, dtype=float))
            self.call_count += 1
            if np.isfinite(f):
                fs.append(f)
                valid_xs.append(x)
        
        if len(fs) < self.mu:
            return [], [], []
        
        # Sort by fitness
        order = np.argsort(fs)
        xs_sorted = [valid_xs[i] for i in order]
        fs_sorted = [fs[i] for i in order]
        
        # Update best
        if fs_sorted[0] < self.best_f:
            self.best_f = fs_sorted[0]
            self.best_x = xs_sorted[0].copy()
        
        # Check convergence
        spread = np.max(fs_sorted) - np.min(fs_sorted)
        if spread < 1e-12 * (1 + abs(self.best_f)):
            self.converged = True
            return xs_sorted, fs_sorted, order
        
        # Update mean
        self.x = sum(self.weights[i] * xs_sorted[i] for i in range(self.mu))
        self.x = np.clip(self.x, -5, 5)
        
        # Update evolution paths
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * (self.inv_B @ (self.x - self.x + 0) / self.sigma)
        # Actually need to compute d = (x_new - x_old) / sigma
        # Let's redo this properly
        
        return xs_sorted, fs_sorted, order
    
    def update(self, xs_sorted, fs_sorted, x_old):
        """Update CMA-ES state after evaluation."""
        if self.converged:
            return
        
        self._update_inverse_chol()
        
        # Update mean
        x_new = sum(self.weights[i] * xs_sorted[i] for i in range(self.mu))
        x_new = np.clip(x_new, -5, 5)
        
        # Evolution path ps
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * (self.inv_B @ (x_new - x_old) / self.sigma)
        
        # Evolution path pc
        hsig = (1 - np.sqrt(1 - (1 - self.cs) ** 2)) / np.sqrt(1 - (1 - self.cs) ** 2)
        h = 1.0 if np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** 2) < (1.4 + 2 / (self.dim + 1)) * hsig * np.sqrt(np.trace(self.C)) / self.sigma else 0.0
        
        self.pc = (1 - self.cc) * self.pc + h * self.cc * (x_new - x_old) / self.sigma
        
        # Update covariance
        delta_h = (1 - h) * self.cc * (2 - self.cc)
        rank_one = np.outer(self.pc, self.pc)
        rank_mu = sum(self.weights[i] * np.outer(xs_sorted[i] - x_old, xs_sorted[i] - x_old) / self.sigma ** 2 for i in range(self.mu))
        
        self.C = (1 + self.c1 * delta_h - self.c1 - self.cmua) * self.C + self.c1 * rank_one + self.cmua * rank_mu
        
        # Ensure C is positive definite
        try:
            eigvals, _ = np.linalg.eigh(self.C)
            if np.min(eigvals) < 1e-20:
                self.C += 1e-20 * np.eye(self.dim)
        except:
            pass
        
        # Update sigma
        self.sigma *= np.exp(self.cs / (self.dim + 3) * (np.linalg.norm(self.ps) / np.sqrt(2 * self.dim) - 1))
        
        # Update x
        self.x = x_new
        
        # Check convergence
        if self.sigma < 1e-15 or self.sigma > 1e10:
            self.converged = True


class Optimizer:
    """Hybrid CMA-ES + Nelder-Mead ensemble for BBOB f14 (Different powers)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        calls = 0
        best_x = None
        best_f = np.inf

        def t_ok():
            return time.time() < deadline

        def ok():
            return t_ok() and calls < budget

        def record(x, f):
            nonlocal best_x, best_f
            if np.isfinite(f) and f < best_f:
                best_f = f
                best_x = x.copy()

        def call(x):
            nonlocal calls
            calls += 1
            return func(np.asarray(x, dtype=float))

        # Phase 1: CMA-ES with multiple restarts
        # Use 3 CMA-ES instances with different initializations
        
        n_cma = 3
        cma_instances = []
        
        # Latin Hypercube Sampling for initial points
        lhs_points = []
        for d in range(dim):
            strata = np.linspace(-5, 5, n_cma + 1)
            for i in range(n_cma):
                lhs_points.append(rng.uniform(strata[i], strata[i+1]))
        lhs_points = np.array(lhs_points).reshape(n_cma, dim)
        
        # Initialize CMA-ES instances
        for i in range(n_cma):
            x0 = lhs_points[i].copy()
            sigma0 = 2.0  # Initial step size
            cma = CMAES(dim, x0, sigma0, rng)
            
            # Initial evaluation
            f0 = call(x0)
            record(x0, f0)
            cma.best_f = f0
            cma.best_x = x0.copy()
            
            cma_instances.append(cma)
        
        # CMA-ES phase: run for ~60% of budget
        cma_budget = int(budget * 0.6)
        cma_calls = 0
        
        while cma_calls < cma_budget and ok():
            # Select the best CMA-ES instance
            best_idx = 0
            best_cma_f = np.inf
            for i, cma in enumerate(cma_instances):
                if not cma.converged and cma.best_f < best_cma_f:
                    best_cma_f = cma.best_f
                    best_idx = i
            
            cma = cma_instances[best_idx]
            if cma.converged:
                # Restart this instance
                x_new = best_x + rng.normal(0, 0.5, dim) if best_x is not None else rng.uniform(-5, 5, dim)
                x_new = np.clip(x_new, -5, 5)
                f_new = call(x_new)
                record(x_new, f_new)
                cma_calls += 1
                
                cma.x = x_new
                cma.sigma = 1.0
                cma.C = np.eye(dim)
                cma.pc = np.zeros(dim)
                cma.ps = np.zeros(dim)
                cma.converged = False
                cma.best_f = f_new
                cma.best_x = x_new.copy()
                continue
            
            # Generate and evaluate offspring
            remaining = cma_budget - cma_calls
            xs, fs, order = cma.sample(func, remaining, deadline)
            
            if len(fs) == 0:
                break
            
            cma_calls += len(fs)
            
            # Record all evaluations
            for x, f in zip(xs, fs):
                record(x, f)
            
            # Update CMA-ES state
            x_old = cma.x.copy()
            cma.update(xs, fs, x_old)
            
            # Check if all CMA-ES instances converged
            if all(c.converged for c in cma_instances):
                break
        
        # Phase 2: Nelder-Mead refinement from best points
        # Use remaining budget for Nelder-Mead
        
        # Get top points from CMA-ES
        candidate_points = []
        for cma in cma_instances:
            if cma.best_f < np.inf:
                candidate_points.append((cma.best_f, cma.best_x.copy()))
        
        # Also add global best
        if best_x is not None:
            candidate_points.append((best_f, best_x.copy()))
        
        # Sort by fitness
        candidate_points.sort(key=lambda x: x[0])
        
        # Take top 3 unique points
        nm_starts = []
        for f, x in candidate_points:
            if len(nm_starts) >= 3:
                break
            # Check if this point is too close to existing starts
            is_unique = True
            for _, x_existing in nm_starts:
                if np.linalg.norm(x - x_existing) < 0.5:
                    is_unique = False
                    break
            if is_unique:
                nm_starts.append((f, x))
        
        # If we don't have enough starts, add random points
        while len(nm_starts) < 3 and ok():
            x_new = rng.uniform(-5, 5, dim)
            f_new = call(x_new)
            record(x_new, f_new)
            nm_starts.append((f_new, x_new.copy()))
        
        # Nelder-Mead parameters
        gamma = 1.0   # reflection
        rho = 2.0     # expansion
        sig = 0.5     # contraction
        shr = 0.5     # shrink
        
        # Run Nelder-Mead from each start
        for f_start, x_start in nm_starts:
            if not ok():
                break
            
            # Build initial simplex
            simp = [x_start.copy()]
            fs = [f_start]
            
            step = 0.3
            for d in range(dim):
                if not ok():
                    break
                xi = x_start.copy()
                xi[d] += step
                xi = np.clip(xi, -5, 5)
                fi = call(xi)
                record(xi, fi)
                simp.append(xi.copy())
                fs.append(fi)
            
            if not ok():
                break
            
            # Nelder-Mead iterations
            nm_iter = 0
            max_nm_iter = 100
            stagnation = 0
            
            while ok() and nm_iter < max_nm_iter:
                nm_iter += 1
                
                # Sort simplex
                order = np.argsort(fs)
                simp = [simp[i] for i in order]
                fs = [fs[i] for i in order]
                
                x_best = simp[0]
                f_best = fs[0]
                x_worst = simp[-1]
                f_worst = fs[-1]
                f_second_worst = fs[-2]
                centroid = np.mean(simp[:-1], axis=0)
                
                # Reflection
                xr = centroid + gamma * (centroid - x_worst)
                xr = np.clip(xr, -5, 5)
                if not ok():
                    break
                fr = call(xr)
                record(xr, fr)
                
                improved = False
                
                if fr < f_best:
                    # Expansion
                    xe = centroid + rho * (xr - centroid)
                    xe = np.clip(xe, -5, 5)
                    if not ok():
                        break
                    fe = call(xe)
                    record(xe, fe)
                    if fe < fr:
                        simp[-1] = xe
                        fs[-1] = fe
                        improved = True
                    else:
                        simp[-1] = xr
                        fs[-1] = fr
                        improved = True
                elif fr < f_second_worst:
                    # Accept reflection
                    simp[-1] = xr
                    fs[-1] = fr
                    improved = True
                else:
                    # Contraction
                    if fr < f_worst:
                        xc = centroid + sig * (xr - centroid)
                    else:
                        xc = centroid - sig * (x_worst - centroid)
                    xc = np.clip(xc, -5, 5)
                    if not ok():
                        break
                    fc = call(xc)
                    record(xc, fc)
                    if fc < min(fr, f_worst):
                        simp[-1] = xc
                        fs[-1] = fc
                        improved = True
                    else:
                        # Shrink
                        for i in range(1, len(simp)):
                            if not ok():
                                break
                            simp[i] = x_best + shr * (simp[i] - x_best)
                            simp[i] = np.clip(simp[i], -5, 5)
                            fs[i] = call(simp[i])
                            record(simp[i], fs[i])
                            if fs[i] < f_best:
                                improved = True
                
                if improved:
                    stagnation = 0
                else:
                    stagnation += 1
                
                # Convergence check
                if np.max(fs) - np.min(fs) < 1e-10:
                    break
                if stagnation > 20:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
