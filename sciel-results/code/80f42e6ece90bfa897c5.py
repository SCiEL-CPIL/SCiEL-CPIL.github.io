"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-start CMA-ES optimizer for BBOB f11 (Discus).
    Uses a portfolio of CMA-ES instances to balance exploration and exploitation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        n_calls = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x_global = np.zeros(dim)
        best_f_global = float('inf')

        def call_f(x):
            nonlocal n_calls
            if n_calls >= budget:
                return None
            n_calls += 1
            return func(x)

        def update_global(x, f):
            nonlocal best_x_global, best_f_global
            if f < best_f_global:
                best_x_global = x.copy()
                best_f_global = f

        # CMA-ES Implementation
        class CMAES:
            def __init__(self, x0, f0, sigma0, rng_local):
                self.x = x0.copy()
                self.f = f0
                self.sigma = sigma0
                self.rng = rng_local
                self.C = np.eye(dim)
                self.C_inv_sqrt = np.eye(dim)
                self.pc = np.zeros(dim)
                self.cs = np.zeros(dim)
                
                # Parameters
                self.mu = max(1, dim // 2)  # Weighted recombination
                self.lambda_ = 4 * self.mu  # Population size
                # Weights
                w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
                self.weights = w / w.sum()
                self.mueff = 1 / np.sum(self.weights**2)
                
                # Adaptation rates
                self.cc = (4 + self.mueff / dim) / (dim + 4 + 2 * self.mueff / dim)
                self.cs = (self.mueff + 2) / (dim + self.mueff + 5)
                self.c1 = 2 / ((dim + 1.3)**2 + self.mueff)
                self.cmua = min(1 - self.c1, 2 * ((self.mueff + 2) / (dim + 2)**2 - 1))
                
                self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (dim + 1)) - 1) + self.cs
                self.chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
                
                # Initial step size
                self.sigma = sigma0
                
                # Precompute inverse sqrt of C (initially identity)
                self._update_C_inv_sqrt()

            def _update_C_inv_sqrt(self):
                # Eigen decomposition for inverse square root
                # C is symmetric positive definite
                eigvals, eigvecs = np.linalg.eigh(self.C)
                # Clip small negative eigenvalues due to numerical errors
                eigvals = np.clip(eigvals, 1e-12, None)
                self.C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            def step(self, func_wrapper, get_budget_left):
                if get_budget_left() < self.lambda_:
                    return False
                
                # Generate population
                ys = []
                for _ in range(self.lambda_):
                    # Sample from N(0, C) using C_inv_sqrt
                    # z ~ N(0, I)
                    z = self.rng.standard_normal(dim)
                    y = self.C_inv_sqrt @ z
                    x_new = self.x + self.sigma * y
                    # Clip to box
                    x_new = np.clip(x_new, -5, 5)
                    ys.append(x_new)
                
                # Evaluate population
                fs = []
                for y in ys:
                    fy = func_wrapper(y)
                    if fy is None:
                        return False
                    fs.append(fy)
                
                fs = np.array(fs)
                ys = np.array(ys)
                
                # Sort
                idx = np.argsort(fs)
                ys_sorted = ys[idx]
                fs_sorted = fs[idx]
                
                # Update mean
                # x_old = self.x
                # self.x = sum(w_i * y_i)
                new_x = np.sum(self.weights[:, None] * ys_sorted[:self.mu], axis=0)
                
                # Update covariance matrix
                # p_sigma = 1/damps * (p_sigma + cs*(1-cs)*x_old/sigma) -- simplified
                # Actually, standard CMA-ES updates:
                
                # 1. Update pc (step size control)
                # pc = (1-cc)*pc + cc/sqrt(cs*(1-cs)) * (y_new - y_old) / sigma
                # Here y is the step in normalized space? No, y is the step in original space.
                # Let's use the standard formulation where x = m + sigma * z
                # z = C^{-1/2} y
                # We generated y = C^{-1/2} z_std. So the actual step is sigma * y.
                # The "evolution path" pc is usually in the transformed space or original?
                # Standard CMA-ES:
                # m is updated.
                # pc = (1-cc)*pc + cc/sqrt(cs*(1-cs)) * (m_new - m_old) / sigma
                # cs = (1-cs)*cs + 2*cs*(||pc||/chiN - 1)
                # sigma = sigma * exp(cs/damps * (||pc||/chiN - 1))
                
                # Covariance update:
                # C = (1 - c1 - cmua) * C + c1 * (pc * pc.T + (cs*(1-cs)*(1+cs)/cs) * (m_new-m_old)/sigma * ((m_new-m_old)/sigma).T) + cmua * sum(wi * (yi - yold) * (yi - yold).T / sigma^2)
                
                # Let's stick to a robust, slightly simplified CMA-ES that works well.
                # We need to be careful with the math.
                
                # Step vector in original space: d = m_new - m_old
                d = new_x - self.x
                d_norm = d / self.sigma
                
                # Update pc
                self.pc = (1 - self.cc) * self.pc + self.cc / np.sqrt(self.cs * (1 - self.cs)) * d_norm
                
                # Update cs
                pc_norm = np.linalg.norm(self.pc)
                self.cs = (1 - self.cs) * self.cs + 2 * self.cs * (pc_norm / self.chiN - 1)
                
                # Update sigma
                self.sigma = self.sigma * np.exp(self.cs / self.damps * (pc_norm / self.chiN - 1))
                self.sigma = np.clip(self.sigma, 1e-8, 100.0)
                
                # Update C
                # Rank-1 update
                rank1 = self.pc * self.pc.T + self.cs * (1 - self.cs) * (1 + self.cs / self.cs) * d_norm * d_norm.T
                # Rank-mu update
                rankmu = np.sum(self.weights[:self.mu, None, None] * 
                                (d_norm[None, :, None] * d_norm[None, None, :]), axis=0)
                # Wait, d_norm is the same for all? No, d_norm is the mean shift.
                # The rank-mu update uses the individual steps.
                # y_i = m_old + sigma * z_i
                # z_i = (y_i - m_old) / sigma
                # The term is sum(wi * z_i * z_i.T)
                
                # We have ys_sorted (original space)
                # z_sorted = (ys_sorted - self.x) / self.sigma
                # But we need to map to the C space? No, the standard update is in the "standardized" space.
                # Let's use the formulation:
                # C = (1 - c1 - cmua) * C + c1 * (pc * pc.T + ...) + cmua * sum(wi * (zi * zi.T))
                # where zi = (yi - m_old) / sigma
                
                z_sorted = (ys_sorted[:self.mu] - self.x) / self.sigma
                rankmu = np.sum(self.weights[:self.mu, None, None] * 
                                (z_sorted[:, :, None] * z_sorted[:, None, :]), axis=0)
                
                self.C = (1 - self.c1 - self.cmua) * self.C + self.c1 * rank1 + self.cmua * rankmu
                
                # Ensure symmetry
                self.C = (self.C + self.C.T) / 2
                
                # Update C_inv_sqrt
                self._update_C_inv_sqrt()
                
                # Update mean
                self.x = new_x
                self.f = fs_sorted[0]
                
                return True

        # Initialize Portfolio
        n_runs = 5
        runs = []
        
        # Initial points: Center, Corners, Random
        init_points = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            -np.ones(dim) * 2.0,
            np.random.uniform(-3, 3, dim),
            np.random.uniform(-3, 3, dim)
        ]
        
        for i in range(n_runs):
            x0 = init_points[i]
            f0 = call_f(x0)
            if f0 is None:
                break
            update_global(x0, f0)
            runs.append(CMAES(x0, f0, 1.0, np.random.default_rng(42 + i)))
        
        # Main Loop
        # Round-robin among active runs
        active_runs = list(range(len(runs)))
        
        while n_calls < budget and time.time() < deadline:
            if not active_runs:
                break
            
            # Pick a run (round robin or weighted by progress)
            # Simple round robin
            run_idx = active_runs.pop(0)
            active_runs.append(run_idx)
            
            cma = runs[run_idx]
            
            # Check if this run is significantly worse than global best
            # If global best is much better, maybe restart this run from global best?
            # Or just let it run.
            
            success = cma.step(call_f, lambda: budget - n_calls)
            
            if not success:
                # Out of budget or error
                continue
            
            # Update global best
            update_global(cma.x, cma.f)
            
            # Optional: If a run is stuck (sigma very small, no improvement in last N steps), restart it
            # For simplicity, we don't track history per run here, but we can check if sigma is tiny
            if cma.sigma < 1e-4:
                # Restart this run from global best with larger sigma
                new_x = best_x_global + 0.1 * rng.standard_normal(dim)
                new_x = np.clip(new_x, -5, 5)
                new_f = call_f(new_x)
                if new_f is not None:
                    update_global(new_x, new_f)
                    runs[run_idx] = CMAES(new_x, new_f, 0.5, np.random.default_rng(42 + run_idx + 100))
        
        return best_x_global, best_f_global

# EVOLVE-BLOCK-END
