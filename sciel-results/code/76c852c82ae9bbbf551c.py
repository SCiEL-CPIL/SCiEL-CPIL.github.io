"""Combined optimizer for BBOB f24 (Lunacek bi-Rastrigin):
coarse structured basin scan + CMA-ES with adaptive restarts and local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coarse scan to locate basins, then CMA-ES with restarts and local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 290.0  # safety margin before the 300s limit

        def safe_func(x):
            """Wrap func to handle potential edge cases."""
            try:
                val = func(x)
                if not np.isfinite(val):
                    return 1e10
                return val
            except Exception:
                return 1e10

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best solution found
        best_x = None
        best_f = np.inf
        call_count = 0

        def record(x, f):
            nonlocal best_x, best_f, call_count
            if f < best_f:
                best_x = x.copy()
                best_f = f
            call_count += 1

        def eval_pt(x):
            x = np.clip(x, -5.0, 5.0)
            f = safe_func(x)
            record(x, f)
            return f

        # Phase 1: Coarse structured scan to locate basins
        # Lunacek bi-Rastrigin has two basins; try to find both
        # Use a grid-like scan with some randomness
        n_scan = min(40, budget // 10)
        
        # Try center and corners of the box
        scan_points = []
        # Center
        scan_points.append(np.zeros(dim))
        # Corners of [-5,5]^dim
        for i in range(2 ** min(dim, 4)):
            pt = np.array([(i >> j) & 1 for j in range(min(dim, 4))], dtype=float)
            pt = pt * 10 - 5
            if len(pt) < dim:
                pt = np.concatenate([pt, np.zeros(dim - len(pt))])
            scan_points.append(pt)
        
        # Add some random points
        for _ in range(n_scan - len(scan_points)):
            scan_points.append(rng.uniform(-5, 5, dim))
        
        # Evaluate scan points
        for pt in scan_points[:n_scan]:
            if call_count >= budget or time.time() > deadline:
                break
            eval_pt(pt)

        # Phase 2: Run CMA-ES from the best found point
        if best_x is not None:
            x_mean = best_x.copy()
            fx = best_f
        else:
            x_mean = rng.uniform(-5, 5, dim)
            fx = eval_pt(x_mean)

        # CMA-ES parameters
        n = dim
        lam = 4 * n  # 20 for dim=5
        mu = lam // 2  # 10
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1 / np.sum(w**2)
        
        chi_n = np.sqrt(n) * (1 - 1/(4*n) - 1/(20*n**2))
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        ds = 1 + 2 * max(0, (mu_eff - 2) / (n + mu_eff)) + cs
        cmu = 2 * (mu_eff - 2 + 1/mu_eff) / ((n + 2)**2 + mu_eff)
        dsigma = 1.0

        # Covariance matrix
        C = np.eye(n)
        Cs = np.zeros(n)
        ps = np.zeros(n)
        sigma = 1.0

        def _cov_to_b_d(C_mat, n_dim):
            """C = B D^2 B^T via Cholesky or eigendecomposition."""
            try:
                L = np.linalg.cholesky(C_mat)
                return L, np.ones(n_dim)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C_mat)
                eigvals = np.clip(eigvals, 1e-12, None)
                return eigvecs, np.sqrt(eigvals)
        
        B, D = _cov_to_b_d(C, n)

        # Adaptive restart counter
        no_improve_count = 0
        max_no_improve = 15
        restart_count = 0
        max_restarts = 3
        
        while call_count < budget and time.time() < deadline:
            # Generate lambda samples
            ys = []
            xs = []
            fs = []
            
            for k in range(lam):
                if time.time() > deadline:
                    break
                if call_count >= budget:
                    break
                    
                z = rng.standard_normal(n)
                y = B @ (D * z)
                x = np.clip(x_mean + sigma * y, -5, 5)
                ys.append(y)
                xs.append(x)
                fs.append(eval_pt(x))
                
            if call_count >= budget:
                break
            
            # Check for timeout
            if time.time() > deadline:
                break
            
            # Sort by fitness
            fs_array = np.array(fs)
            idx = np.argsort(fs_array)
            
            # Track improvement
            if fs_array[idx[0]] < best_f - 1e-10:
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # If no improvement for too long, restart with adjusted sigma
            if no_improve_count > max_no_improve:
                restart_count += 1
                if restart_count > max_restarts:
                    break
                # Try different scales on restart
                if restart_count % 2 == 1:
                    sigma *= 2.0  # Expand
                else:
                    sigma *= 0.5  # Contract
                no_improve_count = 0
                # Reset some internal state
                ps = np.zeros(n)
                Cs = np.zeros(n)
                C = np.eye(n)
                B, D = _cov_to_b_d(C, n)
                continue
            
            # Calculate new mean
            w_sorted = w[:mu]
            selected_x = [xs[i] for i in idx[:mu]]
            new_mean = np.average(selected_x, axis=0, weights=w_sorted)
            
            # Calculate C^(-1/2) for ps update
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = (eigvecs * (1/np.sqrt(eigvals))) @ eigvecs.T
            
            # Update ps
            ps = (1 - cs) * ps + cs * np.sqrt(mu_eff) * (C_inv_sqrt @ (new_mean - x_mean) / sigma)
            
            # Check hsig
            norm_ps = np.linalg.norm(ps)
            hsig = (norm_ps / chi_n - (1 - 0.5 * cs)) / (1 + cs) > 1.4
            hsig = 1.0 if hsig else 0.0
            
            # Update sigma
            sigma = sigma * np.exp((dsigma / ds) * (norm_ps / chi_n - 1))
            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 0.01, 10.0)
            
            # Update Cs
            Cs = (1 - cs) * Cs + hsig * cs * ps
            
            # Update C
            y_sel = np.array([ys[idx[i]] for i in range(mu)])
            y_bar = np.average(y_sel, axis=0, weights=w_sorted)
            
            # Term 1: Decay
            decay = 1 - cs * cs * (1 - ds) - cmu
            # Term 2: Cs Cs^T
            term_cs = cs * cs * (1 - ds) * np.outer(Cs, Cs)
            # Term 3: Weights * Outer products
            term_cmu = cmu * np.average([np.outer(y, y) for y in (y_sel - y_bar)], axis=0, weights=w_sorted)
            
            C = decay * C + term_cs + term_cmu
            
            # Symmetrize C
            C = (C + C.T) / 2
            
            # Update mean
            x_mean = new_mean
            
            # Recompute B, D for next iteration
            B, D = _cov_to_b_d(C, n)
            
            # If sigma is too small, we're likely converged
            if sigma < 0.01:
                break
        
        # Phase 3: Final polish - local refinement around best point
        remaining = budget - call_count
        if remaining > 10 and time.time() < deadline - 5 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            polish_sigma = 0.1
            for _ in range(min(remaining // 2, 50)):
                if time.time() > deadline - 5 or call_count >= budget:
                    break
                y = np.clip(x + rng.normal(0, polish_sigma, n), -5, 5)
                fy = eval_pt(y)
                if fy < fx:
                    x, fx = y, fy
                    polish_sigma *= 1.2
                else:
                    polish_sigma *= 0.9
                polish_sigma = max(1e-8, min(polish_sigma, 1.0))

        return best_x, best_f
# EVOLVE-BLOCK-END
