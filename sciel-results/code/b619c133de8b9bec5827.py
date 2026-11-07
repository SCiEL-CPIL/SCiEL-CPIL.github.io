import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        range_ = hi - lo
        mid = (lo + hi) / 2.0
        
        archive_x = []
        archive_f = []
        evals = 0
        ref_point = np.array([3000.0, 0.06])
        
        def time_ok():
            return time.time() < deadline

        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def is_dominated(f_new, archive_f_list):
            if len(archive_f_list) == 0:
                return False
            for f in archive_f_list:
                if dominates(f, f_new):
                    return True
            return False

        def add_to_archive(x, f):
            nonlocal evals
            # Check if dominated by existing
            if is_dominated(f, archive_f):
                return False
            
            # Remove points dominated by new
            new_archive_x = []
            new_archive_f = []
            for xi, fi in zip(archive_x, archive_f):
                if dominates(f, fi):
                    continue
                new_archive_x.append(xi)
                new_archive_f.append(fi)
                
            new_archive_x.append(x)
            new_archive_f.append(f)
            
            archive_x[:] = new_archive_x
            archive_f[:] = new_archive_f
            return True

        # --- MO-CMA-ES Implementation ---
        
        # Parameters
        N = 12  # Population size
        # Standard CMA-ES parameters for N=12, dim=4
        # mu = N/2
        mu = N // 2
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        # Variance-effectiveness
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size
        sigma = 0.5 * np.mean(range_)
        # Learning rates
        cc = (4 + mu_eff / N) / (N + 4 + 2 * mu_eff / N)
        cs = (mu_eff + 2) / (N + mu_eff + 5)
        c1 = 2 / ((N + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (weights[0] - 0.5) / ((N + 2)**2 + mu_eff))
        
        # State
        mean = mid.copy()
        C = np.eye(self.dim)
        # Cholesky decomposition of C
        # We maintain C explicitly and decompose each iteration for simplicity, 
        # given dim=4 and limited budget, this is fast enough.
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        
        # Initial population using Halton-like or random for diversity
        # Let's use random for the first generation to ensure coverage
        for _ in range(N):
            if not time_ok():
                break
            x = lo + range_ * np.random.rand(self.dim)
            f = problem(x)
            evals += 1
            add_to_archive(x, f)
            
        # Main Loop
        while evals < self.budget and time_ok():
            # 1. Sample population
            pop_x = np.zeros((N, self.dim))
            pop_f = np.zeros((N, self.n_obj))
            
            # Decompose C
            L = np.linalg.cholesky(C)
            
            for i in range(N):
                if not time_ok():
                    break
                z = np.random.randn(self.dim)
                x = mean + sigma * L @ z
                # Clip to bounds
                x = np.clip(x, lo, hi)
                pop_x[i] = x
                pop_f[i] = problem(x)
                evals += 1
                add_to_archive(x, pop_f[i])
                
            if evals >= self.budget or not time_ok():
                break
            
            # 2. Sort by dominance rank (simplified: use a heuristic or just take best mu)
            # For simplicity and speed, we can sort by a scalarization or just pick non-dominated ones.
            # Better: Sort by a simple dominance-based ranking.
            # Given N=12, we can do O(N^2) ranking.
            ranks = np.zeros(N, dtype=int)
            for i in range(N):
                count = 0
                for j in range(N):
                    if i != j and dominates(pop_f[j], pop_f[i]):
                        count += 1
                ranks[i] = count
            
            # Select best mu individuals
            idx_sorted = np.argsort(ranks)
            idx_selected = idx_sorted[:mu]
            
            # 3. Update mean
            old_mean = mean.copy()
            new_mean = np.sum(weights[:, np.newaxis] * pop_x[idx_selected], axis=0)
            
            # 4. Update step size
            sigma_old = sigma
            # Calculate norm of ps
            # ps = 1/sqrt(cs) * (ps - cs * (C^{-1/2} (new_mean - old_mean) / sigma))
            # We need C^{-1/2}. Since C = L @ L.T, C^{-1/2} = (L^{-1})^T @ L^{-1} ? 
            # Actually, it's easier to compute the norm of the step in the transformed space.
            # Let's use the standard update:
            # ps = (1-sqrt(cs)) * ps + sqrt(cs * mu_eff * (1-cs)) * (L^{-1} (new_mean - old_mean) / sigma)
            
            # Solve L @ y = (new_mean - old_mean) / sigma
            diff = (new_mean - old_mean) / sigma
            try:
                y = np.linalg.solve(L, diff)
            except np.linalg.LinAlgError:
                y = np.zeros(self.dim)
            
            ps = (1 - np.sqrt(cs)) * ps + np.sqrt(cs * mu_eff * (1 - cs)) * y
            
            sigma *= np.exp(cs / (N + 2) * (np.linalg.norm(ps) / np.sqrt(2 * self.dim) - 1))
            
            # 5. Update covariance matrix
            # pc = (1-cc) * pc + sqrt(cc * (2-cc) * mu_eff) * (new_mean - old_mean) / sigma
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * (new_mean - old_mean) / sigma
            
            # Rank-mu update
            # C = (1 + c1 * (1 - c1) + cmu) * C
            #    + c1 * (pc @ pc.T + (cc * (2-cc) * hsig) * I)
            #    + cmu * sum(weights[i] * (y_i - mean_w) @ (y_i - mean_w).T)
            
            hsig = 1.0
            # Heuristic for hsig: 1 if norm(ps) < (1.4 + 0.3 * dim) * sqrt((1-cc)/(1+cc))
            threshold = (1.4 + 0.3 * self.dim) * np.sqrt((1 - cc) / (1 + cc))
            if np.linalg.norm(ps) < threshold:
                hsig = 1.0
            else:
                hsig = 0.0
            
            # Base update
            C = (1 + c1 * (1 - c1) + cmu) * C
            C += c1 * (np.outer(pc, pc) + (cc * (2 - cc) * hsig - 1) * np.eye(self.dim))
            
            # Rank-mu update
            # Need y_i for selected individuals
            # y_i = L^{-1} (x_i - old_mean) / sigma
            for i in idx_selected:
                diff_i = (pop_x[i] - old_mean) / sigma
                try:
                    y_i = np.linalg.solve(L, diff_i)
                except np.linalg.LinAlgError:
                    y_i = np.zeros(self.dim)
                C += cmu * weights[i] * np.outer(y_i, y_i)
            
            # Symmetrize and ensure positive definite
            C = (C + C.T) / 2
            # Check eigenvalues
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-10:
                C += 1e-10 * np.eye(self.dim)
            
            mean = new_mean

        return archive_x
