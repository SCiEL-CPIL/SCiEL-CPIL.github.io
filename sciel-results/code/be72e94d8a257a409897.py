import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit
        
        rng = np.random  # Use global np.random seeded by evaluator
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        mid = (lo + hi) / 2.0
        
        # MO-CMA-ES parameters
        N = 8  # Population size (lambda)
        mu = N // 2  # Number of selected solutions (mu)
        dim = self.dim
        n_obj = self.n_obj
        
        # Weights
        w = np.zeros(mu)
        for i in range(mu):
            w[i] = np.log(i + 1.5) - np.log(mu + 0.5)
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        # CMA-ES hyperparameters
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Initial mean: center of bounds
        x_mean = mid.copy()
        sigma = 0.5  # Initial step size
        
        # Covariance matrix: diagonal with span-dependent initial values
        C = np.eye(dim) * (span * 0.3)**2
        
        # Eigen decomposition
        d, B = np.linalg.eigh(C)
        d = np.sqrt(np.maximum(d, 1e-12))
        
        # Path accumulators
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Archive for non-dominated points
        archive_X = []
        archive_F = []
        
        used = 0
        
        def is_dominated(f, F):
            """Check if f is dominated by any point in F."""
            for fi in F:
                if np.all(fi <= f) and np.any(fi < f):
                    return True
            return False
        
        def update_archive(X_new, F_new):
            """Update archive with new points, removing dominated ones."""
            nonlocal archive_X, archive_F
            for x, f in zip(X_new, F_new):
                # Check if new point is dominated by existing archive
                dominated = False
                for af in archive_F:
                    if np.all(af <= f) and np.any(af < f):
                        dominated = True
                        break
                if dominated:
                    continue
                # Remove archive points dominated by new point
                new_archive_X = []
                new_archive_F = []
                for ax, af in zip(archive_X, archive_F):
                    if np.all(f <= af) and np.any(f < af):
                        continue  # Skip dominated
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
                new_archive_X.append(x)
                new_archive_F.append(f)
                archive_X = new_archive_X
                archive_F = new_archive_F
        
        # Initialize: generate first population
        X = np.zeros((N, dim))
        for i in range(N):
            X[i] = x_mean + sigma * (B @ (d * rng.standard_normal(dim)))
            X[i] = np.clip(X[i], lo, hi)
        
        F = np.array([problem(x) for x in X])
        used += N
        update_archive(X, F)
        
        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Select top-mu solutions based on hypervolume contribution
            # Simple approach: select non-dominated, then by crowding-like measure
            # For efficiency, select based on rank and spread
            
            # Compute ranks
            n = len(F)
            ranks = np.zeros(n, dtype=int)
            for i in range(n):
                rank = 0
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        rank += 1
                        break
                ranks[i] = rank
            
            # Select mu solutions: prefer non-dominated, then diverse
            selected_indices = []
            # First, non-dominated
            nd_indices = [i for i in range(n) if ranks[i] == 0]
            
            if len(nd_indices) <= mu:
                selected_indices = nd_indices[:]
                # Fill remaining with next best ranks
                remaining = mu - len(selected_indices)
                for r in range(1, n):
                    rank_indices = [i for i in range(n) if ranks[i] == r and i not in selected_indices]
                    for idx in rank_indices:
                        if remaining <= 0:
                            break
                        selected_indices.append(idx)
                        remaining -= 1
                    if remaining <= 0:
                        break
            else:
                # More than mu non-dominated: select by spread
                # Use simple distance-based selection
                selected_indices = nd_indices[:mu]
            
            # Ensure we have exactly mu
            selected_indices = selected_indices[:mu]
            if len(selected_indices) < mu:
                # Fill with random
                remaining = mu - len(selected_indices)
                available = [i for i in range(n) if i not in selected_indices]
                for _ in range(remaining):
                    if available:
                        idx = rng.choice(available)
                        selected_indices.append(idx)
                        available.remove(idx)
            
            # Reorder F and X for selected
            F_sel = F[selected_indices]
            X_sel = X[selected_indices]
            
            # Compute new mean
            x_new_mean = np.sum(w[:, None] * X_sel, axis=0)
            
            # Standardized differences
            if sigma > 1e-10:
                y_w = (X_sel - x_mean) / sigma
            else:
                y_w = np.zeros_like(X_sel)
            
            # Update path accumulators
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (d * ((y_w - np.mean(y_w, axis=0)) / sigma))) / chi_n if sigma > 1e-10 else ps
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (B @ (d * ((x_new_mean - x_mean) / sigma))) if sigma > 1e-10 else pc
            
            # Update covariance matrix
            delta_h = 1.0
            if sigma > 1e-10:
                norm_ps = np.linalg.norm(ps)
                delta_h = (1 + np.sign(norm_ps**2 - (1 + 4/(dim+2)) * chi_n**2) *
                           (np.sqrt(1 - (1-c_sigma)**(2*max(1, used//N))) - (1-c_sigma)**(used//N)))
            
            C = (1 + cmu * delta_h - cmu) * C
            if sigma > 1e-10:
                for i in range(mu):
                    y_std = (y_w[i] - np.mean(y_w, axis=0)) / sigma
                    C += cmu * w[i] * np.outer(B @ (d * y_std), B @ (d * y_std))
            
            C = (C + C.T) / 2
            
            # Ensure positive definiteness
            try:
                d, B = np.linalg.eigh(C)
                d = np.sqrt(np.maximum(d, 1e-12))
            except np.linalg.LinAlgError:
                # Fallback: reset covariance
                C = np.eye(dim) * (span * 0.3)**2
                d, B = np.linalg.eigh(C)
                d = np.sqrt(np.maximum(d, 1e-12))
            
            # Update sigma
            if sigma > 1e-10:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            sigma = np.clip(sigma, 1e-8, 2.0)
            
            # Update mean
            x_mean = x_new_mean
            
            # Generate new population
            X = np.zeros((N, dim))
            for i in range(N):
                X[i] = x_mean + sigma * (B @ (d * rng.standard_normal(dim)))
                X[i] = np.clip(X[i], lo, hi)
            
            # Evaluate
            F = np.array([problem(x) for x in X])
            used += N
            update_archive(X, F)
            
            # Occasionally add a random restart point to maintain diversity
            if used % (N * 5) < N and used < self.budget:
                if time.time() > deadline:
                    break
                x_rand = lo + span * rng.random(dim)
                f_rand = problem(x_rand)
                used += 1
                update_archive([x_rand], [f_rand])
        
        # Return the archive (best non-dominated points)
        return np.array(archive_X) if archive_X else np.array([mid])
