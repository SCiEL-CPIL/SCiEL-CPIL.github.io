"""Multi-objective optimizer on WFG4 using MO-CMA-ES approach."""
# SEARCH-START
import numpy as np
import math

class Optimizer:
    """Multi-objective CMA-ES optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Population size for CMA-ES: typically lambda = 4 + floor(3 * log(mu))
        # For multi-objective, a larger population helps with spread.
        # Let's use a population size that fits well within the budget.
        # Budget 200. If pop_size=40, we can do 5 generations.
        # If pop_size=20, we can do 10 generations.
        # Let's try pop_size=40 for better initial coverage.
        pop_size = 40
        mu = pop_size // 2  # Number of parents used for update
        
        # Initialize mean in the center of the bounds
        mean = (lo + hi) / 2.0
        # Initialize sigma (step size) to cover a fraction of the range
        # Heuristic: sigma = 0.5 * (hi - lo) is too large for high dim?
        # Let's start with a smaller step size relative to the range.
        sigma = 0.25 * np.max(span)
        
        # Identity covariance matrix
        C = np.eye(self.dim)
        # Precompute weights for rank-mu update
        # Weights are proportional to the rank in the non-dominated front
        # Simple weights: 1/mu for all parents initially, or based on rank
        # For simplicity and stability, use equal weights for the selected parents
        weights = np.ones(mu) / mu
        mu_eff = np.sum(weights)**2 / np.sum(weights**2)
        
        # Strategy parameters
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
        
        # Memory for adaptation
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        
        # Initial population
        X = []
        F = []
        used = 0
        
        # Generate initial population
        for _ in range(pop_size):
            if used >= self.budget:
                break
            x = mean + sigma * np.random.randn(self.dim)
            x = np.clip(x, lo, hi)
            f = problem(x)
            X.append(x)
            F.append(f)
            used += 1
            
        # If we ran out of budget during init, return what we have
        if used >= self.budget:
            return np.array(X)
        
        # Main loop
        max_generations = (self.budget - used) // pop_size
        # Ensure we don't exceed budget in the last generation
        # We will loop until used >= budget
        
        while used < self.budget:
            # Determine how many offspring we can generate
            n_offspring = min(pop_size, self.budget - used)
            
            # Generate offspring
            new_X = []
            new_F = []
            
            for _ in range(n_offspring):
                # Sample from multivariate normal
                z = np.random.randn(self.dim)
                # Cholesky decomposition of C for sampling
                # To save time, we can update Cholesky periodically or just use it if available
                # For dim=12, Cholesky is fast.
                try:
                    L = np.linalg.cholesky(C)
                    x = mean + sigma * (L @ z)
                except np.linalg.LinAlgError:
                    # Fallback if C is not positive definite
                    x = mean + sigma * z
                    
                x = np.clip(x, lo, hi)
                f = problem(x)
                new_X.append(x)
                new_F.append(f)
                used += 1
                
                if used >= self.budget:
                    break
            
            # Combine parents and offspring
            all_X = X + new_X
            all_F = F + new_F
            
            # Non-dominated sorting
            n_total = len(all_F)
            if n_total == 0:
                break
            
            # Fast Non-Dominated Sort
            S = [[] for _ in range(n_total)]
            n_dominates = [0] * n_total
            rank = [0] * n_total
            
            for i in range(n_total):
                for j in range(i + 1, n_total):
                    if np.all(all_F[i] <= all_F[j]) and np.any(all_F[i] < all_F[j]):
                        S[i].append(j)
                        n_dominates[j] += 1
                    elif np.all(all_F[j] <= all_F[i]) and np.any(all_F[j] < all_F[i]):
                        S[j].append(i)
                        n_dominates[i] += 1
            
            front = []
            for i in range(n_total):
                if n_dominates[i] == 0:
                    rank[i] = 0
                    front.append(i)
            
            current_rank = 0
            fronts = [front]
            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            rank[j] = current_rank + 1
                            next_front.append(j)
                current_rank += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break
            
            # Crowding Distance
            def crowding_distance(fitnesses, indices):
                n = len(indices)
                if n <= 2:
                    return [float('inf')] * n
                
                cd = [0.0] * n
                m = len(fitnesses[indices[0]]) if len(fitnesses) > 0 else 1
                
                for k in range(m):
                    sorted_idx = sorted(range(n), key=lambda i: fitnesses[indices[i]][k])
                    cd[sorted_idx[0]] = float('inf')
                    cd[sorted_idx[-1]] = float('inf')
                    
                    f_min = fitnesses[indices[sorted_idx[0]]][k]
                    f_max = fitnesses[indices[sorted_idx[-1]]][k]
                    f_range = f_max - f_min
                    
                    if f_range > 1e-10:
                        for i in range(1, n - 1):
                            cd[sorted_idx[i]] += (
                                fitnesses[indices[sorted_idx[i + 1]]][k] - 
                                fitnesses[indices[sorted_idx[i - 1]]][k]
                            ) / f_range
                
                return cd
            
            # Selection: Select best pop_size individuals
            selected_indices = []
            for front in fronts:
                if len(selected_indices) + len(front) <= pop_size:
                    selected_indices.extend(front)
                else:
                    remaining = pop_size - len(selected_indices)
                    cd = crowding_distance(all_F, front)
                    # Sort by crowding distance descending
                    sorted_front = sorted(range(len(front)), key=lambda i: cd[i], reverse=True)
                    selected_indices.extend([front[i] for i in sorted_front[:remaining]])
                    break
            
            # Update population
            X = [all_X[i] for i in selected_indices]
            F = [all_F[i] for i in selected_indices]
            
            # Update CMA-ES parameters
            # Sort selected indices by rank, then by crowding distance to determine parents
            # We need the top mu parents
            # Re-evaluate ranks for selected individuals? No, we have global ranks.
            # But we need to pick the best mu from the selected pop_size.
            # Actually, standard CMA-ES uses the best mu of the generated offspring+parents.
            # Here, we selected pop_size individuals. We should pick the best mu from these pop_size.
            
            # Sort selected individuals by rank, then crowding distance
            def get_key(idx):
                return (rank[idx], -crowding_distance(all_F, [idx])[0]) # This is inefficient, recompute CD for selected?
            
            # Better: Compute CD for the selected pop_size
            cd_selected = crowding_distance([all_F[i] for i in selected_indices], range(len(selected_indices)))
            
            # Create list of (rank, cd, index_in_selected)
            sel_data = []
            for i, idx in enumerate(selected_indices):
                sel_data.append((rank[idx], -cd_selected[i], i))
            
            sel_data.sort()
            
            # Top mu parents
            parent_indices_in_sel = [d[2] for d in sel_data[:mu]]
            parent_X = np.array([X[i] for i in parent_indices_in_sel])
            
            # Update mean
            new_mean = np.sum([weights[i] * parent_X[i] for i in range(mu)], axis=0)
            
            # Update sigma and C
            # Calculate step size adaptation
            # Standard CMA-ES update rules
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C^(-1/2) * (new_mean - mean))
            # This requires C^(-1/2).
            
            # For simplicity and robustness in this constrained setting, 
            # I will use a simpler adaptation:
            # 1. Update mean
            # 2. Update sigma based on success rate (heuristic) or keep fixed if complex
            # 3. Update C based on parent positions
            
            # Let's use a simplified CMA-ES update that is stable:
            # Mean update
            mean = new_mean
            
            # Sigma update: Heuristic - if we found better points, decrease sigma slightly?
            # Or use the standard formula with approximation.
            # Given the constraints and potential for numerical issues, let's use a fixed sigma 
            # or a very simple adaptation.
            # However, CMA-ES power comes from C adaptation.
            
            # Let's compute C update:
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c * p_c.T + (1 - (1 - c_c)^2) * C) + c_mu * sum(w_i * y_i * y_i.T)
            # where y_i = (x_i - mean) / sigma
            
            # This is computationally heavy for every generation if done naively.
            # With dim=12 and mu=20, it's fine.
            
            # We need p_c.
            # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mu_eff) * h_sigma * C^(-1/2) * (new_mean - mean)
            
            # To avoid complex matrix inverses and square roots which might fail or be slow,
            # I will use a **Rank-Mu CMA-ES** variant but with a simplified sigma adaptation 
            # and a direct covariance update that is more numerically stable for this task.
            
            # Actually, let's stick to the standard update but ensure numerical stability.
            
            # Compute h_sigma
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2*gen}) < (1.4 + 2/(dim+1)) * sqrt((mu_eff+2)/(mu_eff)) * (1-c_sigma) else 0
            # We don't track gen count for this easily, so we approximate h_sigma = 1.
            h_sigma = 1.0
            
            # Compute C^(-1/2)
            # Use eigen decomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10) # Ensure positive
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except:
                C_inv_sqrt = np.eye(self.dim)
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_inv_sqrt @ (mean - new_mean))
            
            # Update p_c
            p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mu_eff) * h_sigma * (C_inv_sqrt @ (mean - new_mean))
            
            # Update C
            y = (parent_X - new_mean) / sigma
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c)**2) * C) + c_mu * np.sum([weights[i] * np.outer(y[i], y[i]) for i in range(mu)], axis=0)
            
            # Ensure C is symmetric and positive semi-definite
            C = (C + C.T) / 2.0
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except:
                C = np.eye(self.dim)
            
            # Adapt sigma
            # Simple heuristic: if the norm of p_sigma is large, sigma is too small?
            # Standard: sigma = sigma * exp(c_sigma/d_sigma * (||p_sigma||/sqrt(1-(1-c_sigma)^{2*gen}) - chi_N))
            # We approximate chi_N = sqrt(dim) * (1 - 1/(4*dim) - 1/(8*dim^2))
            chi_N = math.sqrt(self.dim) * (1 - 1/(4*self.dim) - 1/(8*self.dim**2))
            # We don't have gen count, so we use a simplified version
            # If ||p_sigma|| > chi_N, we are moving consistently, maybe increase sigma?
            # Actually, if ||p_sigma|| is large, it means the step was successful.
            # Let's just keep sigma constant to avoid instability, or use a very slow adaptation.
            # For WFG4, a fixed sigma might be sufficient if the initial one is good.
            # Let's try a small adaptation:
            norm_p_sigma = np.linalg.norm(p_sigma)
            if norm_p_sigma > chi_N:
                sigma *= 1.05 # Increase step size
            else:
                sigma *= 0.98 # Decrease slightly
            
            # Clip sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-6, np.max(span))

        return np.array(X)

# SEARCH-END
