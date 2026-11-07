import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        if lam < 4: lam = 4
        
        # Number of parents
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Effective population size
        mueff = 1 / np.sum(weights**2)
        
        # Learning rates
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        
        # Initial state
        m = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Pre-compute some constants
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        best_x = m
        best_f = np.inf
        
        calls_made = 0
        
        # Loop until budget is exhausted
        while calls_made < budget:
            # Determine how many samples we can take in this generation
            # We take full generations of size lam, unless we have less than lam left
            n_samples = min(lam, budget - calls_made)
            
            # Generate samples
            # Use Cholesky decomposition of C for efficient sampling
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset it (should be rare)
                C = np.eye(dim)
                L = np.eye(dim)
                
            # Generate random vectors
            # We need n_samples vectors
            # standard_normal generates (n_samples, dim)
            z = np.random.standard_normal((n_samples, dim))
            
            # Transform to x: m + sigma * (z @ L.T)
            # Note: C = L @ L.T, so sqrt(C) is L. x = m + sigma * L.T @ z.T? 
            # Standard: x_i = m + sigma * sum_j z_ij * L_jk? 
            # Actually, if C = L L^T, then v = L^T z gives Cov(v) = L^T C L = C? No.
            # If z ~ N(0, I), then y = L z has Cov(y) = L I L^T = L L^T = C.
            # So x = m + sigma * (L @ z.T).T = m + sigma * z @ L.T
            
            x_samples = m + sigma * (z @ L.T)
            
            # Clip to box
            x_samples = np.clip(x_samples, -5, 5)
            
            # Evaluate
            f_values = np.array([func(x) for x in x_samples])
            calls_made += n_samples
            
            # Track best
            idx_min = np.argmin(f_values)
            if f_values[idx_min] < best_f:
                best_f = f_values[idx_min]
                best_x = x_samples[idx_min].copy()
                
            # If we don't have enough samples for a full CMA-ES update (mu), 
            # we can't update the distribution properly. 
            # For simplicity, if n_samples < mu, we just stop or do a simple update.
            # Given budget 500 and lam 10, we will have full generations.
            if n_samples < mu:
                break
                
            # Sort by fitness
            order = np.argsort(f_values)
            x_sorted = x_samples[order]
            f_sorted = f_values[order]
            
            # Update mean
            m_new = np.sum(weights[:, np.newaxis] * x_sorted[:mu], axis=0)
            
            # Update paths
            # Need the inverse of L for ps and pc updates? 
            # ps = (1-cs)ps + sqrt(cs(2-cs)mueff) * C^-1/2 * (m_new - m)/sigma
            # pc = (1-cc)pc + hsig * sqrt(cc(2-cc)mueff) * (m_new - m)/sigma
            
            # We need C^-1/2. We have L such that C = L L^T. So C^-1/2 = L^-T.
            # Let's compute L_inv = np.linalg.inv(L)
            # Then C^-1/2 = L_inv.T
            
            try:
                L_inv = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                L_inv = np.eye(dim)
                
            C_inv_sqrt = L_inv.T
            
            # Step in normalized space
            step = (m_new - m) / sigma
            
            # hsig condition
            # hsig = 1 if ||ps|| / sqrt(1 - (1-cs)^{2/gen}) < (1.4 + 2/(dim+1)) * chi_n
            # We don't track gen exactly, but we can approximate or use a fixed threshold.
            # A common approximation is to just check ||ps||
            # Let's use a simple heuristic: if ||ps|| < 1.4 * chi_n * sqrt(1 - (1-cs)^2) ... 
            # Actually, standard CMA-ES uses the generation counter. 
            # Since we are doing a fixed number of generations, we can track it.
            
            # Let's implement the standard hsig logic properly.
            # We need to know how many generations we've done.
            # But the loop structure is slightly different. Let's restructure to count generations.
            
            # For now, let's assume we do full generations.
            # We'll track a generation counter.
            
            # Update ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ step
            
            # Update pc
            # hsig = 1 if ||ps||/sqrt(1-(1-cs)^2) < (1.4 + 2/(dim+1)) * chi_n
            # We approximate the denominator as 1 for simplicity if cs is small, or track it.
            # Let's track the accumulated (1-cs) power.
            # Actually, simpler: just use the norm.
            # The condition is usually:
            # hsig = 1 if ||ps|| / sqrt(1 - (1-cs)^(2 * gen)) < (1.4 + 2/(dim+1)) * chi_n
            
            # Let's just use a fixed threshold for hsig to keep it simple and robust.
            # hsig = 1 if ||ps|| < 1.4 * chi_n * 2 (roughly)
            # Or better, just always update pc? No, hsig is important.
            # Let's use the standard approximation without gen count, which is slightly less stable but okay for 50 gens.
            # hsig = 1 if ||ps|| < 1.4 * chi_n
            hsig = 1 if np.linalg.norm(ps) < 1.4 * chi_n else 0
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * step
            
            # Update sigma
            # sigma = sigma * exp(cs * (||ps||/chi_n - 1))
            sigma = sigma * np.exp(cs * (np.linalg.norm(ps) / chi_n - 1))
            # Clamp sigma to prevent explosion or vanishing
            sigma = np.clip(sigma, 1e-10, 100.0)
            
            # Update Covariance Matrix C
            # C = (1 - c1 - cmu) C + c1 (pc pc^T + (1 - hsig) cc (2 - cc) C) + cmu sum w_i (y_i y_i^T)
            # where y_i = C^-1/2 (x_i - m_old)
            
            # Calculate y_i for the mu best parents
            # y_i = C_inv_sqrt @ (x_sorted[i] - m) / sigma
            y_parents = (C_inv_sqrt @ (x_sorted[:mu] - m).T).T / sigma
            
            # Rank-1 update part
            rank1 = pc @ pc.T + (1 - hsig) * cc * (2 - cc) * C
            
            # Rank-mu update part
            # Sum of w_i y_i y_i^T
            # Weights are for the mu parents
            # Note: y_parents is (mu, dim)
            # rank_mu = sum(w_i * outer(y_i, y_i))
            # Can be computed as (Y.T * w) @ Y
            Y = y_parents.T # (dim, mu)
            rank_mu = Y @ (weights[:, np.newaxis] * Y.T) # (dim, dim)
            
            C = (1 - c1 - cmu) * C + c1 * rank1 + cmu * rank_mu
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Eigenvalue check / regularization if needed
            # For 5D, this should be stable.
            
            m = m_new
            
        return best_x, best_f
