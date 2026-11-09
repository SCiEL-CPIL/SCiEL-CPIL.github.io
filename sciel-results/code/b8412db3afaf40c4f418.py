"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for BBOB f1 (Sphere)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Standard CMA-ES recommendations
        lam = 4 + int(3 * np.log(dim))  # population size
        mu = lam // 2                     # number of selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))

        # Initialization
        # Start near center, but with some spread to handle shifts
        # The problem states minimizer is in [-4, 4]^5
        x_mean = np.zeros(dim)
        sigma = 2.0  # Initial step size, covers most of [-5,5]
        
        # Covariance matrix C
        C = np.eye(dim)
        # Inverse square root of C
        C_inv_sqrt = np.eye(dim)
        # Eigenvalue decomposition cache
        eigvals = np.ones(dim)
        eigvecs = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # History for progress rate
        norm_sigma = np.zeros(1)
        
        calls = 0
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls += 1
        
        # Pre-allocate population
        pop_x = np.zeros((lam, dim))
        pop_f = np.zeros(lam)
        
        # Counter for eigenvalue update (usually every 1/d_c_1 generations)
        # We'll update eigenvalues every few generations for stability and speed
        gen = 0
        last_eig_update = 0
        eig_update_interval = max(1, int(1 / (c_1 * dim)))  # Heuristic
        
        while calls < budget:
            # Check time
            if time.time() > deadline - 1.0:
                break
                
            # 1. Sample population
            # We need to update C_inv_sqrt if it's stale
            if gen - last_eig_update >= eig_update_interval:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip eigenvalues to avoid numerical issues
                eigvals = np.clip(eigvals, 1e-12, 1e12)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                last_eig_update = gen
            
            for i in range(lam):
                if calls >= budget:
                    break
                z = np.random.randn(dim)
                # Transform z using C_inv_sqrt
                # x = x_mean + sigma * C_inv_sqrt @ z
                step = C_inv_sqrt @ z
                x_new = x_mean + sigma * step
                # Clip to box
                x_new = np.clip(x_new, -5, 5)
                pop_x[i] = x_new
                pop_f[i] = func(x_new)
                calls += 1
            
            if calls >= budget:
                break
                
            # 2. Sort by fitness
            idx = np.argsort(pop_f)
            pop_x_sorted = pop_x[idx]
            pop_f_sorted = pop_f[idx]
            
            # Update best
            if pop_f_sorted[0] < best_f:
                best_f = pop_f_sorted[0]
                best_x = pop_x_sorted[0].copy()
            
            # 3. Update mean
            x_old = x_mean.copy()
            x_mean = np.sum(weights[:, None] * pop_x_sorted[:mu], axis=0)
            
            # 4. Update sigma (step size adaptation)
            # Calculate norm of the step
            # We need the inverse square root of the old C for the path update
            # Using the current C_inv_sqrt is an approximation, but standard CMA-ES
            # uses the one from the previous generation. For simplicity and speed,
            # we use the current one.
            
            # Path update for sigma
            # p_sigma = (1 - c_sigma) * p_sigma + h_sigma * sqrt(mueff) * c_sigma * C_inv_sqrt @ (x_mean - x_old) / sigma
            # h_sigma is 1 if norm(p_sigma) / sqrt(1 - (1-c_sigma)^{2*gen}) < (1.4 + 2/(dim+1)) * chi_n
            # We approximate h_sigma = 1 if the norm is small enough.
            
            diff = x_mean - x_old
            norm_diff = np.linalg.norm(diff)
            
            # Approximate h_sigma
            # If sigma is very small, norm_diff will be small.
            # Standard check:
            # threshold = (1.4 + 2/(dim+1)) * chi_n
            # if norm(p_sigma) / sqrt(1 - (1-c_sigma)^(2*gen)) < threshold: h_sigma = 1 else 0
            # We'll simplify: if norm_diff is reasonable relative to sigma, update.
            # A common simplification in implementations is to just use the update if norm is not exploding.
            
            # More robust simple check:
            # If the step is too large, don't increase sigma?
            # Let's stick to the standard formula but compute h_sigma properly-ish.
            # We need p_sigma history.
            
            # Calculate p_sigma update
            # We need C_inv_sqrt of the *current* C (before update)
            # This is already stored in C_inv_sqrt (approximate)
            
            # h_sigma logic:
            # gen is roughly the number of generations
            # threshold = (1.4 + 2/(dim+1)) * chi_n
            # We need to track the expected norm of p_sigma
            # This is complex. A simpler heuristic for CMA-ES in practice:
            # Update p_sigma always, and check if the norm is within bounds.
            
            # Let's implement the standard update but with a simplified h_sigma
            # h_sigma = 1 if np.linalg.norm(p_sigma) / chi_n < 1.4 + 2/(dim+1)
            # Note: p_sigma is normalized by sigma in the standard derivation?
            # Actually p_sigma is in "sigma units".
            
            # Let's follow the pseudocode:
            # p_sigma = (1-c_sigma) p_sigma + h_sigma * sqrt(mueff) c_sigma * C^(-1/2) (x_mean - x_old) / sigma
            
            # To avoid complex h_sigma calculation which requires tracking history of C,
            # we can use a simplified version:
            # If the progress is consistent, h_sigma=1.
            # A common approximation is to just set h_sigma=1 if the norm of the step
            # (in sigma units) is not too large.
            
            # Let's try:
            # step_in_sigma_units = C_inv_sqrt @ diff
            step_sigma = C_inv_sqrt @ diff
            norm_step_sigma = np.linalg.norm(step_sigma)
            
            # threshold for h_sigma
            # The expected norm of z is chi_n.
            # If norm_step_sigma is close to chi_n * sqrt(mueff) * c_sigma? No.
            # The term added is sqrt(mueff) * c_sigma * step_sigma.
            # So the expected norm of the added term is sqrt(mueff) * c_sigma * chi_n.
            # The existing p_sigma has norm ~ chi_n * sqrt(1 - (1-c_sigma)^(2*gen)) / c_sigma?
            # This is getting complicated.
            
            # Alternative: Use a simpler adaptation rule if CMA-ES implementation is too heavy.
            # But CMA-ES is the best strategy for this.
            # Let's use a simplified h_sigma:
            # h_sigma = 1 if norm_step_sigma < 1.5 * chi_n else 0
            # This is a heuristic.
            
            h_sigma = 1.0 if norm_step_sigma < 1.5 * chi_n else 0.0
            
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(mueff) * c_sigma * step_sigma
            
            # Update sigma
            # sigma = sigma * exp(c_sigma / d_sigma * (||p_sigma|| / chi_n - 1))
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            
            # Clip sigma to prevent explosion or vanishing
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # 5. Update Covariance Matrix C
            # p_c = (1 - c_c) p_c + h_1 * sqrt(mueff) c_c (x_mean - x_old) / sigma
            # h_1 = 1 if ||p_c|| / sqrt(1 - (1-c_c)^(2*gen)) < (1.4 + 2/(dim+1)) * chi_n
            
            # Similar heuristic for h_1
            # The step in sigma units is step_sigma.
            # The term added to p_c is sqrt(mueff) * c_c * step_sigma
            # Expected norm is sqrt(mueff) * c_c * chi_n
            # Existing p_c norm is ~ chi_n * sqrt(1 - (1-c_c)^(2*gen)) / c_c?
            
            # Let's use the same heuristic:
            h_1 = 1.0 if norm_step_sigma < 1.5 * chi_n else 0.0
            
            p_c = (1 - c_c) * p_c + h_1 * np.sqrt(mueff) * c_c * step_sigma
            
            # Rank-one update
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_1) c_c (2 - c_c) C) + c_mu sum w_i (y_i y_i^T)
            # where y_i = (x_i - x_old) / sigma
            
            # Compute y_i
            Y = (pop_x_sorted[:mu] - x_old[None, :]) / sigma
            
            # Rank-mu update
            # sum w_i y_i y_i^T
            # This is a matrix multiplication: Y.T @ (weights[:, None] * Y)
            rank_mu_update = Y.T @ (weights[:, None] * Y)
            
            # Rank-one update
            # p_c p_c^T
            rank_one_update = np.outer(p_c, p_c)
            # Correction term (1 - h_1) c_c (2 - c_c) C
            # We need C here. We have C from previous iteration.
            # If h_1 is 1, this term is 0.
            if h_1 < 1.0:
                rank_one_update += (1 - h_1) * c_c * (2 - c_c) * C
            
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * rank_one_update + c_mu * rank_mu_update
            
            # Symmetrize C
            C = (C + C.T) / 2
            
            gen += 1

        return best_x, best_f

# EVOLVE-BLOCK-END
