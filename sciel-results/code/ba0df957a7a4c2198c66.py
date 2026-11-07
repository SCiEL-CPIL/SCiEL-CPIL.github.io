"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # For dim=5, lambda=10 is a good balance
        lambda_ = 10
        mu = lambda_ // 2
        # Weights for the selected parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Control parameters
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1) - 1) - 1) + cs

        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0  # Initial step size
        C = np.eye(dim)
        Cinv_sqrt = np.eye(dim)
        Cinv = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        def eig_decomp(mat):
            # Compute eigenvalues and eigenvectors
            # Use np.linalg.eigh for symmetric matrices
            vals, vecs = np.linalg.eigh(mat)
            # Ensure positive eigenvalues for sqrt
            vals = np.maximum(vals, 1e-12)
            return vals, vecs

        # Precompute eigendecomposition for initial identity matrix
        # C = I, so eigenvectors are standard basis, eigenvalues 1
        # But we need to update this every step
        
        # Helper to update Cholesky/Eigendecomposition
        # For simplicity and speed in Python, we'll recompute eigendecomposition of C every step
        # This is O(N^3) but N=5 is tiny.
        
        calls = 0
        best_x = x_mean.copy()
        best_f = float('inf')

        # Initial evaluation
        f_mean = func(x_mean)
        calls += 1
        if f_mean < best_f:
            best_f = f_mean
            best_x = x_mean.copy()

        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            # We need to generate lambda samples
            # To save time, we can generate all at once if possible, but we need to evaluate them one by one or in batches.
            # The func call is the bottleneck.
            
            # Check if we have enough time for a full generation
            # Rough estimate: 500 calls / 320s ~ 1.56 calls/s.
            # If each call takes ~0.6s, we are in trouble.
            # But the problem says "Black-box", likely fast.
            # Let's assume we can do many calls.
            
            # Generate lambda candidates
            # We need Cinv_sqrt. 
            # Update Cinv_sqrt based on C
            # For C=I initially, Cinv_sqrt=I.
            # We need to update Cinv_sqrt when C changes.
            
            # Let's update Cinv_sqrt at the start of the loop
            # Actually, it's better to update it after C is updated.
            
            # Generate samples
            # y_i = x_mean + sigma * Cinv_sqrt @ z_i
            # We can generate all z_i at once
            Z = np.random.randn(lambda_, dim)
            # Transform Z
            # Y = X_mean + Sigma * (Cinv_sqrt @ Z.T).T
            # Efficiently: Y = X_mean + Sigma * (Z @ Cinv_sqrt.T)
            
            # Wait, I need to ensure Cinv_sqrt is up to date.
            # In the first iteration, C=I, so Cinv_sqrt=I.
            
            Y = x_mean[None, :] + sigma * (Z @ Cinv_sqrt.T)
            
            # Evaluate all
            F = np.empty(lambda_)
            for i in range(lambda_):
                if time.time() > deadline:
                    # Fill rest with inf to break
                    F[i:] = float('inf')
                    break
                x_i = Y[i]
                # Clip to box? The problem says func clips.
                # So we can just pass x_i.
                F[i] = func(x_i)
                calls += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = x_i.copy()
            
            # If we ran out of time, break
            if time.time() > deadline:
                break
            
            # Select best mu
            idx = np.argsort(F)[:mu]
            Y_sel = Y[idx]
            F_sel = F[idx]
            
            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(weights[:, None] * Y_sel, axis=0)
            
            # Update path evolution strategies
            # ps = (1-cs)ps + cs/sqrt(1-(1-cs)^2) * Cinv_sqrt @ (x_mean - x_mean_old)/sigma
            # pc = (1-cc)pc + sqrt(cc(2-cc)damps) * Cinv_sqrt @ (x_mean - x_mean_old)/sigma
            
            # We need Cinv_sqrt @ (x_mean - x_mean_old)/sigma
            # Let d = (x_mean - x_mean_old) / sigma
            d = (x_mean - x_mean_old) / sigma
            Cinv_sqrt_d = Cinv_sqrt @ d
            
            ps = (1 - cs) * ps + cs / np.sqrt(1 - (1 - cs) ** 2) * Cinv_sqrt_d
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * damps) * Cinv_sqrt_d
            
            # Update covariance matrix
            # C = (1 - c1 - cmu)C + c1 (ps ps^T + (|ps|^2/damps - 1) C) + cmu sum wi (yi x_mean) (yi x_mean)^T
            
            # Calculate rank-one update term
            # ps ps^T + (|ps|^2/damps - 1) C
            ps_norm_sq = np.dot(ps, ps)
            rank_one = np.outer(ps, ps) + (ps_norm_sq / damps - 1) * C
            
            # Calculate rank-mu update term
            # Y_sel - x_mean_old
            Y_sel_diff = Y_sel - x_mean_old[None, :]
            # Weighted outer products
            # sum wi * (yi - x_mean) (yi - x_mean)^T
            # This is (Y_sel_diff.T @ weights) @ Y_sel_diff ? No.
            # It is sum_i wi * outer(Y_sel_diff[i], Y_sel_diff[i])
            # Can be computed as (Y_sel_diff * np.sqrt(weights)[:, None]).T @ (Y_sel_diff * np.sqrt(weights)[:, None])
            # Wait, Y_sel_diff is mu x dim.
            # Weighted sum of outer products:
            # Let A = Y_sel_diff * np.sqrt(weights)[:, None]
            # Then A.T @ A = sum_i A[i].T A[i] = sum_i wi * outer(Y_sel_diff[i], Y_sel_diff[i])
            A = Y_sel_diff * np.sqrt(weights)[:, None]
            rank_mu = A.T @ A
            
            C = (1 - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu
            
            # Symmetrize C to avoid numerical drift
            C = (C + C.T) / 2
            
            # Update Cinv_sqrt
            # We need to decompose C
            # C = V D V^T
            # Cinv_sqrt = V D^-1/2 V^T
            try:
                vals, vecs = np.linalg.eigh(C)
                # Clip small negative eigenvalues
                vals = np.maximum(vals, 1e-12)
                D_inv_sqrt = 1.0 / np.sqrt(vals)
                Cinv_sqrt = (vecs * D_inv_sqrt) @ vecs.T
            except np.linalg.LinAlgError:
                # If decomposition fails, reset to identity? Or break?
                # Resetting might be dangerous. Let's just break or use identity.
                Cinv_sqrt = np.eye(dim)
            
            # Adapt step size
            # sigma = sigma * exp(cs/damps * (||ps||/E||N(0,I)|| - 1))
            # E||N(0,I)|| = sqrt(dim) * (1 - 1/(4dim) + 1/(21dim^2))
            phis = np.sqrt(dim) * (1 - 1.0/(4*dim) + 1.0/(21*dim**2))
            sigma = sigma * np.exp(cs / damps * (np.linalg.norm(ps) / phis - 1))
            
            # Prevent sigma from becoming too large or too small
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Optional: Restart logic?
            # If sigma becomes very small and progress is slow, we might be stuck.
            # But with CMA-ES, it's hard to detect "stuck" easily.
            # Given the budget, let's just run it through.

        return best_x, best_f
# EVOLVE-BLOCK-END
