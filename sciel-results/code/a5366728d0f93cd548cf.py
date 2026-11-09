"""Rewritten optimizer using CMA-ES for BBOB f9 (rotated Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for the rotated Rosenbrock function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin (evaluator kills at 330s)
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))  # ~14 for dim=5
        mu = lambda_ // 2  # number of recombination weights
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mueff = 1.0 / np.sum(w ** 2)
        # Step size adaptation
        cs = (mueff + 1) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) + cs
        # Covariance matrix adaptation
        c1 = (2 + 2 * mueff / (dim + 1) ** 2) / ((dim + 2) ** 2 + 2 * mueff / (dim + 1) ** 2)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + 2))
        # Damping for step size
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial center and step size
        # Start from origin (common for BBOB), but Rosenbrock minimum is often shifted
        # Actually, for BBOB f9, the optimum is in [-4,4]^5. Let's start near center.
        # But Rosenbrock is sensitive. Let's try starting at 0.
        # However, the rotated Rosenbrock can have the valley far from 0.
        # A common heuristic for BBOB is to start at 0 or use a random start.
        # Let's use 0 as it's the standard BBOB initial point, but the shift is hidden.
        # Wait, the task says "minimizer lies in [-4, 4]^5".
        # If we start at 0, and the minimum is at, say, 3, we need to move.
        # CMA-ES is good at this.
        x0 = np.zeros(dim)
        # Initial sigma. For Rosenbrock, 1.0 or 2.0 is often good.
        # Let's try 2.0 to cover the box initially.
        sigma = 2.0

        # State variables
        xmean = x0.copy()
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)  # We will use Cholesky or eig for sampling
        # Actually, standard CMA-ES uses C^{1/2}. We can update C and use Cholesky each step.
        # Or maintain the square root. For dim=5, Cholesky is cheap.
        ps = np.zeros(dim)  # evolution path for sigma
        pc = np.zeros(dim)  # evolution path for C
        diag_C = np.ones(dim)  # for efficient update
        # Precompute constants
        hsig = (1 + 2 / (dim + 1)) * (1 - cs) ** ((budget) / (ds * (dim + dim)))  # This is for termination, we just use time
        # Actually hsig is for checking if ps is consistent.
        hsig = (1 + 2.0 / (dim + 1.0)) * (1 - cs) ** (1.0 / (ds * (dim + dim)))

        best_x = x0.copy()
        best_f = func(x0)
        calls = 1

        # Main loop
        while calls < budget:
            if time.time() > deadline:
                break

            # Sample lambda_ points
            # We need C^{1/2}. Let's compute Cholesky of C.
            # C should be positive definite.
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not PD, reset or fix
                C = np.eye(dim)
                L = np.eye(dim)

            # Sample
            z = rng.standard_normal((lambda_, dim))
            # x_i = xmean + sigma * (L @ z_i)
            # Efficient: (L @ z.T).T
            pop = xmean + sigma * (z @ L.T)

            # Evaluate
            fvals = np.zeros(lambda_)
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                x_i = np.clip(pop[i], -5, 5)
                fvals[i] = func(x_i)
                calls += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = x_i.copy()

            # Sort
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]

            # Recombination
            ymean = np.dot(w, pop[:mu])

            # Update evolution paths
            # p_sigma = (1-cs) p_sigma + sqrt(cs(2-cs)mueff) / sigma * C^{-1/2} (xmean_new - xmean_old)
            # C^{-1/2} (ymean - xmean) / sigma
            # Note: ymean is in the "z" space scaled by sigma? No.
            # x_i = xmean + sigma * z_i
            # ymean = xmean + sigma * z_ymean
            # So (ymean - xmean) / sigma = z_ymean
            # We need C^{-1/2} z_ymean
            # L^{-1} z_ymean
            z_ymean = ymean - xmean  # This is sigma * z_ymean_actual
            z_ymean_actual = z_ymean / sigma
            # Solve L x = z_ymean_actual
            try:
                c_inv_sqrt_z = np.linalg.solve(L, z_ymean_actual)
            except:
                c_inv_sqrt_z = z_ymean_actual

            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * c_inv_sqrt_z

            # Check hsig condition
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (budget - calls))) < (1.4 + 2 / (dim + 1)) * chi_n:
                hsig = 1.0
            else:
                hsig = 0.0

            # Update pc
            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mueff) * c_inv_sqrt_z

            # Update xmean
            xmean_new = xmean + ymean
            # Wait, ymean is the weighted sum of the *positions*.
            # So xmean_new = xmean + (ymean - xmean)? No.
            # ymean is the new mean in the original space.
            # So xmean_new = ymean?
            # Yes, the new mean is the weighted average of the selected points.
            xmean = ymean

            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))

            # Update C
            # rank-mu update
            # C = (1 - c1 - cmu) C + c1 (pc pc^T + (1-hsig) cs(2-cs) C) + cmu sum w_i (y_i - xmean_old)(y_i - xmean_old)^T
            # Note: y_i - xmean_old = sigma * z_i
            # So the outer products are sigma^2 * z_i z_i^T
            # The term c1 (1-hsig) cs(2-cs) C is part of the rank-1 update effectively.
            
            # Let's compute the rank-mu part
            # y_i - xmean_old = pop[i] - xmean_old (before update)
            # We stored pop as the sorted points.
            # xmean_old is the previous xmean.
            # We need to be careful with the update order.
            # Standard CMA-ES:
            # 1. Sample
            # 2. Evaluate
            # 3. Sort
            # 4. Compute ymean
            # 5. Update ps, pc
            # 6. Update xmean = xmean + ymean - xmean_old? No, xmean = ymean.
            # 7. Update sigma
            # 8. Update C

            # The vectors for C update are (x_i - xmean_old)
            # These are pop[i] - xmean_old
            # Let's recompute or store them.
            # pop is sorted. xmean_old is the xmean from before this iteration.
            # We didn't store xmean_old. Let's assume xmean was updated to ymean.
            # So we need to use the old xmean.
            # Let's refactor to keep xmean_old.

            # Actually, let's just use the standard formula with the stored pop.
            # The pop array contains the points.
            # xmean_old is the mean used to generate them.
            # Let's keep a variable xmean_old.

            # I will rewrite the loop logic slightly to track xmean_old.

            # For now, let's approximate or fix the logic.
            # The term is sum_{i=1}^mu w_i (x_i - xmean_old)(x_i - xmean_old)^T
            # x_i - xmean_old = sigma * z_i
            # So it is sigma^2 * sum w_i z_i z_i^T
            
            # Let's compute this.
            # z_i are the rows of z[:mu]
            # But z was the random numbers.
            # pop[i] = xmean_old + sigma * (L @ z_i)
            # So (pop[i] - xmean_old) / sigma = L @ z_i
            # Let v_i = (pop[i] - xmean_old) / sigma
            # Then the term is sigma^2 * sum w_i v_i v_i^T
            
            # We can compute v_i from pop and xmean_old.
            # But we updated xmean.
            # Let's keep xmean_old.

            # Let's restart the loop logic with xmean_old tracking.
            # I'll do a quick fix in the code below.

            # For this iteration, let's assume we can recover it or just use a simpler update.
            # Actually, I'll just write the correct CMA-ES code in the final block.
            pass

        return best_x, best_f


class Optimizer:
    """CMA-ES optimizer for the rotated Rosenbrock function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mueff = 1.0 / np.sum(w ** 2)
        cs = (mueff + 1) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) + cs
        c1 = (2 + 2 * mueff / (dim + 1) ** 2) / ((dim + 2) ** 2 + 2 * mueff / (dim + 1) ** 2)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + 2))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        x0 = np.zeros(dim)
        sigma = 2.0

        xmean = x0.copy()
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        best_x = x0.copy()
        best_f = func(x0)
        calls = 1

        while calls < budget:
            if time.time() > deadline:
                break

            # Sample
            try:
                L = np.linalg.cholesky(C)
            except:
                C = np.eye(dim)
                L = np.eye(dim)

            rng = np.random.default_rng(42 + calls)  # Deterministic seed for reproducibility? Or global rng?
            # Better to use a single rng instance.
            # I'll use a global rng or create one outside.
            # Let's create one outside.
            pass

        # I need to restructure to use a single rng.
        # And fix the C update.

        return best_x, best_f


class Optimizer:
    """CMA-ES optimizer for the rotated Rosenbrock function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mueff = 1.0 / np.sum(w ** 2)
        cs = (mueff + 1) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) + cs
        c1 = (2 + 2 * mueff / (dim + 1) ** 2) / ((dim + 2) ** 2 + 2 * mueff / (dim + 1) ** 2)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + 2))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        x0 = np.zeros(dim)
        sigma = 2.0

        xmean = x0.copy()
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        best_x = x0.copy()
        best_f = func(x0)
        calls = 1

        rng = np.random.default_rng(0)

        while calls < budget:
            if time.time() > deadline:
                break

            # Sample
            try:
                L = np.linalg.cholesky(C)
            except:
                C = np.eye(dim)
                L = np.eye(dim)

            z = rng.standard_normal((lambda_, dim))
            pop = xmean + sigma * (z @ L.T)

            # Evaluate
            fvals = np.zeros(lambda_)
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                x_i = np.clip(pop[i], -5, 5)
                fvals[i] = func(x_i)
                calls += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = x_i.copy()

            # Sort
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]

            # Recombination
            ymean = np.dot(w, pop[:mu])

            # Save old xmean for C update
            xmean_old = xmean.copy()

            # Update evolution paths
            # z_ymean = (ymean - xmean_old) / sigma
            z_ymean = (ymean - xmean_old) / sigma
            try:
                c_inv_sqrt_z = np.linalg.solve(L, z_ymean)
            except:
                c_inv_sqrt_z = z_ymean

            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * c_inv_sqrt_z

            # Check hsig
            # The denominator is sqrt(1 - (1-cs)^{2*iters})
            # Iters is roughly (calls - 1) / lambda_
            # Let's use a simple check
            if np.linalg.norm(ps) < (1.4 + 2 / (dim + 1)) * chi_n:
                hsig = 1.0
            else:
                hsig = 0.0

            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mueff) * c_inv_sqrt_z

            # Update xmean
            xmean = ymean

            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))

            # Update C
            # Rank-mu update
            # v_i = (pop[i] - xmean_old) / sigma
            v = (pop[:mu] - xmean_old) / sigma
            # Outer products
            # sum w_i v_i v_i^T
            # (v.T * w) @ v
            rank_mu = (v.T * w) @ v

            # Rank-1 update
            # pc pc^T + (1-hsig) cs(2-cs) C
            rank_1 = np.outer(pc, pc) + (1 - hsig) * cs * (2 - cs) * C

            C = (1 - c1 - cmu) * C + c1 * rank_1 + cmu * rank_mu

            # Symmetrize
            C = (C + C.T) / 2

        return best_x, best_f
# EVOLVE-BLOCK-END
