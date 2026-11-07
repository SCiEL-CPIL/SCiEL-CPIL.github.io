"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-style (1+1) with covariance adaptation for BBOB f7 (step ellipsoid)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start = time.time()
        deadline = start + 285.0  # safety margin before 330s kill
        rng = np.random.default_rng(12345)

        # --- Phase 1: random sampling to locate a good basin -----------------
        best_x = None
        best_f = np.inf
        n_sample = 200
        for _ in range(n_sample):
            x = rng.uniform(-4.5, 4.5, d)
            f = func(x)
            if f < best_f:
                best_f, best_x = f, x
            if time.time() > deadline:
                break

        # --- Phase 2: adaptive (1+1)-ES with covariance adaptation -----------
        x = best_x.copy()
        fx = best_f
        sigma = 0.4
        # initial covariance: diagonal (isotropic)
        C = np.eye(d)
        Csqrt = np.eye(d)
        # adaptation state
        c_sigma = 0.5
        c_c = 1.0 / (d + 4.0)
        d_c = 1.0
        ps = np.zeros(d)
        pc = np.zeros(d)
        chi_d = np.sqrt(d) * (1 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d * d))
        # stagnation tracking
        no_improve = 0
        stagnation_limit = 80
        restarts = 0

        def sample_and_step(base_x, base_f, sig):
            nonlocal x, fx, C, Csqrt, ps, pc, sigma, no_improve
            z = rng.standard_normal(d)
            y = base_x + sig * (Csqrt @ z)
            y = np.clip(y, -5.0, 5.0)
            fy = func(y)
            if fy < base_f:
                x, fx = y, fy
                sigma *= 1.25
                no_improve = 0
                # rank-1 update
                yv = (y - base_x) / sig
                C = (1 - c_c) * C + c_c * np.outer(yv, yv)
                ps = (1 - c_sigma) * ps + c_sigma * yv
                ps_norm = np.linalg.norm(ps)
                if ps_norm > (1.4 + 2.0 / (d + 1)) * chi_d:
                    pc = (1 - d_c) * pc + d_c * (ps / ps_norm - (1 - c_sigma) * chi_d)
                    C = (1 - c_c) * C + (c_c + d_c) * np.outer(pc, pc)
                # re-orthogonalize Csqrt (Cholesky)
                try:
                    Csqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = C + 1e-6 * np.eye(d)
                    Csqrt = np.linalg.cholesky(C)
                return True
            else:
                sigma *= 0.8
                no_improve += 1
                return False

        for _ in range(budget - n_sample - 1):
            if time.time() > deadline:
                break
            improved = sample_and_step(x, fx, sigma)
            # stagnation restart: jump to a random point but keep best
            if no_improve >= stagnation_limit:
                restarts += 1
                # random restart around current best
                x = np.clip(x + rng.standard_normal(d) * 2.0, -5.0, 5.0)
                fx = func(x)
                # reset adaptation
                C = np.eye(d)
                Csqrt = np.eye(d)
                ps = np.zeros(d)
                pc = np.zeros(d)
                sigma = 1.0
                no_improve = 0
                # if restart got worse, go back to global best
                if fx > best_f:
                    x = best_x.copy()
                    fx = best_f
                else:
                    best_x, best_f = x.copy(), fx

        return best_x, best_f
# EVOLVE-BLOCK-END
