"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Derivative-free optimizer with Nelder-Mead polish for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 280.0

        # Phase 0: Identify the dominant dimension by probing each axis
        x0 = np.zeros(self.dim)
        fx0 = func(x0)
        best_x = x0.copy()
        best_fx = fx0
        calls_used = 1

        # Probe each dimension with a small step to estimate gradient magnitude
        probe_step = 1.0
        dim_gradients = np.zeros(self.dim)
        for d in range(self.dim):
            if time.time() > deadline:
                break
            xp = x0.copy()
            xp[d] += probe_step
            xp = np.clip(xp, -5, 5)
            fpx = func(xp)
            calls_used += 1

            xm = x0.copy()
            xm[d] -= probe_step
            xm = np.clip(xm, -5, 5)
            fxm = func(xm)
            calls_used += 1

            # Gradient estimate (central difference)
            dim_gradients[d] = abs(fpx - fxm) / (2 * probe_step)

            # Update best
            if fpx < best_fx:
                best_fx = fpx
                best_x = xp.copy()
            if fxm < best_fx:
                best_fx = fxm
                best_x = xm.copy()

        # Determine anisotropic sigma: larger step for dominant dimension
        max_grad = np.max(dim_gradients) if np.max(dim_gradients) > 1e-12 else 1.0
        grad_norms = dim_gradients / max_grad
        sigma_vec = np.zeros(self.dim)
        for d in range(self.dim):
            if grad_norms[d] > 0.5:
                sigma_vec[d] = 1.5
            elif grad_norms[d] > 0.1:
                sigma_vec[d] = 0.5
            else:
                sigma_vec[d] = 0.1

        # Phase 1: Anisotropic (1+1)-ES
        x = best_x.copy()
        fx = best_fx
        calls_used_local = calls_used

        for i in range(self.budget - calls_used_local - 10):
            if time.time() > deadline:
                break

            noise = rng.standard_normal(self.dim)
            y = x + sigma_vec * noise
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_used_local += 1

            if fy < fx:
                step_norms = np.abs(y - x)
                for d in range(self.dim):
                    if step_norms[d] > 1e-10:
                        target = 0.3 * np.sqrt(self.dim) * (1.0 if sigma_vec[d] > 0.5 else 0.3)
                        ratio = target / max(step_norms[d], 1e-10)
                        sigma_vec[d] *= 1.0 + 0.2 * np.log(max(ratio, 1e-10))
                        sigma_vec[d] = np.clip(sigma_vec[d], 1e-6, 5.0)
                x, fx = y, fy
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                sigma_vec *= 0.8
                sigma_vec = np.maximum(sigma_vec, 1e-7)

        # Phase 2: Nelder-Mead polish
        remaining = self.budget - calls_used_local
        if remaining > 10 and time.time() < deadline - 5.0:
            # Use Nelder-Mead with a small initial simplex around best_x
            nm_calls = [0]
            
            def nm_func(x_nm):
                if time.time() > deadline:
                    return best_fx
                nm_calls[0] += 1
                x_nm = np.clip(x_nm, -5, 5)
                return func(x_nm)
            
            # Create initial simplex
            simplex = [best_x.copy()]
            for d in range(self.dim):
                perturbed = best_x.copy()
                perturbed[d] += 0.1
                simplex.append(perturbed)
            
            try:
                result = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': min(remaining - 5, 100), 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
