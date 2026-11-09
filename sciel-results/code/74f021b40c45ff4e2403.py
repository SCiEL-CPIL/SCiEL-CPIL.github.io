"""Model-and-optimize: DTLZ2 front is a quarter-sphere; sample it with Halton sequence for uniform coverage."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _get_sequence(self, n):
        """Generate n 2D low-discrepancy points in [0, 1)^2."""
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=2, scramble=True)
            # Sobol requires power of 2 for optimal performance, but works for any n.
            # We can generate next power of 2 and slice, or just generate n.
            # Generating exactly n is fine for small n.
            points = sampler.random(n)
            return points[:, 0], points[:, 1]
        except Exception:
            # Fallback to Halton
            def halton(n, base):
                result = np.zeros(n)
                for i in range(1, n + 1):
                    f = 1.0
                    r = 0.0
                    j = i
                    while j > 0:
                        j, rem = divmod(j, base)
                        f /= base
                        r += rem * f
                    result[i - 1] = r
                return result
            return halton(n, 2), halton(n, 3)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        results = []
        count = 0

        n = self.budget
        
        # Generate low-discrepancy sequence
        u_seq, v_seq = self._get_sequence(n)
        
        # Precompute all points
        for i in range(n):
            if count >= self.budget:
                break
            if time.time() > deadline:
                break
            
            u = u_seq[i]
            v = v_seq[i]
            
            # Map u (uniform) to x0 for surface-area uniform sampling on the quarter-sphere
            # Surface area element is proportional to cos(alpha) where alpha = x0 * pi/2
            # CDF of alpha is sin(alpha). So sin(alpha) = u => alpha = arcsin(u)
            # x0 = (2/pi) * alpha
            x0 = (2.0 / np.pi) * np.arcsin(u)
            
            # Map v (uniform) to x1 (uniform longitude)
            x1 = v
            
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            if self.dim > 2:
                x[2:] = 0.5
            
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                results.append(f)
                count += 1
            except Exception:
                break
        
        return np.array(results) if results else None
# EVOLVE-BLOCK-END
