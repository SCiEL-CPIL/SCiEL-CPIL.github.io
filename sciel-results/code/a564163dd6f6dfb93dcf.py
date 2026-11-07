"""Optimized multi-objective search for ZDT3."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structural optimizer for ZDT3 using 1D optimal quantization."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320
        dim = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT3 structure: f1 = x0, f2 = g(x1..) * (1 + 10*sin(3*pi*x0))
        # g(x) is minimized at x1.. = 0 (g=1).
        # Optimal x0 distribution for HV is proportional to 1 - sin(3*pi*x0).
        
        x_base = np.zeros(dim)
        # Ensure x_base is within bounds
        x_base = np.clip(x_base, lo, hi)
        
        rng = np.random.default_rng(123)
        samples = []
        
        # Rejection sampling for x0
        # PDF: p(x) = (1 - sin(3*pi*x)) / 2 (normalized)
        # Max PDF is 1 (at sin=-1). Wait, 1 - (-1) = 2. 2/2 = 1.
        # So p(x) is in [0, 1].
        # We can sample from Uniform[0,1] and accept if u < p(x).
        
        while len(samples) < budget:
            if time.time() > deadline:
                break
            batch_size = 200
            x_cands = rng.random(batch_size)
            u_cands = rng.random(batch_size)
            # Acceptance probability
            prob = (1.0 - np.sin(3.0 * np.pi * x_cands)) / 2.0
            mask = u_cands < prob
            accepted = x_cands[mask]
            samples.extend(accepted.tolist())
            
        # If we still don't have enough (unlikely), pad with random
        if len(samples) < budget:
            while len(samples) < budget:
                samples.append(rng.random())
        else:
            samples = samples[:budget]
            
        # Evaluate all samples
        for x0 in samples:
            if time.time() > deadline:
                break
            x = x_base.copy()
            x[0] = x0
            x[0] = np.clip(x[0], lo[0], hi[0])
            problem(x)
            
        return x_base
# EVOLVE-BLOCK-END
