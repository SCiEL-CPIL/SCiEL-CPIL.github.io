"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 using analytic structure + gradient search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 270.0
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget
        ref = np.array([11.0, 400.0])
        
        def f0_of(x):
            """Compute f0 analytically for ZDT4."""
            s = np.sum(x**2) + 100.0 * np.sum(np.sin(10.0 * np.pi * x))
            return s * s
        
        def f1_of(x):
            """Compute f1 analytically for ZDT4."""
            x1 = x[0]
            rest = x[1:]
            g = np.sum(rest**2 * (1.0 + 10.0 * np.sin(4.0 * np.pi * rest)**2))
            return 1.0 + np.sin(4.0 * np.pi * x1)**2 + g
        
        def eval_point(x):
            """Evaluate problem and return (f0, f1)."""
            f = problem(x)
            return f
        
        def grad_f0(x):
            """Analytic gradient of f0."""
            s = np.sum(x**2) + 100.0 * np.sum(np.sin(10.0 * np.pi * x))
            ds_dx = 2.0 * x + 100.0 * 10.0 * np.pi * np.cos(10.0 * np.pi * x)
            return 2.0 * s * ds_dx
        
        def grad_f1(x):
            """Analytic gradient of f1."""
            g = np.zeros(dim)
            g[0] = 2.0 * np.sin(4.0 * np.pi * x[0]) * 4.0 * np.pi * np.cos(4.0 * np.pi * x[0])
            for i in range(1, dim):
                xi = x[i]
                sin_term = np.sin(4.0 * np.pi * xi)
                cos_term = np.cos(4.0 * np.pi * xi)
                g[i] = 2.0 * xi * (1.0 + 10.0 * sin_term**2) + xi**2 * 20.0 * sin_term * cos_term * 4.0 * np.pi
            return g
        
        archive = []
        used = 0
        
        def try_add(x):
            nonlocal used
            if used >= budget or time.time() > deadline:
                return False
            x = np.clip(x, lo, hi)
            f = eval_point(x)
            used += 1
            if f[0] < ref[0] and f[1] < ref[1]:
                archive.append((f, x))
                return True
            return False
        
        # Phase 1: Find points where f0 < 11 using structured search
        # Key: need sum(x^2) + 100*sum(sin(10*pi*x)) to be in (-sqrt(11), sqrt(11))
        # Strategy: mix variables with sin = -1 (at 0.15, 0.55, 0.95) and sin = +1 (at 0.35, 0.75)
        # Try various combinations
        
        # Generate candidate base points
        base_points = []
        
        # All zeros
        base_points.append(np.zeros(dim))
        
        # Mix of 0.15 (sin=-1) and 0.35 (sin=+1)
        for k_neg in range(0, dim + 1):
            for k_pos in range(0, dim - k_neg + 1):
                x = np.zeros(dim)
                x[:k_neg] = 0.15
                x[k_neg:k_neg+k_pos] = 0.35
                base_points.append(x)
        
        # Also try 0.55 (sin(10*pi*0.55)=sin(5.5pi)=sin(1.5pi)=-1) and 0.75 (sin(7.5pi)=sin(3.5pi)=1)
        for k_neg in range(0, dim + 1):
            for k_pos in range(0, dim - k_neg + 1):
                x = np.zeros(dim)
                x[:k_neg] = 0.55
                x[k_neg:k_neg+k_pos] = 0.75
                base_points.append(x)
        
        # Evaluate base points and keep good ones
        good_seeds = []
        for x in base_points:
            if time.time() > deadline or used >= budget:
                break
            if not try_add(x):
                # Check if it's close to good
                f = eval_point(np.clip(x, lo, hi))
                used += 1
                if f[0] < 50 and f[1] < 400:
                    good_seeds.append(x.copy())
        
        # Phase 2: Gradient-based refinement from good seeds
        for seed in good_seeds:
            if time.time() > deadline or used >= budget - 10:
                break
            
            x = seed.copy()
            step_size = 0.05
            for it in range(20):
                if time.time() > deadline or used >= budget:
                    break
                
                f0 = f0_of(x)
                f1 = f1_of(x)
                
                if f0 < ref[0] and f1 < ref[1]:
                    # We're in the good region, try to improve hypervolume
                    # Move slightly to explore nearby region
                    pert = 0.01 * rng.standard_normal(dim)
                    x_new = np.clip(x + pert, lo, hi)
                    try_add(x_new)
                else:
                    # Use gradient descent on f0 to get closer to f0 < 11
                    g0 = grad_f0(x)
                    g1 = grad_f1(x)
                    
                    # Combine gradients: primarily reduce f0, secondarily keep f1 low
                    g = g0 + 0.1 * g1
                    
                    # Normalize step
                    g_norm = np.linalg.norm(g)
                    if g_norm > 1e-10:
                        x_new = np.clip(x - step_size * g / g_norm, lo, hi)
                    else:
                        x_new = x + 0.01 * rng.standard_normal(dim)
                        x_new = np.clip(x_new, lo, hi)
                    
                    try_add(x_new)
                    x = x_new
                    
                    # Adaptive step size
                    if f0_of(x) > f0:
                        step_size *= 0.5
                    else:
                        step_size = min(step_size * 1.1, 0.1)
        
        # Phase 3: Random exploration in promising regions
        # Focus on regions where f0 is likely small
        n_random = min(50, budget - used)
        for i in range(n_random):
            if time.time() > deadline or used >= budget:
                break
            
            # Sample from a mixture: some uniform, some biased toward sin=-1 regions
            if rng.random() < 0.5:
                x = rng.random(dim) * (hi - lo) + lo
            else:
                # Biased sampling: each coordinate independently in "good" regions
                x = np.zeros(dim)
                for j in range(dim):
                    r = rng.random()
                    if r < 0.3:
                        x[j] = 0.15 + 0.05 * rng.standard_normal()
                    elif r < 0.6:
                        x[j] = 0.55 + 0.05 * rng.standard_normal()
                    elif r < 0.8:
                        x[j] = 0.35 + 0.05 * rng.standard_normal()
                    else:
                        x[j] = 0.75 + 0.05 * rng.standard_normal()
                    x[j] = np.clip(x[j], lo[j], hi[j])
            
            try_add(x)
        
        # Phase 4: Local refinement around best archive points
        if archive:
            # Sort by f0 to focus on best
            archive.sort(key=lambda p: p[0][0])
            for f, x in archive[:5]:
                if time.time() > deadline or used >= budget:
                    break
                
                for _ in range(10):
                    if time.time() > deadline or used >= budget:
                        break
                    pert = 0.02 * rng.standard_normal(dim)
                    x_new = np.clip(x + pert, lo, hi)
                    try_add(x_new)
        
        return archive
# EVOLVE-BLOCK-END
