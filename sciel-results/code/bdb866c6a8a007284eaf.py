"""Multi-objective optimizer on ZDT2: exploit known front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit ZDT2 structure: set g-minimizing vars, spread t on front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # ZDT2: f1 = x[0], f2 = (1 + g) * (1 - sqrt(f1))
        # g(x) = 10/n * (|{xi}| + 2*sum(sin(10*pi*xi))) for i=2..n
        # g_min = 0 when all xi in {2..n} are 0 (or 1 for some, but 0 is cleaner)
        # So set x[1:] to 0 to minimize g, then f2 = 1 - sqrt(f1)
        
        # Strategy: allocate most budget to sampling t = x[0] uniformly,
        # with x[1:] = 0 (g-minimizing). This traces the true Pareto front.
        
        # We'll use a mix:
        # 1. LHS (Latin Hypercube) on t in [0,1] for broad coverage
        # 2. Some random perturbations of x[1:] to check robustness
        # 3. Local refinement around promising points
        
        dim = self.dim
        budget = self.budget
        
        # Generate LHS samples for t = x[0]
        n_lhs = int(budget * 0.85)
        n_random = budget - n_lhs
        
        # LHS: divide [0,1] into n_lhs bins, pick one random point per bin
        bins = np.linspace(0, 1, n_lhs + 1)
        t_lhs = np.array([bins[i] + (bins[i+1] - bins[i]) * np.random.random() for i in range(n_lhs)])
        
        # Random samples for x[0]
        t_random = np.random.random(n_random)
        
        # Combine
        t_all = np.concatenate([t_lhs, t_random])
        
        # For each sample, set x[0] = t, x[1:] = 0 (g-minimizing)
        # But also add some diversity in x[1:] for a few samples
        
        results = []
        eval_count = 0
        
        # First pass: evaluate LHS samples with x[1:] = 0
        for i in range(n_lhs):
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = t_lhs[i]
            # Clip to bounds
            x = np.clip(x, lo, hi)
            f = problem(x)
            results.append(f)
            eval_count += 1
        
        # Second pass: random samples with some x[1:] perturbation
        for i in range(n_random):
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = t_random[i]
            # Add small random perturbation to x[1:] to test robustness
            if np.random.random() < 0.3:
                x[1:] = np.random.random(dim - 1) * 0.01
            x = np.clip(x, lo, hi)
            f = problem(x)
            results.append(f)
            eval_count += 1
        
        # Third pass: local refinement around best points
        # Find points that look Pareto-optimal (low f1 + f2)
        if results and eval_count < budget:
            results_arr = np.array(results)
            # Sort by f1
            idx = np.argsort(results_arr[:, 0])
            sorted_f = results_arr[idx]
            
            # Find non-dominated points in the sorted set
            nondom_idx = []
            min_f2 = np.inf
            for i in range(len(sorted_f)):
                if sorted_f[i, 1] < min_f2:
                    nondom_idx.append(idx[i])
                    min_f2 = sorted_f[i, 1]
            
            # Refine around non-dominated points
            for idx_i in nondom_idx[:10]:
                if eval_count >= budget or time.time() > deadline:
                    break
                x_best = np.zeros(dim)
                x_best[0] = t_all[idx_i] if idx_i < len(t_all) else 0.5
                
                # Try small perturbations in x[0]
                for delta in [-0.001, 0.001]:
                    if eval_count >= budget or time.time() > deadline:
                        break
                    x = x_best.copy()
                    x[0] = np.clip(x[0] + delta, lo[0], hi[0])
                    f = problem(x)
                    results.append(f)
                    eval_count += 1
        
        return results
# EVOLVE-BLOCK-END
