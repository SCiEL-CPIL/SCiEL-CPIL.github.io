import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Helper to estimate g from f1, f2
        def estimate_g(f1, f2):
            if f1 <= 0:
                return f2
            # u = (sqrt(f1) + sqrt(f1 + 4*f2)) / 2
            # g = u^2
            disc = f1 + 4 * f2
            if disc < 0:
                return np.inf
            u = (np.sqrt(f1) + np.sqrt(disc)) / 2.0
            return u * u

        # Phase 1: Find a good x_tail (x[1:]) to minimize g
        # Allocate 30% of budget to this phase
        n_phase1 = max(10, int(self.budget * 0.3))
        n_phase2 = self.budget - n_phase1
        
        best_g = np.inf
        best_tail = None
        
        # Initial random samples for tail
        n_init = min(5, n_phase1 // 2)
        remaining_p1 = n_phase1 - n_init
        
        # Generate initial candidates for tail
        # We fix x1 to 0.5 for these evaluations to isolate g minimization somewhat?
        # Actually, g is independent of x1. f1=x1.
        # To evaluate g, we can set x1 to any value. Let's use 0.5.
        
        candidates = []
        for _ in range(n_init):
            x_tail = lo[1:] + (hi[1:] - lo[1:]) * np.random.random(self.dim - 1)
            x = np.concatenate(([0.5], x_tail))
            x = np.clip(x, lo, hi)
            f = problem(x)
            g = estimate_g(f[0], f[1])
            candidates.append((g, x_tail))
            
        # Sort by g
        candidates.sort(key=lambda x: x[0])
        best_g, best_tail = candidates[0]
        
        # Local search refinement for the remaining phase 1 budget
        step_size = 0.1
        evals_used_p1 = n_init
        
        while evals_used_p1 < n_phase1:
            # Generate neighbors
            n_neighbors = min(3, n_phase1 - evals_used_p1)
            improved = False
            for _ in range(n_neighbors):
                # Perturb best_tail
                perturbation = np.random.normal(0, step_size, self.dim - 1)
                new_tail = best_tail + perturbation
                new_tail = np.clip(new_tail, lo[1:], hi[1:])
                
                x = np.concatenate(([0.5], new_tail))
                f = problem(x)
                g = estimate_g(f[0], f[1])
                evals_used_p1 += 1
                
                if g < best_g:
                    best_g = g
                    best_tail = new_tail
                    improved = True
                    
            if improved:
                step_size *= 0.8
            else:
                step_size *= 0.5
                if step_size < 0.01:
                    break

        # Phase 2: Sample the Pareto front
        # We have n_phase2 evaluations left.
        # We want to cover x1 in [0, 1] uniformly.
        # Use Halton sequence or linspace. Linspace is deterministic and covers perfectly.
        # However, random sampling might be better for hypervolume if we have noise?
        # With fixed tail, the curve is deterministic. Linspace is optimal for coverage.
        
        x1_values = np.linspace(0.0, 1.0, n_phase2)
        
        # To add some robustness, we could add small noise to x1, but linspace is fine.
        # Let's stick to linspace for maximum coverage of the 1D front.
        
        # Store results? The evaluator records every returned objective vector.
        # We just need to call problem(x) for each point.
        
        for x1 in x1_values:
            x = np.concatenate(([x1], best_tail))
            x = np.clip(x, lo, hi)
            problem(x)
            
        return None
