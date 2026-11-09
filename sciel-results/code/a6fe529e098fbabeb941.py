# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)
        
        # Population size: balance diversity and computational cost
        pop_size = 25
        
        # --- Initialization ---
        # Mix of random and peak-biased initialization for WFG7 multimodality
        X = np.zeros((pop_size, self.dim))
        # Half random
        n_rand = pop_size // 2
        X[:n_rand] = lo + (hi - lo) * rng.random((n_rand, self.dim))
        # Half peak-biased around common WFG7 peaks
        peaks = [0.35, 0.5, 0.65]
        for i in range(n_rand, pop_size):
            peak = peaks[(i - n_rand) % len(peaks)]
            X[i] = lo + (hi - lo) * (peak + 0.15 * rng.standard_normal(self.dim))
            X[i] = np.clip(X[i], lo, hi)
        
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size
        
        # Step size for mutation
        step_scale = 0.15
        min_step = 0.01
        
        # Local polish parameters
        polish_step = 0.05
        polish_iters = 3
        
        while used < self.budget and time.time() < deadline:
            # Find non-dominated points
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                le = np.all(F <= F[i], axis=1)
                lt = np.any(F < F[i], axis=1)
                le[i] = False
                if np.any(le & lt):
                    dominated[i] = True
            
            nd_idx = np.where(~dominated)[0]
            
            # Hypervolume-based parent selection for better focus
            if len(nd_idx) > 0:
                # Simple hypervolume contribution heuristic: prefer points with lower sum
                # but with some diversity
                nd_sums = np.sum(F[nd_idx], axis=1)
                # Weighted selection: lower sum = higher probability
                weights = 1.0 / (nd_sums + 1e-10)
                weights /= weights.sum()
                parent_idx = nd_idx[rng.choice(len(nd_idx), p=weights)]
            else:
                parent_idx = rng.integers(n)
            
            parent = X[parent_idx].copy()
            
            # Mutate with adaptive Gaussian noise
            mutation = step_scale * (hi - lo) * rng.standard_normal(self.dim)
            child = parent + mutation
            child = np.clip(child, lo, hi)
            
            fc = np.array(problem(child), dtype=float)
            used += 1
            
            # Local polish: try small refinements around the child
            for _ in range(polish_iters):
                if used >= self.budget or time.time() >= deadline:
                    break
                polish_mut = polish_step * (hi - lo) * rng.standard_normal(self.dim)
                polish_child = child + polish_mut
                polish_child = np.clip(polish_child, lo, hi)
                fc_polish = np.array(problem(polish_child), dtype=float)
                used += 1
                # Accept if it improves on child in at least one objective without worsening others too much
                if np.any(fc_polish < fc) and np.all(fc_polish <= fc + 0.5 * np.max(fc)):
                    child = polish_child
                    fc = fc_polish
            
            # Replace worst point
            sums = np.sum(F, axis=1)
            worst_idx = int(np.argmax(sums))
            X[worst_idx] = child
            F[worst_idx] = fc
            
            # Adaptive step reduction
            step_scale = max(min_step, step_scale * 0.99)
            polish_step = max(0.01, polish_step * 0.995)
        
        return X
# EVOLVE-BLOCK-END
