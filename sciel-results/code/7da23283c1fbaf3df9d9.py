import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42) # Fixed seed for reproducibility within the run? The evaluator seeds np.random, but we use our own rng. The prompt says "For seeds 0 and 1 the evaluator calls np.random.seed(seed)". It doesn't say we must use np.random. Using our own rng is safer to avoid interference, but we can also use np.random. Let's use np.random to align with the "seed" hint, but default_rng is better practice. I'll stick to default_rng(0) or similar. Actually, the evaluator seeds the global state. If I use np.random, I am using that seed. If I use default_rng, I am independent. The prompt says "For seeds 0 and 1...". It implies the results should vary by seed. If I use a fixed internal seed, both seeds produce the same result. That's bad. I should use the global np.random state.
        
        # Use global np.random to respect the evaluator's seed
        pop_size = 20
        # LHS Initialization
        X = np.zeros((pop_size, self.dim))
        for i in range(self.dim):
            # Latin Hypercube
            samples = np.linspace(0, 1, pop_size, endpoint=False)
            np.random.shuffle(samples)
            X[:, i] = samples
        
        X = lo + (hi - lo) * X
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Archive: list of (x, f)
        archive = []
        for x, f in zip(X, F):
            # Check if non-dominated
            if not any(np.all(f <= af) and np.any(f < af) for _, af in archive):
                archive.append((x, f))
                # Remove dominated points from archive
                archive = [(ax, af) for ax, af in archive if not (np.all(f <= af) and np.any(f < af))]

        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Select parent
            if not archive:
                # Fallback: random
                parent_idx = np.random.randint(len(X))
                x_p = X[parent_idx]
            else:
                # Random selection from archive
                idx = np.random.randint(len(archive))
                x_p = archive[idx][0]
                
            # Mutate
            # Adaptive sigma?
            sigma = 0.1
            x_c = x_p + sigma * (hi - lo) * np.random.standard_normal(self.dim)
            x_c = np.clip(x_c, lo, hi)
            
            f_c = problem(x_c)
            used += 1
            
            # Update Archive
            # Check if f_c is dominated by any in archive
            dominated = False
            for _, af in archive:
                if np.all(af <= f_c) and np.any(af < f_c):
                    dominated = True
                    break
            
            if not dominated:
                # Add to archive
                archive.append((x_c, f_c))
                # Remove points dominated by f_c
                archive = [(ax, af) for ax, af in archive if not (np.all(f_c <= af) and np.any(f_c < af))]
            
            # Also update X, F? Not strictly necessary if we only use archive for selection.
            # But if archive is empty, we need X.
            # Let's just keep X, F updated for fallback.
            # Actually, we can just keep the last evaluated point if archive is empty.
            # But LHS should ensure archive is not empty.
            
        return X # Return value is ignored, but must be returned.
