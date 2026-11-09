# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 exploiting its structure.
    
    ZDT4 has:
      f1(x) = 1 + 10*(n-1) + sum((xi-0.5)^2 - 10*cos(2*pi*(xi-0.5))) for i=2..n
      f2(x) = (f1(x)+1)^(-0.5)
    where x1 in [-5,5] and x2..xn in [0,1].
    
    The Pareto front is achieved when g(x)=1, i.e., all xi (i>=2) are at their
    global minima (xi=0 or xi=1 give g=1 for the standard ZDT4 formulation,
    but actually xi=0.5 gives the local min of each term... let me reconsider).
    
    Actually for ZDT4: each term (xi-0.5)^2 - 10*cos(2*pi*(xi-0.5)) has global
    minimum at xi=0 or xi=1 (value = 0.25 - 10*cos(2*pi*0.5) = 0.25 - 10*(-1) = 10.25)
    Wait, let me recalculate: at xi=0: (0-0.5)^2 - 10*cos(2*pi*(0-0.5)) = 0.25 - 10*cos(-pi) = 0.25 - 10*(-1) = 10.25
    At xi=0.5: (0)^2 - 10*cos(0) = 0 - 10 = -10
    At xi=1: (0.5)^2 - 10*cos(2*pi*0.5) = 0.25 - 10*cos(pi) = 0.25 + 10 = 10.25
    
    So the global minimum of each term is at xi=0.5 with value -10.
    The sum over 9 terms gives minimum g = 9*(-10) = -90, but g is defined as:
    g(x) = 1 + 10*(n-1) + sum(...) = 1 + 90 + sum(...)
    So minimum g = 1 + 90 + 9*(-10) = 1 + 90 - 90 = 1.
    
    So setting all tail variables to 0.5 gives g=1 which is the minimum.
    The Pareto front is f1 = x1 (wait, f1 = g(x) + ... no).
    
    Actually for ZDT4:
    f1(x) = x1 (for x1 in [-5,5])
    f2(x) = (1 + 1/(f1(x)+2)) * g(x)
    
    No wait, let me recall the standard ZDT4 definition:
    f1(x) = x1
    g(x) = 1 + 10*(n-1) + sum_{i=2}^{n} [x_i^2 - 10*cos(2*pi*x_i)]
    f2(x) = (1 - sqrt(f1(x)/x1_max)) * g(x)
    
    Hmm, there are different formulations. Let me just use the knowledge from the
    prompt: setting tail variables to 0 reduces f2 and aligns with Pareto front.
    
    Strategy: Sample x1 uniformly in its range, set tail to 0 (or 0.5), and
    evaluate. Then use remaining budget for local refinement.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, keep 20s margin
        
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Exploit ZDT4 structure
        # x1 (index 0) controls f1, tail variables control g(x)
        # Set tail to 0 (measured to be better than 0.5)
        # Sample x1 across its range to cover the Pareto front
        
        n_phase1 = min(100, budget // 2)
        if n_phase1 < 10:
            n_phase1 = min(10, budget)
        
        x1_range = (lo[0], hi[0])
        # Sample x1 values - include endpoints and uniform samples
        x1_samples = np.linspace(x1_range[0], x1_range[1], n_phase1)
        
        # Also add some random x1 samples
        n_rand = min(20, budget - n_phase1)
        if n_rand > 0:
            x1_rand = rng.uniform(x1_range[0], x1_range[1], n_rand)
            x1_samples = np.concatenate([x1_samples, x1_rand])
        
        # Build solutions with tail = 0
        X = np.zeros((len(x1_samples), dim))
        X[:, 0] = x1_samples
        # Tail variables set to 0 (measured to be better)
        X[:, 1:] = 0.0
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        used = len(X)
        
        # Check deadline
        if time.time() > deadline:
            return X
        
        # Phase 2: Try different tail configurations to find best g
        # Try tail = 0.5, tail = 1, and mixed
        remaining = budget - used
        if remaining > 0:
            # For a subset of x1 values, try different tail configs
            n_tail_test = min(30, remaining)
            if n_tail_test > 0:
                # Pick some x1 values to test
                idx = rng.choice(len(X), min(n_tail_test, len(X)), replace=False)
                
                tail_configs = [
                    np.zeros(dim - 1),       # all zeros
                    np.full(dim - 1, 0.5),   # all 0.5
                    np.ones(dim - 1),        # all ones
                ]
                
                for config in tail_configs:
                    if time.time() > deadline:
                        break
                    for i in idx:
                        if used >= budget:
                            break
                        x_test = X[i].copy()
                        x_test[1:] = config
                        f_test = problem(x_test)
                        used += 1
                        # Keep if it improves
                        if np.all(f_test <= F[i]) and np.any(f_test < F[i]):
                            X[i] = x_test
                            F[i] = f_test
        
        # Phase 3: Local search / refinement
        remaining = budget - used
        if remaining > 0:
            # Non-dominated sort to find Pareto front members
            def is_dominated(f, F):
                return np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))
            
            # Find non-dominated points
            nd_mask = np.array([not is_dominated(F[i], F) for i in range(len(F))])
            
            # Local search on non-dominated points
            n_nd = np.sum(nd_mask)
            if n_nd > 0:
                nd_indices = np.where(nd_mask)[0]
                # Shuffle for diversity
                rng.shuffle(nd_indices)
                
                step_size = 0.01
                for i in nd_indices:
                    if used >= budget or time.time() > deadline:
                        break
                    
                    # Try small perturbations
                    for _ in range(3):
                        if used >= budget or time.time() > deadline:
                            break
                        x_trial = X[i].copy()
                        # Perturb x1
                        x_trial[0] += rng.uniform(-step_size, step_size) * (hi[0] - lo[0])
                        # Perturb tail slightly
                        x_trial[1:] += rng.uniform(-0.01, 0.01, dim - 1)
                        x_trial = np.clip(x_trial, lo, hi)
                        
                        f_trial = problem(x_trial)
                        used += 1
                        
                        if np.all(f_trial <= F[i]) and np.any(f_trial < F[i]):
                            X[i] = x_trial
                            F[i] = f_trial
                            step_size *= 0.5
                        else:
                            step_size *= 0.5
        
        # Phase 4: Fill remaining budget with random restarts focused on front
        remaining = budget - used
        if remaining > 0:
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                x_new = np.random.uniform(lo, hi)
                # Bias tail toward 0
                x_new[1:] = np.random.uniform(0, 0.1, dim - 1)
                x_new = np.clip(x_new, lo, hi)
                f_new = problem(x_new)
                used += 1
                
                # Add if not dominated
                if not np.any(np.all(F <= f_new, axis=1) & np.any(F < f_new, axis=1)):
                    X = np.vstack([X, x_new.reshape(1, -1)])
                    F = np.vstack([F, f_new.reshape(1, -1)])
        
        return X
# EVOLVE-BLOCK-END
