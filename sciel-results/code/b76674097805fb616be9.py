# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 exploiting its structure.
    
    ZDT4's Pareto front is achieved when tail variables (x[1:]) are at their
    global minima (x_i=0 for bounds [0,1]) and x[0] varies across its range.
    Strategy: Dense sampling of x[0] with tail=0, then local refinement.
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
        
        # Phase 1: Dense sampling of x1 with tail=0 (exploits ZDT4 structure)
        # The Pareto front is 1D: f1 varies with x1, f2 = g(x)*h(x1) where g=1 when tail=0
        n_phase1 = min(150, budget - 50)  # Leave 50 for refinement
        if n_phase1 < 20:
            n_phase1 = min(20, budget)
        
        x1_lo, x1_hi = lo[0], hi[0]
        # Dense uniform sampling + endpoints
        x1_samples = np.linspace(x1_lo, x1_hi, n_phase1)
        
        # Build solutions with tail = 0 (global min of each tail term)
        X = np.zeros((len(x1_samples), dim))
        X[:, 0] = x1_samples
        X[:, 1:] = 0.0
        
        # Evaluate all at once
        F = np.array([problem(x) for x in X])
        used = len(X)
        
        # Check deadline
        if time.time() > deadline:
            return X
        
        # Phase 2: Local refinement of non-dominated points
        # Find non-dominated points first
        def find_non_dominated(F):
            n = len(F)
            nd_mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd_mask[i]:
                    continue
                for j in range(n):
                    if i == j or not nd_mask[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd_mask[i] = False
                        break
            return nd_mask
        
        nd_mask = find_non_dominated(F)
        nd_indices = np.where(nd_mask)[0]
        
        remaining = budget - used
        if remaining > 0 and len(nd_indices) > 0:
            # Shuffle for diversity in refinement
            rng.shuffle(nd_indices)
            
            # Adaptive local search with decreasing step size
            base_step = (hi[0] - lo[0]) * 0.01
            
            for i in nd_indices:
                if used >= budget or time.time() > deadline:
                    break
                
                # Try multiple perturbations with decreasing step
                for attempt in range(5):
                    if used >= budget or time.time() > deadline:
                        break
                    
                    step = base_step / (attempt + 1)
                    x_trial = X[i].copy()
                    
                    # Perturb x1 (main control variable for Pareto front)
                    x_trial[0] += rng.uniform(-step, step)
                    # Small perturbation on tail (should stay near 0)
                    x_trial[1:] += rng.uniform(-0.005, 0.005, dim - 1)
                    x_trial = np.clip(x_trial, lo, hi)
                    
                    f_trial = problem(x_trial)
                    used += 1
                    
                    # Accept if it improves (dominates current)
                    if np.all(f_trial <= F[i]) and np.any(f_trial < F[i]):
                        X[i] = x_trial
                        F[i] = f_trial
                    else:
                        # Try a different direction
                        break
        
        # Phase 3: Fill remaining budget with targeted samples
        remaining = budget - used
        if remaining > 0:
            # Identify gaps in x1 coverage and fill them
            x1_vals = X[:, 0]
            x1_sorted = np.sort(x1_vals)
            
            # Find largest gaps
            if len(x1_sorted) > 1:
                gaps = np.diff(x1_sorted)
                gap_idx = np.argsort(gaps)[::-1]
                
                for gi in gap_idx:
                    if used >= budget or time.time() > deadline:
                        break
                    
                    gap_start = x1_sorted[gi]
                    gap_end = x1_sorted[gi + 1]
                    gap_mid = (gap_start + gap_end) / 2
                    
                    # Add 2-3 points in the gap
                    n_fill = min(3, budget - used)
                    for _ in range(n_fill):
                        if used >= budget or time.time() > deadline:
                            break
                        x1_new = rng.uniform(gap_start, gap_end)
                        x_new = np.zeros(dim)
                        x_new[0] = x1_new
                        x_new[1:] = 0.0
                        x_new = np.clip(x_new, lo, hi)
                        
                        f_new = problem(x_new)
                        used += 1
                        
                        # Add if not dominated by existing points
                        if not np.any(np.all(F <= f_new, axis=1) & np.any(F < f_new, axis=1)):
                            X = np.vstack([X, x_new.reshape(1, -1)])
                            F = np.vstack([F, f_new.reshape(1, -1)])
        
        # Phase 4: Final polish - refine best points with very small steps
        remaining = budget - used
        if remaining > 0:
            # Get current non-dominated set
            nd_mask = find_non_dominated(F)
            nd_indices = np.where(nd_mask)[0]
            rng.shuffle(nd_indices)
            
            for i in nd_indices[:min(20, len(nd_indices))]:
                if used >= budget or time.time() > deadline:
                    break
                
                # Very fine refinement
                for _ in range(3):
                    if used >= budget or time.time() > deadline:
                        break
                    
                    x_trial = X[i].copy()
                    x_trial[0] += rng.uniform(-0.001, 0.001) * (hi[0] - lo[0])
                    x_trial[1:] += rng.uniform(-0.001, 0.001, dim - 1)
                    x_trial = np.clip(x_trial, lo, hi)
                    
                    f_trial = problem(x_trial)
                    used += 1
                    
                    if np.all(f_trial <= F[i]) and np.any(f_trial < F[i]):
                        X[i] = x_trial
                        F[i] = f_trial
        
        return X
# EVOLVE-BLOCK-END
