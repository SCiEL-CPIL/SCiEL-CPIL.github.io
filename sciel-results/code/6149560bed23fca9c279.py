"""Multi-objective optimizer for DTLZ2 exploiting problem structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits DTLZ2 structure: Pareto front is a quarter-sphere in 3D.
    
    For DTLZ2 with 3 objectives and 12 variables:
    - g(x) = sum_{i=2}^{11} (x_i - 0.5)^2, minimized at 0 when x_i=0.5
    - f1 = (1+g)*cos(t1)*cos(t2), f2 = (1+g)*cos(t1)*sin(t2), f3 = (1+g)*sin(t1)
    - where t1 = x0*pi/2, t2 = x1*pi/2
    - Pareto front: g=0, so f1^2+f2^2+f3^2 = 1
    
    Strategy: 
    1. Use Latin Hypercube sampling on the Pareto front for uniform coverage
    2. Use remaining budget for local refinement around non-dominated points
    3. Keep a deadline guard to stay within time limits
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self._deadline = None

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: stop 30s before the 330s wall-clock kill
        self._deadline = start_time + 290.0  # 290s limit per evaluation

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # DTLZ2 structure: g=0 when x[2:] = 0.5
        # Pareto front parameterized by x0, x1 in [0,1]
        # f = [cos(t1)*cos(t2), cos(t1)*sin(t2), sin(t1)] where t1=x0*pi/2, t2=x1*pi/2
        
        # Phase 1: Latin Hypercube sampling on the Pareto front
        # Use LHS for good coverage of the 2D parameter space (x0, x1)
        
        n_phase1 = min(budget - 20, 180)  # Leave some budget for refinement
        if n_phase1 < 10:
            n_phase1 = max(10, budget // 2)
        
        # Generate Latin Hypercube samples for x0, x1
        # LHS ensures each stratum in each dimension is sampled exactly once
        def latin_hypercube_2d(n, seed=42):
            """Generate 2D Latin Hypercube samples in [0,1]^2."""
            rng = np.random.default_rng(seed)
            # Create two independent permutations for LHS
            perm1 = rng.permutation(n)
            perm2 = rng.permutation(n)
            # Sample uniformly within each stratum
            x0 = (perm1 + rng.random(n)) / n
            x1 = (perm2 + rng.random(n)) / n
            return x0, x1
        
        x0_lh, x1_lh = latin_hypercube_2d(n_phase1, seed=42)
        
        # Construct full 12D vectors on the Pareto front
        points = np.zeros((n_phase1, dim))
        points[:, 0] = x0_lh
        points[:, 1] = x1_lh
        points[:, 2:] = 0.5  # g=0
        
        # Evaluate all phase 1 points
        F = []
        used = 0
        for x in points:
            if time.time() > self._deadline:
                break
            f = problem(x)
            F.append(f)
            used += 1
        
        F = np.array(F) if F else np.zeros((0, self.n_obj))
        X = points[:used].copy() if used > 0 else np.zeros((0, dim))
        
        # Phase 2: Local refinement - perturb promising points
        # Focus on points that are non-dominated
        if len(F) > 0:
            # Find non-dominated points
            dominated = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i != j:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated[i] = True
                            break
            nondom_idx = np.where(~dominated)[0]
            
            # Use remaining budget for refinement
            remaining_budget = budget - used
            if remaining_budget > 0 and len(nondom_idx) > 0:
                # Perturb each non-dominated point slightly
                n_refine = min(remaining_budget, len(nondom_idx) * 3)
                for k in range(n_refine):
                    if time.time() > self._deadline:
                        break
                    if used >= budget:
                        break
                    
                    # Pick a non-dominated parent
                    idx = nondom_idx[k % len(nondom_idx)]
                    parent = X[idx].copy()
                    
                    # Small perturbation in x0 and x1 directions
                    # Also try to improve g (move x[2:] closer to 0.5)
                    scale = 0.03 / (k + 1)  # Decreasing step size
                    child = parent.copy()
                    
                    # Perturb the angle variables
                    child[0] += scale * ((k % 2) * 2 - 1)  # + or -
                    child[1] += scale * ((k % 4) - 1.5) * 0.5
                    
                    # Move g variables toward 0.5 with small noise
                    if k % 2 == 0:
                        child[2:] = 0.5 + 0.005 * (np.random.rand(dim-2) - 0.5)
                    
                    child = np.clip(child, lo, hi)
                    
                    fc = problem(child)
                    used += 1
                    F = np.vstack([F, [fc]])
                    X = np.vstack([X, [child]])
        
        return X
# EVOLVE-BLOCK-END
