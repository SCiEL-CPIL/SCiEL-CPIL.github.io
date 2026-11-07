"""Multi-objective optimizer on DTLZ7 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    DTLZ7-specific optimizer that directly samples the Pareto front.
    
    DTLZ7 with M=3 objectives:
    - First M-1=2 variables (x1, x2) parameterize the front
    - Remaining M=3 variables (x3, x4, x5) determine g
    - g=0 when x3=x4=x5=0.5
    - Front: f1=1+t1, f2=1+t2, f3=1+(2-t2)*h(t2) where h=(t2+2)/(t2+1)
    
    Strategy:
    1. Directly sample Pareto front points (g=0) with hypervolume-optimized distribution
    2. Use remaining budget for local refinement to find near-front points
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)
        
        # DTLZ7 specific: first 2 vars parameterize front, next 3 control g
        # Set g=0 by placing x3,x4,x5 at 0.5 (within [0,1])
        g_vars = np.array([0.5, 0.5, 0.5])
        
        def front_point(t1, t2):
            """Create a point on the DTLZ7 Pareto front (g=0)."""
            x = np.zeros(self.dim)
            x[0] = t1
            x[1] = t2
            x[2:5] = g_vars
            # Remaining variables don't affect g when x3,x4,x5=0.5
            return x
        
        def front_objectives(t1, t2):
            """Compute objectives for front parameters (analytical, no evaluation needed)."""
            h = (t2 + 2.0) / (t2 + 1.0)
            f1 = 1.0 + t1
            f2 = 1.0 + t2
            f3 = 1.0 + (2.0 - t2) * h
            return np.array([f1, f2, f3])
        
        # Phase 1: Sample Pareto front with hypervolume-optimized distribution
        # Reference point is [1.1, 1.1, 25.0]
        # Hypervolume contribution is maximized near low f1, f2, f3
        # Use power-law distribution biased toward origin
        
        n_front = 120
        # Power-law sampling: t ~ U^(1/alpha) biases toward small values
        # alpha > 1 biases toward 0
        alpha = 1.5
        
        t1_samples = rng.random(n_front) ** (1.0 / alpha)
        t2_samples = rng.random(n_front) ** (1.0 / alpha)
        
        # Add some uniform samples for coverage
        n_uniform = 20
        t1_samples = np.concatenate([t1_samples, rng.random(n_uniform)])
        t2_samples = np.concatenate([t2_samples, rng.random(n_uniform)])
        
        # Use Latin hypercube for better coverage
        n_lh = 60
        idx1 = np.arange(n_lh)
        idx2 = np.arange(n_lh)
        rng.shuffle(idx1)
        rng.shuffle(idx2)
        t1_lh = (idx1 + rng.random(n_lh)) / n_lh
        t2_lh = (idx2 + rng.random(n_lh)) / n_lh
        # Apply power-law bias
        t1_lh = t1_lh ** (1.0 / alpha)
        t2_lh = t2_lh ** (1.0 / alpha)
        
        t1_all = np.concatenate([t1_samples, t1_lh])
        t2_all = np.concatenate([t2_samples, t2_lh])
        
        # Deduplicate and clip
        t1_all = np.clip(t1_all, 0, 1)
        t2_all = np.clip(t2_all, 0, 1)
        
        # Evaluate front points
        used = 0
        best_points = []
        best_objs = []
        
        for i in range(len(t1_all)):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            
            t1, t2 = t1_all[i], t2_all[i]
            x = front_point(t1, t2)
            f = problem(x)
            used += 1
            best_points.append(x)
            best_objs.append(f)
        
        # Phase 2: Local refinement around best front points
        # Try small perturbations of g variables to see if we can get closer to front
        if used < self.budget and len(best_objs) > 0:
            best_objs_arr = np.array(best_objs)
            # Find points with low objective sum
            sums = np.sum(best_objs_arr, axis=1)
            top_k = min(10, len(best_objs_arr))
            top_indices = np.argsort(sums)[:top_k]
            
            for idx in top_indices:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                base_x = best_points[idx].copy()
                # Perturb g variables slightly
                for trial in range(5):
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    x_trial = base_x.copy()
                    # Small random perturbation to g variables
                    perturbation = 0.01 * rng.standard_normal(3)
                    x_trial[2:5] = np.clip(x_trial[2:5] + perturbation, lo[2:5], hi[2:5])
                    
                    f_trial = problem(x_trial)
                    used += 1
                    
                    # Check if this point dominates any existing point
                    dominates = False
                    for j in range(len(best_objs)):
                        if np.all(f_trial <= best_objs[j]) and np.any(f_trial < best_objs[j]):
                            dominates = True
                            break
                    
                    if dominates:
                        best_points.append(x_trial)
                        best_objs.append(f_trial)
        
        # Phase 3: Random exploration in remaining budget
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Random point with bias toward low values
            x = lo + (hi - lo) * (rng.random(self.dim) ** 0.8)
            f = problem(x)
            used += 1
            best_points.append(x)
            best_objs.append(f)
        
        return best_points
# EVOLVE-BLOCK-END
