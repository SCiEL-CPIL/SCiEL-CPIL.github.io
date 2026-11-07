"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid Local-Global Search for Bent Cigar Function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        # 1. Initial Broad Sampling to find a good starting point
        # We use a small number of random points to avoid getting stuck in bad local minima
        # (though Bent Cigar is unimodal, the hidden shift/rotation might make it tricky if we start far away)
        # Actually, Bent Cigar is unimodal. So any point is "good" in terms of basin, but far away is bad for AOCC.
        # We want to get close to the optimum fast.
        
        # Strategy: 
        # Phase 1: Coordinate Descent with adaptive step sizes.
        # Phase 2: Nelder-Mead for final polish.
        
        # Since we don't know the optimum, we start at 0 (center of box) or random?
        # The minimizer is in [-4, 4]^5. 0 is a reasonable guess.
        # Let's try a few random starts and pick the best one quickly, then refine.
        
        rng = np.random.default_rng(42)
        
        # Initial candidates
        candidates = []
        # Center
        candidates.append(np.zeros(self.dim))
        # A few random points
        for _ in range(5):
            candidates.append(rng.uniform(-5, 5, self.dim))
            
        # Evaluate initial candidates
        # We have 500 calls. Using 6 for init is cheap.
        best_x = candidates[0]
        best_f = func(best_x)
        calls_made = 1
        
        for x in candidates[1:]:
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
        if calls_made >= self.budget:
            return best_x, best_f
            
        # Phase 1: Coordinate Descent
        # Bent Cigar is separable in its local coordinates. 
        # Even in global coordinates, coordinate descent often works well for such functions if steps are tuned.
        # We will use a simple adaptive step size per dimension.
        
        x = best_x.copy()
        f = best_f
        
        # Initial step sizes
        steps = np.ones(self.dim) * 1.0
        
        # We will run coordinate descent until budget or time limit
        # To maximize AOCC, we want to improve quickly.
        # Coordinate descent cycles through dimensions.
        
        while calls_made < self.budget and time.time() < deadline:
            improved = False
            for d in range(self.dim):
                if calls_made >= self.budget or time.time() >= deadline:
                    break
                
                # Try positive and negative steps
                # Adaptive step: if we improved, keep step or increase slightly? 
                # If we didn't improve, decrease step.
                
                # Simple heuristic: try step, then -step
                x_pos = x.copy()
                x_pos[d] += steps[d]
                x_pos = np.clip(x_pos, -5, 5)
                f_pos = func(x_pos)
                calls_made += 1
                
                if f_pos < f:
                    x = x_pos
                    f = f_pos
                    best_x = x.copy()
                    best_f = f
                    improved = True
                    # Increase step slightly to explore further in this direction
                    steps[d] *= 1.5
                    # But cap it
                    steps[d] = min(steps[d], 10.0)
                else:
                    # Try negative
                    x_neg = x.copy()
                    x_neg[d] -= steps[d]
                    x_neg = np.clip(x_neg, -5, 5)
                    f_neg = func(x_neg)
                    calls_made += 1
                    
                    if f_neg < f:
                        x = x_neg
                        f = f_neg
                        best_x = x.copy()
                        best_f = f
                        improved = True
                        # Increase step
                        steps[d] *= 1.5
                        steps[d] = min(steps[d], 10.0)
                    else:
                        # No improvement in either direction, shrink step
                        steps[d] *= 0.5
                        # If step is very small, we are likely at a local minimum for this coordinate
                        if steps[d] < 1e-4:
                            steps[d] = 1e-4

            # If no improvement in a full cycle, we might be stuck or converged
            # For Bent Cigar, if we are in the basin, CD should keep improving until very close.
            # However, if the rotation is complex, CD might zig-zag.
            # Let's check if we should switch to Nelder-Mead if CD stalls.
            # Simple heuristic: if steps are all small, switch.
            if np.all(steps < 0.1):
                break
                
        # Phase 2: Nelder-Mead for final polish
        # Nelder-Mead is good for final convergence in low dimensions.
        # We have remaining budget.
        
        if calls_made < self.budget - 10: # Need at least dim+1 points for simplex
            # Initialize simplex around best_x
            # Use current best_x as center
            # Create simplex
            simplex = [best_x.copy()]
            for i in range(self.dim):
                v = best_x.copy()
                # Small perturbation
                if best_x[i] > 0:
                    v[i] -= 0.1
                else:
                    v[i] += 0.1
                v = np.clip(v, -5, 5)
                simplex.append(v)
                
            # Evaluate simplex
            f_simplex = []
            for v in simplex:
                if calls_made >= self.budget:
                    break
                fv = func(v)
                calls_made += 1
                f_simplex.append(fv)
                if fv < best_f:
                    best_f = fv
                    best_x = v.copy()
                    
            if len(f_simplex) == self.dim + 1:
                # Run Nelder-Mead
                # We implement a simple NM here to avoid scipy dependency issues if any, 
                # but scipy is available. Let's use scipy.optimize.minimize for robustness?
                # No, let's stick to numpy for speed and control.
                
                # Sort simplex by function value
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                # NM Parameters
                alpha = 1.0  # Reflection
                gamma = 2.0  # Expansion
                rho = 0.5    # Contraction
                sigma = 0.5  # Shrink
                
                while calls_made < self.budget and time.time() < deadline:
                    # Centroid of best dim points
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    x_r = centroid + alpha * (centroid - simplex[-1])
                    x_r = np.clip(x_r, -5, 5)
                    f_r = func(x_r)
                    calls_made += 1
                    if f_r < best_f:
                        best_f = f_r
                        best_x = x_r.copy()
                        
                    if f_simplex[0] <= f_r < f_simplex[-2]:
                        # Replace worst
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                    elif f_r < f_simplex[0]:
                        # Expansion
                        x_e = centroid + gamma * (x_r - centroid)
                        x_e = np.clip(x_e, -5, 5)
                        f_e = func(x_e)
                        calls_made += 1
                        if f_e < best_f:
                            best_f = f_e
                            best_x = x_e.copy()
                        if f_e < f_r:
                            simplex[-1] = x_e
                            f_simplex[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            f_simplex[-1] = f_r
                    else:
                        # Contraction
                        if f_r < f_simplex[-1]:
                            # Outside contraction
                            x_c = centroid + rho * (x_r - centroid)
                        else:
                            # Inside contraction
                            x_c = centroid - rho * (centroid - simplex[-1])
                        
                        x_c = np.clip(x_c, -5, 5)
                        f_c = func(x_c)
                        calls_made += 1
                        if f_c < best_f:
                            best_f = f_c
                            best_x = x_c.copy()
                            
                        if f_c < min(f_r, f_simplex[-1]):
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                if calls_made >= self.budget:
                                    break
                                f_i = func(simplex[i])
                                calls_made += 1
                                f_simplex[i] = f_i
                                if f_i < best_f:
                                    best_f = f_i
                                    best_x = simplex[i].copy()
                            if calls_made >= self.budget:
                                break
                            
                    # Sort again
                    order = np.argsort(f_simplex)
                    simplex = [simplex[i] for i in order]
                    f_simplex = [f_simplex[i] for i in order]
                    
                    # Convergence check
                    if abs(f_simplex[0] - f_simplex[-1]) < 1e-10:
                        break

        return best_x, best_f

# EVOLVE-BLOCK-END
