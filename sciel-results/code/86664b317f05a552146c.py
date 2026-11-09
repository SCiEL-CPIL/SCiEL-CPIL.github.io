import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        # Initialize at 0 or random
        # 0 is often a good start for BBOB if shift is not huge, but shift is in [-4, 4].
        # Random is safer.
        x = rng.uniform(-5.0, 5.0, dim)
        f_curr = func(x)
        best_x = x.copy()
        best_f = f_curr
        calls = 1
        
        # Step sizes for each dimension
        steps = np.ones(dim) * 1.0
        min_step = 1e-4
        max_step = 10.0
        
        # Number of points to sample per dimension in 1D scan
        n_scan = 10
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(dim):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                # 1D scan along dimension i
                # Create grid points around x[i]
                # We want to cover a range of steps[i]
                # Points: x[i] - steps[i], x[i] - 0.5*steps[i], x[i], x[i] + 0.5*steps[i], x[i] + steps[i]
                # To use n_scan points, we spread them over a wider range or use adaptive range.
                # Let's use a range of [-3*steps[i], 3*steps[i]]
                # This allows moving up to 3 steps in one go.
                
                # If steps[i] is very small, we are in a local search.
                # If steps[i] is large, we are in a global search.
                
                # Generate 1D candidates
                # We evaluate n_scan points.
                # We include the current x to avoid re-evaluating if possible? 
                # But func is black box, we just call it.
                
                # To save calls, we can just evaluate the new points.
                # Let's evaluate 5 points: -2s, -s, 0, s, 2s? 
                # With n_scan=10, we can do -4.5s to 4.5s.
                
                # Let's stick to a simpler adaptive step:
                # Try +step, -step.
                # If success, keep step. If fail, shrink.
                # This is (1+1) coordinate descent.
                # But the lesson said "grid descent".
                # Let's do a small grid: -2s, -s, 0, s, 2s (5 points).
                # This is 5 calls per dim per pass.
                # 5 dims * 5 calls = 25 calls per pass.
                # 500 / 25 = 20 passes.
                
                # This seems reasonable.
                
                candidates = []
                for offset in [-2.0, -1.0, 0.0, 1.0, 2.0]:
                    x_new = x.copy()
                    x_new[i] = x[i] + offset * steps[i]
                    x_new[i] = np.clip(x_new[i], -5.0, 5.0)
                    candidates.append(x_new)
                
                best_candidate = None
                best_candidate_f = f_curr # Use current as baseline? 
                # Actually, we should evaluate all and pick the best among them.
                # This costs 5 calls.
                
                # Optimization: if we are on a plateau, we might want to move even if f doesn't change?
                # For minimization, if f is same, we can move to explore?
                # But "Step" function: if we are on the flat part, moving doesn't change f.
                # We should prefer moves that stay on the plateau or go down.
                
                for xc in candidates:
                    if calls >= budget:
                        break
                    fc = func(xc)
                    calls += 1
                    if fc < best_candidate_f - 1e-12:
                        best_candidate_f = fc
                        best_candidate = xc
                    elif abs(fc - best_candidate_f) < 1e-12 and best_candidate is None:
                        # Tie: pick the one that is different? Or just keep current?
                        # If we are on a plateau, moving might help find the edge.
                        # Let's pick the first one that is different from x?
                        # Actually, if fc == f_curr, and we pick it, we move.
                        # This helps escape plateaus.
                        if not np.allclose(xc, x):
                            best_candidate_f = fc
                            best_candidate = xc

                if best_candidate is not None:
                    x = best_candidate
                    f_curr = best_candidate_f
                    if f_curr < best_f:
                        best_f = f_curr
                        best_x = x.copy()
                    # If we moved, we might want to increase step size slightly?
                    # Or keep it?
                    # If we found a better point, the landscape is sloping.
                    # Let's keep step size.
                    improved = True
                else:
                    # No improvement (or only ties that didn't trigger move? 
                    # If we didn't move, we shrink step.
                    steps[i] *= 0.5
                    if steps[i] < min_step:
                        steps[i] = min_step
                        # If step is too small, we are stuck.
                        # Maybe restart this dim?
            
            if not improved:
                # If no dim improved, we might be stuck.
                # Shrink all steps?
                # Or restart?
                # For now, just continue. The steps will shrink.
                pass

        return best_x, best_f
