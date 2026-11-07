import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        start_time = time.time()
        deadline = start_time + 290
        
        evals_used = 0
        max_evals = self.budget
        archive = []
        
        def evaluate(x):
            nonlocal evals_used
            if evals_used >= max_evals:
                return None
            x_clipped = np.clip(x, lo, hi)
            f = np.array(problem(x_clipped))
            evals_used += 1
            return x_clipped, f

        def is_dominated(f, archive_f):
            for af in archive_f:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def update_archive(x, f):
            if is_dominated(f, [p[1] for p in archive]):
                return
            # Remove points dominated by new point
            new_archive = []
            for p in archive:
                if not (np.all(f <= p[1]) and np.any(f < p[1])):
                    new_archive.append(p)
            new_archive.append((x, f))
            archive.clear()
            archive.extend(new_archive)

        # Stage 1: Coarse Grid Search
        # For dim=4, a 3x3x3x3 grid uses 81 evaluations.
        # This leaves ~119 for refinement.
        grid_points = 3
        if self.dim <= 4:
            grid_points = 3
        elif self.dim <= 6:
            grid_points = 2
        else:
            grid_points = 2
            
        # Generate grid
        grid_coords = np.linspace(0, 1, grid_points)
        
        # We will iterate through all combinations
        # To save time and evals, we stop if we hit deadline or budget
        # But first, let's just do the grid if we have enough budget
        total_grid_evals = grid_points ** self.dim
        if total_grid_evals <= max_evals * 0.4: # Use 40% budget for grid
             for i in range(grid_points):
                for j in range(grid_points):
                   for k in range(grid_points):
                      for l in range(grid_points):
                          if self.dim == 4:
                              x = np.array([lo[0] + (hi[0]-lo[0])*grid_coords[i],
                                             lo[1] + (hi[1]-lo[1])*grid_coords[j],
                                             lo[2] + (hi[2]-lo[2])*grid_coords[k],
                                             lo[3] + (hi[3]-lo[3])*grid_coords[l]])
                          elif self.dim == 3:
                              x = np.array([lo[0] + (hi[0]-lo[0])*grid_coords[i],
                                             lo[1] + (hi[1]-lo[1])*grid_coords[j],
                                             lo[2] + (hi[2]-lo[2])*grid_coords[k]])
                          elif self.dim == 2:
                              x = np.array([lo[0] + (hi[0]-lo[0])*grid_coords[i],
                                             lo[1] + (hi[1]-lo[1])*grid_coords[j]])
                          else:
                              # Fallback for other dims
                              x = np.zeros(self.dim)
                              for d in range(self.dim):
                                  x[d] = lo[d] + (hi[d]-lo[d])*grid_coords[i % grid_points]
                              
                          if time.time() > deadline or evals_used >= max_evals:
                              break
                          res = evaluate(x)
                          if res:
                              update_archive(res[0], res[1])
                          if self.dim == 4 and l == grid_points - 1:
                              if time.time() > deadline or evals_used >= max_evals:
                                  break
                          if self.dim == 4 and k == grid_points - 1 and l == grid_points - 1:
                              if time.time() > deadline or evals_used >= max_evals:
                                  break
                          if self.dim == 4 and j == grid_points - 1 and k == grid_points - 1 and l == grid_points - 1:
                              if time.time() > deadline or evals_used >= max_evals:
                                  break
                          if self.dim == 4 and i == grid_points - 1 and j == grid_points - 1 and k == grid_points - 1 and l == grid_points - 1:
                              if time.time() > deadline or evals_used >= max_evals:
                                  break
        else:
            # If grid is too big, do random initialization
            rng = np.random.default_rng(0)
            for _ in range(min(50, max_evals // 4)):
                x = lo + (hi - lo) * rng.random(self.dim)
                res = evaluate(x)
                if res:
                    update_archive(res[0], res[1])
                if time.time() > deadline or evals_used >= max_evals:
                    break

        # Stage 2: Local Refinement
        # Select top candidates from the archive based on some metric (e.g., distance to ideal point or just random selection from front)
        # To maximize HV, we want to refine points that are on the front.
        # We'll pick a subset of the archive to refine.
        
        if archive:
            # Sort archive by sum of objectives (simple proxy) or just take all
            # Let's take the first 20 points in the archive (which are non-dominated)
            # To be safe, we'll just iterate through the archive
            candidates = list(archive)
            
            # If we have many points, we might need to prioritize.
            # For now, refine all unique points in the archive, but limit total refinement steps.
            # Each refinement step uses ~10-20 evals.
            
            # Shuffle candidates to ensure diversity in refinement
            rng = np.random.default_rng(1)
            rng.shuffle(candidates)
            
            for x0, f0 in candidates:
                if evals_used >= max_evals:
                    break
                if time.time() > deadline:
                    break
                
                # Simple Local Search: Nelder-Mead style or Random Hill Climbing
                # Given the small dim, a simple random perturbation search is robust and fast.
                # We will do a few steps of local search.
                
                current_x = x0.copy()
                current_f = f0.copy()
                
                # Try to improve
                steps = 5
                step_size = 0.1
                for s in range(steps):
                    if evals_used >= max_evals:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Generate a few random perturbations
                    found_improvement = False
                    for _ in range(5):
                        if evals_used >= max_evals:
                            break
                        if time.time() > deadline:
                            break
                            
                        dx = rng.standard_normal(self.dim) * step_size
                        x_new = current_x + dx
                        x_new = np.clip(x_new, lo, hi)
                        
                        res = evaluate(x_new)
                        if not res:
                            break
                        x_new, f_new = res
                        
                        # Check if f_new is better than current_f in a multi-objective sense
                        # We accept if it dominates current_f
                        if np.all(f_new <= current_f) and np.any(f_new < current_f):
                            current_x = x_new
                            current_f = f_new
                            found_improvement = True
                            step_size *= 0.8 # Shrink step
                            break # Restart perturbations with new center
                        # Also accept if it is non-dominated with respect to current_f?
                        # No, strictly better is safer for local convergence.
                        # But for HV, we might want to move sideways.
                        # Let's stick to strict dominance for local search stability.
                    
                    if not found_improvement:
                        step_size *= 0.5
                        if step_size < 0.01:
                            break
                            
                # Update archive with the best found in this local search
                if np.all(current_f <= f0) and np.any(current_f < f0):
                    update_archive(current_x, current_f)
                else:
                    # Even if not strictly better than start, it might be better than other archive points?
                    # update_archive handles dominance check.
                    update_archive(current_x, current_f)

        return [x for x, f in archive]
