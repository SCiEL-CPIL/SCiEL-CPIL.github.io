"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Batched Differential Evolution (DE) optimizer for RE37.
    
    Strategy:
    1. Initialize with a small random population to establish a baseline.
    2. Iteratively generate new candidates using the DE mutation formula:
       v = x_r1 + F * (x_r2 - x_r3)
       This creates new points that are correlated with existing ones but
       explore the neighborhood, which is often more effective for finding
       the Pareto front in low-dimensional bounded spaces than pure random
       or static Sobol sequences, especially when combined with a greedy
       non-dominated set update.
    3. Maintain a greedy non-dominated set (archive) of all evaluated points.
    4. Use the archive for mutation to ensure we explore regions where we
       already have good solutions, promoting convergence to the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.F = 0.8  # DE scaling factor
        self.batch_size = 20  # Number of new candidates per iteration

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        # We will use the global np.random state which is seeded by the evaluator.
        # This ensures reproducibility per seed.
        
        # Archive: list of (x, f) pairs that form the current non-dominated set
        # We store them in a list for easy manipulation.
        # For efficiency in domination checks, we keep it as a list of arrays.
        archive_x = []
        archive_f = []
        
        used_evals = 0
        
        # Helper to check if a point is dominated by any in the archive
        # f_new is dominated if there exists f_arch such that f_arch <= f_new and f_arch < f_new
        def is_dominated(f_new, archive_f_list):
            for f_arch in archive_f_list:
                if np.all(f_arch <= f_new) and np.any(f_arch < f_new):
                    return True
            return False

        # Helper to remove points dominated by f_new from the archive
        def update_archive(f_new, x_new, archive_x_list, archive_f_list):
            # If f_new is dominated, do nothing
            if is_dominated(f_new, archive_f_list):
                return False
            
            # Remove any points in the archive dominated by f_new
            new_x = []
            new_f = []
            for i in range(len(archive_f_list)):
                f_arch = archive_f_list[i]
                # Check if f_arch is dominated by f_new
                if np.all(f_new <= f_arch) and np.any(f_new < f_arch):
                    continue
                new_x.append(archive_x_list[i])
                new_f.append(f_arch)
            
            # Add the new point
            new_x.append(x_new)
            new_f.append(f_new)
            
            # Update lists in place (or return new lists, but in-place is faster if we manage indices carefully)
            # Since we are in a loop, let's just reassign the lists in the outer scope via return
            return True, new_x, new_f

        # Phase 1: Initial Random Sampling
        # Start with a small random population to seed the DE process.
        # Let's use 10% of the budget for initialization, minimum 10, max 50.
        init_size = max(10, min(50, self.budget // 10))
        
        for _ in range(init_size):
            if used_evals >= self.budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.random(self.dim)
            try:
                f = np.array(problem(x), dtype=float)
            except Exception:
                break
            
            used_evals += 1
            
            # Update archive
            if not archive_f:
                archive_x.append(x)
                archive_f.append(f)
            else:
                dominated, new_x, new_f = update_archive(f, x, archive_x, archive_f)
                if dominated:
                    archive_x = new_x
                    archive_f = new_f

        # Phase 2: Differential Evolution Loop
        # Generate candidates in batches.
        # We use points from the current archive as parents.
        # If the archive is small, we might reuse the same parents, which is fine.
        
        while used_evals < self.budget:
            if time.time() > deadline:
                break
                
            # Determine batch size for this step
            current_batch = min(self.batch_size, self.budget - used_evals)
            
            # If archive is empty (shouldn't happen if init succeeded), fall back to random
            if not archive_x:
                 for _ in range(current_batch):
                     x = lo + (hi - lo) * np.random.random(self.dim)
                     f = np.array(problem(x), dtype=float)
                     used_evals += 1
                     archive_x.append(x)
                     archive_f.append(f)
                 continue

            n_archive = len(archive_x)
            
            for _ in range(current_batch):
                if used_evals >= self.budget or time.time() > deadline:
                    break
                
                # Select 3 distinct parents from the archive
                # If archive size < 3, we need to handle it.
                if n_archive < 3:
                    # Fallback to random if we don't have enough parents
                    # This can happen if init was very small and we are just starting
                    x = lo + (hi - lo) * np.random.random(self.dim)
                else:
                    # Pick 3 distinct indices
                    # Using np.random.choice is slow for small n, use manual pick
                    i1 = np.random.randint(n_archive)
                    i2 = np.random.randint(n_archive)
                    i3 = np.random.randint(n_archive)
                    
                    # Ensure distinctness for better exploration, though not strictly required for DE
                    # If they are not distinct, the vector difference might be 0 or smaller
                    # Let's just pick 3 random indices. Collisions are possible but acceptable.
                    # To be slightly more robust, we can retry if i1==i2 or similar, 
                    # but for low dim, the impact is small.
                    
                    r1 = archive_x[i1]
                    r2 = archive_x[i2]
                    r3 = archive_x[i3]
                    
                    # DE Mutation
                    v = r1 + self.F * (r2 - r3)
                    
                    # Clip to bounds
                    x = np.clip(v, lo, hi)
                
                try:
                    f = np.array(problem(x), dtype=float)
                except Exception:
                    break
                
                used_evals += 1
                
                # Update archive
                if not archive_f:
                    archive_x.append(x)
                    archive_f.append(f)
                else:
                    dominated, new_x, new_f = update_archive(f, x, archive_x, archive_f)
                    if dominated:
                        archive_x = new_x
                        archive_f = new_f
                        n_archive = len(archive_x) # Update count for next iteration's parent selection

        # The return value is ignored, but we return the archive as a list of objective vectors
        # to satisfy the "return value" requirement loosely, though the evaluator ignores it.
        # The contract says "Every returned objective vector is recorded". 
        # Wait, the contract says: "Every returned objective vector is recorded... the return value of __call__ is ignored."
        # Actually, re-reading: "Every returned objective vector is recorded" usually implies the evaluator records the outputs of `problem(x)` calls internally or via a wrapper?
        # "The evaluator calls np.random.seed(seed), creates Optimizer... and calls it once."
        # "Every returned objective vector is recorded" -> This likely refers to the outputs of the `problem` function calls made by the Optimizer.
        # The evaluator probably wraps `problem` to record outputs.
        # So the return value of `__call__` is indeed ignored.
        
        return np.array(archive_f) if archive_f else None
# EVOLVE-BLOCK-END
