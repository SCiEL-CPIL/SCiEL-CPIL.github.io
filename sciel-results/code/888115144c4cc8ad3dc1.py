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
            span = hi - lo
            
            # 1. Initialization: Space-filling design
            # Use np.random since evaluator seeds it
            n_init = min(50, self.budget // 4)
            # Latin Hypercube approximation
            X_init = np.zeros((n_init, self.dim))
            for j in range(self.dim):
                perm = np.random.permutation(n_init)
                X_init[:, j] = (perm + 0.5) / n_init
            X_init = lo + X_init * span
            
            F_init = np.array([problem(x) for x in X_init])
            
            archive_X = X_init.copy()
            archive_F = F_init.copy()
            used = n_init
            
            # Helper: Update archive
            def update_archive(new_x, new_f):
                nonlocal archive_X, archive_F
                # Check if new_f is dominated by any in archive
                # If yes, discard (unless it dominates existing ones, but standard NSGA keeps non-dominated)
                # Actually, we want to keep the Pareto front.
                # If new_f is dominated by any archive point, it's not on the front.
                # But it might dominate some archive points.
                
                # Simple O(N) check for dominance
                # new_f is dominated if exists i: F[i] <= new_f and F[i] != new_f
                is_dominated = False
                for i in range(len(archive_F)):
                    if np.all(archive_F[i] <= new_f) and np.any(archive_F[i] < new_f):
                        is_dominated = True
                        break
                
                if is_dominated:
                    return False # Not added to front
                
                # Remove points dominated by new_f
                keep = []
                for i in range(len(archive_F)):
                    # archive_F[i] dominated by new_f?
                    if np.all(new_f <= archive_F[i]) and np.any(new_f < archive_F[i]):
                        continue
                    keep.append(i)
                
                archive_X = np.delete(archive_X, [i for i in range(len(archive_X)) if i not in keep], axis=0)
                archive_F = np.delete(archive_F, [i for i in range(len(archive_F)) if i not in keep], axis=0)
                
                archive_X = np.vstack([archive_X, new_x])
                archive_F = np.vstack([archive_F, new_f])
                return True

            # Apply initial points to archive (some might dominate others)
            # The init is random, so we should filter them
            final_X = []
            final_F = []
            for i in range(n_init):
                if not is_dominated_by_list(F_init[i], final_F):
                    # Remove dominated from final
                    # ... this is getting complex. Let's just keep all initially and filter in the loop?
                    # No, let's just use the list approach for the archive to keep it simple and correct.
                    pass

            # Let's restart the archive logic with a list for simplicity and correctness in small N
            arch_X = []
            arch_F = []
            
            def add_to_archive(x, f):
                # Check if f is dominated by any in arch_F
                for af in arch_F:
                    if np.all(af <= f) and np.any(af < f):
                        return False
                # Remove dominated points from arch
                new_arch_X = []
                new_arch_F = []
                for i in range(len(arch_F)):
                    if np.all(f <= arch_F[i]) and np.any(f < arch_F[i]):
                        continue
                    new_arch_X.append(arch_X[i])
                    new_arch_F.append(arch_F[i])
                new_arch_X.append(x)
                new_arch_F.append(f)
                arch_X[:] = new_arch_X
                arch_F[:] = new_arch_F
                return True

            for i in range(n_init):
                add_to_archive(X_init[i], F_init[i])
            
            # 2. Local Search Phase
            # Budget remaining
            remaining = self.budget - used
            step_size = 0.1 # Initial step size relative to span
            
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                if len(arch_F) == 0:
                    break

                # Select a point from the Pareto front to explore
                # Strategy: Pick the point with the largest "gap" or just random.
                # Random is simple. Let's try to bias towards points that are "extreme" or "central".
                # For now, random selection from the front is a decent baseline.
                idx = np.random.randint(len(arch_F))
                x_base = arch_X[idx]
                
                # Generate candidate
                # Adaptive step: reduce if we are making progress, increase if stuck?
                # Simple approach: fixed step with decay?
                # Let's use a random direction and step.
                noise = np.random.randn(self.dim)
                noise /= (np.linalg.norm(noise) + 1e-9)
                x_cand = x_base + step_size * span * noise
                x_cand = np.clip(x_cand, lo, hi)
                
                f_cand = problem(x_cand)
                used += 1
                
                added = add_to_archive(x_cand, f_cand)
                
                # Adapt step size
                if added:
                    # Good move, keep or slightly increase?
                    # step_size = min(0.2, step_size * 1.1)
                    pass
                else:
                    # Bad move, reduce step size to refine
                    step_size *= 0.9
                    if step_size < 0.001:
                        step_size = 0.1 # Reset if too small? Or keep small for polish.
                        # Resetting might cause jumps. Let's keep it small.
            
            # Return best result (ignored by evaluator, but required)
            if len(arch_X) > 0:
                return np.array(arch_X)
            return X_init
