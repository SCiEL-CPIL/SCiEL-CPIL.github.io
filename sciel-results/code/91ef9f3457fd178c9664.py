"""Multi-objective optimizer on DTLZ1 using MOPSO."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Swarm size
        n_particles = 40
        # Archive size limit
        archive_size = 100
        
        # Initialize swarm
        # Use a fixed seed for reproducibility within the run, but the evaluator sets np.random.seed
        # We use np.random directly as per the task description's implication of seed usage
        X = lo + (hi - lo) * np.random.rand(n_particles, dim)
        V = (hi - lo) * np.random.rand(n_particles, dim) * 0.1 # Small initial velocity
        
        # Evaluate initial swarm
        F = np.array([problem(x) for x in X])
        evaluations = n_particles
        
        # Initialize archive with non-dominated points
        archive = []
        for i in range(n_particles):
            if not self._dominated(F[i], F):
                archive.append((F[i], X[i]))
        
        # Calculate initial crowding distances for archive
        archive = self._update_archive(archive, F, X, archive_size)
        
        # MOPSO Loop
        while evaluations < self.budget:
            if time.time() > deadline:
                break
            
            # Update velocities and positions
            # w: inertia, c1: cognitive, c2: social
            w = 0.5
            c1 = 1.5
            c2 = 1.5
            
            # Select gbest from archive using crowding distance
            if archive:
                # Select index with highest crowding distance
                best_idx = np.argmax([a[2] for a in archive])
                gbest_x = archive[best_idx][1]
                gbest_f = archive[best_idx][0]
            else:
                gbest_x = X[0]
                gbest_f = F[0]
                
            # Personal bests are stored in X_best, F_best
            # We need to maintain personal bests. 
            # For simplicity in this short budget, we can re-evaluate or track.
            # Let's track Pbest.
            if not hasattr(self, '_pbest_x'):
                self._pbest_x = X.copy()
                self._pbest_f = F.copy()
            
            # Update Pbest
            for i in range(n_particles):
                if self._dominated(F[i], self._pbest_f[i]):
                    self._pbest_x[i] = X[i]
                    self._pbest_f[i] = F[i]
            
            r1 = np.random.rand(n_particles, 1)
            r2 = np.random.rand(n_particles, 1)
            
            V = w * V + c1 * r1 * (self._pbest_x - X) + c2 * r2 * (gbest_x - X)
            
            # Clip velocities
            max_v = 0.2 * (hi - lo)
            V = np.clip(V, -max_v, max_v)
            
            X = X + V
            X = np.clip(X, lo, hi)
            
            # Evaluate new positions
            new_F = np.array([problem(x) for x in X])
            evaluations += n_particles
            
            # Update Pbest with new evaluations
            for i in range(n_particles):
                if self._dominated(new_F[i], self._pbest_f[i]):
                    self._pbest_x[i] = X[i]
                    self._pbest_f[i] = new_F[i]
            
            # Update Archive
            # Combine current archive and new non-dominated points
            # To be efficient, we only add points that are non-dominated in the combined set
            # But for small budgets, we can just merge and clean.
            
            # Identify non-dominated points in the new population
            new_non_dominated = []
            for i in range(n_particles):
                # Check if new_F[i] is dominated by any point in current archive
                dominated_by_archive = False
                for a_f, a_x, a_cd in archive:
                    if self._dominated(new_F[i], a_f):
                        dominated_by_archive = True
                        break
                if not dominated_by_archive:
                    new_non_dominated.append((new_F[i], X[i]))
            
            # Add new non-dominated points to archive
            for f, x in new_non_dominated:
                # Check if it dominates any existing archive points
                # And if it is not dominated by any existing archive points (already checked above partially, 
                # but need to check against other new points too? 
                # Simpler: Merge all candidates and run a full non-dominated sort on the union)
                pass
            
            # Efficient Archive Update Strategy:
            # 1. Collect all candidates: current archive + new population points that are not dominated by archive
            # 2. Actually, just take the union of current archive and new population.
            # 3. Filter for non-dominated points.
            # 4. If size > archive_size, use crowding distance to prune.
            
            candidates = []
            # Add current archive
            for a in archive:
                candidates.append((a[0], a[1]))
            # Add new population
            for i in range(n_particles):
                candidates.append((new_F[i], X[i]))
            
            # Filter non-dominated
            non_dominated = []
            for i in range(len(candidates)):
                f_i, x_i = candidates[i]
                is_dominated = False
                for j in range(len(candidates)):
                    if i == j:
                        continue
                    f_j, x_j = candidates[j]
                    if self._dominated(f_i, f_j):
                        is_dominated = True
                        break
                if not is_dominated:
                    non_dominated.append((f_i, x_i))
            
            # Calculate crowding distance and prune if necessary
            if len(non_dominated) > archive_size:
                # Assign crowding distances
                cds = self._crowding_distance(np.array([f for f, x in non_dominated]))
                # Sort by crowding distance descending
                indices = np.argsort(cds)[::-1]
                archive = [(non_dominated[i][0], non_dominated[i][1], cds[i]) for i in indices[:archive_size]]
            else:
                cds = self._crowding_distance(np.array([f for f, x in non_dominated])) if non_dominated else []
                archive = [(non_dominated[i][0], non_dominated[i][1], cds[i]) for i in range(len(non_dominated))]
            
            # Update F and X for next iteration
            F = new_F
            X = X

        # Return the archive points (or a subset)
        # The task says "Every returned objective vector is recorded".
        # We should return the decision variables that produced the best objective vectors.
        # The evaluator calls problem(x) again? No, "problem(x) ... returns 3 objective values".
        # "Every returned objective vector is recorded". This implies the evaluator records the outputs of problem() calls made during __call__.
        # Wait, "The return value of __call__ is ignored."
        # "Every returned objective vector is recorded" -> This refers to the vectors returned by `problem(x)` inside `__call__`.
        # So we don't need to return anything specific, just ensure we have made the evaluations.
        # However, usually these tasks expect the final population to be returned if the evaluator uses it, 
        # but here it explicitly says return value is ignored.
        # So we just need to finish within budget.
        
        return X

    def _dominated(self, f_a, f_b):
        # f_a dominates f_b if f_a <= f_b in all objectives and f_a < f_b in at least one
        # f_a and f_b are 1D arrays
        return np.all(f_a <= f_b) and np.any(f_a < f_b)

    def _crowding_distance(self, F):
        # F is (N, M) array
        N = F.shape[0]
        if N == 0:
            return []
        if N == 1:
            return [np.inf]
        
        cd = np.zeros(N)
        for m in range(F.shape[1]):
            order = np.argsort(F[:, m])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            if F[order[-1], m] - F[order[0], m] > 0:
                for i in range(1, N - 1):
                    cd[order[i]] += (F[order[i+1], m] - F[order[i-1], m]) / (F[order[-1], m] - F[order[0], m])
        return cd

    def _update_archive(self, archive, F, X, max_size):
        # This method is not strictly used in the loop above as I inlined the logic,
        # but kept for clarity if needed.
        pass

# EVOLVE-BLOCK-END
