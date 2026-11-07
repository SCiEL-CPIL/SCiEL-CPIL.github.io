"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """MOPSO multi-objective optimizer for ZDT2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget
        
        # Parameters
        swarm_size = 50
        archive_size = 100
        w = 0.5  # Inertia weight
        c1 = 1.5  # Cognitive coefficient
        c2 = 1.5  # Social coefficient
        
        # Initialize Swarm
        X = lo + (hi - lo) * np.random.random((swarm_size, dim))
        V = np.zeros((swarm_size, dim))
        
        # Initialize Fitness
        F = np.array([problem(x) for x in X], dtype=float)
        evaluations = swarm_size
        
        # Initialize Personal Bests
        Pbest_X = X.copy()
        Pbest_F = F.copy()
        
        # Initialize Archive
        archive_X = []
        archive_F = []
        
        def dominates(f1, f2):
            """Check if f1 dominates f2 (minimization)."""
            return np.all(f1 <= f2) and np.any(f1 < f2)
        
        def add_to_archive(x, f, arch_X, arch_F, max_size):
            """Add solution to archive if non-dominated, prune if too large."""
            # Check if dominated by any in archive
            dominated = False
            for af in arch_F:
                if dominates(af, f):
                    dominated = True
                    break
            if dominated:
                return arch_X, arch_F
            
            # Remove solutions dominated by new one
            new_arch_X = []
            new_arch_F = []
            for i in range(len(arch_X)):
                if not dominates(f, arch_F[i]):
                    new_arch_X.append(arch_X[i])
                    new_arch_F.append(arch_F[i])
            
            new_arch_X.append(x)
            new_arch_F.append(f)
            
            # Prune if exceeds max size
            if len(new_arch_X) > max_size:
                # Calculate crowding distance
                n = len(new_arch_F)
                if n <= 2:
                    # Keep all if very small, but we need to prune
                    # Simplest: just keep the ones with largest spread? 
                    # For simplicity in pruning, let's use a grid or just random if we want speed, 
                    # but crowding is better.
                    pass
                
                # Crowding distance calculation
                F_arr = np.array(new_arch_F)
                cd = np.zeros(n)
                for m in range(self.n_obj):
                    sorted_idx = np.argsort(F_arr[:, m])
                    cd[sorted_idx[0]] = np.inf
                    cd[sorted_idx[-1]] = np.inf
                    f_min = F_arr[sorted_idx[0], m]
                    f_max = F_arr[sorted_idx[-1], m]
                    if f_max - f_min < 1e-15:
                        continue
                    for k in range(1, n - 1):
                        idx = sorted_idx[k]
                        cd[idx] += (F_arr[sorted_idx[k+1], m] - F_arr[sorted_idx[k-1], m]) / (f_max - f_min)
                
                # Remove point with smallest crowding distance
                min_cd_idx = np.argmin(cd)
                new_arch_X.pop(min_cd_idx)
                new_arch_F.pop(min_cd_idx)
                
            return new_arch_X, new_arch_F

        # Populate initial archive
        for i in range(swarm_size):
            archive_X, archive_F = add_to_archive(X[i], F[i], archive_X, archive_F, archive_size)

        # Main Loop
        while evaluations < budget:
            # Update velocity and position
            for i in range(swarm_size):
                if evaluations >= budget:
                    break
                
                # Select gbest from archive using tournament selection on crowding distance
                if len(archive_F) > 0:
                    # Simple tournament: pick 2 random, keep one with larger crowding dist
                    # To avoid recalculating CD every time for every particle, we can precompute or simplify.
                    # For efficiency in this context, let's just pick a random one from the archive 
                    # or use a simple heuristic. 
                    # Actually, proper MOPSO uses CD. Let's compute CD for the current archive once per iteration? 
                    # That might be expensive if archive is large. 
                    # Alternative: Use a grid-based selection or just random if archive is small.
                    # Let's stick to random selection from non-dominated set for simplicity and speed, 
                    # relying on the archive pruning to maintain diversity.
                    # Wait, random selection from archive is a valid strategy (MOEA/D style or simple PSO).
                    # Let's use random selection from the archive to guide the swarm.
                    g_idx = np.random.randint(0, len(archive_F))
                    gbest = archive_X[g_idx]
                else:
                    gbest = Pbest_X[i]
                
                r1 = np.random.random(dim)
                r2 = np.random.random(dim)
                
                V[i] = w * V[i] + c1 * r1 * (Pbest_X[i] - X[i]) + c2 * r2 * (gbest - X[i])
                X[i] = X[i] + V[i]
                X[i] = np.clip(X[i], lo, hi)
                
                # Evaluate
                f_new = problem(X[i])
                evaluations += 1
                F[i] = f_new
                
                # Update Pbest
                if dominates(f_new, Pbest_F[i]):
                    Pbest_X[i] = X[i].copy()
                    Pbest_F[i] = f_new.copy()
                elif not dominates(Pbest_F[i], f_new):
                    # Non-dominated, keep both? 
                    # Standard MOPSO often keeps the one with better crowding or just one.
                    # Let's keep the one that is non-dominated. If both are non-dominated, 
                    # we can keep the one with better "quality" or just one. 
                    # For simplicity, if new is non-dominated by old, and old is non-dominated by new,
                    # we can keep the new one to encourage exploration, or keep old.
                    # Let's keep the one with smaller sum of objectives as a tie-breaker?
                    # Or just keep old to be conservative.
                    pass 
                
                # Update Archive
                archive_X, archive_F = add_to_archive(X[i], f_new, archive_X, archive_F, archive_size)
                
        return None
# EVOLVE-BLOCK-END
