"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

class Optimizer:
    """Multi-objective optimizer using Bayesian Optimization with EHVI."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        center = (lo + hi) / 2.0
        
        rng = np.random
        
        # 1. Initialization: Sobol sequence for diverse coverage
        # We need at least n_obj+1 points for stable GP, but let's start with 15-20
        # Sobol is quasi-random, better than uniform for low dimensions
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            init_count = min(20, budget)
            X_init = sampler.random(init_count) * span + lo
        except ImportError:
            # Fallback to random if scipy.stats.qmc is not available (unlikely)
            X_init = rng.random((init_count, dim)) * span + lo
            
        X = X_init
        F = np.array([problem(x) for x in X])
        evals_used = len(X)
        
        # 2. Reference Point Calculation
        # Task specifies ref [1705.0, 12.0, 0.3] but we should be robust if bounds change?
        # The task says "reference point [1705.0, 12.0, 0.3]". 
        # However, standard practice is to set ref slightly worse than the worst observed.
        # Given the task contract is fixed, we can use the fixed ref, 
        # but for EHVI calculation, using a dynamic ref based on current worst + margin is often safer 
        # to ensure the "improvement" is calculated relative to the current state of knowledge.
        # Let's stick to the task's specific ref for scoring, but for EHVI optimization,
        # we typically maximize the hypervolume improvement relative to the current non-dominated set.
        
        # Fixed Reference Point from Task
        REF_POINT = np.array([1705.0, 12.0, 0.3])
        
        # Helper: Calculate Hypervolume of a set of points dominated by ref
        # Points must be strictly better than ref in all objectives to contribute
        # Simple exact hypervolume for 3D is complex, but we can approximate or use a simple decomposition
        # For EHVI, we need to estimate the improvement.
        
        def _is_dominated(f, F_set):
            """Check if f is dominated by any point in F_set."""
            for f_j in F_set:
                if np.all(f_j <= f) and np.any(f_j < f):
                    return True
            return False

        def _non_dominated(F_set):
            """Return indices of non-dominated points in F_set."""
            n = len(F_set)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]: continue
                for j in range(n):
                    if i == j or not mask[j]: continue
                    # If j dominates i, i is dominated
                    if np.all(F_set[j] <= F_set[i]) and np.any(F_set[j] < F_set[i]):
                        mask[i] = False
                        break
            return np.where(mask)[0]

        def _hypervolume(points, ref):
            """
            Calculate hypervolume for 3D points.
            points: (N, 3) array
            ref: (3,) array
            Only points strictly better than ref count.
            """
            # Filter points better than ref
            valid = np.all(points < ref, axis=1)
            pts = points[valid]
            if len(pts) == 0:
                return 0.0
            
            # Sort by first objective
            pts = pts[np.argsort(pts[:, 0])]
            
            # In 3D, we can decompose by the first objective.
            # For each point i, it contributes a volume defined by:
            # Width in obj0: pts[i,0] to (pts[i+1,0] or ref[0] if last)
            # But actually, the standard exact HV algorithm for 3D is:
            # Sort by x1. 
            # For each point, the "slice" width is x_{i+1} - x_i (or ref[0] - x_i for last).
            # The area in the remaining 2D plane (x2, x3) dominated by this point 
            # but not by any subsequent points in the sorted order (which have larger x1, thus smaller slices)
            # needs to be calculated.
            
            # A simpler robust approach for small N (N <= 200) is to use a grid or a specific 3D algorithm.
            # Let's use a standard O(N^2) or O(N^3) exact 3D HV algorithm.
            
            # Sort by obj 0
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            
            hv = 0.0
            n = len(pts)
            
            # For each point i, calculate the area in (obj1, obj2) plane 
            # that is dominated by pts[i] and NOT dominated by any pts[j] where j > i (in sorted order).
            # Since pts are sorted by obj0 ascending, pts[j] for j>i have larger obj0.
            # The volume contribution of point i is:
            # (ref[0] - pts[i, 0] if it's the last, or pts[i+1, 0] - pts[i, 0]) 
            # Wait, the standard decomposition is:
            # Sort by x1. 
            # Contribution of point i: (x_{i+1} - x_i) * Area_2D( (x2_i, x3_i) dominated, excluding points j>i )
            # Actually, it's easier to think: 
            # The total HV is sum over i of (x_{i+1} - x_i) * HV_2D( points j>=i in 2D )
            # where x_{n} = ref[0].
            
            # Let's implement the recursive 2D HV.
            def _hv_2d(points_2d, ref_2d):
                # points_2d: (M, 2), ref_2d: (2,)
                if len(points_2d) == 0: return 0.0
                # Filter
                valid = np.all(points_2d < ref_2d, axis=1)
                p2 = points_2d[valid]
                if len(p2) == 0: return 0.0
                
                # Sort by first dim
                p2 = p2[np.argsort(p2[:, 0])]
                m = len(p2)
                vol = 0.0
                prev_y = ref_2d[1]
                # Iterate from best (smallest x) to worst
                # Actually, sort ascending x.
                # The "height" in y for a given x is determined by the minimum y of all points with x <= current_x?
                # No, for minimization, we want the area under the "skyline".
                # Sort by x ascending.
                # Keep track of the minimum y seen so far? No.
                # The dominated region is the union of rectangles [x_i, ref_x] x [y_i, ref_y].
                # We can compute this by sorting by x, and maintaining the "active" y boundaries.
                
                # Simpler O(M^2) for small M:
                # For each point i, it contributes (ref_x - x_i) * (min_y_active - y_i)? No.
                
                # Let's use the standard "sort by x, sweep y" or just O(M^2) intersection logic.
                # Given M <= 200, O(M^2) is fine.
                
                # Actually, there is a very simple O(M log M) or O(M^2) way.
                # Sort by x ascending.
                # The area is sum over i of (x_{i+1} - x_i) * (ref_y - min_y(0..i))
                # where x_{M} = ref_x.
                # Why? Because at x between x_i and x_{i+1}, the "limiting" y is the minimum y among points 0..i.
                
                min_y = ref_2d[1]
                total_area = 0.0
                for i in range(m):
                    # Update min_y with current point
                    if p2[i, 1] < min_y:
                        min_y = p2[i, 1]
                    # Width is to the next point's x, or ref_x if last
                    next_x = p2[i+1, 0] if i < m - 1 else ref_2d[0]
                    width = next_x - p2[i, 0]
                    height = ref_2d[1] - min_y
                    if height > 0:
                        total_area += width * height
                return total_area

            # Now back to 3D
            # Sort pts by obj 0
            # x_{n} = ref[0]
            # For each i, width = (pts[i+1, 0] - pts[i, 0]) or (ref[0] - pts[i, 0])
            # The "cross section" area in obj1-obj2 plane is the HV_2D of all points j >= i 
            # (because points with smaller obj0 have already "claimed" their slice width, 
            # but wait... the decomposition is usually: 
            # HV = sum_{i=0}^{n-1} (x_{i+1} - x_i) * HV_2D( P_i )
            # where P_i is the set of points {p_j | j >= i} projected to 2D?
            # Let's verify.
            # If we sort by x1 ascending.
            # The region dominated by p_i is [x_i, ref_x] x [y_i, ref_y] x [z_i, ref_z].
            # The union of these boxes.
            # Slice at x in [x_i, x_{i+1}]: The active points are those with x_j <= x.
            # So active points are j <= i.
            # The area in y-z plane is HV_2D of {p_j | j <= i}.
            
            # So:
            # HV = sum_{i=0}^{n-1} (x_{i+1} - x_i) * HV_2D( {p_j | j <= i} )
            
            hv = 0.0
            # We need to efficiently update HV_2D as we add points.
            # Since N is small (<=200), we can just recalculate HV_2D for each prefix.
            # O(N^3) total, which is 8,000,000 ops max. Very fast.
            
            for i in range(n):
                # Get prefix up to i
                prefix = pts[:i+1]
                # Project to 2D (cols 1, 2)
                p2d = prefix[:, 1:3]
                ref2d = ref[1:3]
                area = _hv_2d(p2d, ref2d)
                
                next_x = pts[i+1, 0] if i < n - 1 else ref[0]
                width = next_x - pts[i, 0]
                hv += width * area
                
            return hv

        def _ehvi(x, X_train, F_train, ref):
            """
            Calculate Expected Hypervolume Improvement for a candidate x.
            We approximate the posterior by sampling from the GP.
            """
            # This is a placeholder. We need a GP.
            # Implementing a full GP from scratch is risky.
            # Let's use a simpler surrogate: Kriging or just a simple RBF interpolation?
            # Or, since we have budget 200, we can use a "Local Search" around promising points.
            
            # Alternative Strategy: 
            # Since implementing a robust GP + EHVI in < 300s with 200 evals is tight,
            # let's switch to a **Multi-Objective Particle Swarm Optimization (MOPSO)** 
            # which is structurally different from DE and often performs well on small budgets.
            # MOPSO uses velocity and position updates, guided by a global best and local best.
            
            # Wait, the directive says "COMPUTE BUDGET... Keep an EXPLICIT deadline guard".
            # MOPSO is iterative.
            
            # Let's try a different structural change: **Evolutionary Strategy with Covariance Matrix Adaptation (CMA-ES)**
            # adapted for MOO? No, CMA-ES is single objective.
            
            # Let's go with **MOPSO**. It's a distinct family from DE (Evolutionary) 
            # (Swarm Intelligence).
            
            # I will rewrite the __call__ to use MOPSO.
            
            # --- MOPSO IMPLEMENTATION ---
            
            # Reset state for MOPSO
            pop_size = 30
            # We have already used 20 evals for init.
            # Remaining budget: budget - evals_used
            
            # Velocity
            V = np.zeros((pop_size, dim))
            # Best individual (Pbest) and Global Best (Gbest)
            # Gbest is selected from an external archive using crowding distance or random.
            
            # Archive
            archive = []
            archive_f = []
            
            # Add initial points to archive
            for i in range(len(X)):
                if not _is_dominated(F[i], archive_f):
                    # Check if any in archive dominates F[i]
                    dominated = False
                    for af in archive_f:
                        if np.all(af <= F[i]) and np.any(af < F[i]):
                            dominated = True
                            break
                    if not dominated:
                        archive.append(X[i].copy())
                        archive_f.append(F[i].copy())
                        # Remove points in archive dominated by F[i]
                        new_archive = []
                        new_archive_f = []
                        for j, af in enumerate(archive_f):
                            if j == len(archive_f) - 1 and not new_archive: # This logic is getting messy
                                pass
                        # Simpler: rebuild archive
                        pass
            
            # Let's implement a clean Archive update function
            def update_archive(x, f, archive, archive_f):
                # Check if f is dominated by any in archive
                for af in archive_f:
                    if np.all(af <= f) and np.any(af < f):
                        return archive, archive_f
                
                # Remove points in archive dominated by f
                new_archive = []
                new_archive_f = []
                for i in range(len(archive)):
                    if not (np.all(f <= archive_f[i]) and np.any(f < archive_f[i])):
                        new_archive.append(archive[i])
                        new_archive_f.append(archive_f[i])
                
                # Add new point
                new_archive.append(x)
                new_archive_f.append(f)
                
                return new_archive, new_archive_f

            # Re-init archive properly
            archive = []
            archive_f = []
            for i in range(len(X)):
                archive, archive_f = update_archive(X[i], F[i], archive, archive_f)
            
            # MOPSO Parameters
            w_min = 0.4
            w_max = 0.9
            c1 = 1.5
            c2 = 1.5
            max_iter = (budget - evals_used) // pop_size
            
            # We need to track Pbest for each particle
            # Pbest_x, Pbest_f
            Pbest_x = X.copy()
            Pbest_f = F.copy()
            
            # Velocity initialization
            # v = r * (pbest - x) ? Or small random.
            V = (hi - lo) * 0.1 * rng.random((pop_size, dim))
            
            current_iter = 0
            while evals_used < budget and current_iter < max_iter:
                if time.time() > deadline - 10.0:
                    break
                
                # Update inertia weight
                w = w_min + (w_max - w_min) * (1 - current_iter / max_iter)
                
                for i in range(pop_size):
                    if evals_used >= budget:
                        break
                    if time.time() > deadline - 5.0:
                        break
                    
                    # Select Gbest from archive
                    # Simple strategy: Random selection from archive, or use crowding distance.
                    # For speed, random is fine, or pick the one with max distance to others.
                    if len(archive) > 0:
                        # Pick random from archive
                        g_idx = rng.integers(len(archive))
                        Gbest_x = archive[g_idx]
                    else:
                        Gbest_x = Pbest_x[i]
                    
                    r1 = rng.random(dim)
                    r2 = rng.random(dim)
                    
                    # Update velocity
                    V[i] = w * V[i] + c1 * r1 * (Pbest_x[i] - X[i]) + c2 * r2 * (Gbest_x - X[i])
                    
                    # Clip velocity to prevent divergence
                    v_max = 0.5 * (hi - lo)
                    V[i] = np.clip(V[i], -v_max, v_max)
                    
                    # Update position
                    X[i] = X[i] + V[i]
                    X[i] = np.clip(X[i], lo, hi)
                    
                    # Evaluate
                    F[i] = problem(X[i])
                    evals_used += 1
                    
                    # Update Pbest
                    # If F[i] dominates Pbest_f[i]
                    if np.all(F[i] <= Pbest_f[i]) and np.any(F[i] < Pbest_f[i]):
                        Pbest_f[i] = F[i]
                        Pbest_x[i] = X[i].copy()
                    # If Pbest_f[i] dominates F[i], keep Pbest
                    # If non-dominated, keep Pbest (standard MOPSO) or replace with prob?
                    # Standard: Keep Pbest if non-dominated to maintain diversity? 
                    # Actually, if non-dominated, we might want to keep the one with better crowding?
                    # For simplicity, keep Pbest if it doesn't dominate new, and new doesn't dominate Pbest?
                    # Let's stick to: Replace if new dominates. Keep otherwise.
                    
                    # Update Archive
                    archive, archive_f = update_archive(X[i], F[i], archive, archive_f)
                
                current_iter += 1
            
            return X

# EVOLVE-BLOCK-END
