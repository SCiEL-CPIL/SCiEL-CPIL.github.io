import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 295.0  # Safety margin before 300s limit

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        rng = np.random

        used = 0

        def dominates(a, b):
            """Returns True if a dominates b (a <= b in all, a < b in at least one)"""
            return np.all(a <= b) and np.any(a < b)

        def dominates_any(f, archive_F):
            """Check if f is dominated by any point in archive_F (list of arrays)"""
            for af in archive_F:
                if dominates(af, f):
                    return True
            return False

        def remove_dominated(f, archive_X, archive_F):
            """Remove points dominated by f from archive. Returns new lists."""
            new_X = []
            new_F = []
            for i in range(len(archive_X)):
                if not dominates(f, archive_F[i]):
                    new_X.append(archive_X[i])
                    new_F.append(archive_F[i])
            return new_X, new_F

        def hypervolume_3d(points, ref):
            """Compute 3D hypervolume for non-dominated points using inclusion-exclusion."""
            n = len(points)
            if n == 0:
                return 0.0
            
            # Filter points that are better than reference in all objectives
            valid = []
            for p in points:
                if all(p[i] < ref[i] for i in range(n_obj)):
                    valid.append(p)
            
            if len(valid) == 0:
                return 0.0
            
            # Sort by first objective
            valid = sorted(valid, key=lambda x: x[0])
            
            hv = 0.0
            for i, p in enumerate(valid):
                next_obj0 = ref[0] if i == len(valid) - 1 else valid[i + 1][0]
                
                # Find min obj1 and obj2 among points i..end
                min_obj1 = min([q[1] for q in valid[i:]])
                min_obj2 = min([q[2] for q in valid[i:]])
                
                if p[0] < next_obj0 and min_obj1 < ref[1] and min_obj2 < ref[2]:
                    vol = (next_obj0 - p[0]) * (ref[1] - min_obj1) * (ref[2] - min_obj2)
                    hv += max(0, vol)
            
            return hv

        def crowding_distance(F_arr):
            """Calculate crowding distance for a set of objective vectors F_arr (N, n_obj)."""
            n = F_arr.shape[0]
            if n <= 2:
                return np.full(n, np.inf)

            cd = np.zeros(n)
            for m in range(n_obj):
                sorted_indices = np.argsort(F_arr[:, m])
                cd[sorted_indices[0]] = np.inf
                cd[sorted_indices[-1]] = np.inf
                f_range = F_arr[sorted_indices[-1], m] - F_arr[sorted_indices[0], m]
                if f_range == 0:
                    continue
                for i in range(1, n - 1):
                    idx = sorted_indices[i]
                    cd[idx] += (F_arr[sorted_indices[i + 1], m] - F_arr[sorted_indices[i - 1], m]) / f_range
            return cd

        # Archive as lists of numpy arrays
        archive_X = []
        archive_F = []

        def evaluate(x):
            """Evaluate x and return f, incrementing used."""
            nonlocal used
            if used >= budget:
                return None
            if time.time() - start_time > deadline:
                return None

            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            used += 1

            if np.any(~np.isfinite(f)):
                return None
            return x, f

        def add_to_archive(x, f):
            """Add point to archive if non-dominated, removing dominated points."""
            nonlocal archive_X, archive_F

            # Check if dominated by existing archive
            if dominates_any(f, archive_F):
                return False

            # Remove points dominated by f
            archive_X, archive_F = remove_dominated(f, archive_X, archive_F)

            archive_X.append(x)
            archive_F.append(f)
            return True

        # --- Initialization: diverse starting points ---
        # Center
        x0 = (lo + hi) / 2.0
        res = evaluate(x0)
        if res:
            add_to_archive(res[0], res[1])

        # Vertices of the hypercube
        for mask in range(2 ** dim):
            if used >= budget:
                break
            x = np.array([hi[j] if (mask >> j) & 1 else lo[j] for j in range(dim)])
            res = evaluate(x)
            if res:
                add_to_archive(res[0], res[1])

        # Additional random points for diversity
        while used < 15:
            x = lo + (hi - lo) * rng.random(dim)
            res = evaluate(x)
            if res:
                add_to_archive(res[0], res[1])

        # Reference point for hypervolume calculation
        ref_point = np.array([1705.0, 12.0, 0.3])

        # --- Multi-Objective Simulated Annealing with DE-inspired mutation ---
        # Temperature schedule
        T_initial = 1.0
        T_final = 0.01
        cooling_rate = 0.995

        # Current state: start from a random archive point or center
        if len(archive_X) > 0:
            idx = rng.randint(len(archive_X))
            current_x = archive_X[idx].copy()
            current_f = np.array(problem(current_x))
            used += 1
        else:
            current_x = (lo + hi) / 2.0
            current_f = np.array(problem(current_x))
            used += 1
            add_to_archive(current_x, current_f)

        # Track best hypervolume
        best_hv = hypervolume_3d(archive_F, ref_point)
        
        # MOSA loop
        T = T_initial
        no_improve_count = 0
        
        while used < budget:
            if time.time() - start_time > deadline:
                break

            # Generate a candidate using DE-inspired mutation from archive
            if len(archive_X) >= 3:
                # Pick 3 parents from archive
                p1, p2, p3 = rng.choice(len(archive_X), size=3, replace=False)
                F_factor = 0.5 + 0.3 * rng.random()
                CR = 0.7 + rng.random() * 0.3
                r = rng.random(dim)
                candidate_x = archive_X[p1].copy()
                for j in range(dim):
                    if r[j] < CR:
                        candidate_x[j] = archive_X[p1][j] + F_factor * (archive_X[p2][j] - archive_X[p3][j])
                    else:
                        candidate_x[j] = archive_X[p1][j]
            else:
                # Fallback: perturb current point
                step_size = T * (hi - lo) * 0.1
                candidate_x = current_x + rng.normal(0, 1, dim) * step_size
            
            candidate_x = np.clip(candidate_x, lo, hi)

            res = evaluate(candidate_x)
            if res is None:
                break
            
            candidate_x, candidate_f = res

            # Determine acceptance
            # 1. If candidate is non-dominated and improves hypervolume, accept
            # 2. If candidate is dominated, accept with probability exp(-delta/T)
            # 3. If candidate is non-dominated but doesn't improve HV, accept with probability exp(-small_delta/T)
            
            # Check if dominated by archive
            is_dominated = dominates_any(candidate_f, archive_F)
            
            # Compute hypervolume improvement
            archive_F_copy = list(archive_F)
            archive_F_copy.append(candidate_f)
            # Remove dominated points from the copy
            new_archive_F = []
            for i, f in enumerate(archive_F_copy):
                dominated = False
                for j, g in enumerate(archive_F_copy):
                    if i != j and dominates(g, f):
                        dominated = True
                        break
                if not dominated:
                    new_archive_F.append(f)
            
            new_hv = hypervolume_3d(new_archive_F, ref_point)
            hv_improvement = new_hv - best_hv
            
            if not is_dominated and hv_improvement > 0:
                # Accept: improves hypervolume and non-dominated
                accept = True
                delta = -hv_improvement  # negative means improvement
            elif not is_dominated:
                # Non-dominated but no HV improvement
                accept = True
                delta = 0.1  # small penalty
            else:
                # Dominated: compute how much worse it is
                # Find the point in archive that dominates it
                min_delta = float('inf')
                for af in archive_F:
                    if dominates(af, candidate_f):
                        delta_val = np.max(candidate_f - af)
                        min_delta = min(min_delta, delta_val)
                
                if min_delta == float('inf'):
                    accept = True
                    delta = 0
                else:
                    # Accept with probability exp(-delta/T)
                    if T > 1e-10:
                        prob = np.exp(-min_delta / T)
                        accept = rng.random() < prob
                        delta = min_delta
                    else:
                        accept = False
                        delta = min_delta

            if accept:
                current_x = candidate_x
                current_f = candidate_f
                add_to_archive(current_x, current_f)
                
                # Update best HV
                current_hv = hypervolume_3d(archive_F, ref_point)
                if current_hv > best_hv:
                    best_hv = current_hv
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1

            # Cool down
            T *= cooling_rate
            T = max(T, T_final)

            # If stuck for too long, restart from a random archive point
            if no_improve_count > 50 and len(archive_X) > 1:
                idx = rng.randint(len(archive_X))
                current_x = archive_X[idx].copy()
                res = evaluate(current_x)
                if res:
                    current_x, current_f = res
                no_improve_count = 0

        # Final local polishing: perturb archive points with small steps
        if len(archive_X) > 0:
            for _ in range(min(20, budget - used)):
                if time.time() - start_time > deadline:
                    break
                idx = rng.randint(len(archive_X))
                base_x = archive_X[idx]
                step_size = 0.01 * (hi - lo)
                x = base_x + rng.normal(0, 1, dim) * step_size
                res = evaluate(x)
                if res:
                    add_to_archive(res[0], res[1])

        return None
