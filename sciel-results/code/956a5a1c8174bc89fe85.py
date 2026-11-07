import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin: evaluator kills at 330s
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        mid = (lo + hi) / 2.0
        
        archive_x = []
        archive_f = []
        evals = 0
        ref_point = np.array([3000.0, 0.06])
        
        def time_ok():
            return time.time() < deadline

        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def is_dominated(f_new, archive_f_list):
            if len(archive_f_list) == 0:
                return False
            for f in archive_f_list:
                if dominates(f, f_new):
                    return True
            return False

        def add_to_archive(x, f):
            nonlocal evals
            if is_dominated(f, archive_f):
                return False
            
            new_archive_x = []
            new_archive_f = []
            for xi, fi in zip(archive_x, archive_f):
                if dominates(f, fi):
                    continue
                new_archive_x.append(xi)
                new_archive_f.append(fi)
                
            new_archive_x.append(x)
            new_archive_f.append(f)
            
            archive_x[:] = new_archive_x
            archive_f[:] = new_archive_f
            return True

        def compute_hv_2d(f_list, ref_point):
            if not f_list:
                return 0.0
            pts = [f for f in f_list if np.all(f < ref_point)]
            if not pts:
                return 0.0
            pts = sorted(pts, key=lambda x: x[0])
            hv = 0.0
            for i in range(len(pts)):
                f1, f2 = pts[i]
                next_f1 = pts[i+1][0] if i < len(pts) - 1 else ref_point[0]
                width = next_f1 - f1
                height = ref_point[1] - f2
                if width > 0 and height > 0:
                    hv += width * height
            return hv

        def halton_sequence(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_point(n, dim):
            p = np.zeros(dim)
            primes = [2, 3, 5, 7]
            for d in range(dim):
                p[d] = halton_sequence(n, primes[d])
            return p

        def local_simplex_search(x0, f0, max_iters=10):
            """Local simplex search to improve a point."""
            nonlocal evals
            if evals >= self.budget or not time_ok():
                return x0, f0
            
            # Initialize simplex
            alpha = 0.05
            simplex = [x0.copy()]
            obj_values = [f0.copy()]
            
            for d in range(self.dim):
                x_new = x0.copy()
                # Step in dimension d
                step = range_[d] * alpha
                x_new[d] = x0[d] + step
                x_new = np.clip(x_new, lo, hi)
                simplex.append(x_new)
                
                if evals >= self.budget or not time_ok():
                    break
                f_new = problem(x_new)
                evals += 1
                obj_values.append(f_new)
            
            # Simplex iterations
            for _ in range(max_iters):
                if evals >= self.budget or not time_ok():
                    break
                if len(simplex) < 2:
                    break
                
                # Sort by sum of objectives (simple scalarization)
                indices = sorted(range(len(simplex)), key=lambda i: np.sum(obj_values[i]))
                simplex = [simplex[i] for i in indices]
                obj_values = [obj_values[i] for i in indices]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = obj_values[-1]
                
                # Reflection
                x_reflect = centroid + (centroid - worst)
                x_reflect = np.clip(x_reflect, lo, hi)
                
                if evals >= self.budget or not time_ok():
                    break
                f_reflect = problem(x_reflect)
                evals += 1
                
                # Accept if better than worst
                if np.all(f_reflect <= worst_f) and np.any(f_reflect < worst_f):
                    simplex[-1] = x_reflect
                    obj_values[-1] = f_reflect
                else:
                    # Try contraction
                    x_contract = centroid + 0.5 * (worst - centroid)
                    x_contract = np.clip(x_contract, lo, hi)
                    
                    if evals >= self.budget or not time_ok():
                        break
                    f_contract = problem(x_contract)
                    evals += 1
                    
                    if np.all(f_contract <= worst_f) and np.any(f_contract < worst_f):
                        simplex[-1] = x_contract
                        obj_values[-1] = f_contract
                    else:
                        # Shrink towards best
                        best = simplex[0]
                        for i in range(1, len(simplex)):
                            simplex[i] = best + 0.5 * (simplex[i] - best)
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            if evals >= self.budget or not time_ok():
                                break
                            f_new = problem(simplex[i])
                            evals += 1
                            obj_values[i] = f_new
            
            # Return best point
            best_idx = np.argmin([np.sum(f) for f in obj_values])
            return simplex[best_idx], obj_values[best_idx]

        # 1. Initialization: Use Halton sequence for better coverage
        init_size = 30
        for i in range(1, init_size + 1):
            if not time_ok():
                break
            p = halton_point(i, self.dim)
            x = lo + range_ * p
            f = problem(x)
            evals += 1
            add_to_archive(x, f)
        
        if not time_ok():
            return archive_x

        # 2. Main search: Adaptive random restart with local search
        # We will use the remaining budget for the main loop.
        
        # Determine reference point based on current best
        if archive_f:
            current_best = min(archive_f, key=lambda x: x[0] + x[1]) # Simple heuristic
        else:
            current_best = [1.0, 1.0]
        
        # Strategy:
        # - Use a mix of global exploration (random) and local exploitation (Gaussian around best points)
        # - Periodically restart from a new random point if stuck
        # - Use a simple adaptation of step size based on improvement
        # - Occasionally apply local simplex search to promising points
        
        best_hv_so_far = compute_hv_2d(archive_f, ref_point)
        last_improvement_eval = evals
        
        while evals < self.budget and time_ok():
            # Decide whether to explore or exploit
            # If we haven't improved in a while, explore more
            no_improvement = evals - last_improvement_eval
            
            if no_improvement > 50:
                # Force exploration
                x = lo + range_ * np.random.rand(self.dim)
            elif archive_x and np.random.rand() < 0.7:
                # Exploit: pick a random point from archive
                idx = np.random.randint(len(archive_x))
                parent = archive_x[idx]
                # Adaptive step size: smaller if we are exploiting a good point
                # Step size proportional to range, decaying over time
                progress = evals / self.budget
                step_scale = 0.1 * (1.0 - 0.5 * progress)
                step = step_scale * range_
                x = parent + step * np.random.normal(size=self.dim)
                x = np.clip(x, lo, hi)
            else:
                # Moderate exploration: random point near center or random
                if np.random.rand() < 0.5:
                    x = lo + range_ * np.random.rand(self.dim)
                else:
                    # Gaussian around center
                    x = mid + 0.2 * range_ * np.random.normal(size=self.dim)
                    x = np.clip(x, lo, hi)
            
            f_child = problem(x)
            evals += 1
            
            # Check if this point is better than reference point
            # And if it improves the archive
            was_added = add_to_archive(x, f_child)
            
            if was_added:
                # Recalculate HV to see if it improved
                current_hv = compute_hv_2d(archive_f, ref_point)
                if current_hv > best_hv_so_far:
                    best_hv_so_far = current_hv
                    last_improvement_eval = evals
                    
                    # Occasionally apply local search to improve the new point
                    if evals < self.budget - 15 and time_ok() and np.random.rand() < 0.3:
                        x_opt, f_opt = local_simplex_search(x, f_child, max_iters=5)
                        if evals < self.budget and time_ok():
                            # Check if the optimized point is better
                            if not is_dominated(f_opt, archive_f):
                                add_to_archive(x_opt, f_opt)
                                current_hv = compute_hv_2d(archive_f, ref_point)
                                if current_hv > best_hv_so_far:
                                    best_hv_so_far = current_hv
                                    last_improvement_eval = evals

        return archive_x
