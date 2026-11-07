"""Improved optimizer for BBOB f22 using direct structural approach:
Halton-based broad sampling + Nelder-Mead local refinement + coordinate polish."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        n = self.dim
        budget = self.budget
        total_calls = 0
        best_x = np.zeros(n)
        best_f = np.inf

        def try_eval(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return None
            if time.time() > deadline - 2:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            total_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_vector(index, dims):
            point = []
            bases = [2, 3, 5, 7, 11]
            for d in range(dims):
                point.append(halton(index, bases[d]))
            return np.array(point)

        # Phase 1: Global Search
        n_phase1 = 40
        phase1_results = []
        for i in range(1, n_phase1 + 1):
            if total_calls >= budget or time.time() > deadline - 10:
                break
            h_point = halton_vector(i, n)
            x_try = -4.5 + 9.0 * h_point
            f_try = try_eval(x_try)
            if f_try is None:
                break
            phase1_results.append((f_try, x_try.copy()))
        
        # Add structured points
        structured = [np.zeros(n), np.ones(n)*2.0, -np.ones(n)*2.0, np.ones(n)*-2.0]
        for p in structured:
            if total_calls >= budget or time.time() > deadline - 10:
                break
            f_s = try_eval(p)
            if f_s is not None:
                phase1_results.append((f_s, p.copy()))

        if not phase1_results:
            return best_x, best_f

        phase1_results.sort(key=lambda x: x[0])
        start_x = phase1_results[0][1]

        # Phase 2: Powell's Method
        def powell_refine(x0):
            nonlocal best_x, best_f
            x = x0.copy()
            f = try_eval(x)
            if f is None: return
            if f < best_f:
                best_f = f
                best_x = x.copy()
            
            directions = np.eye(n)
            # 2 cycles to save budget
            for cycle in range(2):
                if total_calls >= budget - 50 or time.time() > deadline - 10:
                    break
                x_old = x.copy()
                max_delta = 0
                max_idx = 0
                
                for i in range(n):
                    if total_calls >= budget - 50 or time.time() > deadline - 5:
                        break
                    dir_vec = directions[i]
                    
                    def line_func(t):
                        x_new = x + t * dir_vec
                        x_new = np.clip(x_new, -5.0, 5.0)
                        ft = try_eval(x_new)
                        if ft is None:
                            return 1e10
                        return ft
                    
                    try:
                        res = minimize_scalar(line_func, bounds=(-2.0, 2.0), method='bounded', options={'maxiter': 15})
                        if res.success:
                            t_opt = res.x
                            x_new = x + t_opt * dir_vec
                            x_new = np.clip(x_new, -5.0, 5.0)
                            f_new = try_eval(x_new)
                            if f_new is not None and f_new < f:
                                delta = f - f_new
                                if delta > max_delta:
                                    max_delta = delta
                                    max_idx = i
                                x = x_new
                                f = f_new
                                if f < best_f:
                                    best_f = f
                                    best_x = x.copy()
                    except Exception:
                        pass
                
                if max_delta > 1e-6:
                    new_dir = x - x_old
                    norm = np.linalg.norm(new_dir)
                    if norm > 1e-6:
                        new_dir = new_dir / norm
                        directions[max_idx] = new_dir

        powell_refine(start_x)

        # Phase 3: Nelder-Mead
        def nelder_mead(x0):
            nonlocal best_x, best_f
            if total_calls >= budget - 10:
                return
            
            simplex = [x0.copy()]
            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.5 if x0[i] < 4.0 else -0.5
                simplex.append(xi)
            
            simplex_f = []
            for i in range(n + 1):
                if total_calls >= budget - 5 or time.time() > deadline - 5:
                    return
                f_i = try_eval(simplex[i])
                if f_i is None:
                    return
                simplex_f.append(f_i)
            
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            gamma = 1.5
            rho = 2.0
            sigma_nm = 0.5
            no_improve = 0
            patience = 30
            
            while total_calls < budget - 5 and time.time() < deadline - 5:
                f_range = simplex_f[-1] - simplex_f[0]
                x_range = max(np.linalg.norm(simplex[-1] - simplex[0]), 1e-12)
                if f_range < 1e-8 and x_range < 1e-5:
                    break
                if no_improve >= patience:
                    break
                
                centroid = np.mean([simplex[i] for i in range(n)], axis=0)
                x_r = centroid + gamma * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                if total_calls >= budget - 3 or time.time() > deadline - 3:
                    break
                f_r = try_eval(x_r)
                if f_r is None:
                    break
                
                if simplex_f[0] <= f_r < simplex_f[-1]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                    no_improve = 0 if f_r < simplex_f[-2] else no_improve + 1
                elif f_r < simplex_f[0]:
                    x_e = centroid + rho * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    if total_calls >= budget - 3 or time.time() > deadline - 3:
                        break
                    f_e = try_eval(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                    no_improve = 0
                else:
                    x_c = centroid + sigma_nm * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    if total_calls >= budget - 3 or time.time() > deadline - 3:
                        break
                    f_c = try_eval(x_c)
                    if f_c is None:
                        break
                    if f_c < simplex_f[-1]:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                        no_improve += 1
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            if total_calls >= budget - 3 or time.time() > deadline - 3:
                                return
                            f_i = try_eval(simplex[i])
                            if f_i is None:
                                return
                            simplex_f[i] = f_i
                        no_improve += 1
                
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
            
            best_idx = int(np.argmin(simplex_f))
            if simplex_f[best_idx] < best_f:
                best_f = simplex_f[best_idx]
                best_x = simplex[best_idx].copy()

        if total_calls < budget - 10 and time.time() < deadline - 10:
            nelder_mead(best_x.copy())

        return best_x, best_f
