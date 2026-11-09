"""Combined optimizer: Rastrigin-structure-aware search + Nelder-Mead ensemble."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Rastrigin-aware optimizer with integer-grid exploitation and NM refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def call_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            val = float(func(x))
            call_count += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Exploit Rastrigin's periodic structure via integer-aligned sampling
        # The global minimum of Rastrigin is at integer-aligned points (after shift)
        
        # 1a: Evaluate origin and immediate integer neighbors
        candidates = [np.zeros(dim)]
        
        # Single-coordinate integer offsets
        for d in range(dim):
            for sign in [1, -1]:
                pt = np.zeros(dim)
                pt[d] = sign
                candidates.append(pt)
        
        # Two-coordinate integer offsets
        for d1 in range(dim):
            for d2 in range(d1 + 1, dim):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        pt = np.zeros(dim)
                        pt[d1] = s1
                        pt[d2] = s2
                        candidates.append(pt)
        
        for pt in candidates:
            if call_count >= budget or time.time() > deadline:
                break
            fx = call_func(pt)
            if fx is None:
                break
        
        # 1b: Broader integer grid sampling - more samples for better coverage
        int_range = list(range(-4, 5))
        n_int_samples = 200
        for _ in range(n_int_samples):
            if call_count >= budget or time.time() > deadline:
                break
            pt = np.array([rng.choice(int_range) for _ in range(dim)], dtype=float)
            fx = call_func(pt)
            if fx is None:
                break
        
        # Phase 1c: Integer coordinate refinement - try each integer coordinate in each dimension
        if call_count < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            
            for d in range(dim):
                if call_count >= budget or time.time() > deadline:
                    break
                
                best_coord_val = fx
                best_coord = x[d]
                
                # Search integer coordinates near current position
                for ic in range(int(x[d]) - 2, int(x[d]) + 3):
                    if call_count >= budget or time.time() > deadline:
                        break
                    pt = x.copy()
                    pt[d] = float(ic)
                    pt = np.clip(pt, -5.0, 5.0)
                    fx_test = call_func(pt)
                    if fx_test is None:
                        break
                    if fx_test < best_coord_val - 1e-10:
                        best_coord_val = fx_test
                        best_coord = float(ic)
                
                if best_coord_val < fx:
                    x[d] = best_coord
                    fx = best_coord_val
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
        
        # Phase 2: Nelder-Mead ensemble refinement around best point
        if call_count < budget and time.time() < deadline:
            # Initialize multiple simplexes around the best point found
            num_runs = 4
            simplex_step = 0.5
            
            # First simplex centered on best point
            simplex_starts = [best_x.copy()]
            
            # Additional simplexes at slightly offset positions
            for i in range(1, num_runs):
                offset = rng.uniform(-1.0, 1.0, dim)
                s = best_x + offset
                s = np.clip(s, -5.0, 5.0)
                simplex_starts.append(s)
            
            simplexes = []
            for s in simplex_starts:
                simplex = [(s.copy(), np.inf)]
                for j in range(dim):
                    point = s.copy()
                    point[j] += simplex_step
                    point = np.clip(point, -5.0, 5.0)
                    simplex.append((point, np.inf))
                simplexes.append(simplex)
            
            # Evaluate all initial simplex points
            for simplex in simplexes:
                for idx, (pt, _) in enumerate(simplex):
                    if call_count >= budget:
                        break
                    f = call_func(pt)
                    if f is None:
                        break
                    simplex[idx] = (pt, f)
                if call_count >= budget:
                    break
            
            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5
            
            # Interleave iterations across simplexes
            iteration = 0
            active_simplexes = list(range(num_runs))
            
            while call_count < budget and active_simplexes:
                if time.time() > deadline:
                    break
                
                si = active_simplexes[iteration % len(active_simplexes)]
                simplex = simplexes[si]
                
                if len(simplex) < 2:
                    continue
                
                # Sort by function value
                simplex.sort(key=lambda pf: pf[1])
                
                # Check convergence
                x_best_s = simplex[0][0]
                x_worst_s = simplex[-1][0]
                if np.max(np.abs(x_best_s - x_worst_s)) < 1e-7:
                    active_simplexes.remove(si)
                    continue
                
                x_worst = simplex[-1][0]
                f_worst = simplex[-1][1]
                
                # Centroid of all points except worst
                centroid = np.zeros(dim)
                for i in range(len(simplex) - 1):
                    centroid += simplex[i][0]
                centroid /= (len(simplex) - 1)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - x_worst)
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = call_func(x_ref)
                if f_ref is None:
                    break
                
                if f_ref < simplex[0][1]:
                    # Try expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    f_exp = call_func(x_exp)
                    if f_exp is None:
                        break
                    if f_exp < f_ref:
                        simplex[-1] = (x_exp, f_exp)
                    else:
                        simplex[-1] = (x_ref, f_ref)
                elif f_ref < simplex[-2][1]:
                    # Accept reflection
                    simplex[-1] = (x_ref, f_ref)
                else:
                    # Contraction
                    if f_ref < f_worst:
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        x_con = centroid - rho * (x_worst - centroid)
                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = call_func(x_con)
                    if f_con is None:
                        break
                    if f_con <= min(f_ref, f_worst):
                        simplex[-1] = (x_con, f_con)
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            pt, fv = simplex[i]
                            new_pt = x_best_s + sigma_shrink * (pt - x_best_s)
                            new_pt = np.clip(new_pt, -5.0, 5.0)
                            new_f = call_func(new_pt)
                            if new_f is None:
                                break
                            simplex[i] = (new_pt, new_f)
                        if call_count >= budget:
                            break
                
                iteration += 1
        
        # Phase 3: Fine coordinate descent polish
        if best_x is not None and call_count < budget and time.time() < deadline:
            polish_x = best_x.copy()
            polish_f = best_f
            step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
            
            for step in step_sizes:
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                improved = True
                while improved and call_count < budget:
                    improved = False
                    for d in range(dim):
                        if call_count >= budget:
                            break
                        for direction in [-1, 1]:
                            if call_count >= budget:
                                break
                            trial = polish_x.copy()
                            trial[d] += direction * step
                            trial = np.clip(trial, -5.0, 5.0)
                            f_trial = call_func(trial)
                            if f_trial is None:
                                break
                            if f_trial < polish_f - 1e-12:
                                polish_x = trial
                                polish_f = f_trial
                                improved = True
                                best_x = polish_x.copy()
                                best_f = polish_f
                            elif time.time() > deadline:
                                break
        
        # Phase 4: If still not good enough, do adaptive restarts with increasing jumps
        if call_count < budget and time.time() < deadline and best_f > 3.0:
            n_restarts = min(20, (budget - call_count) // 12)
            for r in range(n_restarts):
                if call_count >= budget or time.time() > deadline:
                    break
                
                # Adaptive jump: increase distance from current best as restarts continue
                jump_scale = 1.0 + 0.5 * r
                pt = best_x + rng.uniform(-jump_scale, jump_scale, dim)
                pt = np.clip(pt, -5.0, 5.0)
                
                fx = call_func(pt)
                if fx is None:
                    break
                
                if fx < best_f:
                    # Quick local refinement from this point
                    x = pt.copy()
                    fx_cur = fx
                    for step in [0.2, 0.1, 0.05, 0.02, 0.01]:
                        if call_count >= budget or time.time() > deadline:
                            break
                        improved = True
                        while improved and call_count < budget and time.time() < deadline:
                            improved = False
                            for d in range(dim):
                                if call_count >= budget:
                                    break
                                for direction in [1, -1]:
                                    if call_count >= budget:
                                        break
                                    x_new = x.copy()
                                    x_new[d] += direction * step
                                    x_new = np.clip(x_new, -5.0, 5.0)
                                    fx_new = call_func(x_new)
                                    if fx_new is None:
                                        break
                                    if fx_new < fx_cur - 1e-12:
                                        x = x_new
                                        fx_cur = fx_new
                                        improved = True
                                        if fx_cur < best_f:
                                            best_f = fx_cur
                                            best_x = x.copy()
                                        break
                            if time.time() > deadline:
                                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
