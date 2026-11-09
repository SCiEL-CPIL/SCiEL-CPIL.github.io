"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def design(problem):
    """Design a broadband antireflection coating by optimizing layer thicknesses
    directly against the exact characteristic-matrix reflectance objective."""
    
    t_start = time.time()
    DEADLINE = 115.0  # 120s limit, leave 5s safety margin
    
    ambient = problem.get("ambient_index", 1.0)
    substrate = problem.get("substrate_index", 1.52)
    materials = problem.get("materials", {"L": 1.38, "H": 2.35})
    n_L = materials["L"]
    n_H = materials["H"]
    band = problem.get("band_nm", [400, 800])
    lam_min, lam_max = band[0], band[1]
    max_layers = problem.get("max_layers", 20)
    max_total = problem.get("max_total_nm", 1000)
    max_layer = problem.get("max_layer_nm", 400)
    
    n_wl = 401
    wavelengths = np.linspace(lam_min, lam_max, n_wl)
    
    def compute_reflectance(labels, thicknesses_nm):
        """Exact characteristic-matrix method for normal incidence, lossless non-dispersive.
        Returns mean reflectance in percent over the band."""
        n_layers = len(labels)
        M = np.eye(2, dtype=complex)
        for i in range(n_layers):
            n_j = materials[labels[i]]
            d = thicknesses_nm[i]
            delta = 2.0 * np.pi * n_j * d / wavelengths
            c = np.cos(delta)
            s = np.sin(delta)
            i_c = 1j
            M00 = c
            M01 = i_c / (n_j * s)
            M10 = i_c * n_j * s
            M11 = c
            new00 = M[0,0]*M00 + M[0,1]*M10
            new01 = M[0,0]*M01 + M[0,1]*M11
            new10 = M[1,0]*M00 + M[1,1]*M10
            new11 = M[1,0]*M01 + M[1,1]*M11
            M = np.array([[new00, new01],[new10, new11]])
        Y_sub = substrate
        Y = (M[0,0] + M[0,1]*Y_sub) / (M[1,0] + M[1,1]*Y_sub)
        r = (Y - ambient) / (Y + ambient)
        R = np.abs(r)**2
        return float(np.mean(R) * 100.0)
    
    def make_symmetric_labels(n_layers):
        if n_layers % 2 != 0:
            n_layers -= 1
        half = n_layers // 2
        half_labels = []
        for i in range(half):
            half_labels.append("L" if i % 2 == 0 else "H")
        full = half_labels + list(reversed(half_labels))
        return full
    
    def make_random_labels(n_layers):
        """Generate a random valid label sequence."""
        return [random.choice(["L", "H"]) for _ in range(n_layers)]
    
    def project_to_feasible(x):
        """Project x to feasible region: 0 <= x_i <= max_layer, sum(x) <= max_total."""
        x = np.clip(x, 0.0, max_layer)
        s = np.sum(x)
        if s > max_total:
            x = x * (max_total / s)
            x = np.clip(x, 0.0, max_layer)
        return x
    
    def objective(x, labels):
        x = project_to_feasible(x)
        return compute_reflectance(labels, list(x))
    
    def optimize_from_start(labels, x0, time_budget):
        """Run L-BFGS-B then Nelder-Mead polish from a given start."""
        from scipy.optimize import minimize
        n_var = len(labels)
        x0 = project_to_feasible(x0.copy())
        best_x = x0.copy()
        best_val = objective(x0, labels)
        
        bounds = [(0.0, max_layer)] * n_var
        
        try:
            res = minimize(lambda x: objective(x, labels), x0, method='L-BFGS-B', 
                           bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
            if res.fun < best_val:
                best_x = project_to_feasible(res.x)
                best_val = objective(best_x, labels)
        except Exception:
            pass
        
        # Nelder-Mead polish with bounds enforcement
        if time.time() < t_start + time_budget - 2:
            try:
                def nm_objective(x):
                    x = project_to_feasible(x)
                    return objective(x, labels)
                res2 = minimize(nm_objective, best_x, method='Nelder-Mead',
                                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-14,
                                         'adaptive': True})
                if res2.fun < best_val:
                    best_x = project_to_feasible(res2.x)
                    best_val = objective(best_x, labels)
            except Exception:
                pass
        
        return best_x, best_val
    
    def optimize_stack(n_layers, n_starts=3, time_budget=30.0):
        """Try multiple starts for a given layer count and label pattern."""
        labels = make_symmetric_labels(n_layers)
        if len(labels) == 0:
            return None
        n_var = len(labels)
        center = 0.5*(lam_min + lam_max)
        
        best_x = None
        best_val = float('inf')
        
        starts = []
        # Start 1: quarter-wave at band center
        x0 = np.zeros(n_var)
        for i, lab in enumerate(labels):
            n_i = materials[lab]
            x0[i] = center / (4.0 * n_i)
        starts.append(x0)
        
        # Start 2: slightly perturbed quarter-wave
        x1 = x0.copy()
        for i in range(n_var):
            x1[i] = x0[i] * (0.8 + 0.4 * random.random())
        starts.append(x1)
        
        # Start 3: random feasible
        x2 = np.random.uniform(0.1, max_layer * 0.8, n_var)
        starts.append(x2)
        
        for x0 in starts:
            if time.time() >= t_start + time_budget:
                break
            x_opt, val = optimize_from_start(labels, x0, time_budget)
            if val < best_val:
                best_val = val
                best_x = x_opt
        
        if best_x is None:
            return None
        return labels, list(best_x), best_val
    
    def optimize_stack_random_labels(n_layers, n_starts=2, time_budget=25.0):
        """Try random label patterns for a given layer count."""
        best_labels = None
        best_thick = None
        best_val = float('inf')
        
        for _ in range(n_starts):
            if time.time() >= t_start + time_budget:
                break
            labels = list(make_random_labels(n_layers))
            n_var = len(labels)
            center = 0.5*(lam_min + lam_max)
            
            # Quarter-wave initial guess
            x0 = np.zeros(n_var)
            for i, lab in enumerate(labels):
                n_i = materials[lab]
                x0[i] = center / (4.0 * n_i)
            
            x_opt, val = optimize_from_start(labels, x0, time_budget)
            if val < best_val:
                best_val = val
                best_labels = labels
                best_thick = list(x_opt)
        
        if best_labels is None:
            return None
        return best_labels, best_thick, best_val
    
    # Try several configurations
    best_labels = None
    best_thick = None
    best_score = float('inf')
    
    # Symmetric patterns with multiple starts
    for n_layers in [10, 8, 12, 14, 6, 16]:
        if time.time() >= DEADLINE - 10:
            break
        remaining = DEADLINE - 10 - time.time()
        if remaining < 5:
            break
        time_budget = min(remaining * 0.8, 35.0)
        res = optimize_stack(n_layers, n_starts=3, time_budget=time_budget)
        if res is None:
            continue
        labels, th, val = res
        if len(labels) > max_layers:
            continue
        th = [max(0.0, min(t, max_layer)) for t in th]
        s = sum(th)
        if s > max_total:
            scale = max_total / s
            th = [t*scale for t in th]
            if any(t > max_layer for t in th):
                continue
        val2 = compute_reflectance(labels, th)
        if val2 < best_score:
            best_score = val2
            best_labels = labels
            best_thick = th
    
    # Random label patterns
    for n_layers in [8, 10, 12]:
        if time.time() >= DEADLINE - 10:
            break
        remaining = DEADLINE - 10 - time.time()
        if remaining < 5:
            break
        time_budget = min(remaining * 0.6, 25.0)
        res = optimize_stack_random_labels(n_layers, n_starts=2, time_budget=time_budget)
        if res is None:
            continue
        labels, th, val = res
        if len(labels) > max_layers:
            continue
        th = [max(0.0, min(t, max_layer)) for t in th]
        s = sum(th)
        if s > max_total:
            scale = max_total / s
            th = [t*scale for t in th]
            if any(t > max_layer for t in th):
                continue
        val2 = compute_reflectance(labels, th)
        if val2 < best_score:
            best_score = val2
            best_labels = labels
            best_thick = th
    
    # Final high-precision polish on the best solution
    if best_labels is not None and time.time() < DEADLINE - 3:
        from scipy.optimize import minimize
        remaining = DEADLINE - 3 - time.time()
        if remaining > 5:
            try:
                def final_obj(x):
                    x = project_to_feasible(x)
                    return compute_reflectance(best_labels, list(x))
                res = minimize(final_obj, np.array(best_thick), method='Nelder-Mead',
                               options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-16,
                                        'adaptive': True})
                if res.fun < best_score:
                    best_thick = list(project_to_feasible(res.x))
                    best_score = compute_reflectance(best_labels, best_thick)
            except Exception:
                pass
    
    # Fallback
    if best_labels is None:
        center = 0.5*(lam_min + lam_max)
        best_labels = ["L"]
        best_thick = [center/(4.0*n_L)]
    
    return best_labels, best_thick
# EVOLVE-BLOCK-END
