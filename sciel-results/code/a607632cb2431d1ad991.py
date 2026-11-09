# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

def design(problem):
    """Design a broadband antireflection coating by optimizing layer thicknesses
    directly against the exact characteristic-matrix reflectance objective."""
    
    t_start = time.time()
    DEADLINE = 140.0  # leave 10s safety margin before 150s kill
    
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
        if n_layers == 0:
            r0 = (ambient - substrate) / (ambient + substrate)
            return float(np.abs(r0)**2 * 100.0)
            
        M = np.eye(2, dtype=complex)
        for i in range(n_layers):
            n_j = materials[labels[i]]
            d = thicknesses_nm[i]
            if d < 1e-9:
                continue
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
    
    def make_alternating_labels(n_layers):
        """Build an alternating label sequence starting with L."""
        labels = []
        for i in range(n_layers):
            labels.append("L" if i % 2 == 0 else "H")
        return labels
    
    def make_symmetric_labels(n_layers):
        """Build a symmetric label sequence. For even n, half-stack mirrored."""
        if n_layers % 2 != 0:
            n_layers -= 1
        if n_layers <= 0:
            return []
        half = n_layers // 2
        half_labels = []
        for i in range(half):
            half_labels.append("L" if i % 2 == 0 else "H")
        full = half_labels + list(reversed(half_labels))
        return full
    
    def objective_wrapper(labels, x):
        """Objective function with constraint projection."""
        x = np.clip(x, 0.0, max_layer)
        s = np.sum(x)
        if s > max_total:
            x = x * (max_total / s)
            x = np.clip(x, 0.0, max_layer)
        return compute_reflectance(labels, list(x))
    
    def optimize_stack(n_layers, n_restarts=3):
        """Optimize a stack of given size using differential evolution then local polish."""
        labels = make_alternating_labels(n_layers)
        if len(labels) == 0:
            return None
        n_var = len(labels)
        
        center = 0.5*(lam_min + lam_max)
        
        def objective(x):
            return objective_wrapper(labels, x)
        
        best_x = None
        best_val = float('inf')
        
        bounds = [(0.0, max_layer)] * n_var
        
        # Try differential evolution first for global search
        if time.time() < DEADLINE - 30:
            try:
                # Use a constrained objective that penalizes sum > max_total
                def de_objective(x):
                    x = np.clip(x, 0.0, max_layer)
                    s = np.sum(x)
                    if s > max_total:
                        # Heavy penalty
                        return compute_reflectance(labels, list(x)) + 10.0 * (s - max_total)
                    return compute_reflectance(labels, list(x))
                
                # Run differential evolution with limited budget
                de_result = differential_evolution(
                    de_objective, bounds,
                    maxiter=50, popsize=15, seed=42,
                    tol=1e-6, atol=1e-6,
                    polish=False,
                    init='sobol',
                    mutation=(0.3, 1.0, 'bin'),
                    recombination=0.7
                )
                
                if de_result.fun < best_val:
                    best_val = de_result.fun
                    best_x = de_result.x
            except Exception:
                pass
        
        # Local polish with L-BFGS-B
        if best_x is not None and time.time() < DEADLINE - 15:
            try:
                res = minimize(objective, best_x, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 30})
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            except Exception:
                pass
        
        # Nelder-Mead polish for robustness
        if best_x is not None and time.time() < DEADLINE - 10:
            try:
                res2 = minimize(objective, best_x, method='Nelder-Mead',
                                options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-10, 'adaptive': True})
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_x = res2.x
            except Exception:
                pass
        
        if best_x is None:
            return None
        
        # Final projection
        best_x = np.clip(best_x, 0.0, max_layer)
        s = np.sum(best_x)
        if s > max_total:
            best_x = best_x * (max_total / s)
            best_x = np.clip(best_x, 0.0, max_layer)
        
        return labels, list(best_x), best_val
    
    # Try several layer counts
    best_labels = None
    best_thick = None
    best_score = float('inf')
    
    # Prioritize layer counts that are likely to work well
    layer_counts = [12, 14, 10, 16, 8, 18, 20]
    
    for n_layers in layer_counts:
        if time.time() >= DEADLINE - 15:
            break
        res = optimize_stack(n_layers, n_restarts=1)
        if res is None:
            continue
        labels, th, val = res
        # Validate
        if len(labels) > max_layers:
            continue
        if any(t < -1e-9 or t > max_layer + 1e-9 for t in th):
            continue
        
        # Final projection and check
        th_np = np.array(th)
        th_np = np.clip(th_np, 0.0, max_layer)
        s = np.sum(th_np)
        if s > max_total:
            th_np = th_np * (max_total / s)
            th_np = np.clip(th_np, 0.0, max_layer)
        
        th_list = list(th_np)
        if any(t > max_layer for t in th_list):
            continue
        
        val2 = compute_reflectance(labels, th_list)
        if val2 < best_score:
            best_score = val2
            best_labels = labels
            best_thick = th_list
    
    # Fallback
    if best_labels is None:
        center = 0.5*(lam_min + lam_max)
        best_labels = ["L"]
        best_thick = [center/(4.0*n_L)]
    
    return best_labels, best_thick
# EVOLVE-BLOCK-END
