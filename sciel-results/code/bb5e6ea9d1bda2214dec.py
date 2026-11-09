"""Improved broadband AR coating design using combinatorial layer-assignment search
informed by continuous index profile, followed by thickness optimization."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def design(problem):
    """Design a broadband antireflection coating by searching over discrete L/H
    layer assignments informed by a continuous index profile, then optimizing
    thicknesses for promising patterns."""
    
    t_start = time.time()
    DEADLINE = 150.0 - 15.0  # 15s safety margin
    
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
    
    def project(x):
        """Project x onto feasible region: 0 <= x_i <= max_layer, sum <= max_total."""
        x = np.clip(x, 0.0, max_layer)
        s = np.sum(x)
        if s > max_total:
            x = x * (max_total / s)
            x = np.clip(x, 0.0, max_layer)
        return x
    
    def objective(x, labels):
        x = project(x)
        return compute_reflectance(labels, list(x))
    
    def optimize_thicknesses(labels, x0, max_iter=100):
        """Run L-BFGS-B to optimize thicknesses for a fixed layer pattern."""
        from scipy.optimize import minimize
        
        x0 = project(x0)
        best_x = x0.copy()
        best_val = objective(best_x, labels)
        
        bounds = [(0.0, max_layer)] * len(labels)
        
        try:
            res = minimize(lambda x: objective(x, labels), x0, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_val:
                best_x = project(res.x)
                best_val = objective(best_x, labels)
        except Exception:
            pass
        
        return best_x, best_val
    
    def compute_continuous_profile(n_layers):
        """Compute the ideal continuous refractive index profile for an AR coating.
        For a linear taper from ambient to substrate, the index at position i is:
        n_i = ambient + (substrate - ambient) * (i / n_layers)
        Return a list of 'L' or 'H' labels by comparing each n_i to the midpoint."""
        n_mid = 0.5 * (n_L + n_H)
        labels = []
        for i in range(n_layers):
            n_i = ambient + (substrate - ambient) * (i / n_layers)
            if n_i < n_mid:
                labels.append("L")
            else:
                labels.append("H")
        return labels
    
    def generate_initial_guess(labels):
        """Generate initial thickness guess: quarter-wave at center wavelength."""
        center = 0.5 * (lam_min + lam_max)
        x0 = np.zeros(len(labels))
        for i, lab in enumerate(labels):
            n_i = materials[lab]
            x0[i] = center / (4.0 * n_i)
        return project(x0)
    
    best_labels = None
    best_thick = None
    best_score = float('inf')
    
    # Phase 1: Generate diverse candidate patterns
    candidates = []
    
    # Pattern type 1: Continuous profile discretization (directive strategy)
    for n_layers in [8, 10, 12, 14, 16, 18, 20]:
        if n_layers > max_layers:
            continue
        labels = compute_continuous_profile(n_layers)
        x0 = generate_initial_guess(labels)
        candidates.append((labels, x0, "continuous"))
    
    # Pattern type 2: Symmetric alternating patterns
    for n_layers in [10, 12, 14, 16, 18]:
        if n_layers > max_layers:
            continue
        half = n_layers // 2
        half_labels = []
        for i in range(half):
            half_labels.append("L" if i % 2 == 0 else "H")
        labels = half_labels + list(reversed(half_labels))
        x0 = generate_initial_guess(labels)
        candidates.append((labels, x0, "symmetric_alt"))
        
    # Pattern type 3: Wood's coating style (HL pairs)
    for n_pairs in [5, 6, 7, 8, 9, 10]:
        n_layers = n_pairs * 2
        if n_layers > max_layers:
            continue
        labels = []
        for i in range(n_pairs):
            labels.append("H")
            labels.append("L")
        x0 = generate_initial_guess(labels)
        candidates.append((labels, x0, "woods"))
        
    # Pattern type 4: Geometric progression thicknesses with alternating indices
    for n_layers in [12, 14, 16]:
        if n_layers > max_layers:
            continue
        labels = []
        for i in range(n_layers):
            labels.append("L" if i % 2 == 0 else "H")
        for r in [1.1, 1.15, 1.2]:
            x0 = np.zeros(n_layers)
            sum_r = np.sum([r**i for i in range(n_layers)])
            d_0 = max_total / sum_r
            for i in range(n_layers):
                x0[i] = d_0 * (r ** i)
            x0 = project(x0)
            candidates.append((labels, x0, "geometric"))
    
    # Shuffle for diversity
    random.shuffle(candidates)
    
    # Phase 2: Evaluate candidates with quick optimization
    for labels, x0, ptype in candidates:
        if time.time() > DEADLINE - 30:  # Leave 30s for final polish
            break
        
        opt_x, opt_val = optimize_thicknesses(labels, x0, max_iter=80)
        
        # Validate
        if any(t < -1e-9 or t > max_layer + 1e-9 for t in opt_x):
            continue
        if sum(opt_x) > max_total + 1e-6:
            opt_x = opt_x * (max_total / sum(opt_x))
        
        val = compute_reflectance(labels, list(opt_x))
        if val < best_score:
            best_score = val
            best_labels = labels
            best_thick = list(opt_x)
    
    # Phase 3: Local search around best pattern - try small mutations
    if best_labels is not None and time.time() < DEADLINE - 20:
        current_labels = list(best_labels)
        current_thick = np.array(best_thick)
        current_score = best_score
        
        for mutation_iter in range(20):
            if time.time() > DEADLINE - 10:
                break
            
            # Try mutating one layer's index
            if len(current_labels) < max_layers:
                # Try inserting a layer
                for pos in range(len(current_labels) + 1):
                    if time.time() > DEADLINE - 10:
                        break
                    for new_idx in ["L", "H"]:
                        new_labels = current_labels[:pos] + [new_idx] + current_labels[pos:]
                        new_thick = np.zeros(len(new_labels))
                        # Distribute thickness proportionally
                        total_old = sum(current_thick)
                        if total_old > 0:
                            for i, lab in enumerate(new_labels):
                                if i < pos:
                                    new_thick[i] = current_thick[i]
                                elif i == pos:
                                    new_thick[i] = max_layer * 0.1  # Small initial
                                else:
                                    new_thick[i] = current_thick[i-1]
                        new_thick = project(new_thick)
                        
                        opt_x, opt_val = optimize_thicknesses(new_labels, new_thick, max_iter=50)
                        val = compute_reflectance(new_labels, list(opt_x))
                        if val < current_score:
                            current_labels = new_labels
                            current_thick = opt_x
                            current_score = val
                            best_labels = list(current_labels)
                            best_thick = list(current_thick)
                            best_score = current_score
            
            # Try changing one layer's index
            for pos in range(len(current_labels)):
                if time.time() > DEADLINE - 10:
                    break
                new_idx = "H" if current_labels[pos] == "L" else "L"
                new_labels = current_labels[:]
                new_labels[pos] = new_idx
                new_thick = current_thick.copy()
                
                opt_x, opt_val = optimize_thicknesses(new_labels, new_thick, max_iter=50)
                val = compute_reflectance(new_labels, list(opt_x))
                if val < current_score:
                    current_labels = new_labels
                    current_thick = opt_x
                    current_score = val
                    best_labels = list(current_labels)
                    best_thick = list(current_thick)
                    best_score = current_score
    
    # Phase 4: Final polish with Nelder-Mead
    if best_labels is not None and time.time() < DEADLINE - 5:
        try:
            from scipy.optimize import minimize
            labels = best_labels
            x0 = np.array(best_thick)
            bounds = [(0.0, max_layer)] * len(labels)
            res = minimize(lambda x: objective(x, labels), x0, method='Nelder-Mead',
                           options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-12})
            if res.fun < best_score:
                best_x = project(res.x)
                best_val = compute_reflectance(labels, list(best_x))
                if best_val < best_score:
                    best_score = best_val
                    best_thick = list(best_x)
        except Exception:
            pass
            
    # Fallback
    if best_labels is None:
        center = 0.5 * (lam_min + lam_max)
        best_labels = ["L"]
        best_thick = [center / (4.0 * n_L)]
    
    return best_labels, best_thick
# EVOLVE-BLOCK-END
