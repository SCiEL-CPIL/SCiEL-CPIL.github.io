"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START

def _reflectance_spectrum(materials, thicknesses, wls, ambient_index=1.0, substrate_index=1.52, materials_dict=None):
    """
    Compute reflectance at all wavelengths using the characteristic matrix method (vectorized).
    """
    if materials_dict is None:
        materials_dict = {'L': 1.38, 'H': 2.35}
    
    n_layers = len(materials)
    if n_layers == 0:
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return np.full_like(wls, r*r)
        
    M11 = np.ones_like(wls, dtype=complex)
    M12 = np.zeros_like(wls, dtype=complex)
    M21 = np.zeros_like(wls, dtype=complex)
    M22 = np.ones_like(wls, dtype=complex)
    
    for mat, thick in zip(materials, thicknesses):
        n = materials_dict[mat]
        if thick < 1e-9:
            continue
            
        beta = 2.0 * np.pi * n * thick / wls
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        L11 = cos_b
        L12 = 1.0j * sin_b / n
        L21 = n * 1.0j * sin_b
        L22 = cos_b
        
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22
        
    Y_s = substrate_index
    Y = (M21 + M22 * Y_s) / (M11 + M12 * Y_s)
    
    Y_0 = ambient_index
    r = (Y_0 - Y) / (Y_0 + Y)
    R = np.abs(r) ** 2
    return R

def _mean_reflectance(materials, thicknesses, band_nm=(400, 800), n_points=401,
                       ambient_index=1.0, substrate_index=1.52, materials_dict=None):
    """Compute mean reflectance in percent over the band."""
    if materials_dict is None:
        materials_dict = {'L': 1.38, 'H': 2.35}
    
    wls = np.linspace(band_nm[0], band_nm[1], n_points)
    refls = _reflectance_spectrum(materials, thicknesses, wls, ambient_index, substrate_index, materials_dict)
    return 100.0 * np.mean(refls)

def _is_valid(materials, thicknesses, problem):
    """Check if the design satisfies all constraints."""
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < -1e-9 or t > max_layer_nm + 1e-9:
            return False
    if sum(thicknesses) > max_total_nm + 1e-9:
        return False
    return True

def _penalized_score(materials, thicknesses, problem, band_nm=(400, 800), n_points=401):
    """
    Compute a penalized mean reflectance score.
    Adds penalty for constraint violations.
    """
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    materials_dict = problem["materials"]
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    
    penalty = 0.0
    total = sum(thicknesses)
    if total > max_total_nm:
        penalty += (total - max_total_nm) * 0.1
    for t in thicknesses:
        if t < 0:
            penalty += abs(t) * 0.1
        elif t > max_layer_nm:
            penalty += (t - max_layer_nm) * 0.1
    
    base_score = _mean_reflectance(materials, thicknesses, band_nm, n_points, ambient_index, substrate_index, materials_dict)
    return base_score + penalty

def _nelder_mead(f, x0, bounds, max_iter=200, tol=1e-5, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    """
    Simple Nelder-Mead implementation with box constraints.
    """
    n = len(x0)
    simplex = [x0.copy()]
    for i in range(n):
        x_new = x0.copy()
        step = 0.05 * (bounds[i][1] - bounds[i][0]) + 0.01
        x_new[i] += step
        x_new[i] = np.clip(x_new[i], bounds[i][0], bounds[i][1])
        simplex.append(x_new)
        
    simplex_f = [f(x) for x in simplex]
    
    for it in range(max_iter):
        order = np.argsort(simplex_f)
        simplex = [simplex[i] for i in order]
        simplex_f = [simplex_f[i] for i in order]
        
        if abs(simplex_f[-1] - simplex_f[0]) < tol:
            break
            
        centroid = np.mean(simplex[:-1], axis=0)
        
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, [b[0] for b in bounds], [b[1] for b in bounds])
        f_r = f(x_r)
        
        if f_r < simplex_f[0]:
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, [b[0] for b in bounds], [b[1] for b in bounds])
            f_e = f(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                simplex_f[-1] = f_e
            else:
                simplex[-1] = x_r
                simplex_f[-1] = f_r
        elif f_r < simplex_f[-2]:
            simplex[-1] = x_r
            simplex_f[-1] = f_r
        else:
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c = np.clip(x_c, [b[0] for b in bounds], [b[1] for b in bounds])
            f_c = f(x_c)
            if f_c < simplex_f[-1]:
                simplex[-1] = x_c
                simplex_f[-1] = f_c
            else:
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    simplex_f[i] = f(simplex[i])
                    
    best_idx = np.argmin(simplex_f)
    return simplex[best_idx], simplex_f[best_idx]

def design(problem):
    """
    Design using multi-start Nelder-Mead for thickness optimization
    over a diverse set of material sequences.
    """
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill
    
    materials_dict = problem["materials"]
    nL = materials_dict["L"]
    nH = materials_dict["H"]
    n_sub = problem["substrate_index"]
    n_amb = problem["ambient_index"]
    band_nm = problem["band_nm"]
    wl_center = 0.5 * (band_nm[0] + band_nm[1])
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    wls = np.linspace(band_nm[0], band_nm[1], 401)
    
    best_score = 100.0
    best_materials = ["L"]
    best_thicknesses = [wl_center / (4.0 * nL)]
    
    # Generate diverse patterns
    patterns = []
    
    # 1. Single layer
    for mat in ['L', 'H']:
        n = nL if mat == 'L' else nH
        d0 = wl_center / (4.0 * n)
        patterns.append(([mat], [d0]))
        
    # 2. Two layers
    for p in [('L', 'H'), ('H', 'L')]:
        n1 = nL if p[0]=='L' else nH
        n2 = nL if p[1]=='L' else nH
        d1 = wl_center / (4.0 * n1)
        d2 = wl_center / (4.0 * n2)
        patterns.append((list(p), [d1, d2]))
        
    # 3. Three layers
    for p in [('L', 'H', 'L'), ('H', 'L', 'H')]:
        n1 = nL if p[0]=='L' else nH
        n2 = nL if p[1]=='L' else nH
        n3 = nL if p[2]=='L' else nH
        d1 = wl_center / (4.0 * n1)
        d2 = wl_center / (4.0 * n2)
        d3 = wl_center / (4.0 * n3)
        patterns.append((list(p), [d1, d2, d3]))
        
    # 4. Four layers
    for p in [('L', 'H', 'L', 'H'), ('H', 'L', 'H', 'L')]:
        thicks = []
        for m in p:
            n = nL if m=='L' else nH
            thicks.append(wl_center / (4.0 * n))
        patterns.append((list(p), thicks))
        
    # 5. Five layers
    for p in [('L', 'H', 'L', 'H', 'L'), ('H', 'L', 'H', 'L', 'H')]:
        thicks = []
        for m in p:
            n = nL if m=='L' else nH
            thicks.append(wl_center / (4.0 * n))
        patterns.append((list(p), thicks))
        
    # 6. Six layers
    for p in [('L', 'H', 'L', 'H', 'L', 'H'), ('H', 'L', 'H', 'L', 'H', 'L')]:
        thicks = []
        for m in p:
            n = nL if m=='L' else nH
            thicks.append(wl_center / (4.0 * n))
        patterns.append((list(p), thicks))
        
    # 7. Seven layers
    for p in [('L', 'H', 'L', 'H', 'L', 'H', 'L'), ('H', 'L', 'H', 'L', 'H', 'L', 'H')]:
        thicks = []
        for m in p:
            n = nL if m=='L' else nH
            thicks.append(wl_center / (4.0 * n))
        patterns.append((list(p), thicks))
        
    # 8. Eight layers
    for p in [('L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'), ('H', 'L', 'H', 'L', 'H', 'L', 'H', 'L')]:
        thicks = []
        for m in p:
            n = nL if m=='L' else nH
            thicks.append(wl_center / (4.0 * n))
        patterns.append((list(p), thicks))
        
    # 9. Ten layers
    for p in [('L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'), ('H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L')]:
        thicks = []
        for m in p:
            n = nL if m=='L' else nH
            thicks.append(wl_center / (4.0 * n))
        patterns.append((list(p), thicks))
        
    # Add some random initializations for variety
    np.random.seed(42)
    random.seed(42)
    for _ in range(10):
        n_layers = np.random.randint(3, 10)
        mats = []
        thicks = []
        for i in range(n_layers):
            m = np.random.choice(['L', 'H'])
            n = nL if m == 'L' else nH
            d = np.random.uniform(50, 200)
            mats.append(m)
            thicks.append(d)
        if sum(thicks) <= max_total_nm:
            patterns.append((mats, thicks))
            
    # Optimize each pattern using Nelder-Mead
    for mats, init_thicks in patterns:
        if time.time() - start_time > deadline - 10:
            break
            
        current_materials = mats
        bounds = [(0.0, max_layer_nm) for _ in range(len(mats))]
        x0 = np.array(init_thicks)
        
        def objective(thicknesses):
            return _penalized_score(mats, thicknesses, problem, band_nm)
        
        try:
            x_opt, f_opt = _nelder_mead(objective, x0, bounds, max_iter=500)
            if f_opt < best_score:
                best_score = f_opt
                best_materials = mats
                best_thicknesses = list(x_opt)
        except Exception:
            pass
            
    # Final local refinement on the best solution found
    if time.time() - start_time < deadline - 5 and len(best_materials) > 0:
        current_materials = best_materials
        bounds = [(0.0, max_layer_nm) for _ in range(len(best_materials))]
        x0 = np.array(best_thicknesses)
        
        def objective_final(thicknesses):
            return _penalized_score(best_materials, thicknesses, problem, band_nm)
        
        try:
            x_opt, f_opt = _nelder_mead(objective_final, x0, bounds, max_iter=1000, tol=1e-7)
            if f_opt < best_score:
                best_score = f_opt
                best_thicknesses = list(x_opt)
        except Exception:
            pass
            
    # Ensure validity
    if not _is_valid(best_materials, best_thicknesses, problem):
        # Fallback to a valid single layer
        best_materials = ["L"]
        best_thicknesses = [wl_center / (4.0 * nL)]
        
    # Clip thicknesses to valid range
    best_thicknesses = [max(0.0, min(max_layer_nm, t)) for t in best_thicknesses]
    
    # Check total thickness constraint and scale if necessary
    total = sum(best_thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        best_thicknesses = [t * scale for t in best_thicknesses]
        
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
