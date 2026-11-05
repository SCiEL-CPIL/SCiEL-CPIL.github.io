"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _mean_reflectance_vectorized(mats, thicknesses, n0, ns, band_nm):
    """
    Compute mean reflectance percent over the band using vectorized operations.
    mats: list of refractive indices
    thicknesses: list of thicknesses
    n0: ambient index
    ns: substrate index
    band_nm: [min_nm, max_nm]
    """
    lam_min, lam_max = band_nm
    wls = np.linspace(lam_min, lam_max, 401)
    
    # Initialize M components as arrays of ones/zeros
    # M = [[M11, M12], [M21, M22]]
    M11 = np.ones_like(wls, dtype=complex)
    M12 = np.zeros_like(wls, dtype=complex)
    M21 = np.zeros_like(wls, dtype=complex)
    M22 = np.ones_like(wls, dtype=complex)
    
    for i in range(len(mats)):
        n = mats[i]
        d = thicknesses[i]
        
        # beta = 2 * pi * n * d / lam
        beta = 2.0 * np.pi * n * d / wls
        cb = np.cos(beta)
        sb = np.sin(beta)
        
        # M_i components
        Mi11 = cb
        Mi12 = 1.0j * sb / n
        Mi21 = 1.0j * n * sb
        Mi22 = cb
        
        # M_new = M_old @ M_i
        # M11_new = M11*Mi11 + M12*Mi21
        # M12_new = M11*Mi12 + M12*Mi22
        # M21_new = M21*Mi11 + M22*Mi21
        # M22_new = M21*Mi12 + M22*Mi22
        
        new_M11 = M11 * Mi11 + M12 * Mi21
        new_M12 = M11 * Mi12 + M12 * Mi22
        new_M21 = M21 * Mi11 + M22 * Mi21
        new_M22 = M21 * Mi12 + M22 * Mi22
        
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
        
    # Calculate Y
    Y = (M11 + M12 * ns) / (M21 + M22 * ns)
    r = (n0 - Y) / (n0 + Y)
    R = np.abs(r)**2
    
    return np.mean(R) * 100.0

def _project_to_feasible(x, max_layer_nm, max_total_nm):
    """Project point x to satisfy box and sum constraints."""
    x = np.array(x, dtype=float)
    x = np.clip(x, 0.0, max_layer_nm)
    total = np.sum(x)
    if total > max_total_nm:
        x = x * (max_total_nm / total)
        x = np.clip(x, 0.0, max_layer_nm)
    return x

def _nelder_mead_fast(objective, x0, max_iter=200, timeout=10.0):
    """
    Optimized Nelder-Mead with timeout.
    """
    start_time = time.time()
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    simplex = [x0.copy()]
    fvals = [objective(x0)]
    
    for i in range(n):
        xi = x0.copy()
        step = 0.1 if x0[i] == 0 else 0.1 * x0[i]
        # Ensure step is not too large initially to stay in bounds roughly
        if step > 10: step = 10
        xi[i] += step
        simplex.append(xi)
        fvals.append(objective(xi))
        
    best_idx = 0
    for i in range(1, n+1):
        if fvals[i] < fvals[best_idx]:
            best_idx = i
            
    for k in range(max_iter):
        if time.time() - start_time > timeout:
            break
            
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[-1] - fvals[0]) < 1e-5:
            break
            
        centroid = np.mean(simplex[:-1], axis=0)
        
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            if fr < fvals[-1]:
                xc = centroid + rho * (xr - centroid)
            else:
                xc = centroid + rho * (simplex[-1] - centroid)
            fc = objective(xc)
            
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
                
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def _design_quarterwave_stack(n_layers, nL, nH, band_nm):
    """Design a quarter-wave stack for the center wavelength."""
    center = 0.5 * (band_nm[0] + band_nm[1])
    labels = []
    thicknesses = []
    for i in range(n_layers):
        if i % 2 == 0:
            labels.append('L')
            thicknesses.append(center / (4.0 * nL))
        else:
            labels.append('H')
            thicknesses.append(center / (4.0 * nH))
    return labels, thicknesses

def _design_magic_3layer(nL, nH, ns):
    """
    Design the classic 3-layer 'magic' antireflection coating.
    Layer 1 (air side): nL, thickness for quarter wave at center
    Layer 2: nH, thickness for quarter wave at center  
    Layer 3: nL, thickness for quarter wave at center
    This is a well-known broadband AR structure.
    """
    # Use center wavelength 600 nm as a good compromise for 400-800 nm
    center = 600.0
    labels = ['L', 'H', 'L']
    thicknesses = [
        center / (4.0 * nL),
        center / (4.0 * nH),
        center / (4.0 * nL)
    ]
    return labels, thicknesses

def _design_alternating_stack(n_layers, nL, nH, band_nm, scale_factor=1.0):
    """Design an alternating L/H stack with quarter-wave thicknesses."""
    center = 0.5 * (band_nm[0] + band_nm[1])
    labels = []
    thicknesses = []
    for i in range(n_layers):
        if i % 2 == 0:
            labels.append('L')
            thicknesses.append((center / (4.0 * nL)) * scale_factor)
        else:
            labels.append('H')
            thicknesses.append((center / (4.0 * nH)) * scale_factor)
    return labels, thicknesses

def _ga_optimize(objective, n_layers, max_layer_nm, max_total_nm, 
                 pop_size=30, n_gen=50, timeout=30.0):
    """
    Genetic Algorithm for thickness optimization.
    """
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Initialize population
    pop = np.zeros((pop_size, n_layers))
    for i in range(pop_size):
        pop[i] = rng.uniform(0, max_layer_nm, n_layers)
        pop[i] = _project_to_feasible(pop[i], max_layer_nm, max_total_nm)
    
    fitness = np.array([objective(pop[i]) for i in range(pop_size)])
    
    best_idx = np.argmin(fitness)
    best_x = pop[best_idx].copy()
    best_f = fitness[best_idx]
    
    for gen in range(n_gen):
        if time.time() - start_time > timeout:
            break
            
        # Selection: keep top half
        sort_idx = np.argsort(fitness)
        survivors = pop[sort_idx[:pop_size//2]].copy()
        surv_fit = fitness[sort_idx[:pop_size//2]].copy()
        
        # Update best
        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_f:
            best_f = fitness[gen_best_idx]
            best_x = pop[gen_best_idx].copy()
        
        # Create new population
        new_pop = np.zeros((pop_size, n_layers))
        new_pop[:pop_size//2] = survivors
        
        for i in range(pop_size//2, pop_size):
            # Crossover: blend two parents
            p1_idx = rng.integers(0, pop_size//2)
            p2_idx = rng.integers(0, pop_size//2)
            alpha = rng.uniform(0.2, 0.8)
            child = alpha * survivors[p1_idx] + (1-alpha) * survivors[p2_idx]
            child = _project_to_feasible(child, max_layer_nm, max_total_nm)
            
            # Mutation
            if rng.random() < 0.3:
                mut_idx = rng.integers(0, n_layers)
                child[mut_idx] += rng.normal(0, 20)
                child = _project_to_feasible(child, max_layer_nm, max_total_nm)
            
            new_pop[i] = child
        
        pop = new_pop
        fitness = np.array([objective(pop[i]) for i in range(pop_size)])
    
    return best_x, best_f

def design(problem):
    """Design an antireflection coating."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    best_labels = None
    best_thicknesses = None
    best_score = float('inf')
    
    def update_best(labels, thicknesses, score):
        nonlocal best_labels, best_thicknesses, best_score
        if score < best_score:
            best_labels = labels[:]
            best_thicknesses = thicknesses[:]
            best_score = score
    
    # Phase 1: Try known good designs
    # Magic 3-layer
    labels_3, thick_3 = _design_magic_3layer(nL, nH, ns)
    thick_3 = _project_to_feasible(thick_3, max_layer_nm, max_total_nm)
    score_3 = _mean_reflectance_vectorized(
        [nL if l == 'L' else nH for l in labels_3], 
        thick_3, n0, ns, band_nm
    )
    update_best(labels_3, thick_3.tolist(), score_3)
    
    # Alternating stacks with various layer counts
    for n_layers in [4, 6, 8, 10, 12, 14, 16, 18, 20]:
        if n_layers > max_layers:
            break
        if time.time() > deadline - 60:
            break
        
        labels_alt, thick_alt = _design_alternating_stack(n_layers, nL, nH, band_nm)
        thick_alt = _project_to_feasible(thick_alt, max_layer_nm, max_total_nm)
        score_alt = _mean_reflectance_vectorized(
            [nL if l == 'L' else nH for l in labels_alt], 
            thick_alt, n0, ns, band_nm
        )
        update_best(labels_alt, thick_alt.tolist(), score_alt)
    
    # Phase 2: Local refinement with Nelder-Mead on best candidates
    if time.time() < deadline - 30:
        # Refine the best solution found so far
        if best_labels is not None:
            mats = [nL if l == 'L' else nH for l in best_labels]
            n_layers = len(best_labels)
            
            def objective(x):
                t = _project_to_feasible(x, max_layer_nm, max_total_nm)
                return _mean_reflectance_vectorized(mats, t, n0, ns, band_nm)
            
            x0 = np.array(best_thicknesses)
            x0 = _project_to_feasible(x0, max_layer_nm, max_total_nm)
            
            try:
                x_opt, f_opt = _nelder_mead_fast(objective, x0, max_iter=150, timeout=20.0)
                x_opt = _project_to_feasible(x_opt, max_layer_nm, max_total_nm)
                f_opt = objective(x_opt)
                update_best(best_labels, x_opt.tolist(), f_opt)
            except Exception:
                pass
        
        # Try GA for a few layer counts
        for n_layers in [4, 6, 8, 10]:
            if time.time() > deadline - 15:
                break
            
            labels_ga = []
            for i in range(n_layers):
                if i % 2 == 0:
                    labels_ga.append('L')
                else:
                    labels_ga.append('H')
            
            mats_ga = [nL if l == 'L' else nH for l in labels_ga]
            
            def objective_ga(x):
                t = _project_to_feasible(x, max_layer_nm, max_total_nm)
                return _mean_reflectance_vectorized(mats_ga, t, n0, ns, band_nm)
            
            try:
                x_ga, f_ga = _ga_optimize(objective_ga, n_layers, max_layer_nm, 
                                          max_total_nm, pop_size=20, n_gen=30, timeout=10.0)
                x_ga = _project_to_feasible(x_ga, max_layer_nm, max_total_nm)
                f_ga = objective_ga(x_ga)
                update_best(labels_ga, x_ga.tolist(), f_ga)
            except Exception:
                pass
    
    # Phase 3: Final refinement if time remains
    if time.time() < deadline - 5 and best_labels is not None:
        mats_final = [nL if l == 'L' else nH for l in best_labels]
        
        def objective_final(x):
            t = _project_to_feasible(x, max_layer_nm, max_total_nm)
            return _mean_reflectance_vectorized(mats_final, t, n0, ns, band_nm)
        
        x0_final = np.array(best_thicknesses)
        x0_final = _project_to_feasible(x0_final, max_layer_nm, max_total_nm)
        
        try:
            x_final, f_final = _nelder_mead_fast(objective_final, x0_final, 
                                                 max_iter=100, timeout=8.0)
            x_final = _project_to_feasible(x_final, max_layer_nm, max_total_nm)
            f_final = objective_final(x_final)
            update_best(best_labels, x_final.tolist(), f_final)
        except Exception:
            pass
    
    # Fallback
    if best_labels is None:
        labels, thicknesses = _design_quarterwave_stack(4, nL, nH, band_nm)
        thicknesses = _project_to_feasible(thicknesses, max_layer_nm, max_total_nm)
        best_labels = labels
        best_thicknesses = thicknesses.tolist()
    
    # Final constraint check
    best_thicknesses = [max(0, min(t, max_layer_nm)) for t in best_thicknesses]
    total = sum(best_thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    return best_labels, best_thicknesses
# EVOLVE-BLOCK-END
