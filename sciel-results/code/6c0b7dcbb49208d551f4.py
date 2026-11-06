"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def _compute_reflectance(labels, thicknesses, materials, ambient_index, substrate_index, band_nm):
    """Compute mean reflectance in percent over the band using characteristic matrix method."""
    if len(labels) == 0:
        r0 = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return r0 * r0 * 100.0
    
    n_L = materials['L']
    n_H = materials['H']
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    n_layers = len(labels)
    
    # Build index array
    n_indices = np.empty(n_layers + 2, dtype=float)
    n_indices[0] = ambient_index
    n_indices[-1] = substrate_index
    for i, lab in enumerate(labels):
        n_indices[i + 1] = n_L if lab == 'L' else n_H
    
    d = np.array(thicknesses, dtype=float)
    
    # Vectorized computation over all wavelengths
    delta = 2.0 * np.pi * np.outer(n_indices[1:-1] * d, 1.0 / wavelengths)
    cos_d = np.cos(delta)
    sin_d = np.sin(delta)
    
    m11 = np.ones(401, dtype=complex)
    m12 = np.zeros(401, dtype=complex)
    m21 = np.zeros(401, dtype=complex)
    m22 = np.ones(401, dtype=complex)
    
    for j in range(n_layers):
        n_j = n_indices[j + 1]
        lm11 = cos_d[j]
        lm12 = 1j * sin_d[j] / n_j
        lm21 = 1j * n_j * sin_d[j]
        lm22 = cos_d[j]
        
        new_m11 = m11 * lm11 + m12 * lm21
        new_m12 = m11 * lm12 + m12 * lm22
        new_m21 = m21 * lm11 + m22 * lm21
        new_m22 = m21 * lm12 + m22 * lm22
        
        m11 = new_m11
        m12 = new_m12
        m21 = new_m21
        m22 = new_m22
    
    Y = (m21 + m22 * substrate_index) / (m11 + m12 * substrate_index)
    r = (ambient_index - Y) / (ambient_index + Y)
    reflectances = np.abs(r) ** 2
    
    return np.mean(reflectances) * 100.0

def _nelder_mead(objective, x0, step=10.0, max_iter=100, tol=1e-5, bounds=None):
    """
    Nelder-Mead optimizer with bounds.
    x0: initial point
    bounds: list of (min, max) for each dimension
    """
    n = len(x0)
    
    # Initialize simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        xi = np.array(x0, dtype=float)
        xi[i] += step
        if bounds:
            if xi[i] > bounds[i][1]:
                xi[i] = bounds[i][1]
            elif xi[i] < bounds[i][0]:
                xi[i] = bounds[i][0]
        simplex.append(xi)
    
    # Evaluate
    fvals = [objective(x) for x in simplex]
    
    for _ in range(max_iter):
        # Sort by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        xr = centroid + alpha * (centroid - simplex[-1])
        if bounds:
            for i in range(n):
                xr[i] = np.clip(xr[i], bounds[i][0], bounds[i][1])
        fr = objective(xr)
        
        if fr < fvals[0]:
            # Expansion
            gamma = 2.0
            xe = centroid + gamma * (centroid - simplex[-1])
            if bounds:
                for i in range(n):
                    xe[i] = np.clip(xe[i], bounds[i][0], bounds[i][1])
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
            # Contraction
            rho = 0.5
            xc = centroid + rho * (simplex[-1] - centroid)
            if bounds:
                for i in range(n):
                    xc[i] = np.clip(xc[i], bounds[i][0], bounds[i][1])
            fc = objective(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    if bounds:
                        for j in range(n):
                            simplex[i][j] = np.clip(simplex[i][j], bounds[j][0], bounds[j][1])
                    fvals[i] = objective(simplex[i])
    
    # Return best
    order = np.argsort(fvals)
    return simplex[order[0]], fvals[order[0]]

def design(problem):
    """Design an antireflection coating stack using multi-start Nelder-Mead."""
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    materials = problem["materials"]
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    n_L = materials['L']
    n_H = materials['H']
    
    # Candidate label sequences
    # We focus on symmetric and alternating patterns which are known to be effective
    candidate_labels = [
        ['L'],
        ['H'],
        ['L', 'H'],
        ['H', 'L'],
        ['L', 'H', 'L'],
        ['H', 'L', 'H'],
        ['L', 'H', 'L', 'H'],
        ['H', 'L', 'H', 'L'],
        ['L', 'H', 'L', 'H', 'L'],
        ['H', 'L', 'H', 'L', 'H'],
        ['L', 'H', 'L', 'H', 'L', 'H'],
        ['H', 'L', 'H', 'L', 'H', 'L'],
        ['L', 'H', 'L', 'H', 'L', 'H', 'L'],
        ['H', 'L', 'H', 'L', 'H', 'L', 'H'],
    ]
    
    # Filter candidates by max_layers
    candidate_labels = [lab for lab in candidate_labels if len(lab) <= max_layers]
    
    best_score = float('inf')
    best_labels = ['L']
    best_thicknesses = [600.0 / (4.0 * n_L)]
    
    # Initial evaluation for fallback
    fallback_score = _compute_reflectance(best_labels, best_thicknesses, materials, ambient_index, substrate_index, band_nm)
    if fallback_score < best_score:
        best_score = fallback_score
    
    for labels in candidate_labels:
        if time.time() > deadline:
            break
            
        n_layers = len(labels)
        bounds = [(0.1, max_layer_nm) for _ in range(n_layers)]
        
        # Define objective function
        def objective(thicknesses):
            if any(t < 0.1 or t > max_layer_nm for t in thicknesses):
                return 1000.0
            if sum(thicknesses) > max_total_nm:
                return 1000.0
            return _compute_reflectance(labels, thicknesses, materials, ambient_index, substrate_index, band_nm)
        
        # Multi-start
        # We try a few starting points
        # 1. Quarter wave at 600nm
        # 2. Quarter wave at 500nm
        # 3. Random
        starts = []
        
        # Start 1: 600nm QW
        t_600 = []
        for lab in labels:
            n = n_L if lab == 'L' else n_H
            t_600.append(600.0 / (4.0 * n))
        starts.append(t_600)
        
        # Start 2: 500nm QW
        t_500 = []
        for lab in labels:
            n = n_L if lab == 'L' else n_H
            t_500.append(500.0 / (4.0 * n))
        starts.append(t_500)
        
        # Start 3: Random
        t_rand = []
        for lab in labels:
            t_rand.append(random.uniform(50, max_layer_nm))
        starts.append(t_rand)
        
        for x0 in starts:
            if time.time() > deadline:
                break
            
            # Check if start is valid
            if sum(x0) > max_total_nm:
                # Scale down
                scale = max_total_nm / sum(x0)
                x0 = [t * scale for t in x0]
            
            try:
                x_opt, f_opt = _nelder_mead(objective, x0, step=20.0, max_iter=50, tol=1e-4, bounds=bounds)
                
                # Validate result
                if f_opt < best_score:
                    # Ensure constraints are met
                    if sum(x_opt) <= max_total_nm + 1e-6 and all(0.1 <= t <= max_layer_nm for t in x_opt):
                        best_score = f_opt
                        best_labels = labels[:]
                        best_thicknesses = x_opt.tolist()
            except Exception:
                pass
    
    # Final validation
    if sum(best_thicknesses) > max_total_nm:
        best_labels = ['L']
        best_thicknesses = [600.0 / (4.0 * n_L)]
        best_score = _compute_reflectance(best_labels, best_thicknesses, materials, ambient_index, substrate_index, band_nm)
    
    if any(t < 0 or t > max_layer_nm for t in best_thicknesses):
        best_labels = ['L']
        best_thicknesses = [600.0 / (4.0 * n_L)]
        best_score = _compute_reflectance(best_labels, best_thicknesses, materials, ambient_index, substrate_index, band_nm)
        
    return best_labels, best_thicknesses

# EVOLVE-BLOCK-END
