"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _reflectance_vectorized(materials, thicknesses, wls, n_amb, n_sub, mat_dict):
    """
    Vectorized reflectance calculation for a stack of layers.
    Returns an array of reflectance values (fraction, not percent) for the given wavelengths.
    """
    n_layers = len(materials)
    if n_layers == 0:
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return np.full_like(wls, r * r)
    
    # Precompute n and d for each layer
    ns = np.array([mat_dict[m] for m in materials])
    ds = np.array(thicknesses)
    
    # Initialize total matrix as identity for all wavelengths
    # Shape: (n_wls, 2, 2)
    M_total = np.zeros((len(wls), 2, 2), dtype=complex)
    M_total[:, 0, 0] = 1.0
    M_total[:, 1, 1] = 1.0
    
    for i in range(n_layers):
        n = ns[i]
        d = ds[i]
        if d < 1e-9:
            continue
        beta = 2.0 * np.pi * n * d / wls
        c = np.cos(beta)
        s = np.sin(beta)
        
        # Characteristic matrix for layer i:
        # [[cos(beta), i*sin(beta)/n], [i*n*sin(beta), cos(beta)]]
        # M_layer shape: (n_wls, 2, 2)
        M_layer = np.zeros((len(wls), 2, 2), dtype=complex)
        M_layer[:, 0, 0] = c
        M_layer[:, 0, 1] = 1j * s / n
        M_layer[:, 1, 0] = 1j * n * s
        M_layer[:, 1, 1] = c
        
        # M_total = M_total @ M_layer
        # Using einsum for batched matrix multiplication
        # (n_wls, 2, 2) @ (n_wls, 2, 2) -> (n_wls, 2, 2)
        M_total = np.einsum('wij,wjk->wik', M_total, M_layer)
        
    # Calculate input admittance Y
    # Y = (M[1,0] + M[1,1]*n_sub) / (M[0,0] + M[0,1]*n_sub)
    Y = (M_total[:, 1, 0] + M_total[:, 1, 1] * n_sub) / (M_total[:, 0, 0] + M_total[:, 0, 1] * n_sub)
    
    # Reflectance
    r = (n_amb - Y) / (n_amb + Y)
    return np.abs(r) ** 2

def _mean_reflectance(materials, thicknesses, wls, n_amb, n_sub, mat_dict):
    """Calculate mean reflectance in percent."""
    refls = _reflectance_vectorized(materials, thicknesses, wls, n_amb, n_sub, mat_dict)
    return 100.0 * np.mean(refls)

def _is_valid(materials, thicknesses, problem):
    """Check constraints."""
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False
    if sum(thicknesses) > max_total_nm:
        return False
    return True

def design(problem):
    """
    Design a broadband antireflection coating.
    Strategy:
    1. Use Differential Evolution (DE) to globally optimize a fixed-length parameter vector.
    2. The vector encodes material indices (0=L, 1=H) and thicknesses.
    3. DE is robust to multimodality and handles discrete/continuous mixed spaces well via rounding.
    4. Vectorized physics allows fast population evaluation.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    mat_dict = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wls = np.linspace(band_nm[0], band_nm[1], 401)
    
    # We will optimize a stack of exactly max_layers. 
    # Unused layers can have thickness 0.
    # Parameter vector: [mat_0, th_0, mat_1, th_1, ..., mat_N-1, th_N-1]
    # mat_i is continuous [0, 1], rounded to 0 or 1.
    # th_i is continuous [0, max_layer_nm].
    
    N = max_layers
    dim = 2 * N
    
    # Initial best: try a few standard designs
    best_materials = []
    best_thicknesses = []
    best_score = 100.0 # Start high
    
    def try_candidate(materials, thicknesses):
        nonlocal best_materials, best_thicknesses, best_score
        if not _is_valid(materials, thicknesses, problem):
            return
        # Trim trailing zero-thickness layers for cleaner output (optional, but keeps len <= 20)
        # Actually, the evaluator accepts up to max_layers. We can return the full list with zeros.
        # But it's cleaner to trim.
        mats_clean = []
        ths_clean = []
        for m, t in zip(materials, thicknesses):
            if t > 1e-6:
                mats_clean.append(m)
                ths_clean.append(t)
        
        if not mats_clean:
            mats_clean = ['L']
            ths_clean = [0.0] # Avoid empty? Or just return empty? 
            # Contract says "list of ... labels". Empty list might be valid?
            # "minimizes ... bare glass: 4.26%". 
            # If we return empty, score is 4.26. 
            # Let's just return the clean lists. If empty, it's valid.
            
        score = _mean_reflectance(mats_clean, ths_clean, wls, n_amb, n_sub, mat_dict)
        if score < best_score:
            best_score = score
            best_materials = mats_clean[:]
            best_thicknesses = ths_clean[:]

    # Seed with quarter-wave stack of 10 layers (5L5H alternating)
    # Center wavelength 600nm
    wl_ref = 600.0
    seed_mats = []
    seed_ths = []
    for i in range(10):
        mat = 'L' if i % 2 == 0 else 'H'
        seed_mats.append(mat)
        seed_ths.append(wl_ref / (4.0 * mat_dict[mat]))
    try_candidate(seed_mats, seed_ths)

    # Differential Evolution Setup
    # Bounds:
    # mat_i: [0, 1]
    # th_i: [0, max_layer_nm]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # Material
        bounds.append((0.0, float(max_layer_nm))) # Thickness

    pop_size = 30
    # Initial population
    population = np.random.uniform(
        [b[0] for b in bounds], 
        [b[1] for b in bounds], 
        size=(pop_size, dim)
    )
    
    # Evaluate initial population
    fitnesses = np.zeros(pop_size)
    for i in range(pop_size):
        vec = population[i]
        mats = []
        ths = []
        for j in range(N):
            mat_val = vec[2*j]
            th_val = vec[2*j+1]
            # Round material to 0 or 1
            mat_idx = 0 if mat_val < 0.5 else 1
            mat_char = 'L' if mat_idx == 0 else 'H'
            if th_val > 1e-6:
                mats.append(mat_char)
                ths.append(th_val)
        
        if not mats:
            mats = ['L']
            ths = [0.0]
            
        if _is_valid(mats, ths, problem):
            fitnesses[i] = _mean_reflectance(mats, ths, wls, n_amb, n_sub, mat_dict)
        else:
            fitnesses[i] = 100.0 # Penalty

    # Update best
    min_idx = np.argmin(fitnesses)
    if fitnesses[min_idx] < best_score:
        vec = population[min_idx]
        mats = []
        ths = []
        for j in range(N):
            mat_val = vec[2*j]
            th_val = vec[2*j+1]
            mat_idx = 0 if mat_val < 0.5 else 1
            mat_char = 'L' if mat_idx == 0 else 'H'
            if th_val > 1e-6:
                mats.append(mat_char)
                ths.append(th_val)
        if mats:
            try_candidate(mats, ths)

    # DE Loop
    F = 0.8 # Scaling factor
    CR = 0.9 # Crossover rate
    
    while time.time() - start_time < deadline:
        # Generate trial population
        for i in range(pop_size):
            if time.time() - start_time > deadline:
                break
                
            # Select three distinct individuals
            idxs = np.random.choice(pop_size, 3, replace=False)
            r1, r2, r3 = idxs
            
            # Don't use i itself if possible, but DE allows it in some variants.
            # Standard DE: r1, r2, r3 != i
            while i in idxs:
                idxs = np.random.choice(pop_size, 3, replace=False)
                r1, r2, r3 = idxs
            
            # Mutation: v = x_r1 + F * (x_r2 - x_r3)
            v = population[r1] + F * (population[r2] - population[r3])
            
            # Crossover
            trial = np.zeros_like(v)
            j_rand = np.random.randint(0, dim)
            for j in range(dim):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = population[i][j]
            
            # Bounds repair
            for j in range(dim):
                if trial[j] < bounds[j][0]:
                    trial[j] = bounds[j][0]
                elif trial[j] > bounds[j][1]:
                    trial[j] = bounds[j][1]
            
            # Evaluate trial
            mats = []
            ths = []
            for j in range(N):
                mat_val = trial[2*j]
                th_val = trial[2*j+1]
                mat_idx = 0 if mat_val < 0.5 else 1
                mat_char = 'L' if mat_idx == 0 else 'H'
                if th_val > 1e-6:
                    mats.append(mat_char)
                    ths.append(th_val)
            
            if not mats:
                mats = ['L']
                ths = [0.0]
                
            if _is_valid(mats, ths, problem):
                trial_fit = _mean_reflectance(mats, ths, wls, n_amb, n_sub, mat_dict)
            else:
                trial_fit = 100.0
            
            # Selection
            if trial_fit <= fitnesses[i]:
                population[i] = trial
                fitnesses[i] = trial_fit
                
                if trial_fit < best_score:
                    best_score = trial_fit
                    best_materials = mats[:]
                    best_thicknesses = ths[:]

    # Final check: if best_materials is empty or bad, return a safe default
    if not best_materials:
        return ['L'], [600.0 / (4.0 * mat_dict['L'])]
        
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
