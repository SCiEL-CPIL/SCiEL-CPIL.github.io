"""
Broadband antireflection coating design via discrete Genetic Algorithm + Nelder-Mead polish.

Strategy:
- Use a Genetic Algorithm to search over discrete layer sequences (L/H) and discrete thickness indices.
- The discrete thicknesses are chosen from a set of physically motivated values (quarter-wave multiples).
- After GA identifies a promising discrete structure, refine the thicknesses continuously using Nelder-Mead.
- This hybrid approach leverages global combinatorial search and local continuous optimization.
"""
import numpy as np
import time

def _compute_reflectance(wavelengths_nm, indices, thicknesses_nm, ambient_idx, substrate_idx):
    """
    Compute reflectance (fraction, 0-1) at each wavelength using the characteristic matrix method.
    """
    n_layers = len(indices)
    if n_layers == 0:
        r0 = (ambient_idx - substrate_idx) / (ambient_idx + substrate_idx)
        return np.full(len(wavelengths_nm), r0 * r0)
    
    k = 2.0 * np.pi / wavelengths_nm
    
    M00 = np.ones(len(wavelengths_nm), dtype=complex)
    M01 = np.zeros(len(wavelengths_nm), dtype=complex)
    M10 = np.zeros(len(wavelengths_nm), dtype=complex)
    M11 = np.ones(len(wavelengths_nm), dtype=complex)
    
    for i in range(n_layers):
        n_i = indices[i]
        d_i = thicknesses_nm[i]
        delta = k * n_i * d_i
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        B00 = cos_d
        B01 = 1j * sin_d / n_i
        B10 = 1j * n_i * sin_d
        B11 = cos_d
        
        new_M00 = M00 * B00 + M01 * B10
        new_M01 = M00 * B01 + M01 * B11
        new_M10 = M10 * B00 + M11 * B10
        new_M11 = M10 * B01 + M11 * B11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    Y_out = substrate_idx
    Y_in = (M10 + M11 * Y_out) / (M00 + M01 * Y_out)
    
    r = (ambient_idx - Y_in) / (ambient_idx + Y_in)
    R = np.abs(r) ** 2
    
    return R

def _decode_and_evaluate(seq, thick_idx, candidates, materials, wavelengths_nm, ambient_idx, substrate_idx, max_total_nm, max_layer_nm):
    """
    Decode discrete representation and evaluate.
    seq: array of 0 (L) or 1 (H)
    thick_idx: array of indices into candidates
    Returns: (score, labels, thicknesses)
    """
    n_layers = len(seq)
    labels = []
    thicknesses = []
    total_thick = 0.0
    
    for i in range(n_layers):
        t = candidates[thick_idx[i]]
        if t > max_layer_nm:
            t = max_layer_nm
        if t > 0: # Only include non-zero layers in the list? 
            # The problem allows 0 thickness, but it's cleaner to omit them for the output list
            # However, the evaluator expects lists of labels and thicknesses.
            # If we omit, the count changes.
            # Let's keep all layers in the internal representation for optimization,
            # but for the final output, we can filter out zero-thickness layers if needed,
            # or just return them. The constraint is on the sum and individual values.
            # Returning 0 thickness is valid.
            pass
        
        mat_label = 'L' if seq[i] == 0 else 'H'
        labels.append(mat_label)
        thicknesses.append(t)
        total_thick += t
    
    if total_thick > max_total_nm:
        # Penalty or scaling? Scaling is better for feasibility.
        # But for GA fitness, we want to penalize or handle carefully.
        # Let's scale down to fit, then evaluate.
        scale = max_total_nm / total_thick
        thicknesses = [t * scale for t in thicknesses]
        # Note: scaling might make some layers very thin, but valid.
    
    indices = [materials[label] for label in labels]
    R = _compute_reflectance(wavelengths_nm, indices, thicknesses, ambient_idx, substrate_idx)
    score = np.mean(R) * 100.0
    
    return score, labels, thicknesses

def _nelder_mead(f, x0, bounds, max_iter=2000, tol=1e-10, time_deadline=None):
    """
    Nelder-Mead simplex with bounds.
    """
    n = len(x0)
    lo_bounds = np.array([b[0] for b in bounds])
    hi_bounds = np.array([b[1] for b in bounds])
    x0 = np.clip(np.asarray(x0, dtype=float), lo_bounds, hi_bounds)
    
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    
    for i in range(n):
        simplex[i + 1] = x0.copy()
        lo, hi = bounds[i]
        step = 0.05 * (hi - lo) if hi > lo else 0.05
        simplex[i + 1][i] += step
    
    fvals = np.array([f(simplex[i]) for i in range(n + 1)])
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for it in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break
        
        idx = np.argsort(fvals)
        simplex = simplex[idx]
        fvals = fvals[idx]
        
        if np.max(fvals) - np.min(fvals) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, lo_bounds, hi_bounds)
        f_r = f(x_r)
        
        if f_r < fvals[0]:
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, lo_bounds, hi_bounds)
            f_e = f(x_e)
            
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
        else:
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c = np.clip(x_c, lo_bounds, hi_bounds)
            f_c = f(x_c)
            
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], lo_bounds, hi_bounds)
                    fvals[i] = f(simplex[i])
    
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def _ga_search(pop_size, max_generations, max_layers, candidates, materials, wavelengths_nm, ambient_idx, substrate_idx, max_total_nm, max_layer_nm, time_deadline, rng):
    """
    Genetic Algorithm for discrete layer sequence and thickness indices.
    """
    n_cand = len(candidates)
    
    # Initialize population
    # seq: (pop_size, max_layers) of 0/1
    # thick_idx: (pop_size, max_layers) of 0..n_cand-1
    pop_seq = rng.randint(0, 2, size=(pop_size, max_layers))
    pop_thick = rng.randint(0, n_cand, size=(pop_size, max_layers))
    
    # Evaluate initial population
    scores = np.zeros(pop_size)
    best_labels = None
    best_thicknesses = None
    best_score = np.inf
    
    for i in range(pop_size):
        score, labels, thicknesses = _decode_and_evaluate(
            pop_seq[i], pop_thick[i], candidates, materials, wavelengths_nm, 
            ambient_idx, substrate_idx, max_total_nm, max_layer_nm
        )
        scores[i] = score
        if score < best_score:
            best_score = score
            best_labels = labels
            best_thicknesses = thicknesses
    
    # GA Loop
    for gen in range(max_generations):
        if time.time() > time_deadline:
            break
        
        # Selection: Tournament
        new_pop_seq = np.zeros_like(pop_seq)
        new_pop_thick = np.zeros_like(pop_thick)
        
        for i in range(pop_size):
            # Pick 3 random parents
            parents = rng.choice(pop_size, size=3, replace=False)
            parent_scores = scores[parents]
            winner_idx = parents[np.argmin(parent_scores)]
            
            # Crossover: Uniform
            mask = rng.random(max_layers) > 0.5
            
            # Parent 1 and 2
            p1_idx = rng.choice(pop_size)
            p2_idx = rng.choice(pop_size)
            
            # For seq
            new_pop_seq[i, mask] = pop_seq[p1_idx, mask]
            new_pop_seq[i, ~mask] = pop_seq[p2_idx, ~mask]
            
            # For thick
            new_pop_thick[i, mask] = pop_thick[p1_idx, mask]
            new_pop_thick[i, ~mask] = pop_thick[p2_idx, ~mask]
            
            # Mutation
            # Seq mutation: flip bits
            mut_mask_seq = rng.random(max_layers) < 0.1
            new_pop_seq[i, mut_mask_seq] = 1 - new_pop_seq[i, mut_mask_seq]
            
            # Thick mutation: change index
            mut_mask_thick = rng.random(max_layers) < 0.1
            if np.any(mut_mask_thick):
                new_pop_thick[i, mut_mask_thick] = rng.randint(0, n_cand, size=np.sum(mut_mask_thick))
            
            # Elitism: Copy best from previous gen to new gen slot 0
            if i == 0:
                best_idx_prev = np.argmin(scores)
                new_pop_seq[i] = pop_seq[best_idx_prev]
                new_pop_thick[i] = pop_thick[best_idx_prev]
        
        # Evaluate new population
        new_scores = np.zeros(pop_size)
        for i in range(pop_size):
            if time.time() > time_deadline:
                break
            score, labels, thicknesses = _decode_and_evaluate(
                new_pop_seq[i], new_pop_thick[i], candidates, materials, wavelengths_nm, 
                ambient_idx, substrate_idx, max_total_nm, max_layer_nm
            )
            new_scores[i] = score
            if score < best_score:
                best_score = score
                best_labels = labels
                best_thicknesses = thicknesses
        
        # Replace population
        pop_seq = new_pop_seq
        pop_thick = new_pop_thick
        scores = new_scores
    
    return best_score, best_labels, best_thicknesses

def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (labels, thicknesses)
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    # Define discrete thickness candidates
    # Include 0 to allow "removing" a layer
    # Include QWOTs and variations
    # L: 1.38, H: 2.35
    # QWOT L: 600/(4*1.38) ~ 109
    # QWOT H: 600/(4*2.35) ~ 64
    candidates = np.unique(np.concatenate([
        [0],
        np.arange(20, 401, 20),
        [109, 64, 50, 150, 200, 250, 300]
    ])).astype(float)
    candidates = candidates[candidates <= max_layer_nm]
    
    rng = np.random.RandomState(42)
    
    # Run GA
    pop_size = 60
    max_gens = 150
    
    best_score, best_labels, best_thicknesses = _ga_search(
        pop_size, max_gens, max_layers, candidates, materials, wavelengths_nm, 
        ambient_idx, substrate_idx, max_total_nm, max_layer_nm, deadline, rng
    )
    
    # If GA didn't find anything (shouldn't happen), fallback
    if best_labels is None:
        best_labels = ['L']
        best_thicknesses = [center_wl / (4.0 * materials['L'])]
        best_score = 100.0
    
    # Local Refinement with Nelder-Mead
    # We need to optimize the continuous thicknesses for the fixed sequence
    n_opt = len(best_labels)
    if n_opt > 0:
        indices = [materials[label] for label in best_labels]
        
        # Initial guess is the discrete thicknesses
        x0 = np.array(best_thicknesses, dtype=float)
        
        # Bounds for NM
        bounds = [(0.0, float(max_layer_nm))] * n_opt
        
        def make_f(indices):
            def f(x):
                thicknesses = np.asarray(x, dtype=float)
                if np.any(thicknesses < -1e-6) or np.any(thicknesses > max_layer_nm + 1e-6):
                    return 100.0
                if np.sum(thicknesses) > max_total_nm + 1e-6:
                    return 100.0
                
                thicknesses_clipped = np.clip(thicknesses, 0.0, max_layer_nm)
                if np.sum(thicknesses_clipped) > max_total_nm:
                    return 100.0
                
                R = _compute_reflectance(wavelengths_nm, indices, thicknesses_clipped, 
                                         ambient_idx, substrate_idx)
                return np.mean(R) * 100.0
            return f
        
        f = make_f(indices)
        
        # Run NM
        nm_deadline = min(time.time() + 20.0, deadline)
        try:
            x_nm, f_nm = _nelder_mead(f, x0, bounds, max_iter=1000, tol=1e-10, time_deadline=nm_deadline)
            if f_nm < best_score:
                best_score = f_nm
                best_thicknesses = x_nm.tolist()
        except Exception:
            pass
    
    # Final constraint check and cleanup
    thicknesses = np.array(best_thicknesses)
    thicknesses = np.clip(thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    # Ensure lists are of correct type
    return best_labels, thicknesses.tolist()
