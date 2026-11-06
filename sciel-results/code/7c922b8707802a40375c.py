"""Rewritten program for broadband antireflection coating design."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def _fast_compute_mean_reflectance(labels, thicknesses, n_amb, n_sub, n_L, n_H, band_nm):
    """Compute mean reflectance in percent over 401 equally spaced wavelengths in band_nm."""
    lam_min, lam_max = band_nm
    wavelengths = np.linspace(lam_min, lam_max, 401)
    
    n_mat = np.array([n_L if l == 'L' else n_H for l in labels], dtype=float)
    d = np.array(thicknesses, dtype=float)
    
    if len(labels) == 0:
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return np.abs(r) ** 2 * 100.0
    
    # Pre-compute 2*pi*n*d for each layer
    beta = 2.0 * np.pi * n_mat * d  # beta[i] = 2*pi*n_mat[i]*d[i]
    
    total_r = 0.0
    n_wl = len(wavelengths)
    chunk = 40
    for start in range(0, n_wl, chunk):
        end = min(start + chunk, n_wl)
        lams = wavelengths[start:end]
        nl = len(lams)
        
        # M as (nl, 2, 2) complex array, initialized to identity
        M00 = np.ones(nl, dtype=complex)
        M01 = np.zeros(nl, dtype=complex)
        M10 = np.zeros(nl, dtype=complex)
        M11 = np.ones(nl, dtype=complex)
        
        for i in range(len(labels)):
            delta = beta[i] / lams
            c = np.cos(delta)
            s = np.sin(delta)
            
            a = c
            b = 1j * s / n_mat[i]
            cc = 1j * n_mat[i] * s
            dd = c
            
            new_M00 = M00 * a + M01 * cc
            new_M01 = M00 * b + M01 * dd
            new_M10 = M10 * a + M11 * cc
            new_M11 = M10 * b + M11 * dd
            M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
        
        Y = M10 / M00
        r = (n_amb - Y) / (n_amb + Y)
        R = np.abs(r) ** 2 * 100.0
        total_r += np.sum(R)
    
    return total_r / n_wl


def design(problem):
    """Design a broadband antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin for safety
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    best_labels = None
    best_thicknesses = None
    best_score = float('inf')
    
    centre = 0.5 * (band_nm[0] + band_nm[1])
    
    def _obj(labels, x):
        """Objective with soft penalty for total thickness violation."""
        x = np.clip(x, 0.0, max_layer_nm)
        penalty = 0.0
        total = np.sum(x)
        if total > max_total_nm:
            penalty = 10.0 * (total - max_total_nm) / max_total_nm
        val = _fast_compute_mean_reflectance(labels, list(x), n_amb, n_sub, n_L, n_H, band_nm)
        return val + penalty
    
    def _is_feasible(x):
        return all(0.0 <= t <= max_layer_nm for t in x) and sum(x) <= max_total_nm + 1e-9
    
    # Evaluate a few initial configurations
    initial_configs = [
        (["L"], [centre / (4.0 * n_L)]),
        (["L", "H"], [centre / (4.0 * n_L), centre / (4.0 * n_H)]),
        (["H", "L"], [centre / (4.0 * n_H), centre / (4.0 * n_L)]),
        (["L", "H", "L"], [centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L)]),
        (["H", "L", "H"], [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H)]),
        (["L", "H", "L", "H"], [centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H)]),
        (["H", "L", "H", "L"], [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L)]),
        (["L", "H", "L", "H", "L"], [centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L)]),
    ]
    
    for labels, thicknesses in initial_configs:
        if len(labels) > max_layers:
            continue
        if sum(thicknesses) > max_total_nm:
            continue
        if any(t < 0 or t > max_layer_nm for t in thicknesses):
            continue
        score = _fast_compute_mean_reflectance(labels, thicknesses, n_amb, n_sub, n_L, n_H, band_nm)
        if score < best_score:
            best_score = score
            best_labels = labels[:]
            best_thicknesses = thicknesses[:]
    
    import random
    random.seed(42)
    
    # Build candidate sequences: alternating + some structured non-alternating
    candidate_sequences = []
    seen = set()
    
    def _add_seq(seq):
        key = tuple(seq)
        if key not in seen and len(seq) <= max_layers:
            seen.add(key)
            candidate_sequences.append(list(seq))
    
    # Alternating sequences of various lengths
    for n_layers in range(1, min(max_layers + 1, 13)):
        seq_L = ["L" if i % 2 == 0 else "H" for i in range(n_layers)]
        seq_H = ["H" if i % 2 == 0 else "L" for i in range(n_layers)]
        _add_seq(seq_L)
        _add_seq(seq_H)
    
    # Some structured non-alternating patterns (e.g., LHL, HHL, etc.)
    extra_patterns = [
        ["L", "H", "L", "L", "H"],
        ["H", "L", "H", "H", "L"],
        ["L", "L", "H", "H", "L"],
        ["H", "H", "L", "L", "H"],
        ["L", "H", "H", "L"],
        ["H", "L", "L", "H"],
        ["L", "L", "H"],
        ["H", "H", "L"],
        ["L", "H", "L", "L"],
        ["H", "L", "H", "H"],
        ["L", "H", "H", "L", "H"],
        ["H", "L", "L", "H", "L"],
        ["L", "H", "L", "H", "L", "H"],
        ["H", "L", "H", "L", "H", "L"],
        ["L", "H", "L", "H", "L", "H", "L"],
        ["H", "L", "H", "L", "H", "L", "H"],
    ]
    for p in extra_patterns:
        _add_seq(p)
    
    def optimize_thicknesses(labels, max_iter=200):
        """Optimize thicknesses for a given label sequence using Nelder-Mead with soft constraints."""
        n = len(labels)
        if n == 0:
            return None, float('inf')
        
        # Initial guess: quarter-wave at center wavelength
        x0 = np.array([centre / (4.0 * (n_L if l == 'L' else n_H)) for l in labels])
        x0 = np.clip(x0, 0.0, max_layer_nm)
        
        # If initial exceeds total, scale down
        if np.sum(x0) > max_total_nm:
            x0 = x0 * (max_total_nm / np.sum(x0))
            x0 = np.clip(x0, 0.0, max_layer_nm)
        
        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initial simplex: perturb each dimension
        simplex = [x0.copy()]
        for i in range(n):
            perturbed = x0.copy()
            step = 0.15 * max_layer_nm
            perturbed[i] = np.clip(perturbed[i] + step, 0.0, max_layer_nm)
            simplex.append(perturbed)
        
        # Evaluate initial simplex
        fvals = [_obj(labels, x) for x in simplex]
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Sort by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            if abs(fvals[-1] - fvals[0]) < 1e-7:
                break
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, 0.0, max_layer_nm)
            fr = _obj(labels, xr)
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, 0.0, max_layer_nm)
                fe = _obj(labels, xe)
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
                if fr < fvals[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (simplex[-1] - centroid)
                xc = np.clip(xc, 0.0, max_layer_nm)
                fc = _obj(labels, xc)
                
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink all points toward best
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], 0.0, max_layer_nm)
                        fvals[i] = _obj(labels, simplex[i])
        
        # Return best feasible point
        best_idx = np.argmin(fvals)
        best_x = simplex[best_idx]
        best_f = fvals[best_idx]
        
        # Only return if feasible
        if _is_feasible(best_x):
            actual_score = _fast_compute_mean_reflectance(labels, list(best_x), n_amb, n_sub, n_L, n_H, band_nm)
            return list(best_x), actual_score
        return None, float('inf')
    
    # Optimize for each candidate sequence
    for labels in candidate_sequences:
        if time.time() > deadline - 8.0:
            break
        
        if len(labels) > max_layers:
            continue
        
        # Quick feasibility check on initial quarter-wave
        init_thick = [centre / (4.0 * (n_L if l == 'L' else n_H)) for l in labels]
        if sum(init_thick) > max_total_nm * 1.5:
            continue
        
        opt_thick, opt_score = optimize_thicknesses(labels, max_iter=180)
        
        if opt_thick is not None and opt_score < best_score:
            if _is_feasible(opt_thick):
                best_score = opt_score
                best_labels = labels[:]
                best_thicknesses = opt_thick[:]
    
    # Random restarts with different sequences
    if time.time() < deadline - 12.0:
        for trial in range(80):
            if time.time() > deadline - 8.0:
                break
            
            n_layers = random.randint(2, min(10, max_layers))
            labels = [random.choice(['L', 'H']) for _ in range(n_layers)]
            
            key = tuple(labels)
            if key in seen:
                continue
            seen.add(key)
            
            init_thick = [centre / (4.0 * (n_L if l == 'L' else n_H)) for l in labels]
            if sum(init_thick) > max_total_nm * 1.5:
                continue
            
            opt_thick, opt_score = optimize_thicknesses(labels, max_iter=120)
            
            if opt_thick is not None and opt_score < best_score:
                if _is_feasible(opt_thick):
                    best_score = opt_score
                    best_labels = labels[:]
                    best_thicknesses = opt_thick[:]
    
    # If we have more time, do a local refinement on the best solution
    if time.time() < deadline - 10.0 and best_labels is not None:
        refined, refined_score = optimize_thicknesses(best_labels, max_iter=200)
        if refined is not None and refined_score < best_score:
            if _is_feasible(refined):
                best_score = refined_score
                best_thicknesses = refined[:]
    
    # Return the best result found
    if best_labels is None:
        centre_val = 0.5 * (band_nm[0] + band_nm[1])
        best_labels = ["L"]
        best_thicknesses = [centre_val / (4.0 * n_L)]
    
    # Final constraint enforcement
    best_thicknesses = [max(0.0, min(t, max_layer_nm)) for t in best_thicknesses]
    if sum(best_thicknesses) > max_total_nm:
        scale = max_total_nm / sum(best_thicknesses)
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    return best_labels, best_thicknesses
# EVOLVE-BLOCK-END
