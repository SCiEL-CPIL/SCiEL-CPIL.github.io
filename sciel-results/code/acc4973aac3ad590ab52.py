"""Rewritten program for broadband antireflection coating design."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def _compute_mean_reflectance(labels, thicknesses, n_amb, n_sub, n_L, n_H, band_nm):
    """Compute mean reflectance in percent over 401 equally spaced wavelengths in band_nm."""
    lam_min, lam_max = band_nm
    wavelengths = np.linspace(lam_min, lam_max, 401)
    
    n_mat = np.array([n_L if l == 'L' else n_H for l in labels], dtype=float)
    d = np.array(thicknesses, dtype=float)
    
    total_r = 0.0
    for lam in wavelengths:
        # Build characteristic matrix
        # M = product of layer matrices from first layer (air side) to last (substrate side)
        # Each layer matrix: [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
        # where delta = 2*pi*n*d/lam
        # For normal incidence, s and p are same
        
        # Start with identity
        M = np.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j]])
        
        for i in range(len(labels)):
            delta = 2.0 * np.pi * n_mat[i] * d[i] / lam
            c = np.cos(delta)
            s = np.sin(delta)
            # Layer matrix
            Mi = np.array([[c, 1j * s / n_mat[i]], [1j * n_mat[i] * s, c]])
            M = M @ Mi
        
        # Admittance: Y = M[1,0] / M[0,0]
        Y = M[1, 0] / M[0, 0]
        
        # Reflectance
        # r = (Y_amb - Y) / (Y_amb + Y) where Y_amb is ambient index (for normal incidence)
        # Actually, the characteristic matrix method:
        # Y_sub = n_sub (for the substrate)
        # The total matrix M maps from air to substrate
        # Y_total = M[1,0]/M[0,0]
        # But we need to account for the ambient and substrate properly.
        
        # Standard formulation:
        # Let M be the product of all layer matrices.
        # Then the effective admittance at the air interface is Y = M[1,0]/M[0,0]
        # The reflectance is |r|^2 where r = (n_amb - Y)/(n_amb + Y)
        # Wait, let me be more careful.
        
        # The characteristic matrix for a stack is:
        # [E]   = M [E_sub]
        # [H]       [H_sub]
        # where M = M1 * M2 * ... * Mm
        # For normal incidence, H = E/n, so Y = H/E = 1/n
        # The admittance at the input plane: Y_in = M[1,0]/M[0,0]
        # The reflection coefficient: r = (Y_amb - Y_in) / (Y_amb + Y_in)
        # where Y_amb = n_amb
        
        r = (n_amb - Y) / (n_amb + Y)
        R = np.abs(r) ** 2
        total_r += R * 100.0  # in percent
    
    return total_r / len(wavelengths)


def _fast_compute_mean_reflectance(labels, thicknesses, n_amb, n_sub, n_L, n_H, band_nm):
    """Faster version using vectorization over wavelengths."""
    lam_min, lam_max = band_nm
    wavelengths = np.linspace(lam_min, lam_max, 401)
    
    n_mat = np.array([n_L if l == 'L' else n_H for l in labels], dtype=float)
    d = np.array(thicknesses, dtype=float)
    
    if len(labels) == 0:
        # Bare substrate
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return np.abs(r) ** 2 * 100.0
    
    # Compute delta for all layers and wavelengths
    # delta[i, j] = 2*pi*n_mat[i]*d[i]/wavelengths[j]
    # Use matrix multiplication approach for characteristic matrices
    
    total_r = 0.0
    # Process in chunks to avoid memory issues
    for start in range(0, len(wavelengths), 50):
        end = min(start + 50, len(wavelengths))
        lams = wavelengths[start:end]
        
        # Initialize M as identity for each wavelength
        M = np.zeros((len(lams), 2, 2), dtype=complex)
        M[:, 0, 0] = 1.0
        M[:, 1, 1] = 1.0
        
        for i in range(len(labels)):
            delta = 2.0 * np.pi * n_mat[i] * d[i] / lams
            c = np.cos(delta)
            s = np.sin(delta)
            
            # Mi for each wavelength
            Mi_00 = c
            Mi_01 = 1j * s / n_mat[i]
            Mi_10 = 1j * n_mat[i] * s
            Mi_11 = c
            
            # M_new = M @ Mi
            M_new = np.zeros_like(M)
            M_new[:, 0, 0] = M[:, 0, 0] * Mi_00 + M[:, 0, 1] * Mi_10
            M_new[:, 0, 1] = M[:, 0, 0] * Mi_01 + M[:, 0, 1] * Mi_11
            M_new[:, 1, 0] = M[:, 1, 0] * Mi_00 + M[:, 1, 1] * Mi_10
            M_new[:, 1, 1] = M[:, 1, 0] * Mi_01 + M[:, 1, 1] * Mi_11
            M = M_new
        
        Y = M[:, 1, 0] / M[:, 0, 0]
        r = (n_amb - Y) / (n_amb + Y)
        R = np.abs(r) ** 2 * 100.0
        total_r += np.sum(R)
    
    return total_r / len(wavelengths)


def design(problem):
    """Design a broadband antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
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
    
    # Start with a reasonable initial guess: multiple quarter-wave layers
    centre = 0.5 * (band_nm[0] + band_nm[1])
    
    # Try a few initial configurations
    initial_configs = []
    
    # Single quarter-wave L
    initial_configs.append((["L"], [centre / (4.0 * n_L)]))
    
    # Two quarter-wave layers: L H or H L
    initial_configs.append((["L", "H"], [centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    initial_configs.append((["H", "L"], [centre / (4.0 * n_H), centre / (4.0 * n_L)]))
    
    # Three quarter-wave layers
    initial_configs.append((["L", "H", "L"], [centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L)]))
    initial_configs.append((["H", "L", "H"], [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    
    # Four quarter-wave layers
    initial_configs.append((["L", "H", "L", "H"], [centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    
    # Evaluate initial configs
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
    
    # Use a gradient-free optimization: Nelder-Mead style with random restarts
    # We'll optimize the thicknesses for a given sequence of materials
    
    import random
    random.seed(42)
    
    # Define candidate material sequences
    # For broadband AR, alternating L and H layers work well
    candidate_sequences = []
    
    # Add sequences of various lengths
    for n_layers in range(1, min(max_layers + 1, 11)):
        # Alternating starting with L
        seq_L = ["L" if i % 2 == 0 else "H" for i in range(n_layers)]
        # Alternating starting with H
        seq_H = ["H" if i % 2 == 0 else "L" for i in range(n_layers)]
        candidate_sequences.append(seq_L)
        if seq_H != seq_L:
            candidate_sequences.append(seq_H)
    
    # Also try some non-alternating patterns for longer stacks
    # But keep it simple for now
    
    def optimize_thicknesses(labels, max_iter=200, tol=1e-6):
        """Optimize thicknesses for a given label sequence using Nelder-Mead."""
        n = len(labels)
        if n == 0:
            return None, float('inf')
        
        # Initial guess: quarter-wave at center wavelength
        centre = 0.5 * (band_nm[0] + band_nm[1])
        x0 = np.array([centre / (4.0 * (n_L if l == 'L' else n_H)) for l in labels])
        
        # Ensure within bounds
        x0 = np.clip(x0, 0.0, max_layer_nm)
        
        # Nelder-Mead implementation
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initial simplex
        simplex = [x0.copy()]
        for i in range(n):
            perturbed = x0.copy()
            step = 0.1 * max_layer_nm if x0[i] == 0 else 0.1 * x0[i]
            perturbed[i] += step
            perturbed = np.clip(perturbed, 0.0, max_layer_nm)
            simplex.append(perturbed)
        
        # Evaluate initial simplex
        fvals = []
        for x in simplex:
            # Check total thickness constraint
            if np.sum(x) > max_total_nm:
                fvals.append(float('inf'))
            else:
                fvals.append(_fast_compute_mean_reflectance(labels, list(x), n_amb, n_sub, n_L, n_H, band_nm))
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Sort by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            if abs(fvals[-1] - fvals[0]) < tol:
                break
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, 0.0, max_layer_nm)
            if np.sum(xr) > max_total_nm:
                fr = float('inf')
            else:
                fr = _fast_compute_mean_reflectance(labels, list(xr), n_amb, n_sub, n_L, n_H, band_nm)
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, 0.0, max_layer_nm)
                if np.sum(xe) > max_total_nm:
                    fe = float('inf')
                else:
                    fe = _fast_compute_mean_reflectance(labels, list(xe), n_amb, n_sub, n_L, n_H, band_nm)
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
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (simplex[-1] - centroid)
                xc = np.clip(xc, 0.0, max_layer_nm)
                if np.sum(xc) > max_total_nm:
                    fc = float('inf')
                else:
                    fc = _fast_compute_mean_reflectance(labels, list(xc), n_amb, n_sub, n_L, n_H, band_nm)
                
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], 0.0, max_layer_nm)
                        if np.sum(simplex[i]) > max_total_nm:
                            fvals[i] = float('inf')
                        else:
                            fvals[i] = _fast_compute_mean_reflectance(labels, list(simplex[i]), n_amb, n_sub, n_L, n_H, band_nm)
        
        # Return best point
        best_idx = np.argmin(fvals)
        return list(simplex[best_idx]), fvals[best_idx]
    
    # Optimize for each candidate sequence
    for labels in candidate_sequences:
        if time.time() > deadline - 10.0:
            break
        
        if len(labels) > max_layers:
            continue
        
        # Check if initial quarter-wave stack is feasible
        centre = 0.5 * (band_nm[0] + band_nm[1])
        init_thick = [centre / (4.0 * (n_L if l == 'L' else n_H)) for l in labels]
        if sum(init_thick) > max_total_nm:
            continue
        
        opt_thick, opt_score = optimize_thicknesses(labels, max_iter=150)
        
        if opt_thick is not None and opt_score < best_score:
            # Verify constraints
            if all(0 <= t <= max_layer_nm for t in opt_thick) and sum(opt_thick) <= max_total_nm:
                best_score = opt_score
                best_labels = labels[:]
                best_thicknesses = opt_thick
    
    # If we have time, try some random sequences
    if time.time() < deadline - 15.0:
        for trial in range(50):
            if time.time() > deadline - 10.0:
                break
            
            n_layers = random.randint(2, min(8, max_layers))
            labels = [random.choice(['L', 'H']) for _ in range(n_layers)]
            
            # Skip if same as already tried
            if labels in [c for c in candidate_sequences]:
                continue
            
            centre = 0.5 * (band_nm[0] + band_nm[1])
            init_thick = [centre / (4.0 * (n_L if l == 'L' else n_H)) for l in labels]
            if sum(init_thick) > max_total_nm:
                continue
            
            opt_thick, opt_score = optimize_thicknesses(labels, max_iter=100)
            
            if opt_thick is not None and opt_score < best_score:
                if all(0 <= t <= max_layer_nm for t in opt_thick) and sum(opt_thick) <= max_total_nm:
                    best_score = opt_score
                    best_labels = labels[:]
                    best_thicknesses = opt_thick
    
    # Return the best result found
    if best_labels is None:
        # Fallback: single quarter-wave L layer
        centre = 0.5 * (band_nm[0] + band_nm[1])
        best_labels = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]
    
    # Final constraint check
    best_thicknesses = [max(0.0, min(t, max_layer_nm)) for t in best_thicknesses]
    if sum(best_thicknesses) > max_total_nm:
        # Scale down
        scale = max_total_nm / sum(best_thicknesses)
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    return best_labels, best_thicknesses
# EVOLVE-BLOCK-END
