"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _mean_reflectance(thicknesses, materials, substrate_index, band_nm, n_wavelengths=401):
    """Compute mean reflectance (as a fraction, 0-1) over the band using the characteristic matrix method."""
    lam = np.linspace(band_nm[0], band_nm[1], n_wavelengths)
    n_air = 1.0
    n_sub = substrate_index
    n_L = materials['L']
    n_H = materials['H']
    
    # We need to know which layer is L and which is H. 
    # The input to this helper assumes we pass in the actual indices for each layer in order.
    # Let's refactor to take a list of indices and thicknesses.
    # But the interface is design(problem) -> (materials, thicknesses).
    # So inside design, we will construct the index list.
    pass

def _mean_reflectance_from_indices(indices, thicknesses, substrate_index, band_nm, n_wavelengths=401):
    """
    indices: list of refractive indices for each layer (air -> substrate)
    thicknesses: list of physical thicknesses in nm
    substrate_index: index of the glass
    band_nm: [lambda_min, lambda_max]
    Returns: mean reflectance as a fraction (0 to 1)
    """
    lam = np.linspace(band_nm[0], band_nm[1], n_wavelengths)
    n_air = 1.0
    n_sub = substrate_index
    n_layers = len(indices)
    
    # Pre-compute sin^2 terms? No, cos is needed.
    # Vectorize over wavelengths for speed.
    # k_j = 2 * pi * n_j / lambda
    # d_j is thickness
    
    # Characteristic matrix M = M_1 * M_2 * ... * M_N
    # M_j = [[cos(d_j k_j), i sin(d_j k_j)/n_j], [i n_j sin(d_j k_j), cos(d_j k_j)]]
    
    # We can compute M for all wavelengths at once using numpy broadcasting.
    # M is a 2x2 matrix for each wavelength.
    # Let's store M as a 2x2xN array.
    
    # Initialize M as identity for all wavelengths
    M = np.ones((2, 2, n_wavelengths), dtype=complex)
    
    for i in range(n_layers):
        n_j = indices[i]
        d_j = thicknesses[i]
        
        k_j = 2.0 * np.pi * n_j / lam  # shape (N,)
        cos_dk = np.cos(d_j * k_j)      # shape (N,)
        sin_dk = np.sin(d_j * k_j)      # shape (N,)
        
        # M_j for each wavelength
        # [[cos, i*sin/n], [i*n*sin, cos]]
        M_j = np.ones((2, 2, n_wavelengths), dtype=complex)
        M_j[0, 0, :] = cos_dk
        M_j[0, 1, :] = 1j * sin_dk / n_j
        M_j[1, 0, :] = 1j * n_j * sin_dk
        M_j[1, 1, :] = cos_dk
        
        # M = M_j * M
        M = M_j @ M
    
    # Y = n_air * M[1, 0] / M[0, 0]
    # Wait, the formula is Y = n_0 * (M_11 + M_12 * n_s) / (M_21 + M_22 * n_s)
    # Where M = [[M_11, M_12], [M_21, M_22]]
    
    M11 = M[0, 0, :]
    M12 = M[0, 1, :]
    M21 = M[1, 0, :]
    M22 = M[1, 1, :]
    
    Y = n_air * (M11 + M12 * n_sub) / (M21 + M22 * n_sub)
    
    r = (n_air - Y) / (n_air + Y)
    R = np.abs(r)**2
    
    # Ensure R is real and within [0, 1]
    R = np.real(R)
    R = np.clip(R, 0.0, 1.0)
    
    return np.mean(R)

def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (materials, thicknesses)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for the 150s limit
    
    materials_dict = problem['materials']
    n_L = materials_dict['L']
    n_H = materials_dict['H']
    n_sub = problem['substrate_index']
    band_nm = problem['band_nm']
    max_layers = problem['max_layers']
    max_total_nm = problem['max_total_nm']
    max_layer_nm = problem['max_layer_nm']
    
    # Strategy:
    # 1. Generate a set of diverse initial designs (random layer counts, random sequences, random thicknesses).
    # 2. Use a local optimization method (e.g., Nelder-Mead or a simple gradient-free search) to refine each.
    # 3. Keep track of the best design found.
    
    # To ensure speed, we will use a simplified local search:
    # - Perturb thicknesses and layer types.
    # - Accept if better.
    # This is essentially a stochastic local search / hill climbing.
    
    # Even better: Use Nelder-Mead on the thicknesses for a fixed layer sequence.
    # We can try a few fixed sequences (e.g., alternating L/H, graded) and optimize their thicknesses.
    
    # Let's define a function to evaluate a design given layer types and thicknesses.
    def evaluate(layer_types, thicknesses):
        # layer_types: list of 'L' or 'H'
        # thicknesses: list of floats
        indices = [n_L if t == 'L' else n_H for t in layer_types]
        return _mean_reflectance_from_indices(indices, thicknesses, n_sub, band_nm)
    
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    # Generate diverse initial sequences
    # 1. Single layer
    # 2. Two layers
    # 3. Alternating L/H
    # 4. Graded index (approximated by alternating)
    # 5. Random sequences
    
    import random
    random.seed(42)
    
    sequences = []
    
    # Fixed length sequences
    for n_layers in range(1, max_layers + 1):
        # Alternating starting with L
        seq = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        sequences.append(seq)
        # Alternating starting with H
        seq = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
        sequences.append(seq)
        
        # If we have time, add some random sequences
        if time.time() < deadline - 30.0:
            for _ in range(2):
                seq = [random.choice(['L', 'H']) for _ in range(n_layers)]
                sequences.append(seq)
    
    # To avoid too many sequences, limit the number of initial points
    # We will pick a subset or just iterate through them quickly.
    # Given 20 layers, 40 fixed + 80 random = 120 sequences.
    # Optimizing 120 sequences might be too slow if Nelder-Mead is heavy.
    # Let's use a faster local search: Coordinate descent or simple random perturbation.
    
    # Let's implement a simple local search for each sequence.
    # For each sequence:
    #   1. Initialize thicknesses (e.g., quarter-wave at center wavelength).
    #   2. Iterate:
    #      - Pick a random layer, perturb its thickness.
    #      - If better, accept.
    #      - If no improvement for X steps, reset or stop.
    
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    for seq in sequences:
        if time.time() > deadline - 5.0:
            break
            
        n_layers = len(seq)
        # Initial thicknesses: quarter-wave
        thicknesses = [center_wl / (4.0 * (n_L if t == 'L' else n_H)) for t in seq]
        
        # Clip to max_layer_nm
        thicknesses = [min(t, max_layer_nm) for t in thicknesses]
        
        # Check total thickness constraint
        if sum(thicknesses) > max_total_nm:
            # Scale down
            scale = max_total_nm / sum(thicknesses)
            thicknesses = [t * scale for t in thicknesses]
            
        current_score = evaluate(seq, thicknesses)
        
        if current_score < best_score:
            best_score = current_score
            best_materials = seq[:]
            best_thicknesses = thicknesses[:]
            
        # Local optimization: Hill Climbing
        no_improve_count = 0
        max_no_improve = 200  # Limit iterations per sequence to save time
        
        while no_improve_count < max_no_improve and time.time() < deadline - 5.0:
            # Pick a random layer to perturb
            idx = random.randint(0, n_layers - 1)
            
            # Perturbation size: proportional to current thickness or fixed
            # Let's use a relative perturbation
            perturbation = random.uniform(-0.2, 0.2) * thicknesses[idx]
            new_thickness = thicknesses[idx] + perturbation
            
            # Clip to valid range
            new_thickness = max(0.0, min(new_thickness, max_layer_nm))
            
            # Check total thickness constraint
            new_total = sum(thicknesses) - thicknesses[idx] + new_thickness
            if new_total > max_total_nm:
                # Reject if it violates total thickness
                # Or scale down? Scaling down changes other layers, which is complex.
                # Let's just reject for simplicity in this local search.
                continue
                
            # Evaluate
            new_thicknesses = thicknesses[:]
            new_thicknesses[idx] = new_thickness
            new_score = evaluate(seq, new_thicknesses)
            
            if new_score < current_score:
                current_score = new_score
                thicknesses = new_thicknesses
                no_improve_count = 0
                
                if current_score < best_score:
                    best_score = current_score
                    best_materials = seq[:]
                    best_thicknesses = thicknesses[:]
            else:
                no_improve_count += 1
                
            # Occasionally try a larger perturbation (restart-like)
            if no_improve_count % 50 == 0 and no_improve_count > 0:
                # Try a random restart for this sequence?
                # Or just increase perturbation size?
                pass

    # Final check: Ensure constraints are met
    if best_materials is None:
        # Fallback to single quarter-wave layer
        best_materials = ['L']
        best_thicknesses = [center_wl / (4.0 * n_L)]
        
    # Double check constraints
    total_thickness = sum(best_thicknesses)
    if total_thickness > max_total_nm:
        scale = max_total_nm / total_thickness
        best_thicknesses = [t * scale for t in best_thicknesses]
        
    for i in range(len(best_thicknesses)):
        if best_thicknesses[i] > max_layer_nm:
            best_thicknesses[i] = max_layer_nm
        if best_thicknesses[i] < 0:
            best_thicknesses[i] = 0.0
            
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
