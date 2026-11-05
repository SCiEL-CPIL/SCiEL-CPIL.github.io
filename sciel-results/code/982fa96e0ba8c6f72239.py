import numpy as np

def design(problem):
    ambient_index = problem.get("ambient_index", 1.0)
    substrate_index = problem.get("substrate_index", 1.52)
    materials = problem.get("materials", {"L": 1.38, "H": 2.35})
    band_nm = problem.get("band_nm", [400, 800])
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    nL = materials["L"]
    nH = materials["H"]
    n0 = ambient_index
    ns = substrate_index
    
    # Wavelengths
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Precompute 2*pi / lambda
    two_pi_over_lambda = 2.0 * np.pi / wavelengths
    
    def compute_reflectance(mat_types, thicknesses):
        """
        mat_types: list of 'L' or 'H'
        thicknesses: list of floats
        Returns mean reflectance in percent.
        """
        N = len(mat_types)
        if N == 0:
            # Bare substrate
            y_in = 1.0 / ns
            r = (y_in - n0) / (y_in + n0)
            return np.mean(np.abs(r)**2) * 100.0
            
        # Initialize total matrix M as identity for each wavelength
        # M is shape (401, 2, 2)
        m11 = np.ones_like(wavelengths)
        m12 = np.zeros_like(wavelengths)
        m21 = np.zeros_like(wavelengths)
        m22 = np.ones_like(wavelengths)
        
        for i in range(N):
            n_j = nL if mat_types[i] == 'L' else nH
            d_j = thicknesses[i]
            delta = two_pi_over_lambda * n_j * d_j
            
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            
            # Matrix for layer i
            # M_i = [[cos, i sin/n], [i n sin, cos]]
            # We accumulate M = M * M_i
            # M_new_11 = M_11 * cos + M_12 * (i sin/n)
            # M_new_12 = M_11 * (i n sin) + M_12 * cos  -- Wait, order of multiplication?
            # Standard: M_total = M_1 M_2 ... M_N
            # If we iterate 1 to N, we do M_acc = M_acc @ M_i
            # M_acc = [[m11, m12], [m21, m22]]
            # M_i = [[c, i s/n], [i n s, c]]
            
            # m11_new = m11*c + m12*(i s/n)
            # m12_new = m11*(i n s) + m12*c
            # m21_new = m21*c + m22*(i s/n)
            # m22_new = m21*(i n s) + m22*c
            
            i_s_over_n = 1j * sin_d / n_j
            i_n_s = 1j * n_j * sin_d
            
            new_m11 = m11 * cos_d + m12 * i_s_over_n
            new_m12 = m11 * i_n_s + m12 * cos_d
            new_m21 = m21 * cos_d + m22 * i_s_over_n
            new_m22 = m21 * i_n_s + m22 * cos_d
            
            m11, m12, m21, m22 = new_m11, new_m12, new_m21, new_m22
            
        # Calculate Y_in
        # Y_in = (M11 + M12 * ns) / (M21 + M22 * ns)
        y_in = (m11 + m12 * ns) / (m21 + m22 * ns)
        
        # Reflectance
        r = (y_in - n0) / (y_in + n0)
        refl = np.abs(r)**2
        
        return np.mean(refl) * 100.0

    # Optimization: Simulated Annealing
    # State: (mat_types, thicknesses)
    
    def get_cost(mat_types, thicknesses):
        # Check constraints
        if len(mat_types) > max_layers:
            return 1000.0
        if sum(thicknesses) > max_total_nm:
            return 1000.0
        if any(t < 0 or t > max_layer_nm for t in thicknesses):
            return 1000.0
            
        return compute_reflectance(mat_types, thicknesses)

    # Initial state: Random valid stack
    import random
    random.seed(42)
    
    def random_state():
        n_layers = random.randint(3, 10)
        types = [random.choice(['L', 'H']) for _ in range(n_layers)]
        # Random thicknesses, ensure sum < max_total
        # Start with small thicknesses
        ths = [random.uniform(10, 100) for _ in range(n_layers)]
        # Scale to fit if needed, but keep simple
        total = sum(ths)
        if total > max_total_nm:
            scale = max_total_nm / total
            ths = [t * scale for t in ths]
        return types, ths

    best_types, best_ths = random_state()
    best_cost = get_cost(best_types, best_ths)
    
    current_types, current_ths = random_state()
    current_cost = get_cost(current_types, current_ths)
    
    T = 1.0
    T_min = 0.01
    alpha = 0.995
    max_iter = 5000 # Adjust based on time
    
    import time
    start_time = time.time()
    
    for i in range(max_iter):
        if time.time() - start_time > 45: # Leave buffer for return
            break
            
        # Generate neighbor
        new_types = current_types[:]
        new_ths = current_ths[:]
        
        move_type = random.random()
        
        if move_type < 0.3 and len(new_types) > 0:
            # Perturb thickness
            idx = random.randint(0, len(new_ths)-1)
            new_ths[idx] += random.gauss(0, 20)
            new_ths[idx] = max(0, min(new_ths[idx], max_layer_nm))
        elif move_type < 0.5 and len(new_types) > 0:
            # Change type
            idx = random.randint(0, len(new_types)-1)
            new_types[idx] = 'L' if new_types[idx] == 'H' else 'H'
        elif move_type < 0.7 and len(new_types) < max_layers:
            # Add layer
            new_types.insert(random.randint(0, len(new_types)), random.choice(['L', 'H']))
            new_ths.insert(random.randint(0, len(new_ths)), random.uniform(10, 50))
        elif move_type < 0.8 and len(new_types) > 1:
            # Remove layer
            idx = random.randint(0, len(new_types)-1)
            new_types.pop(idx)
            new_ths.pop(idx)
        else:
            # Swap adjacent
            if len(new_types) > 1:
                idx = random.randint(0, len(new_types)-2)
                new_types[idx], new_types[idx+1] = new_types[idx+1], new_types[idx]
                new_ths[idx], new_ths[idx+1] = new_ths[idx+1], new_ths[idx]
        
        new_cost = get_cost(new_types, new_ths)
        
        # Acceptance
        delta = new_cost - current_cost
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_types = new_types
            current_ths = new_ths
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_types = current_types[:]
                best_ths = current_ths[:]
        
        T *= alpha
        if T < T_min:
            T = T_min
            
    # Final projection to ensure validity
    # The search already maintains validity mostly, but let's double check
    if sum(best_ths) > max_total_nm:
        scale = max_total_nm / sum(best_ths)
        best_ths = [t * scale for t in best_ths]
    best_ths = [max(0, min(t, max_layer_nm)) for t in best_ths]
    
    return best_types, best_ths
