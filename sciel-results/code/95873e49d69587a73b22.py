"""Broadband antireflection coating design via Global Stochastic Search with Vectorized Evaluation."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _vectorized_mean_reflectance_pct(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls):
    """
    Compute mean reflectance in percent for a single design across all wavelengths.
    Vectorized over wavelengths for speed.
    
    thicknesses: numpy array of physical thicknesses in nm
    materials: numpy array of 'L' or 'H' strings
    wls: numpy array of wavelengths in nm
    """
    n_layers = len(thicknesses)
    if n_layers == 0:
        # Bare substrate
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        R = (r * r).real
        return float(R * 100.0)
    
    # Initialize characteristic matrix M = I
    M11 = np.ones_like(wls, dtype=complex)
    M12 = np.zeros_like(wls, dtype=complex)
    M21 = np.zeros_like(wls, dtype=complex)
    M22 = np.ones_like(wls, dtype=complex)
    
    for i in range(n_layers):
        mat = materials[i]
        if mat == 'H':
            n = n_H
        else:
            n = n_L
            
        d = thicknesses[i]
        # beta = 2*pi*n*d / wl
        beta = 2.0 * np.pi * n * d / wls
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        # Layer matrix components
        L11 = cos_b
        L12 = -1j * sin_b / n
        L21 = 1j * n * sin_b
        L22 = cos_b
        
        # M = M @ L
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
    
    # Admittance Y
    Y = (M11 * n_substrate + M12) / (M21 * n_substrate + M22)
    
    # Reflection coefficient r
    r = (n_ambient - Y) / (n_ambient + Y)
    
    # Reflectance R
    R = (r * r.conjugate()).real
    
    # Mean reflectance in percent
    return float(np.mean(R) * 100.0)

def design(problem):
    """Design a broadband antireflection coating."""
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    wl_min, wl_max = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 401 equally spaced wavelengths
    wls = np.linspace(wl_min, wl_max, 401)
    wl_center = 0.5 * (wl_min + wl_max)
    
    # Quarter-wave thicknesses at center wavelength
    qw_L = wl_center / (4.0 * n_L)
    qw_H = wl_center / (4.0 * n_H)
    
    def is_valid(thicknesses, materials):
        if len(thicknesses) > max_layers:
            return False
        if len(thicknesses) != len(materials):
            return False
        for t in thicknesses:
            if t < 0 or t > max_layer_nm:
                return False
        if sum(thicknesses) > max_total_nm:
            return False
        return True
    
    def evaluate(thicknesses, materials):
        if not is_valid(thicknesses, materials):
            return float('inf')
        return _vectorized_mean_reflectance_pct(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls)
    
    # --- Global Search Strategy ---
    # We will use a combination of:
    # 1. Random restarts with different layer counts and material sequences.
    # 2. Simulated Annealing / Hill Climbing for each restart.
    
    best_score = float('inf')
    best_thicknesses = None
    best_materials = None
    
    # Pre-compute some common material patterns
    # Pattern 1: Alternating starting with L (L-H-L-H...)
    # Pattern 2: Alternating starting with H (H-L-H-L...)
    # Pattern 3: All L, All H (less likely but check)
    # Pattern 4: Random sequences
    
    def random_materials(n_layers):
        # 50% chance alternating L, 50% chance alternating H, 10% random
        r = np.random.rand()
        if r < 0.4:
            return np.array(['L' if i % 2 == 0 else 'H' for i in range(n_layers)])
        elif r < 0.8:
            return np.array(['H' if i % 2 == 0 else 'L' for i in range(n_layers)])
        else:
            return np.array([np.random.choice(['L', 'H']) for _ in range(n_layers)])
    
    def local_optimize(thicknesses, materials, t_limit):
        """
        Perform hill climbing / simulated annealing on thicknesses.
        Returns best thicknesses and score found.
        """
        current_th = np.array(thicknesses, dtype=float)
        current_mat = materials
        current_score = evaluate(current_th, current_mat)
        
        if current_score == float('inf'):
            return current_th, current_mat, current_score
            
        best_local_th = current_th.copy()
        best_local_score = current_score
        
        # Annealing parameters
        T = 1.0  # Temperature
        T_min = 0.01
        alpha = 0.95  # Cooling rate
        
        while time.time() < t_limit:
            if T < T_min:
                break
                
            # Propose a move: perturb one thickness
            i = np.random.randint(len(current_th))
            delta = np.random.normal(0, 20.0)  # Gaussian perturbation
            new_th = current_th.copy()
            new_th[i] += delta
            
            # Clamp to valid range [0, max_layer_nm]
            new_th[i] = np.clip(new_th[i], 0.0, max_layer_nm)
            
            # Check total thickness constraint
            if np.sum(new_th) > max_total_nm:
                # Scale down if needed? Or just reject. Rejecting is simpler.
                # Let's try scaling down proportionally to stay in bounds
                scale = max_total_nm / np.sum(new_th)
                new_th *= scale
            
            new_score = evaluate(new_th, current_mat)
            
            # Acceptance criterion
            if new_score < current_score:
                current_th = new_th
                current_score = new_score
                if current_score < best_local_score:
                    best_local_score = current_score
                    best_local_th = current_th.copy()
            else:
                # Accept worse solution with probability exp(-deltaE / T)
                delta_E = new_score - current_score
                if delta_E > 0:
                    p = np.exp(-delta_E / T)
                    if np.random.rand() < p:
                        current_th = new_th
                        current_score = new_score
            
            T *= alpha
            
        return best_local_th, current_mat, best_local_score

    # Main search loop
    # Try different layer counts
    possible_layer_counts = list(range(1, max_layers + 1))
    
    # Shuffle layer counts to explore broadly
    np.random.shuffle(possible_layer_counts)
    
    # We will allocate time evenly among a subset of layer counts
    # Let's try up to 10 different configurations if time permits
    configs_to_try = []
    
    # Ensure we try some standard ones first
    standard_counts = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    for c in standard_counts:
        if c <= max_layers:
            configs_to_try.append(c)
            
    # Add random ones
    while len(configs_to_try) < 15:
        c = np.random.randint(1, max_layers + 1)
        if c not in configs_to_try:
            configs_to_try.append(c)
    
    # Sort by some heuristic? Let's just iterate.
    # Time per config: (deadline - start_time - 5) / len(configs_to_try)
    
    for n_layers in configs_to_try:
        if time.time() > deadline - 5.0:
            break
            
        time_remaining = deadline - time.time() - 2.0
        if time_remaining < 1.0:
            break
            
        # Allocate a portion of remaining time to this config
        # Assume we have ~10 configs left, so take 1/10 of remaining time minus overhead
        t_alloc = max(1.0, (time_remaining - 2.0) / max(1, len(configs_to_try) - configs_to_try.index(n_layers)))
        t_alloc = min(t_alloc, time_remaining - 1.0)
        
        # Try 2-3 random material sequences for this layer count
        for attempt in range(3):
            if time.time() > deadline - 2.0:
                break
                
            materials = random_materials(n_layers)
            
            # Initialize thicknesses with quarter-wave values
            thicknesses = np.array([qw_H if m == 'H' else qw_L for m in materials], dtype=float)
            
            # Randomize slightly
            thicknesses *= np.random.uniform(0.8, 1.2, size=n_layers)
            
            # Check validity
            if not is_valid(thicknesses, materials):
                # Scale to fit
                s = max_total_nm / np.sum(thicknesses)
                if s < 1:
                    thicknesses *= s
                # Clip individual layers
                thicknesses = np.clip(thicknesses, 0, max_layer_nm)
                if not is_valid(thicknesses, materials):
                    continue
                    
            # Local optimization
            t_start_local = time.time()
            best_th, best_mat, local_score = local_optimize(thicknesses, materials, deadline - 2.0)
            
            if local_score < best_score:
                best_score = local_score
                best_thicknesses = best_th
                best_materials = best_mat
                
    # If no valid solution found, fallback to single layer
    if best_thicknesses is None:
        return ["L"], [qw_L]
        
    # Final validity check and cleanup
    final_th = best_thicknesses
    final_mat = best_materials
    
    # Ensure types are correct
    final_th_list = [float(t) for t in final_th]
    final_mat_list = [str(m) for m in final_mat]
    
    if not is_valid(final_th_list, final_mat_list):
        # Fallback
        return ["L"], [qw_L]
        
    return final_mat_list, final_th_list

# EVOLVE-BLOCK-END
