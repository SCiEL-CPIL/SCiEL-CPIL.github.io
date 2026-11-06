import numpy as np
import time
import random

def _compute_reflectance(wavelengths, materials, thicknesses, substrate_index):
    """
    Compute reflectance for a given stack.
    wavelengths: array of wavelengths in nm
    materials: list of indices (floats)
    thicknesses: list of thicknesses (floats) in nm
    substrate_index: float
    """
    N = len(materials)
    if N == 0:
        # Bare substrate
        r = (1.0 - substrate_index) / (1.0 + substrate_index)
        return np.abs(r)**2 * 100.0

    # Precompute phase terms for all layers and wavelengths
    # delta[j, k] = 2 * pi * n_j * d_j / lambda_k
    # n: (N, 1), d: (N, 1), lambda: (1, M)
    n_arr = np.array(materials)[:, np.newaxis]
    d_arr = np.array(thicknesses)[:, np.newaxis]
    lam_arr = np.array(wavelengths)[np.newaxis, :]
    
    delta = 2.0 * np.pi * n_arr * d_arr / lam_arr
    
    # Initialize total matrix M as identity
    # M is (N, M, 2, 2) effectively, but we can do it layer by layer or vectorized.
    # Vectorized approach:
    # M_total = M_1 M_2 ... M_N
    # M_j = [[cos(d_j), i/n_j sin(d_j)], [i n_j sin(d_j), cos(d_j)]]
    
    # We can compute the product of matrices.
    # Let's use complex numbers.
    cos_d = np.cos(delta)
    sin_d = np.sin(delta)
    
    # M_j elements
    M11 = cos_d
    M12 = 1j / n_arr * sin_d
    M21 = 1j * n_arr * sin_d
    M22 = cos_d
    
    # Product of matrices
    # Start with identity
    # P = I
    # P = P @ M_1
    # P = P @ M_2
    # ...
    
    # To save memory and time, we can iterate or use a reduce.
    # Given N <= 20 and M <= 401, full array is small.
    
    P11 = np.ones_like(M11)
    P12 = np.zeros_like(M11)
    P21 = np.zeros_like(M11)
    P22 = np.ones_like(M11)
    
    for j in range(N):
        # P = P @ M_j
        # [P11 P12; P21 P22] @ [M11 M12; M21 M22]
        new_P11 = P11 * M11[j] + P12 * M21[j]
        new_P12 = P11 * M12[j] + P12 * M22[j]
        new_P21 = P21 * M11[j] + P22 * M21[j]
        new_P22 = P21 * M12[j] + P22 * M22[j]
        
        P11 = new_P11
        P12 = new_P12
        P21 = new_P21
        P22 = new_P22
        
    # Admittance Y = (P11 + i P12) / (P21 + i P22)
    # Wait, the standard formula uses the matrix product from air side.
    # The total matrix M is defined such that [E; H]_0 = M [E; H]_N.
    # Actually, the characteristic matrix is usually defined as:
    # M = M_1 M_2 ... M_N
    # And the relation is [E; H]_0 = M [E; H]_N.
    # At the substrate, H_N = n_s E_N / n_0 (if E is normalized).
    # Let's use the standard formula:
    # Y = (M11 + i M12) / (M21 + i M22) is incorrect if M is defined as above.
    # Let's check the definition.
    # Common definition:
    # M_j = [[cos delta_j, i/n_j sin delta_j], [i n_j sin delta_j, cos delta_j]]
    # Total M = M_1 ... M_N.
    # Then Y = (M11 + i M12) / (M21 + i M22) is NOT correct.
    # Correct relation:
    # [E; H]_0 = M [E; H]_N
    # E_0 = M11 E_N + M12 H_N
    # H_0 = M21 E_N + M22 H_N
    # At substrate: H_N = (n_s / n_0) E_N (assuming E_N is the field, H is related to E by impedance).
    # Actually, usually H is normalized such that H = (n/n_0) E.
    # So H_N = (n_s / n_0) E_N.
    # E_0 = M11 E_N + M12 (n_s/n_0) E_N = (M11 + M12 n_s/n_0) E_N
    # H_0 = M21 E_N + M22 (n_s/n_0) E_N = (M21 + M22 n_s/n_0) E_N
    # Y_0 = H_0 / E_0 = (M21 + M22 n_s/n_0) / (M11 + M12 n_s/n_0)
    # But Y_0 is also n_0 E_0 / E_0? No, Y is admittance.
    # In air, Y_0 = n_0 E_0 / E_0 = n_0? No.
    # Let's use the reflectance formula directly.
    # r = (n_0 Y - 1) / (n_0 Y + 1) ? No.
    # Let's use:
    # r = (M11 n_0 + M12 n_s - M21 n_0 - M22 n_s) / (M11 n_0 + M12 n_s + M21 n_0 + M22 n_s)
    # This is derived from:
    # E_0 = (M11 + M12 n_s) E_s
    # H_0 = (M21 + M22 n_s) E_s
    # Y_0 = H_0/E_0 = (M21 + M22 n_s) / (M11 + M12 n_s)
    # r = (Y_0 - n_0) / (Y_0 + n_0)
    # r = ( (M21 + M22 n_s) / (M11 + M12 n_s) - n_0 ) / ( (M21 + M22 n_s) / (M11 + M12 n_s) + n_0 )
    # r = ( M21 + M22 n_s - n_0 (M11 + M12 n_s) ) / ( M21 + M22 n_s + n_0 (M11 + M12 n_s) )
    # r = ( M21 + M22 n_s - n_0 M11 - n_0 M12 n_s ) / ( M21 + M22 n_s + n_0 M11 + n_0 M12 n_s )
    
    n_0 = 1.0
    n_s = substrate_index
    
    numerator = M21 + M22 * n_s - n_0 * M11 - n_0 * M12 * n_s
    denominator = M21 + M22 * n_s + n_0 * M11 + n_0 * M12 * n_s
    
    r = numerator / denominator
    R = np.abs(r)**2 * 100.0
    return R

def design(problem):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    materials_map = problem["materials"]
    n_L = materials_map["L"]
    n_H = materials_map["H"]
    n_sub = problem["substrate_index"]
    n_air = problem["ambient_index"]
    
    band = problem["band_nm"]
    wl_min, wl_max = band[0], band[1]
    wavelengths = np.linspace(wl_min, wl_max, 401)
    
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Initial guess: 10 layers, alternating H and L, quarter wave at 600nm
    # Center wavelength 600nm
    lam_c = 600.0
    d_q_L = lam_c / (4.0 * n_L)
    d_q_H = lam_c / (4.0 * n_H)
    
    # Start with a reasonable stack
    # H L H L H L H L H L
    init_labels = ['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L']
    init_thick = [d_q_H, d_q_L, d_q_H, d_q_L, d_q_H, d_q_L, d_q_H, d_q_L, d_q_H, d_q_L]
    
    # Check validity
    def is_valid(labels, thicks):
        if len(labels) != len(thicks):
            return False
        if len(labels) > max_layers:
            return False
        if sum(thicks) > max_total_nm:
            return False
        for t in thicks:
            if t < 0 or t > max_layer_nm:
                return False
        return True
    
    # Objective function
    def objective(labels, thicks):
        if not is_valid(labels, thicks):
            return float('inf')
        mats = [materials_map[l] for l in labels]
        R = _compute_reflectance(wavelengths, mats, thicks, n_sub)
        return np.mean(R)
    
    current_labels = list(init_labels)
    current_thicks = list(init_thick)
    current_score = objective(current_labels, current_thicks)
    
    best_labels = list(current_labels)
    best_thicks = list(current_thicks)
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.001
    
    # Random seed for reproducibility in this run
    random.seed(42)
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # Generate a neighbor
        new_labels = list(current_labels)
        new_thicks = list(current_thicks)
        
        move_type = random.random()
        
        if move_type < 0.6:
            # Perturb thickness of a random layer
            if len(new_thicks) > 0:
                idx = random.randint(0, len(new_thicks) - 1)
                # Gaussian perturbation
                delta_t = random.gauss(0, 10.0)
                new_thicks[idx] += delta_t
                # Clip
                if new_thicks[idx] < 0: new_thicks[idx] = 0
                if new_thicks[idx] > max_layer_nm: new_thicks[idx] = max_layer_nm
                
        elif move_type < 0.8:
            # Change material of a random layer
            if len(new_labels) > 0:
                idx = random.randint(0, len(new_labels) - 1)
                if new_labels[idx] == 'L':
                    new_labels[idx] = 'H'
                else:
                    new_labels[idx] = 'L'
                # Adjust thickness to maintain quarter-wave approximation? 
                # Or just let it be. Let's keep thickness same for now.
                
        elif move_type < 0.9:
            # Insert a layer
            if len(new_labels) < max_layers:
                pos = random.randint(0, len(new_labels))
                new_mat = random.choice(['L', 'H'])
                new_thick = random.uniform(10.0, 50.0)
                new_labels.insert(pos, new_mat)
                new_thicks.insert(pos, new_thick)
                
        else:
            # Delete a layer
            if len(new_labels) > 1:
                idx = random.randint(0, len(new_labels) - 1)
                new_labels.pop(idx)
                new_thicks.pop(idx)
        
        new_score = objective(new_labels, new_thicks)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_labels = new_labels
            current_thicks = new_thicks
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_labels = list(current_labels)
                best_thicks = list(current_thicks)
                
        # Cool down
        T *= cooling_rate
        if T < min_T:
            T = min_T
            # If we've cooled too much, maybe restart with a small T or stop?
            # Let's just keep it at min_T and continue exploring locally.
            
    # Final polish: Coordinate descent on the best solution
    # Try to optimize each thickness individually
    polish_labels = list(best_labels)
    polish_thicks = list(best_thicks)
    polish_score = best_score
    
    for _ in range(50): # Max polish iterations
        improved = False
        for i in range(len(polish_thicks)):
            if time.time() > deadline - 5.0: break
            orig_t = polish_thicks[i]
            # Try small steps
            for step in [1.0, -1.0, 5.0, -5.0, 0.1, -0.1]:
                candidate_t = orig_t + step
                if candidate_t < 0 or candidate_t > max_layer_nm: continue
                if sum(polish_thicks[:i] + [candidate_t] + polish_thicks[i+1:]) > max_total_nm: continue
                
                old_t = polish_thicks[i]
                polish_thicks[i] = candidate_t
                s = objective(polish_labels, polish_thicks)
                if s < polish_score:
                    polish_score = s
                    improved = True
                else:
                    polish_thicks[i] = old_t
            
        if not improved:
            break
            
    if polish_score < best_score:
        best_labels = polish_labels
        best_thicks = polish_thicks
        best_score = polish_score
        
    # Convert to required format
    # Labels are already 'L'/'H'
    # Thicknesses are floats
    
    return best_labels, best_thicks

