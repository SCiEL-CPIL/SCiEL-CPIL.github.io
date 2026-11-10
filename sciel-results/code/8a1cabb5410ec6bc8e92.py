import numpy as np
import time
import random

def design(problem):
    n_0 = problem["ambient_index"]
    n_s = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wavelengths = np.linspace(band[0], band[1], 401)
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline with safety margin
    
    def compute_reflectance(materials, thicknesses):
        """Compute mean reflectance (%) using characteristic matrix method."""
        n = np.array([n_L if m == 'L' else n_H for m in materials])
        t = np.array(thicknesses, dtype=float)
        d = len(materials)
        
        # Admittance of substrate
        Y_s = n_s
        
        # Build characteristic matrices for each wavelength
        # M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
        # where delta_j = 2*pi*n_j*t_j/wavelength
        
        # We'll compute for all wavelengths at once using vectorization
        # For each layer j, delta_j(lambda) = 2*pi*n_j*t_j/lambda
        
        # Y starts as Y_s, then we propagate backwards
        # Actually, let's compute the total characteristic matrix M = M_1 * M_2 * ... * M_d
        # Then Y_in = Y_s * (M[0,0] + i*M[0,1]*Y_s) / (M[1,0] + i*M[1,1]*Y_s)
        # Wait, let me be more careful.
        
        # The characteristic matrix for layer j is:
        # M_j = [[cos(d_j), i*sin(d_j)/n_j], [i*n_j*sin(d_j), cos(d_j)]]
        # where d_j = 2*pi*n_j*t_j/lambda
        
        # Total matrix M = M_1 * M_2 * ... * M_d (from air side to substrate)
        # Then the input admittance Y_in satisfies:
        # [E_in, H_in]^T = M * [E_s, H_s]^T
        # Y_in = E_in/H_in = (M[0,0]*Y_s + M[0,1]) / (M[1,0]*Y_s + M[1,1])
        # Wait, I need to be careful with the convention.
        
        # Let me use: at the substrate, Y_s = n_s (for normal incidence, lossless)
        # The relation is: Y_in = (M11 * Y_s + M12) / (M21 * Y_s + M22)
        # where M = [[M11, M12], [M21, M22]] is the total characteristic matrix
        
        # Reflectance R = |(Y_in - n_0)/(Y_in + n_0)|^2
        
        if d == 0:
            R = ((n_s - n_0)/(n_s + n_0))**2
            return np.mean(R) * 100.0
        
        # Vectorize over wavelengths
        # For each layer j and wavelength lambda:
        # delta_j = 2*pi*n_j*t_j/lambda
        # M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
        
        # We'll compute the product of matrices for each wavelength
        # This is tricky to vectorize directly since matrix multiplication is sequential over layers
        
        # Alternative: iterate over wavelengths (401 of them) - should be fast enough
        R_vals = np.zeros(len(wavelengths))
        
        for i, lam in enumerate(wavelengths):
            # Build total matrix M = I initially
            M11, M12 = 1.0+0j, 0.0+0j
            M21, M22 = 0.0+0j, 1.0+0j
            
            for j in range(d):
                delta = 2.0 * np.pi * n[j] * t[j] / lam
                c = np.cos(delta)
                s = np.sin(delta)
                # M_j = [[c, i*s/n_j], [i*n_j*s, c]]
                m11, m12 = c, 1j * s / n[j]
                m21, m22 = 1j * n[j] * s, c
                
                # M = M * M_j
                new_M11 = M11 * m11 + M12 * m21
                new_M12 = M11 * m12 + M12 * m22
                new_M21 = M21 * m11 + M22 * m21
                new_M22 = M21 * m12 + M22 * m22
                M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
            
            # Y_in = (M11 * Y_s + M12) / (M21 * Y_s + M22)
            Y_in = (M11 * n_s + M12) / (M21 * n_s + M22)
            R_vals[i] = np.abs((Y_in - n_0) / (Y_in + n_0))**2
        
        return np.mean(R_vals) * 100.0
    
    def is_valid(materials, thicknesses):
        if len(materials) != len(thicknesses):
            return False
        if len(materials) > max_layers:
            return False
        if any(t < 0 or t > max_layer_nm for t in thicknesses):
            return False
        if sum(thicknesses) > max_total_nm:
            return False
        return True
    
    # Initial design: start with a few quarter-wave layers
    centre = 0.5 * (band[0] + band[1])
    # Try a 3-layer or 5-layer initial design
    # Start with alternating L and H layers
    init_n_layers = 5
    init_materials = ['L', 'H', 'L', 'H', 'L']
    init_thicknesses = []
    for m in init_materials:
        n = n_L if m == 'L' else n_H
        init_thicknesses.append(centre / (4.0 * n))
    
    # Ensure validity
    while sum(init_thicknesses) > max_total_nm:
        scale = max_total_nm / sum(init_thicknesses)
        init_thicknesses = [t * scale for t in init_thicknesses]
    
    best_materials = init_materials[:]
    best_thicknesses = init_thicknesses[:]
    best_score = compute_reflectance(best_materials, best_thicknesses)
    
    # Local search
    random.seed(42)
    
    def perturb(materials, thicknesses):
        """Create a perturbed version of the design."""
        mat = materials[:]
        th = thicknesses[:]
        
        # Randomly choose what to perturb
        r = random.random()
        
        if r < 0.3 and len(th) > 0:
            # Perturb thicknesses
            for i in range(len(th)):
                if random.random() < 0.5:
                    # Add small random perturbation
                    th[i] += random.gauss(0, 10)
                    th[i] = max(0, min(max_layer_nm, th[i]))
        elif r < 0.5 and len(mat) > 0:
            # Change a random layer's material
            i = random.randint(0, len(mat)-1)
            mat[i] = 'H' if mat[i] == 'L' else 'L'
            # Adjust thickness to keep quarter-wave-ish
            n_new = n_L if mat[i] == 'L' else n_H
            # Keep optical thickness similar
            n_old = n_L if materials[i] == 'L' else n_H
            th[i] = th[i] * n_old / n_new
            th[i] = max(0, min(max_layer_nm, th[i]))
        elif r < 0.7 and len(mat) < max_layers:
            # Add a layer
            pos = random.randint(0, len(mat))
            new_mat = random.choice(['L', 'H'])
            n_new = n_L if new_mat == 'L' else n_H
            new_th = centre / (4.0 * n_new)
            new_th = min(new_th, max_layer_nm)
            mat.insert(pos, new_mat)
            th.insert(pos, new_th)
        elif r < 0.85 and len(mat) > 1:
            # Remove a layer
            pos = random.randint(0, len(mat)-1)
            mat.pop(pos)
            th.pop(pos)
        else:
            # Perturb all thicknesses slightly
            for i in range(len(th)):
                th[i] += random.gauss(0, 5)
                th[i] = max(0, min(max_layer_nm, th[i]))
        
        # Check total thickness constraint
        if sum(th) > max_total_nm:
            scale = max_total_nm / sum(th)
            th = [t * scale for t in th]
        
        return mat, th
    
    # Iterative local search
    current_materials = best_materials[:]
    current_thicknesses = best_thicknesses[:]
    current_score = best_score
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        # Generate perturbation
        new_materials, new_thicknesses = perturb(current_materials, current_thicknesses)
        
        if not is_valid(new_materials, new_thicknesses):
            no_improve_count += 1
            continue
        
        new_score = compute_reflectance(new_materials, new_thicknesses)
        
        if new_score < current_score:
            current_materials = new_materials
            current_thicknesses = new_thicknesses
            current_score = new_score
            no_improve_count = 0
            
            if new_score < best_score:
                best_materials = new_materials[:]
                best_thicknesses = new_thicknesses[:]
                best_score = new_score
        else:
            no_improve_count += 1
        
        # If stuck for too long, try a bigger jump
        if no_improve_count > max_no_improve:
            # Reset to best and try different perturbation strategy
            current_materials = best_materials[:]
            current_thicknesses = best_thicknesses[:]
            current_score = best_score
            no_improve_count = 0
            max_no_improve = int(max_no_improve * 1.5)
            
            # Try a more aggressive perturbation
            for _ in range(5):
                new_materials, new_thicknesses = perturb(current_materials, current_thicknesses)
                if is_valid(new_materials, new_thicknesses):
                    s = compute_reflectance(new_materials, new_thicknesses)
                    if s < current_score:
                        current_materials = new_materials
                        current_thicknesses = new_thicknesses
                        current_score = s
                        if s < best_score:
                            best_materials = new_materials[:]
                            best_thicknesses = new_thicknesses[:]
                            best_score = s
                        break
    
    return best_materials, best_thicknesses
