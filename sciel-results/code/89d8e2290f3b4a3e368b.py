"""Broadband AR coating design: continuous admittance profile optimization via L-BFGS + quantization."""
import numpy as np
import time
import random
import sys

# EVOLVE-BLOCK-START
def design(problem):
    """Design broadband AR coating using continuous profile optimization + quantization."""
    start_time = time.time()
    DEADLINE = 145.0
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    lambdas = np.linspace(band[0], band[1], 401)
    lambda_c = 0.5 * (band[0] + band[1])
    n_wl = len(lambdas)
    
    def reflectance_from_profile(n_layers, thicknesses):
        """Compute mean reflectance in percent for given layer indices and thicknesses."""
        n = len(thicknesses)
        if n == 0:
            r = (n0 - ns) / (n0 + ns)
            return (np.abs(r) ** 2) * 100.0
        
        m_total = np.zeros((n_wl, 2, 2), dtype=complex)
        m_total[:, 0, 0] = 1.0
        m_total[:, 1, 1] = 1.0
        
        for i in range(n):
            ni = n_layers[i]
            di = thicknesses[i]
            delta = 2.0 * np.pi * ni * di / lambdas
            c = np.cos(delta)
            s = np.sin(delta)
            
            m_layer = np.zeros((n_wl, 2, 2), dtype=complex)
            m_layer[:, 0, 0] = c
            m_layer[:, 0, 1] = 1j * s / ni
            m_layer[:, 1, 0] = 1j * ni * s
            m_layer[:, 1, 1] = c
            
            m_total = np.einsum('wij,wjk->wik', m_total, m_layer)
        
        A = m_total[:, 0, 0]
        B = m_total[:, 0, 1]
        C = m_total[:, 1, 0]
        D = m_total[:, 1, 1]
        
        y_in = (C + D * ns) / (A + B * ns)
        r = (n0 - y_in) / (n0 + y_in)
        R = np.abs(r) ** 2
        
        return float(np.mean(R) * 100.0)
    
    def is_valid(materials, thicknesses):
        if len(materials) != len(thicknesses):
            return False
        if len(materials) > max_layers:
            return False
        if any(t < -1e-9 or t > max_layer_nm + 1e-9 for t in thicknesses):
            return False
        if sum(thicknesses) > max_total_nm + 1e-9:
            return False
        return True
    
    def clip_thicknesses(thicknesses):
        t = np.array(thicknesses, dtype=float)
        t = np.clip(t, 0.0, max_layer_nm)
        total = np.sum(t)
        if total > max_total_nm:
            scale = max_total_nm / total
            t = t * scale
        return [float(x) for x in t]
    
    def quantize_profile(n_layers, thicknesses, target_layers):
        """Quantize a continuous profile to discrete L/H layers."""
        n = len(thicknesses)
        if n == 0:
            return [], []
        
        # Map each layer to nearest material based on index
        materials = []
        for i in range(n):
            if n_layers[i] < (nL + nH) / 2:
                materials.append('L')
            else:
                materials.append('H')
        
        # Adjust thicknesses to match target layer count
        if len(materials) > target_layers:
            # Merge adjacent layers with same material
            new_mats = []
            new_thicks = []
            i = 0
            while i < n:
                mat = materials[i]
                thick = thicknesses[i]
                j = i + 1
                while j < n and materials[j] == mat and len(new_mats) < target_layers - 1:
                    thick += thicknesses[j]
                    j += 1
                new_mats.append(mat)
                new_thicks.append(thick)
                i = j
            materials = new_mats
            thicknesses = new_thicks
        
        # If still too many, drop smallest layers
        while len(materials) > target_layers:
            min_idx = np.argmin(thicknesses)
            materials.pop(min_idx)
            thicknesses.pop(min_idx)
        
        # If too few, add quarter-wave layers
        while len(materials) < target_layers:
            # Add alternating L/H at quarter-wave
            if len(materials) == 0:
                materials.append('L')
                thicknesses.append(lambda_c / (4 * nL))
            else:
                last_mat = materials[-1]
                new_mat = 'H' if last_mat == 'L' else 'L'
                materials.append(new_mat)
                thicknesses.append(lambda_c / (4 * (nH if new_mat == 'H' else nL)))
        
        # Clip thicknesses
        thicknesses = clip_thicknesses(thicknesses)
        
        return materials, thicknesses
    
    # Strategy 1: Generate diverse initial candidates
    qw_L = lambda_c / (4.0 * nL)
    qw_H = lambda_c / (4.0 * nH)
    
    candidates = []
    
    # Standard alternating structures
    for n_layers in range(1, 13):
        # LHLH...
        mat = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        thick = [qw_L if i % 2 == 0 else qw_H for i in range(n_layers)]
        candidates.append((mat, thick))
        
        # HLHL...
        mat = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
        thick = [qw_H if i % 2 == 0 else qw_L for i in range(n_layers)]
        candidates.append((mat, thick))
    
    # Random structures
    for _ in range(20):
        n_layers = random.randint(3, 15)
        mat = [random.choice(['L', 'H']) for _ in range(n_layers)]
        thick = [random.uniform(qw_L*0.4, qw_L*1.6) if m=='L' else random.uniform(qw_H*0.4, qw_H*1.6) for m in mat]
        thick = clip_thicknesses(thick)
        candidates.append((mat, thick))
    
    # Evaluate all candidates
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    scored_candidates = []
    
    for mat, thick in candidates:
        if not is_valid(mat, thick):
            continue
        score = reflectance_from_profile(
            np.array([nL if m == 'L' else nH for m in mat], dtype=float),
            np.array(thick, dtype=float)
        )
        scored_candidates.append((score, mat, thick))
        if score < best_score:
            best_score = score
            best_materials = mat
            best_thicknesses = thick
    
    if best_materials is None:
        return ["L"], [qw_L]
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Strategy 2: Local refinement using coordinate descent with gradient estimation
    def refine_thicknesses(materials, thicknesses, max_iter=200):
        """Refine thicknesses using coordinate descent."""
        n = len(thicknesses)
        n_layers_arr = np.array([nL if m == 'L' else nH for m in materials], dtype=float)
        current_thick = list(thicknesses)
        current_f = reflectance_from_profile(n_layers_arr, np.array(current_thick))
        best_thick = list(current_thick)
        best_f = current_f
        
        for it in range(max_iter):
            if (time.time() - start_time) > DEADLINE:
                break
            
            improved = False
            for i in range(n):
                # Try multiple step sizes
                for step in [5, 10, 20, 30, 50]:
                    for direction in [-1, 1]:
                        new_thick = list(current_thick)
                        new_thick[i] += direction * step
                        new_thick = clip_thicknesses(new_thick)
                        
                        if not is_valid(materials, new_thick):
                            continue
                        
                        new_f = reflectance_from_profile(n_layers_arr, np.array(new_thick))
                        if new_f < current_f:
                            current_f = new_f
                            current_thick = new_thick
                            improved = True
                            if current_f < best_f:
                                best_f = current_f
                                best_thick = list(current_thick)
                            break
            
            if not improved:
                break
        
        return best_thick, best_f
    
    # Refine top candidates
    for score, materials, thick in scored_candidates[:5]:
        if (time.time() - start_time) > DEADLINE:
            break
        
        refined_thick, refined_f = refine_thicknesses(materials, thick)
        
        if refined_f < best_score:
            best_score = refined_f
            best_materials = materials
            best_thicknesses = refined_thick
    
    # Strategy 3: Try different layer counts with optimization
    for target_n in [8, 10, 12, 14, 16]:
        if (time.time() - start_time) > DEADLINE:
            break
        
        # Start with alternating pattern
        mat = ['L' if i % 2 == 0 else 'H' for i in range(target_n)]
        thick = [qw_L if i % 2 == 0 else qw_H for i in range(target_n)]
        thick = clip_thicknesses(thick)
        
        if not is_valid(mat, thick):
            continue
        
        refined_thick, refined_f = refine_thicknesses(mat, thick)
        
        if refined_f < best_score:
            best_score = refined_f
            best_materials = mat
            best_thicknesses = refined_thick
    
    # Final validation
    best_thicknesses = clip_thicknesses(best_thicknesses)
    
    if not is_valid(best_materials, best_thicknesses):
        for mat, thick in scored_candidates:
            if is_valid(mat, thick):
                return mat, thick
        return ["L"], [qw_L]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
