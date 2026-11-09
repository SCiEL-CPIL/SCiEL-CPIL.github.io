"""Broadband antireflection coating design: Simulated Annealing on full 20-layer space."""
import time
import random
import numpy as np

def _mean_reflectance(materials, thicknesses, problem):
    """
    Compute mean reflectance (percent) over the band using characteristic matrices.
    Vectorized for speed.
    """
    if not materials:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2
    
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_mat = {k: v for k, v in problem["materials"].items()}
    
    # Filter out zero-thickness layers to speed up calculation if many are zero
    # But we must keep the order. 
    # Actually, for a fixed length of 20, it's safer to just compute all.
    # Let's optimize: if thickness is 0, the matrix is identity.
    
    ns = np.array([n_mat[m] for m in materials], dtype=float)
    ds = np.asarray(thicknesses, dtype=float)
    
    # delta_j = 2*pi*n_j*d_j/lambda
    # Shape: (N, K)
    delta = 2.0 * np.pi * (ns[:, None] * ds[:, None]) / wavs[None, :]
    
    c = np.cos(delta)
    s = np.sin(delta)
    
    # Initialize characteristic matrix product
    # M = [[A, B], [C, D]]
    A = np.ones_like(wavs)
    B = np.zeros_like(wavs)
    C = np.zeros_like(wavs)
    D = np.ones_like(wavs)
    
    i = 1.0j
    
    # Multiply matrices for each layer
    for j in range(len(materials)):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        
        # Matrix for layer j:
        # [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
        # Wait, standard form is:
        # M_j = [[cos(d), i/sin(d)/n_j], [i*n_j*sin(d), cos(d)]]
        # Let's re-verify the characteristic matrix convention.
        # Usually: M = [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
        # Or sometimes scaled. The ratio r = (n0(A+Bns) - (C+Dns)) / (n0(A+Bns) + (C+Dns))
        # is invariant to scaling of M.
        
        A_new = A * cj + 1j * B * sj / n_j
        B_new = 1j * A * n_j * sj + B * cj
        C_new = C * cj + 1j * D * sj / n_j
        D_new = 1j * C * n_j * sj + D * cj
        
        A, B, C, D = A_new, B_new, C_new, D_new

    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)

def design(problem):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s kill
    
    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]
    
    # Quarter wave thicknesses
    qw = {m: wl_center / (4.0 * n) for m, n in materials_dict.items()}
    
    def constrain(ths):
        """Ensure thicknesses are within [0, max_layer] and sum <= max_total."""
        ths = np.array(ths, dtype=float)
        # Cap individual layers
        ths = np.clip(ths, 0.0, max_layer)
        total = np.sum(ths)
        if total > max_total:
            scale = max_total / total
            ths = ths * scale
            # Re-cap in case scaling pushed something weird (it won't, but safe)
            ths = np.clip(ths, 0.0, max_layer)
        return ths.tolist()
        
    def local_polish(materials, ths, problem, deadline, max_iters=200):
        """Simple coordinate descent polish."""
        ths = list(ths)
        best_score = _mean_reflectance(materials, ths, problem)
        n = len(ths)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                orig = ths[i]
                # Try small perturbations
                for delta in [1.0, -1.0, 5.0, -5.0, 10.0, -10.0]:
                    new_ths = list(ths)
                    new_ths[i] = max(0.0, min(max_layer, orig + delta))
                    new_ths = constrain(new_ths)
                    score = _mean_reflectance(materials, new_ths, problem)
                    if score < best_score:
                        best_score = score
                        ths = new_ths
                        improved = True
                        break # Restart coordinate descent from new point
                if not improved:
                    # Try a random jump in this coordinate
                    new_val = random.uniform(0, max_layer)
                    new_ths = list(ths)
                    new_ths[i] = new_val
                    new_ths = constrain(new_ths)
                    score = _mean_reflectance(materials, new_ths, problem)
                    if score < best_score:
                        best_score = score
                        ths = new_ths
                        improved = True
            if not improved:
                break
        return ths, best_score

    # Initial setup: 20 layers
    current_mats = ['L' if random.random() < 0.5 else 'H' for _ in range(max_layers)]
    current_ths = [qw[m] for m in current_mats]
    current_ths = constrain(current_ths)
    current_score = _mean_reflectance(current_mats, current_ths, problem)
    
    best_mats = list(current_mats)
    best_ths = list(current_ths)
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 10.0  # Initial temperature
    T_min = 0.01
    cooling = 0.999
    steps = 5000
    
    for step in range(steps):
        if time.time() > deadline:
            break
            
        # Generate neighbor
        new_mats = list(current_mats)
        new_ths = list(current_ths)
        
        if random.random() < 0.4:
            # Flip a layer type
            idx = random.randint(0, max_layers - 1)
            new_mats[idx] = 'L' if new_mats[idx] == 'H' else 'H'
            # Keep thickness same initially, or adjust to quarter wave?
            # Keeping same thickness is a valid move in the space.
        else:
            # Perturb thicknesses
            # Perturb 1 or 2 random layers
            num_perturb = random.randint(1, 2)
            for _ in range(num_perturb):
                idx = random.randint(0, max_layers - 1)
                # Gaussian perturbation
                delta = random.gauss(0, 20.0)
                new_ths[idx] += delta
        
        new_ths = constrain(new_ths)
        
        new_score = _mean_reflectance(new_mats, new_ths, problem)
        
        # Acceptance criterion
        delta_e = new_score - current_score
        if delta_e < 0 or random.random() < np.exp(-delta_e / T):
            current_mats = new_mats
            current_ths = new_ths
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_mats = list(current_mats)
                best_ths = list(current_ths)
                
                # Periodic polish to refine the best found so far
                if step % 500 == 0 and step > 0:
                    polished_ths, polished_score = local_polish(
                        best_mats, best_ths, problem, deadline, max_iters=50
                    )
                    if polished_score < best_score:
                        best_score = polished_score
                        best_ths = polished_ths
                        # Update current state to the polished one to continue from there
                        current_ths = list(best_ths)
                        current_score = best_score
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    # Final polish on the best solution found
    final_ths, final_score = local_polish(best_mats, best_ths, problem, deadline, max_iters=300)
    
    # Ensure strict constraints one last time
    final_ths = constrain(final_ths)
    
    return best_mats, final_ths
