import time
import numpy as np
import random

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (percent) over the band using characteristic matrices."""
    if not materials:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = [n_mat[m] for m in materials]
    ds = np.asarray(thicknesses, dtype=float)

    # Build per-layer optical path lengths
    n_ds = np.array([n * d for n, d in zip(ns, ds)])  # (N,)
    delta = 2.0 * np.pi * n_ds[:, None] / wavs[None, :]  # (N, K)
    c = np.cos(delta)
    s = np.sin(delta)

    # Initialize characteristic matrix product as identity for all wavelengths
    A = np.ones_like(wavs)
    B = np.zeros_like(wavs)
    C = np.zeros_like(wavs)
    D = np.ones_like(wavs)

    i = 1.0j
    for j in range(len(materials)):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        # Update A,B,C,D
        A_new = A * cj + 1j * B * n_j * sj
        B_new = 1j * A * sj / n_j + B * cj
        C_new = C * cj + 1j * D * n_j * sj
        D_new = 1j * C * sj / n_j + D * cj
        A, B, C, D = A_new, B_new, C_new, D_new

    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    # Reflectance: r = (n_0 * (A + B * n_s) - (C + D * n_s)) / (n_0 * (A + B * n_s) + (C + D * n_s))
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)

def design(problem):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin for 150s limit

    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]
    
    # Precompute quarter-wave thicknesses at center wavelength for initialization
    qw = {}
    for m, n in materials_dict.items():
        qw[m] = wl_center / (4.0 * n)

    best_result = None
    best_score = float('inf')
    
    # Simulated Annealing strategy:
    # State: (materials_list, thicknesses_list)
    # Moves:
    #   1. Change thickness of a random layer (Gaussian perturbation)
    #   2. Change material of a random layer (L<->H)
    #   3. Insert a new layer at a random position
    #   4. Delete a random layer (if len > 1)
    #   5. Swap two adjacent layers
    
    def get_score(mats, ths):
        if not mats:
            return _mean_reflectance(mats, ths, problem)
        total = sum(ths)
        if total > max_total:
            # Penalty for violating total thickness constraint
            # Scale down to valid and add penalty
            scale = max_total / total
            ths_valid = [t * scale for t in ths]
            return _mean_reflectance(mats, ths_valid, problem) + 0.1 * (total - max_total) / max_total
        # Check individual layer constraints
        for t in ths:
            if t < 0 or t > max_layer:
                return 1000.0
        return _mean_reflectance(mats, ths, problem)

    # Initialize with a reasonable starting point: alternating L/H, quarter-wave
    N_init = 9
    mats_init = []
    ths_init = []
    for i in range(N_init):
        m = 'L' if i % 2 == 0 else 'H'
        mats_init.append(m)
        ths_init.append(qw[m])
    
    # Ensure initial state is valid
    total_init = sum(ths_init)
    if total_init > max_total:
        scale = max_total / total_init
        ths_init = [t * scale for t in ths_init]
    
    cur_mats = mats_init.copy()
    cur_ths = ths_init.copy()
    cur_score = get_score(cur_mats, cur_ths)
    
    if cur_score < best_score:
        best_score = cur_score
        best_result = (cur_mats.copy(), cur_ths.copy())
    
    # Annealing parameters
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.995
    T = T_start
    
    # Number of iterations per temperature step
    iters_per_step = 20
    
    step = 0
    while time.time() < deadline:
        if T <= T_end:
            break
        
        for _ in range(iters_per_step):
            if time.time() > deadline:
                break
            
            # Choose a move
            move_type = random.randint(0, 4)
            new_mats = cur_mats.copy()
            new_ths = cur_ths.copy()
            
            if move_type == 0:
                # Change thickness
                if not cur_ths:
                    continue
                idx = random.randint(0, len(cur_ths) - 1)
                # Gaussian perturbation
                sigma = 20.0
                new_ths[idx] = cur_ths[idx] + random.gauss(0, sigma)
                # Clip to [0, max_layer]
                new_ths[idx] = max(0.0, min(max_layer, new_ths[idx]))
                
            elif move_type == 1:
                # Change material
                if not cur_mats:
                    continue
                idx = random.randint(0, len(cur_mats) - 1)
                new_mats[idx] = 'H' if cur_mats[idx] == 'L' else 'L'
                
            elif move_type == 2:
                # Insert layer
                if len(cur_mats) >= max_layers:
                    continue
                pos = random.randint(0, len(cur_mats))
                m = random.choice(['L', 'H'])
                d = random.uniform(20, 100)
                new_mats.insert(pos, m)
                new_ths.insert(pos, d)
                
            elif move_type == 3:
                # Delete layer
                if len(cur_mats) <= 1:
                    continue
                pos = random.randint(0, len(cur_mats) - 1)
                new_mats.pop(pos)
                new_ths.pop(pos)
                
            else:
                # Swap adjacent layers
                if len(cur_mats) < 2:
                    continue
                pos = random.randint(0, len(cur_mats) - 2)
                new_mats[pos], new_mats[pos+1] = new_mats[pos+1], new_mats[pos]
                new_ths[pos], new_ths[pos+1] = new_ths[pos+1], new_ths[pos]
            
            # Evaluate new state
            new_score = get_score(new_mats, new_ths)
            
            # Acceptance criterion
            delta = new_score - cur_score
            if delta < 0 or random.random() < np.exp(-delta / T):
                cur_mats = new_mats
                cur_ths = new_ths
                cur_score = new_score
                
                if cur_score < best_score:
                    best_score = cur_score
                    best_result = (cur_mats.copy(), cur_ths.copy())
        
        # Cool down
        T *= cooling_rate
        step += 1
    
    # Final validation
    if best_result is None:
        # Fallback: simple 1-layer AR
        return ["L"], [qw["L"]]
    
    mats, ths = best_result
    # Ensure validity
    total = sum(ths)
    if total > max_total:
        scale = max_total / total
        ths = [t * scale for t in ths]
    ths = [max(0.0, min(max_layer, t)) for t in ths]
    
    # Final score check
    final_score = _mean_reflectance(mats, ths, problem)
    if final_score < best_score:
        best_score = final_score
        best_result = (mats, ths)
    
    return list(best_result[0]), list(best_result[1])
