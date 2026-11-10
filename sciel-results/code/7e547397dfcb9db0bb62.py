"""Broadband antireflection coating design using multi-start local search with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _reflectance_vectorized(labels, thicknesses, materials, ambient_index, substrate_index, wavelengths):
    """Compute reflectance spectrum using characteristic-matrix method (vectorized over wavelengths)."""
    n_L = materials['L']
    n_H = materials['H']
    n_s = substrate_index
    n_0 = ambient_index
    
    N = len(wavelengths)
    m00 = np.ones(N, dtype=complex)
    m01 = np.zeros(N, dtype=complex)
    m10 = np.zeros(N, dtype=complex)
    m11 = np.ones(N, dtype=complex)
    
    for j, (label, d) in enumerate(zip(labels, thicknesses)):
        n_j = n_L if label == 'L' else n_H
        delta = 2.0 * np.pi * n_j * d / wavelengths
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        m00_j = cos_d
        m01_j = 1j * sin_d / n_j
        m10_j = 1j * n_j * sin_d
        m11_j = cos_d
        
        nm00 = m00 * m00_j + m01 * m10_j
        nm01 = m00 * m01_j + m01 * m11_j
        nm10 = m10 * m00_j + m11 * m10_j
        nm11 = m10 * m01_j + m11 * m11_j
        
        m00, m01, m10, m11 = nm00, nm01, nm10, nm11
    
    Y = (m11 + n_s * m10) / (m01 + n_s * m00)
    R = np.abs((Y - n_0) / (Y + n_0)) ** 2
    return R * 100.0


def _local_search(labels, x0, materials, ambient_index, substrate_index, wavelengths, 
                  max_total_nm, max_layer_nm, timeout_check, max_iter=50):
    """Local search with adaptive step sizes."""
    n = len(x0)
    x = np.array(x0, dtype=float).copy()
    f_current = _reflectance_vectorized(labels, x, materials, ambient_index, substrate_index, wavelengths).mean()
    if np.sum(x) > max_total_nm:
        f_current = 100.0
    
    # Adaptive step sizes per dimension
    steps = np.full(n, 10.0)
    
    for iteration in range(max_iter):
        if timeout_check():
            break
        
        improved = False
        for i in range(n):
            if timeout_check():
                break
            
            for direction in [-1, 1]:
                x_trial = x.copy()
                x_trial[i] += direction * steps[i]
                x_trial[i] = max(0.0, min(max_layer_nm, x_trial[i]))
                
                if np.sum(x_trial) > max_total_nm:
                    continue
                
                f_trial = _reflectance_vectorized(labels, x_trial, materials, ambient_index, 
                                                   substrate_index, wavelengths).mean()
                if f_trial < f_current:
                    x = x_trial
                    f_current = f_trial
                    steps[i] *= 1.5
                    improved = True
                    break
            else:
                steps[i] *= 0.7
        
        if not improved:
            break
    
    return x, f_current


def design(problem):
    """Design broadband AR coating using multi-start local search."""
    start_time = time.time()
    deadline = start_time + 125.0  # 25s safety margin before 150s kill
    
    materials = problem['materials']
    ambient_index = problem['ambient_index']
    substrate_index = problem['substrate_index']
    band = problem['band_nm']
    max_layers = problem['max_layers']
    max_total_nm = problem['max_total_nm']
    max_layer_nm = problem['max_layer_nm']
    
    wavelengths = np.linspace(band[0], band[1], 401)
    centre = 0.5 * (band[0] + band[1])
    
    def check_timeout():
        return time.time() > deadline
    
    best_score = float('inf')
    best_labels = None
    best_thick = None
    
    # Generate diverse initial configurations
    configs = []
    
    # Quarter-wave stacks (even layers, alternating)
    for n in range(2, min(max_layers + 1, 15), 2):
        dL = centre / (4.0 * materials['L'])
        dH = centre / (4.0 * materials['H'])
        
        for start_char in ['L', 'H']:
            labels = [start_char if i % 2 == 0 else ('H' if start_char == 'L' else 'L') for i in range(n)]
            thicks = [centre / (4.0 * materials[l]) for l in labels]
            if sum(thicks) <= max_total_nm:
                configs.append((labels, thicks))
    
    # Geometric progression thicknesses for broad spectral coverage
    for n in [4, 6, 8, 10, 12]:
        if n > max_layers:
            continue
        labels = ['L' if i % 2 == 0 else 'H' for i in range(n)]
        
        # Try different geometric ratios
        for ratio in [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]:
            base = 50.0
            thicks = [base * (ratio ** i) for i in range(n)]
            total = sum(thicks)
            if total > max_total_nm:
                scale = max_total_nm / total
                thicks = [t * scale for t in thicks]
            thicks = [max(0.0, min(max_layer_nm, t)) for t in thicks]
            if sum(thicks) <= max_total_nm and sum(thicks) > 0:
                configs.append((labels, thicks))
    
    # Linear progression thicknesses
    for n in [4, 6, 8, 10]:
        if n > max_layers:
            continue
        labels = ['L' if i % 2 == 0 else 'H' for i in range(n)]
        
        for start_t, end_t in [(30, 200), (50, 150), (20, 250), (40, 180)]:
            thicks = list(np.linspace(start_t, end_t, n))
            thicks = [max(0.0, min(max_layer_nm, t)) for t in thicks]
            if sum(thicks) <= max_total_nm:
                configs.append((labels, thicks))
    
    # Pseudo-random deterministic configs
    rng = np.random.RandomState(12345)
    for _ in range(15):
        n = rng.randint(4, min(max_layers + 1, 13))
        labels = [rng.choice(['L', 'H']) for _ in range(n)]
        thicks = [rng.uniform(30, min(max_layer_nm, 350)) for _ in range(n)]
        total = sum(thicks)
        if total > max_total_nm:
            scale = max_total_nm / total
            thicks = [t * scale for t in thicks]
        thicks = [max(0.0, min(max_layer_nm, t)) for t in thicks]
        if sum(thicks) <= max_total_nm and sum(thicks) > 0:
            configs.append((labels, thicks))
    
    # Sort configs by initial reflectance to prioritize promising ones
    def initial_score(config):
        labels, thicks = config
        return _reflectance_vectorized(labels, np.array(thicks), materials, 
                                       ambient_index, substrate_index, wavelengths).mean()
    
    configs.sort(key=initial_score)
    
    # Optimize each configuration
    for labels, thicks in configs:
        if check_timeout():
            break
        
        x0 = np.array(thicks, dtype=float)
        x_opt, f_opt = _local_search(labels, x0, materials, ambient_index, substrate_index,
                                      wavelengths, max_total_nm, max_layer_nm, check_timeout,
                                      max_iter=40)
        
        if f_opt < best_score:
            best_score = f_opt
            best_labels = labels[:]
            best_thick = [float(t) for t in x_opt]
    
    # Final refinement on best solution
    if best_labels is not None and not check_timeout():
        labels = best_labels
        thicks = np.array(best_thick, dtype=float)
        x_opt, f_opt = _local_search(labels, thicks, materials, ambient_index, substrate_index,
                                      wavelengths, max_total_nm, max_layer_nm, check_timeout,
                                      max_iter=60)
        if f_opt < best_score:
            best_score = f_opt
            best_thick = [float(t) for t in x_opt]
    
    # Fallback
    if best_labels is None:
        return ["L"], [centre / (4.0 * materials['L'])]
    
    # Validate and clamp
    best_thick = [max(0.0, min(max_layer_nm, t)) for t in best_thick]
    total = sum(best_thick)
    if total > max_total_nm:
        scale = max_total_nm / total
        best_thick = [t * scale for t in best_thick]
    
    return best_labels, best_thick
# EVOLVE-BLOCK-END
