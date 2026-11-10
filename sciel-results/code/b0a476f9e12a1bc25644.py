"""Broadband antireflection coating design using Differential Evolution."""
import numpy as np
import time
from scipy.optimize import differential_evolution

# EVOLVE-BLOCK-START

def _reflectance_for_design(labels, thicknesses, problem, wavelengths):
    """
    Compute reflectance spectrum using characteristic matrix method.
    """
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    n_layers = problem["materials"]
    N = len(thicknesses)
    
    layer_indices = np.array([n_layers[lbl] for lbl in labels])
    
    reflectance = np.zeros(len(wavelengths), dtype=float)
    
    for i, lam in enumerate(wavelengths):
        M11 = 1.0 + 0j
        M12 = 0.0 + 0j
        M21 = 0.0 + 0j
        M22 = 1.0 + 0j
        
        for j in range(N):
            nj = layer_indices[j]
            dj = thicknesses[j]
            delta = 2.0 * np.pi * nj * dj / lam
            
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            
            m11 = cos_d
            m12 = 1j * sin_d / nj
            m21 = 1j * nj * sin_d
            m22 = cos_d
            
            new_M11 = M11 * m11 + M12 * m21
            new_M12 = M11 * m12 + M12 * m22
            new_M21 = M21 * m11 + M22 * m21
            new_M22 = M21 * m12 + M22 * m22
            
            M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
        
        numerator = n0 * M11 + M12 - M21 - M22
        denominator = n0 * M11 + M12 + M21 + M22
        r = numerator / denominator
        reflectance[i] = np.abs(r) ** 2
    
    return reflectance


def _objective(x, problem, wavelengths, deadline):
    """
    Objective function for differential evolution.
    x: array of 2*N values: [label_0, thick_0, label_1, thick_1, ...]
    """
    if time.time() > deadline[0]:
        return 100.0
    
    N = len(x) // 2
    labels = []
    thicknesses = []
    
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    for i in range(N):
        lbl_code = x[2*i]
        thick = x[2*i+1]
        
        if lbl_code < 0.5:
            labels.append('L')
        else:
            labels.append('H')
        thicknesses.append(thick)
    
    total = sum(thicknesses)
    if total > max_total:
        return 100.0 + (total - max_total) / max_total
    for t in thicknesses:
        if t < 0 or t > max_layer:
            return 100.0
    
    refl = _reflectance_for_design(labels, thicknesses, problem, wavelengths)
    return 100.0 * np.mean(refl)


def design(problem):
    """
    Design a broadband antireflection coating using Differential Evolution.
    """
    start_time = time.time()
    deadline_val = start_time + 50.0
    deadline = [deadline_val]
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    materials = problem["materials"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wavelengths = np.linspace(band[0], band[1], 401)
    
    N = max_layers
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, max_layer_nm))
    
    center_wl = 0.5 * (band[0] + band[1])
    
    # Initial heuristic: alternating quarter-wave
    init_x = np.zeros(2 * N)
    for i in range(N):
        if i % 2 == 0:
            init_x[2*i] = 0.0
            init_x[2*i+1] = center_wl / (4.0 * materials['L'])
        else:
            init_x[2*i] = 1.0
            init_x[2*i+1] = center_wl / (4.0 * materials['H'])
    
    best_result = None
    best_score = 100.0
    
    labels = ['L' if init_x[2*i] < 0.5 else 'H' for i in range(N)]
    thicknesses = list(init_x[1::2])
    if sum(thicknesses) <= max_total_nm:
        refl = _reflectance_for_design(labels, thicknesses, problem, wavelengths)
        score = 100.0 * np.mean(refl)
        if score < best_score:
            best_score = score
            best_result = (labels[:], thicknesses[:])
    
    try:
        result = differential_evolution(
            _objective,
            bounds,
            args=(problem, wavelengths, deadline),
            maxiter=300,
            popsize=12,
            tol=1e-5,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            polish=False
        )
        
        x_best = result.x
        labels = ['L' if x_best[2*i] < 0.5 else 'H' for i in range(N)]
        thicknesses = list(x_best[1::2])
        
        if sum(thicknesses) <= max_total_nm:
            refl = _reflectance_for_design(labels, thicknesses, problem, wavelengths)
            score = 100.0 * np.mean(refl)
            if score < best_score:
                best_score = score
                best_result = (labels[:], thicknesses[:])
        
    except Exception:
        pass
    
    if best_result is None:
        center = 0.5 * (band[0] + band[1])
        return ["L"], [center / (4.0 * materials["L"])]
    
    labels, thicknesses = best_result
    
    final_labels = []
    final_thicknesses = []
    for lbl, t in zip(labels, thicknesses):
        if t > 1e-6:
            final_labels.append(lbl)
            final_thicknesses.append(t)
    
    if not final_labels:
        center = 0.5 * (band[0] + band[1])
        return ["L"], [center / (4.0 * materials["L"])]
    
    return final_labels, final_thicknesses

# EVOLVE-BLOCK-END
