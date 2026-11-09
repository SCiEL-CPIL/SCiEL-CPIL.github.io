"""Broadband AR coating design via simulated annealing."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def _reflectance(materials, thicknesses, wavelengths, n0, ns):
    """Compute reflectance spectrum using characteristic matrix method."""
    n_layers = len(materials)
    # Build layer indices
    n_layers_arr = np.array([materials[m] for m in materials])
    d = np.array(thicknesses)
    # wavelengths: (N,)
    # delta: (N, n_layers)
    delta = 2.0 * np.pi * n_layers_arr[None, :] * d[None, :] / wavelengths[:, None]
    cos_d = np.cos(delta)
    sin_d = np.sin(delta)
    
    # Start with identity matrix for each wavelength
    # M_total = (N, 2, 2) complex
    M = np.zeros((len(wavelengths), 2, 2), dtype=np.complex128)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = 1.0
    
    for i in range(n_layers):
        # M_i = [[cos, i*sin/n], [i*n*sin, cos]]
        M_i = np.zeros((len(wavelengths), 2, 2), dtype=np.complex128)
        M_i[:, 0, 0] = cos_d[:, i]
        M_i[:, 0, 1] = 1j * sin_d[:, i] / n_layers_arr[i]
        M_i[:, 1, 0] = 1j * n_layers_arr[i] * sin_d[:, i]
        M_i[:, 1, 1] = cos_d[:, i]
        # M = M @ M_i  (matrix multiplication along last two dims)
        M = M @ M_i
    
    # Admittance Y = (M11 + M12 * ns) / (M21 + M22 * ns)
    M11 = M[:, 0, 0]
    M12 = M[:, 0, 1]
    M21 = M[:, 1, 0]
    M22 = M[:, 1, 1]
    
    Y = (M11 + M12 * ns) / (M21 + M22 * ns)
    R = np.abs((Y - n0) / (Y + n0)) ** 2
    return R


def _mean_reflectance_pct(materials, thicknesses, problem):
    """Mean reflectance in percent over the band."""
    band = problem["band_nm"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    mat = problem["materials"]
    wl = np.linspace(band[0], band[1], 401)
    R = _reflectance(mat, thicknesses, wl, n0, ns)
    return float(np.mean(R) * 100.0)


def _is_valid(materials, thicknesses, problem):
    """Check constraints."""
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > problem["max_layers"]:
        return False
    for m in materials:
        if m not in problem["materials"]:
            return False
    for t in thicknesses:
        if t < 0 or t > problem["max_layer_nm"]:
            return False
    if sum(thicknesses) > problem["max_total_nm"] + 1e-6:
        return False
    return True


def _quarter_wave_thickness(n, center_wl):
    return center_wl / (4.0 * n)


def _sa_optimize(materials, thicknesses, problem, start_time, deadline):
    """Simulated annealing to optimize thicknesses for a fixed layer pattern."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    mat = problem["materials"]
    band = problem["band_nm"]
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    
    wl = np.linspace(band[0], band[1], 401)
    center_wl = 0.5 * (band[0] + band[1])
    
    # Initial thicknesses: quarter-wave at center
    if thicknesses is None:
        thicknesses = [_quarter_wave_thickness(mat[m], center_wl) for m in materials]
    
    n_layers = len(materials)
    n_mat = np.array([mat[m] for m in materials])
    
    current = np.array(thicknesses, dtype=np.float64)
    current_cost = _mean_reflectance_pct(materials, current, problem)
    best = current.copy()
    best_cost = current_cost
    
    # SA parameters
    T0 = 0.5  # initial temperature (in percent reflectance units)
    T_end = 0.001
    cooling = 0.995
    T = T0
    
    rng = np.random.default_rng(42)
    
    while T > T_end:
        if time.time() > deadline:
            break
        
        # Propose a move: perturb one or more thicknesses
        n_perturb = rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
        idx = rng.choice(n_layers, size=n_perturb, replace=False)
        step_scale = T * 50.0  # step size scales with temperature
        new = current.copy()
        for i in idx:
            new[i] += rng.normal(0, step_scale)
            new[i] = np.clip(new[i], 0.0, max_layer)
        
        # Check total thickness constraint
        if np.sum(new) > max_total:
            # Scale down if needed
            scale = max_total / np.sum(new)
            new = new * scale
        
        new_cost = _mean_reflectance_pct(materials, new, problem)
        
        delta = new_cost - current_cost
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current = new
            current_cost = new_cost
            if current_cost < best_cost:
                best = current.copy()
                best_cost = current_cost
        
        T *= cooling
    
    return materials, best, best_cost


def _try_pattern(pattern, problem, start_time, deadline):
    """Try optimizing a given layer pattern."""
    if time.time() > deadline:
        return None
    materials = list(pattern)
    thicknesses, cost = _sa_optimize(materials, None, problem, start_time, deadline)
    if thicknesses is not None and _is_valid(materials, thicknesses, problem):
        return materials, thicknesses, cost
    return None


def design(problem):
    """Design broadband AR coating using simulated annealing."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    mat = problem["materials"]
    band = problem["band_nm"]
    center_wl = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    
    # Candidate patterns: alternating L/H starting with L or H, various lengths
    # Also include some non-alternating patterns
    patterns = []
    
    # Alternating patterns
    for n in range(1, min(max_layers, 15) + 1):
        # Start with L
        pat_l = ['L' if i % 2 == 0 else 'H' for i in range(n)]
        patterns.append(pat_l)
        # Start with H
        if n > 1:
            pat_h = ['H' if i % 2 == 0 else 'L' for i in range(n)]
            patterns.append(pat_h)
    
    # Some specific known-good patterns for broadband AR
    # Classic 5-layer: L H L H L (or H L H L H)
    # 7-layer: L H L H L H L
    # Add some mixed patterns
    patterns.append(['L', 'H', 'L', 'H', 'L'])
    patterns.append(['H', 'L', 'H', 'L', 'H'])
    patterns.append(['L', 'H', 'L', 'H', 'L', 'H', 'L'])
    patterns.append(['H', 'L', 'H', 'L', 'H', 'L', 'H'])
    patterns.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])
    patterns.append(['L', 'L', 'H', 'H', 'L', 'L', 'H', 'H'])
    patterns.append(['H', 'H', 'L', 'L', 'H', 'H', 'L', 'L'])
    
    # Remove duplicates
    seen = set()
    unique_patterns = []
    for p in patterns:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique_patterns.append(p)
    
    best_result = None
    best_cost = float('inf')
    
    # Sort by number of layers (start with moderate ones that are likely good)
    # Actually, start with patterns known to be good for broadband AR
    # Prioritize: 5, 7, 9 layer alternating patterns first
    priority = []
    for n in [5, 7, 9, 3, 11, 13, 1, 2, 4, 6, 8, 10, 12, 14, 15]:
        for p in unique_patterns:
            if len(p) == n:
                priority.append(p)
    
    for pattern in priority:
        if time.time() > deadline:
            break
        result = _try_pattern(pattern, problem, start_time, deadline)
        if result is not None:
            materials, thicknesses, cost = result
            if cost < best_cost:
                best_cost = cost
                best_result = (materials, thicknesses)
    
    if best_result is None:
        # Fallback: single quarter-wave L layer
        return ["L"], [center_wl / (4.0 * mat["L"])]
    
    return best_result
# EVOLVE-BLOCK-END
