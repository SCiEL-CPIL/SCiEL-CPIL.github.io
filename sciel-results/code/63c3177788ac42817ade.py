"""Broadband antireflection coating design via sequence grid-search + L-BFGS-B."""
import time
import numpy as np
from itertools import product

def _fast_mean_R(labels, thicknesses, n_amb, n_sub, n_L, n_H, band_nm):
    """Compute mean reflectance in percent over 401 equally spaced wavelengths (vectorized)."""
    lam_min, lam_max = band_nm
    wavelengths = np.linspace(lam_min, lam_max, 401)
    n_wl = len(wavelengths)
    
    if len(labels) == 0:
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return float(np.abs(r) ** 2 * 100.0)
    
    n_mat = np.array([n_L if l == 'L' else n_H for l in labels], dtype=float)
    d = np.array(thicknesses, dtype=float)
    
    # Precompute for each layer: n_i, and the phase factor will depend on wavelength
    # M_total = prod(M_i)
    # M_i = [[cos(d_i), i*sin(d_i)/n_i], [i*n_i*sin(d_i), cos(d_i)]] where d_i = 2*pi*n_i*t_i/lambda
    
    # For efficiency, compute at all wavelengths at once
    # delta[i, w] = 2*pi*n_i*t_i / lambda[w]
    delta = 2.0 * np.pi * np.outer(n_mat * d, 1.0 / wavelengths)  # (n_layers, n_wl)
    
    c = np.cos(delta)  # (n_layers, n_wl)
    s = np.sin(delta)  # (n_layers, n_wl)
    
    # Initialize M_total = I
    M00 = np.ones(n_wl, dtype=complex)
    M01 = np.zeros(n_wl, dtype=complex)
    M10 = np.zeros(n_wl, dtype=complex)
    M11 = np.ones(n_wl, dtype=complex)
    
    for i in range(len(labels)):
        a = c[i]
        b = 1j * s[i] / n_mat[i]
        cc = 1j * n_mat[i] * s[i]
        dd = c[i]
        
        new_M00 = M00 * a + M01 * cc
        new_M01 = M00 * b + M01 * dd
        new_M10 = M10 * a + M11 * cc
        new_M11 = M10 * b + M11 * dd
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    Y = M10 / M00
    r = (n_amb - Y) / (n_amb + Y)
    R = np.abs(r) ** 2 * 100.0
    return float(np.mean(R))


def design(problem):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    centre = 0.5 * (band_nm[0] + band_nm[1])
    
    def _obj(thicknesses):
        """Objective for L-BFGS-B: mean reflectance + penalty for constraint violations."""
        t = np.clip(thicknesses, 0.0, max_layer_nm)
        total = np.sum(t)
        penalty = 0.0
        if total > max_total_nm:
            penalty = 100.0 * (total - max_total_nm) / max_total_nm
        val = _fast_mean_R(labels, list(t), n_amb, n_sub, n_L, n_H, band_nm)
        return val + penalty
    
    def _is_feasible(t):
        return all(0.0 <= x <= max_layer_nm for x in t) and sum(t) <= max_total_nm + 1e-9
    
    best_labels = None
    best_thicknesses = None
    best_score = float('inf')
    
    # --- Generate candidate sequences ---
    # Include structured and random sequences
    candidate_sequences = []
    
    # Quarter-wave designs (good starting points)
    for n_l in range(1, min(max_layers + 1, 13)):
        # Alternating starting with L
        seq1 = ['L' if i % 2 == 0 else 'H' for i in range(n_l)]
        candidate_sequences.append(seq1)
        # Alternating starting with H
        seq2 = ['H' if i % 2 == 0 else 'L' for i in range(n_l)]
        candidate_sequences.append(seq2)
    
    # Add some random sequences
    import random
    random.seed(42)
    for _ in range(20):
        n_l = random.randint(3, min(max_layers, 12))
        seq = [random.choice(['L', 'H']) for _ in range(n_l)]
        candidate_sequences.append(seq)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_sequences = []
    for seq in candidate_sequences:
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            unique_sequences.append(seq)
    
    # --- Optimize each sequence using multi-start L-BFGS-B ---
    from scipy.optimize import minimize
    
    for seq in unique_sequences:
        if time.time() > deadline - 5.0:
            break
        
        n_l = len(seq)
        if n_l > max_layers:
            continue
        
        # Initial guesses
        initial_guesses = []
        
        # Quarter-wave thicknesses
        qw = [centre / (4.0 * (n_L if l == 'L' else n_H)) for l in seq]
        if sum(qw) <= max_total_nm and all(0 <= t <= max_layer_nm for t in qw):
            initial_guesses.append(qw)
        
        # Scaled quarter-wave
        total_qw = sum(qw)
        if total_qw > 0:
            scale = min(1.0, max_total_nm / total_qw)
            scaled = [t * scale for t in qw]
            if all(0 <= t <= max_layer_nm for t in scaled):
                initial_guesses.append(scaled)
        
        # Random initial guesses
        for _ in range(3):
            guess = []
            for l in seq:
                n_mat = n_L if l == 'L' else n_H
                t = centre / (4.0 * n_mat) * random.uniform(0.5, 1.5)
                t = max(0.0, min(t, max_layer_nm))
                guess.append(t)
            total = sum(guess)
            if total > max_total_nm:
                s = max_total_nm / total
                guess = [t * s for t in guess]
            if all(0 <= t <= max_layer_nm for t in guess):
                initial_guesses.append(guess)
        
        if not initial_guesses:
            # Fallback: very thin layers
            initial_guesses.append([max_total_nm / n_l / 4.0] * n_l)
        
        bounds = [(0.0, max_layer_nm)] * n_l
        constraints = [{'type': 'ineq', 'fun': lambda t: max_total_nm - np.sum(t)}]
        
        for x0 in initial_guesses:
            if time.time() > deadline - 5.0:
                break
            
            try:
                result = minimize(
                    _obj,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                
                if result.success or result.fun < best_score:
                    t_opt = np.clip(result.x, 0.0, max_layer_nm)
                    if _is_feasible(t_opt):
                        score = _fast_mean_R(seq, list(t_opt), n_amb, n_sub, n_L, n_H, band_nm)
                        if score < best_score:
                            best_score = score
                            best_labels = seq[:]
                            best_thicknesses = list(t_opt)
            except Exception:
                pass
    
    # --- Final constraint enforcement ---
    if best_labels is None:
        # Fallback: single quarter-wave L layer
        best_labels = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]
        best_score = _fast_mean_R(best_labels, best_thicknesses, n_amb, n_sub, n_L, n_H, band_nm)
    
    best_thicknesses = [max(0.0, min(t, max_layer_nm)) for t in best_thicknesses]
    if sum(best_thicknesses) > max_total_nm:
        scale = max_total_nm / sum(best_thicknesses)
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    return best_labels, best_thicknesses
