"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 105
    
    n = 24
    n_samples = 2048
    theta = 2 * np.pi * np.arange(n_samples) / n_samples
    z_powers = np.exp(1j * theta)[None, :] ** np.arange(n)[:, None]
    
    def fast_score(coeffs):
        vals = np.dot(coeffs, z_powers)
        return np.max(np.abs(vals))
    
    # Start from Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:24], dtype=float)
    best_score = fast_score(best_coeffs)
    
    current = best_coeffs.copy()
    current_score = best_score
    
    random.seed(42)
    temp = 0.05
    cooling = 0.99
    
    step = 0
    while time.time() < deadline:
        step += 1
        idx = random.randint(0, n - 1)
        new_coeffs = current.copy()
        new_coeffs[idx] = -new_coeffs[idx]
        
        new_score = fast_score(new_coeffs)
        delta = new_score - current_score
        
        if delta < 0 or random.random() < np.exp(-delta / temp):
            current = new_coeffs
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_coeffs = current.copy()
        
        temp *= cooling
    
    # Local search: try all single flips
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(n):
            new_coeffs = best_coeffs.copy()
            new_coeffs[idx] = -new_coeffs[idx]
            new_score = fast_score(new_coeffs)
            if new_score < best_score - 1e-12:
                best_score = new_score
                best_coeffs = new_coeffs
                improved = True
    
    return list(best_coeffs.astype(int))
# EVOLVE-BLOCK-END
