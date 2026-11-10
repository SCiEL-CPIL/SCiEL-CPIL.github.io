"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random
from scipy.optimize import minimize_scalar

# EVOLVE-BLOCK-START
import sys

def _eval_poly(coeffs, z):
    """Evaluate polynomial P(z) = sum c_k z^k at complex z using Horner's method."""
    result = 0.0
    for c in reversed(coeffs):
        result = result * z + c
    return result

def _sup_norm_fast(coeffs, num_samples=1024):
    """
    Compute sup norm of P(z) over |z|=1 via dense sampling.
    This is used for the local search steps for speed.
    """
    n = len(coeffs)
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    
    # Vectorized evaluation using Horner's method
    # coeffs is a list of real numbers, zs is a complex array
    # We want P(zs) for all zs.
    # P(z) = c0 + c1*z + ... + c_{n-1}*z^{n-1}
    # We can compute this by iterating coefficients from high to low.
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * zs + c
        
    mag = np.abs(vals)
    return float(np.max(mag))

def _sup_norm_refined(coeffs, num_samples=4096):
    """
    Compute sup norm with golden-section refinement of top peaks.
    This is used for the final score to match evaluator accuracy.
    """
    n = len(coeffs)
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * zs + c
        
    mag = np.abs(vals)
    
    # Find top peaks for refinement
    top_k = 5
    if num_samples > top_k:
        peak_indices = np.argsort(mag)[-top_k:][::-1]
    else:
        peak_indices = np.arange(num_samples)
        
    best_val = float(np.max(mag))
    window = 2 * np.pi / num_samples
    
    for idx in peak_indices:
        center = angles[idx]
        # Ensure center is within [0, 2pi)
        center = center % (2 * np.pi)
        
        def obj(theta):
            z = np.exp(1j * theta)
            return -abs(_eval_poly(coeffs, z))
        
        # Bounded minimization (maximization of -abs)
        # We search in a small window around the peak
        lower = center - window
        upper = center + window
        
        # Handle wrapping around 2pi
        # If lower < 0 or upper > 2pi, we might need to be careful, 
        # but minimize_scalar handles bounds. 
        # However, the function is periodic. 
        # For simplicity, we just search in the local window.
        try:
            res = minimize_scalar(obj, bounds=(lower, upper), 
                                  method='bounded', options={'xatol': 1e-10})
            if res.success:
                peak_mag = -res.fun
                if peak_mag > best_val:
                    best_val = peak_mag
        except Exception:
            pass
            
    return best_val

def _score(coeffs, num_samples=1024, refine=False):
    """Score: max |P(z)| / sqrt(n)"""
    n = len(coeffs)
    if refine:
        return _sup_norm_refined(coeffs, num_samples=4096) / np.sqrt(n)
    else:
        return _sup_norm_fast(coeffs, num_samples=num_samples) / np.sqrt(n)

def _local_search(coeffs, time_limit=5.0):
    """
    Greedy local search: try flipping single coefficients.
    Returns the best coeffs found and its fast score.
    """
    start_time = time.time()
    current = list(coeffs)
    n = len(current)
    current_score = _score(current, num_samples=1024)
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
            
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            old_val = current[i]
            current[i] = -old_val
            new_score = _score(current, num_samples=1024)
            if new_score < current_score - 1e-12:
                current_score = new_score
                improved = True
            else:
                current[i] = old_val
                
    return current, current_score

def _grs_sequence(length=24):
    """Generate Golay-Rudin-Shapiro sequence truncated to given length."""
    p, q = [1], [1]
    while len(p) < length:
        p, q = p + q, p + [-c for c in q]
    return p[:length]

def construct():
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    # Initialize with GRS
    best_coeffs = _grs_sequence(24)
    best_score = _score(best_coeffs, num_samples=1024)
    
    # Run local search from GRS
    coeffs, score = _local_search(best_coeffs, time_limit=10.0)
    if score < best_score:
        best_score = score
        best_coeffs = coeffs
    
    # Random restarts
    # The search space is 2^24. 
    # Each local search takes a few seconds.
    # We have ~100s left.
    # Let's try as many restarts as possible.
    
    # To speed up, we can use a slightly lower sample count for the 
    # "fast" score during local search if needed, but 1024 is fast enough.
    
    num_restarts = 0
    while time.time() < deadline - 20:
        # Random initialization
        coeffs = [random.choice([-1, 1]) for _ in range(24)]
        # Local search
        coeffs, score = _local_search(coeffs, time_limit=5.0)
        if score < best_score:
            best_score = score
            best_coeffs = coeffs
            num_restarts += 1
            # Optional: If we find a very good score, we can stop or continue
            # The optimum is 1.2583758. 
            # If we get close, we might still want to verify with refinement.
            
    # Final refinement of the best candidate
    # We use high-precision evaluation to get the true score
    final_score = _score(best_coeffs, refine=True)
    
    # If the fast score was misleading, the refined score might be higher.
    # But we want to minimize the refined score.
    # The local search minimized the fast score. 
    # Ideally, the fast score and refined score are correlated.
    
    return best_coeffs

# EVOLVE-BLOCK-END
