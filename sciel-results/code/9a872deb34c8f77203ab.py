"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random
from scipy.optimize import minimize_scalar

# EVOLVE-BLOCK-START
import sys
import math

def _eval_poly(coeffs, z):
    """Evaluate polynomial P(z) = sum c_k z^k at complex z using Horner's method."""
    result = 0.0
    for c in reversed(coeffs):
        result = result * z + c
    return result

def _soft_max(log_mags, beta=10.0):
    """
    Compute soft-max of magnitudes using log-sum-exp for smoothness.
    This provides a differentiable surrogate for the hard max.
    """
    # log_mags is an array of log(|P(z)|) values
    max_log = np.max(log_mags)
    # For numerical stability
    if max_log > 1000:
        log_mags = log_mags - max_log
        max_log = 0
    
    # log-sum-exp: log(sum(exp(x_i))) = max + log(sum(exp(x_i - max)))
    shifted = log_mags - max_log
    sum_exp = np.sum(np.exp(shifted))
    if sum_exp < 1e-300:
        return max_log + 0.0
    return max_log + np.log(sum_exp)

def _sup_norm_smooth(coeffs, num_samples=1024, beta=10.0):
    """
    Compute a smooth approximation of sup norm using log-sum-exp.
    This is differentiable and better for optimization.
    """
    n = len(coeffs)
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    
    # Vectorized evaluation using Horner's method
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * zs + c
        
    mags = np.abs(vals)
    
    # Use log-sum-exp for smooth max approximation
    # beta controls smoothness: higher beta -> closer to hard max
    # We use: soft_max = (1/beta) * log(sum(exp(beta * log(mag))))
    # But we need to be careful with zeros and very small values
    
    # Clip to avoid log(0)
    mags_clipped = np.clip(mags, 1e-10, None)
    log_mags = np.log(mags_clipped)
    
    # Soft max approximation
    soft_max_val = (1.0 / beta) * _soft_max(log_mags * beta)
    
    return float(soft_max_val)

def _sup_norm_fast(coeffs, num_samples=1024):
    """
    Compute sup norm of P(z) over |z|=1 via dense sampling.
    This is used for the local search steps for speed.
    """
    n = len(coeffs)
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    
    # Vectorized evaluation using Horner's method
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

def _score_smooth(coeffs, num_samples=1024, beta=10.0):
    """Score using smooth surrogate: soft_max |P(z)| / sqrt(n)"""
    n = len(coeffs)
    return _sup_norm_smooth(coeffs, num_samples=num_samples, beta=beta) / np.sqrt(n)

def _local_search_smooth(coeffs, time_limit=5.0, beta=10.0):
    """
    Local search using smooth surrogate for better guidance.
    Try flipping single coefficients and accept if smooth score improves.
    """
    start_time = time.time()
    current = list(coeffs)
    n = len(current)
    current_score = _score_smooth(current, num_samples=1024, beta=beta)
    
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
            new_score = _score_smooth(current, num_samples=1024, beta=beta)
            if new_score < current_score - 1e-12:
                current_score = new_score
                improved = True
            else:
                current[i] = old_val
                
    return current, current_score

def _simulated_annealing(coeffs, time_limit=10.0, beta=10.0):
    """
    Simulated annealing to escape local minima.
    Uses smooth surrogate for fast evaluation.
    """
    start_time = time.time()
    current = list(coeffs)
    n = len(current)
    current_score = _score_smooth(current, num_samples=1024, beta=beta)
    
    # Initial and final temperatures
    T_start = 0.1
    T_end = 0.001
    total_time = time_limit
    
    best_coeffs = list(current)
    best_score = current_score
    
    elapsed = 0.0
    step = 0
    
    while time.time() - start_time < time_limit:
        # Compute current temperature (exponential cooling)
        progress = min(1.0, (time.time() - start_time) / total_time)
        T = T_start * (T_end / T_start) ** progress
        
        # Random move: flip a random coefficient
        i = random.randint(0, n - 1)
        old_val = current[i]
        current[i] = -old_val
        
        new_score = _score_smooth(current, num_samples=1024, beta=beta)
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_coeffs = list(current)
        else:
            current[i] = old_val
        
        step += 1
        
    return best_coeffs, best_score

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
    
    # Phase 1: Smooth local search from GRS
    coeffs, smooth_score = _local_search_smooth(best_coeffs, time_limit=8.0, beta=10.0)
    hard_score = _score(coeffs, num_samples=1024)
    if hard_score < best_score:
        best_score = hard_score
        best_coeffs = coeffs
    
    # Phase 2: Simulated annealing to escape local minima
    remaining = deadline - time.time()
    sa_time = min(20.0, remaining * 0.4)
    if sa_time > 2.0:
        sa_coeffs, sa_score = _simulated_annealing(best_coeffs, time_limit=sa_time, beta=10.0)
        sa_hard_score = _score(sa_coeffs, num_samples=1024)
        if sa_hard_score < best_score:
            best_score = sa_hard_score
            best_coeffs = sa_coeffs
    
    # Phase 3: Random restarts with smooth local search
    num_restarts = 0
    while time.time() < deadline - 25:
        # Random initialization
        coeffs = [random.choice([-1, 1]) for _ in range(24)]
        # Smooth local search
        coeffs, smooth_score = _local_search_smooth(coeffs, time_limit=3.0, beta=10.0)
        hard_score = _score(coeffs, num_samples=1024)
        if hard_score < best_score:
            best_score = hard_score
            best_coeffs = coeffs
            num_restarts += 1
    
    # Phase 4: Final polish with greedy hard-max local search
    remaining = deadline - time.time()
    if remaining > 5.0:
        # Greedy local search with hard max for final polish
        current = list(best_coeffs)
        n = len(current)
        current_score = _score(current, num_samples=1024)
        
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            for i in range(n):
                if time.time() > deadline - 15:
                    break
                old_val = current[i]
                current[i] = -old_val
                new_score = _score(current, num_samples=1024)
                if new_score < current_score - 1e-12:
                    current_score = new_score
                    improved = True
                else:
                    current[i] = old_val
                    
        if current_score < best_score:
            best_score = current_score
            best_coeffs = current
    
    # Final refinement of the best candidate
    # We use high-precision evaluation to get the true score
    final_score = _score(best_coeffs, refine=True)
    
    return best_coeffs

# EVOLVE-BLOCK-END
