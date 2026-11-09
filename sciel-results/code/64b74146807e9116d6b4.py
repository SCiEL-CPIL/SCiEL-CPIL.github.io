"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.special import roots_genlaguerre

def _compute_laguerre_zeros(n, alpha=0):
    """Compute zeros of the generalized Laguerre polynomial L_n^{(alpha)}(x)."""
    try:
        roots, _ = roots_genlaguerre(n, alpha)
        real_roots = [float(r) for r in roots if r > 0]
        real_roots.sort()
        return real_roots
    except Exception:
        # Fallback: compute via numpy
        c = []
        for k in range(n + 1):
            val = ((-1) ** k) * math.comb(n + k, k) / math.factorial(k)
            c.append(val)
        c_rev = list(reversed(c))
        roots = np.roots(c_rev)
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-10 and r.real > 0]
        real_roots.sort()
        return real_roots

def _compute_c4_bound(roots, t_max=100.0, n_samples=20000):
    """
    Compute the C4 upper bound for the Cohn-Goncalves construction.
    Uses the L4/L2 norm ratio of the sum of exponentials.
    """
    roots = np.array(roots, dtype=float)
    n = len(roots)
    
    t = np.linspace(0, t_max, n_samples)
    
    # Sum of exponentials: S(t) = sum_i exp(-i * a_i * t)
    S = np.zeros_like(t, dtype=complex)
    for a in roots:
        S += np.exp(-1j * a * t)
    
    S2 = np.abs(S) ** 2
    S4 = np.abs(S) ** 4
    
    # Integrate using trapezoidal rule
    int_S2 = np.trapz(S2, t)
    int_S4 = np.trapz(S4, t)
    
    if int_S2 > 1e-30:
        bound = math.sqrt(int_S4 / int_S2)
    else:
        bound = float('inf')
    
    return bound

def _compute_c4_bound_fast(roots, t_max=50.0, n_samples=10000):
    """Faster version for initial screening."""
    roots = np.array(roots, dtype=float)
    
    t = np.linspace(0, t_max, n_samples)
    
    S = np.zeros_like(t, dtype=complex)
    for a in roots:
        S += np.exp(-1j * a * t)
    
    S2 = np.abs(S) ** 2
    S4 = np.abs(S) ** 4
    
    int_S2 = np.trapz(S2, t)
    int_S4 = np.trapz(S4, t)
    
    if int_S2 > 1e-30:
        bound = math.sqrt(int_S4 / int_S2)
    else:
        bound = float('inf')
    
    return bound

def construct():
    # Deadline guard
    t_start = time.time()
    DEADLINE = 600.0
    
    # Compute Laguerre zeros of L_12^{(1)} - the generalized Laguerre with alpha=1
    # These are theoretically optimal for the weight x*exp(-x)
    try:
        lag_roots_alpha1 = _compute_laguerre_zeros(12, alpha=1)
        if len(lag_roots_alpha1) != 12:
            raise ValueError("Expected 12 Laguerre zeros")
    except Exception:
        lag_roots_alpha1 = [4.0 * (i + 1) for i in range(12)]
    
    # Also compute standard Laguerre zeros L_12^{(0)}
    try:
        lag_roots_alpha0 = _compute_laguerre_zeros(12, alpha=0)
        if len(lag_roots_alpha0) != 12:
            raise ValueError("Expected 12 Laguerre zeros")
    except Exception:
        lag_roots_alpha0 = [4.0 * (i + 1) for i in range(12)]
    
    # Seed roots: arithmetic progression
    seed_roots = [4.0 * (i + 1) for i in range(12)]
    
    best_roots = seed_roots
    best_bound = _compute_c4_bound_fast(seed_roots)
    
    # Try the generalized Laguerre roots with alpha=1 (theoretically optimal)
    bound_a1 = _compute_c4_bound_fast(lag_roots_alpha1)
    if bound_a1 < best_bound:
        best_bound = bound_a1
        best_roots = lag_roots_alpha1[:]
    
    # Try the standard Laguerre roots with alpha=0
    bound_a0 = _compute_c4_bound_fast(lag_roots_alpha0)
    if bound_a0 < best_bound:
        best_bound = bound_a0
        best_roots = lag_roots_alpha0[:]
    
    # Try various scalings of the alpha=1 Laguerre roots
    lag_min = min(lag_roots_alpha1)
    lag_max = max(lag_roots_alpha1)
    lag_range = lag_max - lag_min
    
    # Grid search over scaling parameters
    for target_min in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]:
        if time.time() - t_start > DEADLINE - 30:
            break
        for target_max in [20.0, 30.0, 40.0, 48.0, 60.0, 80.0, 100.0, 120.0, 150.0, 200.0]:
            if target_max <= target_min:
                continue
            scale = (target_max - target_min) / lag_range
            scaled = [(r - lag_min) * scale + target_min for r in lag_roots_alpha1]
            if min(scaled) <= 0:
                continue
            bound = _compute_c4_bound_fast(scaled)
            if bound < best_bound:
                best_bound = bound
                best_roots = scaled[:]
    
    # Try scalings of the alpha=0 Laguerre roots
    lag_min0 = min(lag_roots_alpha0)
    lag_max0 = max(lag_roots_alpha0)
    lag_range0 = lag_max0 - lag_min0
    
    for target_min in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]:
        if time.time() - t_start > DEADLINE - 30:
            break
        for target_max in [20.0, 30.0, 40.0, 48.0, 60.0, 80.0, 100.0, 120.0, 150.0, 200.0]:
            if target_max <= target_min:
                continue
            scale = (target_max - target_min) / lag_range0
            scaled = [(r - lag_min0) * scale + target_min for r in lag_roots_alpha0]
            if min(scaled) <= 0:
                continue
            bound = _compute_c4_bound_fast(scaled)
            if bound < best_bound:
                best_bound = bound
                best_roots = scaled[:]
    
    # Refine around the best found using finer scaling
    if best_roots != seed_roots:
        best_min = min(best_roots)
        best_max = max(best_roots)
        
        # Determine which base to use for refinement
        # Use the one that gave the best result
        if best_roots == lag_roots_alpha1 or abs(best_bound - bound_a1) < abs(best_bound - bound_a0):
            base_roots = lag_roots_alpha1
            base_min = lag_min
            base_max = lag_max
            base_range = lag_range
        else:
            base_roots = lag_roots_alpha0
            base_min = lag_min0
            base_max = lag_max0
            base_range = lag_range0
        
        # Fine-tune around the best scaling
        for frac in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0]:
            if time.time() - t_start > DEADLINE - 30:
                break
            new_min = best_min * frac
            new_max = best_max * frac
            if new_min <= 0 or new_max <= new_min:
                continue
            scale = (new_max - new_min) / base_range
            scaled = [(r - base_min) * scale + new_min for r in base_roots]
            if min(scaled) <= 0:
                continue
            bound = _compute_c4_bound_fast(scaled)
            if bound < best_bound:
                best_bound = bound
                best_roots = scaled[:]
    
    # Final high-precision evaluation of the best candidate
    if time.time() - t_start < DEADLINE - 10:
        final_bound = _compute_c4_bound(best_roots, t_max=100.0, n_samples=50000)
        if final_bound < best_bound:
            best_bound = final_bound
    
    # Safety check: ensure all roots are positive
    if any(r <= 0 for r in best_roots):
        return seed_roots
    
    if time.time() - t_start > DEADLINE:
        return seed_roots
    
    return best_roots
# EVOLVE-BLOCK-END
