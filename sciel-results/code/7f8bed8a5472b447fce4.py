"""Cap set greedy builder with frequency-domain algebraic priority over Z_3."""
import math
import time

_start_time = None
_DEADLINE = 330 * 0.95

# Precompute omega components: omega = e^(2*pi*i/3)
_OMEGA_REAL = -0.5
_OMEGA_IMAG = math.sqrt(3) / 2.0

# EVOLVE-BLOCK-START
def priority(vector, n):
    """
    Priority based on frequency-domain structure over Z_3.
    
    Structural change: treat each vector v as a polynomial P(x) = sum v_i * x^i
    over the complex numbers, and evaluate at the primitive cube root of unity
    omega = e^(2*pi*i/3). The magnitude |P(omega)| measures how aligned the
    vector is with the "line-direction" character of the affine space.
    
    Vectors with small |P(omega)| are orthogonal to the primary line-formation
    mode, making them less likely to complete arithmetic progressions with
    other vectors of similar structure. This is a global frequency-domain
    signal, fundamentally different from local coordinate statistics.
    
    Additionally incorporates:
    - Coordinate spread (distinct count)
    - Degeneracy penalty
    - Deterministic hash tiebreaking
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        # Safety: return a neutral priority if we're running out of time
        return 0.5
    
    v = tuple(vector)
    
    # Frequency-domain score: evaluate P(omega) = sum v_i * omega^i
    # omega^k cycles with period 3: omega^0=1, omega^1=omega, omega^2=conj(omega)
    real_part = 0.0
    imag_part = 0.0
    
    for i in range(n):
        d = v[i]
        if d == 0:
            continue
        k = i % 3
        if k == 0:
            real_part += d
        elif k == 1:
            real_part += d * _OMEGA_REAL
            imag_part += d * _OMEGA_IMAG
        else:
            real_part += d * _OMEGA_REAL
            imag_part -= d * _OMEGA_IMAG
    
    mag_sq = real_part * real_part + imag_part * imag_part
    
    # Normalize by n to get per-coordinate frequency energy
    norm_energy = mag_sq / n if n > 0 else 0.0
    
    # Lower energy = more orthogonal to line direction = better
    # Use exponential decay for smooth prioritization
    freq_score = math.exp(-norm_energy * 0.5)
    
    # Coordinate spread: more distinct values = less confined to subspaces
    distinct = len(set(v))
    spread_score = distinct / n if n > 0 else 0.0
    
    # Degeneracy penalty: all-same vectors are in 1D subspaces
    degenerate = 0.4 if distinct == 1 else 0.0
    
    # Mild coset diversifier: sum mod 3
    sum_mod = sum(v) % 3
    coset_boost = {0: 0.0, 1: 0.04, 2: 0.02}[sum_mod]
    
    # Deterministic hash tiebreaker
    h = 0
    for i, x in enumerate(v):
        h = (h * 31 + x + i * 7) % 997
    hash_score = h / 997.0 * 0.05
    
    total = (0.50 * freq_score + 0.25 * spread_score + 
             0.05 * coset_boost + hash_score - degenerate)
    
    return total
# EVOLVE-BLOCK-END
