"""Cap set greedy construction with time-varying dynamic priority."""
import time
import math
import random
from collections import defaultdict

_START_TIME = time.time()
TIME_LIMIT = 280  # seconds

# Global state for dynamic conflict tracking
_lsh_buckets = defaultdict(int)
_lsh_num_hashes = 8
_lsh_bucket_size = 64
_phase_duration = 15.0  # seconds per phase
_phase_start = _START_TIME

def _get_current_phase():
    """Get current time-based phase for dynamic behavior."""
    return int((time.time() - _phase_start) / _phase_duration)

def _update_lsh_buckets(vector, n):
    """Update LSH buckets with the selected vector."""
    phase = _get_current_phase()
    # Use phase to vary the hash functions
    for i in range(_lsh_num_hashes):
        # Create hash based on vector and phase
        h = 0
        for j in range(n):
            # Vary the mixing based on phase and hash index
            h = (h * 31 + vector[j] * (i + 1) + phase * 7) % _lsh_bucket_size
        _lsh_buckets[(phase, i, h)] += 1

def _get_conflict_density(vector, n):
    """Estimate conflict density using LSH buckets."""
    phase = _get_current_phase()
    density = 0
    for i in range(_lsh_num_hashes):
        h = 0
        for j in range(n):
            h = (h * 31 + vector[j] * (i + 1) + phase * 7) % _lsh_bucket_size
        density += _lsh_buckets.get((phase, i, h), 0)
    return density

def priority(vector, n):
    """
    Dynamic priority function with time-varying behavior and conflict density.
    
    Strategy:
    1. Time-varying hash to explore different regions in different phases.
    2. Conflict density approximation using LSH to avoid clustering.
    3. Balance and structure penalties as secondary signals.
    4. Dynamic weighting that shifts emphasis over time.
    """
    # Time-varying component
    elapsed = time.time() - _phase_start
    phase = _get_current_phase()
    phase_fraction = (elapsed % _phase_duration) / _phase_duration
    
    # Compute basic statistics
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # 1. Balance Penalty
    t0 = n / 6.0
    t1 = n / 3.0
    t2 = n / 2.0
    balance_penalty = (c0 - t0) ** 2 + (c1 - t1) ** 2 + (c2 - t2) ** 2
    
    # 2. Structure Penalty
    def comb2(k):
        return k * (k - 1) // 2
    equal_pairs = comb2(c0) + comb2(c1) + comb2(c2)
    
    # 3. Sum Deviation
    s = c1 + 2 * c2
    sum_penalty = (s - 1.5 * n) ** 2
    
    # 4. Conflict density (LSH-based)
    conflict = _get_conflict_density(vector, n)
    
    # 5. Time-varying hash for exploration
    h = 0
    for i in range(n):
        # Phase-dependent mixing
        h = (h * 31 + vector[i] * (phase + 1) + i * phase) % (10**9 + 7)
    
    # Dynamic weights that shift over time
    # Early: favor balance and low structure
    # Mid: increase conflict avoidance
    # Late: more aggressive exploration
    t = phase_fraction
    w_balance = 800.0 * (1 - 0.5 * t)
    w_structure = 40.0 * (1 - 0.3 * t)
    w_sum = 30.0 * (1 - 0.5 * t)
    w_conflict = 10.0 * (0.5 + 0.5 * t)  # Increases over time
    w_hash = 1e-6 * (1 + phase)  # Stronger exploration later
    
    score = -balance_penalty * w_balance
    score -= equal_pairs * w_structure
    score -= sum_penalty * w_sum
    score -= conflict * w_conflict
    score += h * w_hash
    
    return score
