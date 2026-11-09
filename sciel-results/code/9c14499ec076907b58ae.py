import time
import math
import random
from itertools import product

_START_TIME = time.time()
TIME_LIMIT = 280

def priority(vector, n):
    """
    Structural change: Use a time-varying randomized hash-based priority
    that explores different "views" of the vector space over time.
    
    Instead of static heuristics, we use:
    1. A base hash that changes with elapsed time (different "strategy" per phase)
    2. Weighted combination of multiple hash functions for diversity
    3. Light balance preference to avoid extreme vectors
    
    This allows the greedy algorithm to explore different regions
    of the cap set space as time progresses.
    """
    elapsed = time.time() - _START_TIME
    
    # Phase-based strategy: different randomization seeds per phase
    # This causes the priority ordering to shift, exploring different vectors
    phase = int(elapsed / 20)  # New strategy every 20 seconds
    
    # Multiple hash functions with phase-dependent weights
    # Hash 1: Polynomial hash
    h1 = 0
    for v in vector:
        h1 = (h1 * 31 + v) & 0xFFFFFFFF
    
    # Hash 2: XOR-based hash
    h2 = 0
    for i, v in enumerate(vector):
        h2 ^= (v * (i + 1 + phase))
    h2 = h2 & 0xFFFFFFFF
    
    # Hash 3: Position-weighted sum
    h3 = 0
    for i, v in enumerate(vector):
        h3 += v * (i + 1) * (phase + 1)
    h3 = h3 & 0xFFFFFFFF
    
    # Combine hashes with phase-dependent weights
    # Different phases emphasize different hash functions
    w1 = 1.0 + 0.5 * math.sin(phase * 0.7)
    w2 = 1.0 + 0.5 * math.cos(phase * 0.7)
    w3 = 1.0 + 0.5 * math.sin(phase * 1.3)
    
    total_weight = w1 + w2 + w3
    combined_hash = (w1 * h1 + w2 * h2 + w3 * h3) / total_weight
    
    # Light balance component (not dominant, just a soft preference)
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    target = n / 3.0
    balance_penalty = (c0 - target) ** 2 + (c1 - target) ** 2 + (c2 - target) ** 2
    
    # Score: primarily driven by the time-varying hash, with small balance preference
    # The hash term dominates to create diverse exploration
    score = (combined_hash / 4294967295.0) - 0.01 * balance_penalty
    
    return score
