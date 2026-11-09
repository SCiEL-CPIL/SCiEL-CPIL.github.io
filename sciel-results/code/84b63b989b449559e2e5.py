"""Cap set greedy construction with improved static priority heuristic."""
import time
import math
import random
from itertools import product

# Deadline guard
_START_TIME = time.time()
TIME_LIMIT = 280  # seconds, with 20s safety margin before 300s limit

def _spatial_spread(positions):
    """Compute sum of pairwise distances between positions in the list."""
    m = len(positions)
    if m < 2:
        return 0.0
    total = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            total += abs(positions[i] - positions[j])
    return total

def priority(vector, n):
    """
    Static priority function for greedy cap set construction.
    
    Prioritize vectors that are balanced in digit distribution,
    with tie-breakers that encourage diversity and spatial spread
    of both 0-positions (additive identity barriers) and 1-positions.
    """
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    target = n / 3.0
    
    # Primary: minimize deviation from balanced distribution
    penalty = (c0 - target) ** 2 + (c1 - target) ** 2 + (c2 - target) ** 2
    
    # Secondary: sum of squared digits (encourages spread of non-zero values)
    sq_sum = sum(v * v for v in vector)
    
    # Tertiary: spatial spread of 0-positions (additive identity barrier)
    # Vectors with zeros spread across many positions are less likely to
    # form symmetric triples required for 3-term APs
    zeros_positions = [i for i, v in enumerate(vector) if v == 0]
    zero_spread = _spatial_spread(zeros_positions)
    
    # Quaternary: spatial spread of 1-positions (breaks ties, reduces local clustering)
    ones_positions = [i for i, v in enumerate(vector) if v == 1]
    one_spread = _spatial_spread(ones_positions)
    
    # Quinternary: lexicographic tie-breaker encoded as float
    lex_val = 0.0
    scale = 1.0
    for i in range(n):
        lex_val += vector[i] * scale
        scale *= 1.1  # Non-integer base to avoid collisions
    
    # Normalize spreads by maximum possible value for comparability
    max_spread = n * (n - 1) / 2.0
    if max_spread > 0:
        zero_spread_norm = zero_spread / max_spread
        one_spread_norm = one_spread / max_spread
    else:
        zero_spread_norm = 0.0
        one_spread_norm = 0.0
    
    # Combine: balance dominates, then zero-spread (AP barrier), then sq_sum,
    # then one-spread, then lex
    return (-penalty * 10000.0 
            + zero_spread_norm * 5000.0 
            + sq_sum * 100.0 
            + one_spread_norm * 500.0 
            + lex_val)
