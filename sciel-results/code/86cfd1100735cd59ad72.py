import time
import math

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s

def priority(vector, n):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    if time.time() - _start_time > _DEADLINE:
        return 0.0
    
    # Count occurrences of each symbol
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Sum of squares of coordinate values: favors 1s and 2s over 0s
    # 0^2=0, 1^2=1, 2^2=4
    # This breaks the symmetry of the standard balance heuristic and
    # exploits the non-linear distribution of affine lines in the ternary grid
    sum_sq = c1 * 1 + c2 * 4
    
    # Normalize sum_sq: max is 4n (all 2s), min is 0 (all 0s)
    norm_sum_sq = sum_sq / (4.0 * n) if n > 0 else 0.0
    
    # Shannon entropy of the coordinate distribution
    # Higher entropy = more balanced/spread out = less likely to form lines
    total = n
    entropy = 0.0
    for c in (c0, c1, c2):
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    
    # Max possible entropy for ternary is log2(3) ≈ 1.585
    # Normalize to [0, 1]
    max_entropy = math.log2(3)
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    
    # Number of distinct symbols present
    distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    
    # Pairwise disagreement: count positions where adjacent coords differ
    # Higher disagreement = more "mixed" vector
    disagreement = 0
    for i in range(n - 1):
        if vector[i] != vector[i + 1]:
            disagreement += 1
    # Normalize by max possible (n-1)
    if n > 1:
        norm_disagree = disagreement / (n - 1)
    else:
        norm_disagree = 0.0
    
    # Hash-based tiebreaker for deterministic diversity
    # Use a simple hash of the vector to break ties
    h = 0
    for x in vector:
        h = (h * 31 + x) & 0xFFFFFFFF
    hash_frac = (h % 10000) / 10000.0 * 0.01  # small perturbation
    
    # Combine: entropy as primary, sum_sq as secondary (new), 
    # distinct symbols as tertiary, disagreement as quaternary,
    # hash as tiny tiebreaker
    return float(10.0 * norm_entropy + 3.0 * norm_sum_sq + 2.0 * distinct / 3.0 + 1.0 * norm_disagree + hash_frac)
