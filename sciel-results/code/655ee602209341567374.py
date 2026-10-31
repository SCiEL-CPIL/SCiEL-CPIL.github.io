"""Cap set greedy priority: structural change using sum-of-squares weighting + deadline guard."""
import time
from collections import Counter

# Deadline guard: kill before 330s wall-clock
_START = time.time()
_DEADLINE = _START + 280.0  # 50s safety margin

def priority(vector, n):
    """
    Return a float priority for a vector in {0,1,2}^n.
    
    Strategy: Combine sum of squares (energy) with distinct symbol count and balance.
    
    1. Primary: Distinct symbol count (3 > 2 > 1). Vectors using all 3 symbols are 
       more "generic" and less likely to form affine lines with other generic vectors.
    2. Secondary: Sum of squares (energy). Among vectors with the same distinct count,
       favor those with higher energy (more 1s and 2s). This breaks symmetry and
       exploits the non-linear distribution of affine lines in the ternary grid.
    3. Tertiary: Balance. Among vectors with the same distinct count and energy,
       favor those with symbol counts closer to uniform.
    4. Quaternary: Lexicographic tiebreaker for determinism.
    """
    # Deadline check
    if time.time() - _START > _DEADLINE:
        return 0.0
    
    # Fast symbol counting using Counter (C-implemented)
    c = Counter(vector)
    c0 = c.get(0, 0)
    c1 = c.get(1, 0)
    c2 = c.get(2, 0)
    
    # Distinct count (3 is best, 2 is middle, 1 is worst)
    distinct_count = sum(1 for c_val in [c0, c1, c2] if c_val > 0)
    
    # Sum of squares: 0^2=0, 1^2=1, 2^2=4
    # This gives distinct weights: a vector with k ones and m twos has sum_sq = k + 4m
    sum_sq = c1 + 4 * c2
    
    # Balance: L1 deviation from uniform (faster than chi2, same ordering for small n)
    ideal = n / 3.0
    dev = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    
    # Fast hash: simple polynomial rolling hash
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    
    # Combine scores with appropriate weights:
    # - Distinct count dominates (gap of 10000 between levels)
    # - Sum of squares is secondary (max 4n, so scale by 100 to stay within distinct gaps)
    # - Balance is tertiary (dev max ~n, scale by 10)
    # - Hash is quaternary (small perturbation for tie-breaking)
    
    distinct_score = distinct_count * 10000.0
    energy_score = sum_sq * 100.0
    balance_score = -dev * 10.0
    hash_score = h / 0x7FFFFFFF
    
    return float(distinct_score + energy_score + balance_score + hash_score)
