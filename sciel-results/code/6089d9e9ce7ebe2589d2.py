import time
import heapq
from itertools import product

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy:
    1. Primary: Smoothly prioritize vectors with more distinct symbols (quadratic in presence).
    2. Secondary: Quadratic balance penalty (smoother than L1, provides better gradient).
    3. Tertiary: Center-targeting via distance-from-boundary product to avoid AP-prone regions.
    4. Quaternary: Fast structural hash for deterministic tie-breaking.
    
    Uses smooth surrogates to provide more informative ordering signals.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        # Past deadline: return a constant to signal "stop caring about ordering"
        return 0.0

    # Fast symbol counting
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Smooth distinct score: reward each symbol's presence quadratically
    # This creates a smooth landscape where having all 3 is best, but
    # the penalty for missing one is less abrupt than the discrete version
    presence0 = 1.0 if c0 > 0 else 0.0
    presence1 = 1.0 if c1 > 0 else 0.0
    presence2 = 1.0 if c2 > 0 else 0.0
    
    # Quadratic in number of distinct symbols: 0, 1, 2, or 3 present
    num_distinct = presence0 + presence1 + presence2
    distinct_score = 10000.0 * num_distinct * num_distinct
    
    # Balance: Quadratic deviation from uniform (smoother than L1)
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0
    
    # Center-targeting: reward vectors that are "central" in each coordinate
    # A coordinate value of 1 is central, 0 or 2 are boundary.
    # Vectors with many 1s are less likely to form APs with other vectors.
    # Use a smooth product: for each position, score = 1 - |val - 1| (i.e., 1 if val==1, 0 otherwise)
    # Then take the geometric mean (log-sum) to get a smooth center score.
    # But for speed, use a simpler approach: count of 1s weighted by position diversity.
    # Actually, use: sum over positions of (1 - abs(val - 1)) which is just c1.
    # Better: use a quadratic penalty for being at boundary in many positions.
    # Number of boundary positions = c0 + c2 = n - c1
    boundary_count = c0 + c2
    # Smooth center score: quadratic in boundary count, penalizing vectors that are "far" from center
    center_score = -boundary_count * boundary_count * 2.0
    
    # Diversity bonus: reward vectors that have a mix across positions
    # Count how many positions have value 0, 1, 2 respectively (already have c0, c1, c2)
    # The "diversity" is captured by the entropy-like measure
    # Use: -sum(p_i * log(p_i)) as a smooth diversity measure, but that's expensive.
    # Instead, use a simpler proxy: reward having all three symbols present in a balanced way.
    # This is already partially captured by distinct_score and balance_score.
    # Add a small bonus for vectors where no single symbol dominates too much.
    max_count = max(c0, c1, c2)
    dominance_penalty = -max_count * max_count * 1.0
    
    # Fast hash: simple polynomial rolling hash
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    
    hash_score = h / 0x7FFFFFFF
    
    return float(distinct_score + balance_score + center_score + dominance_penalty + hash_score)
