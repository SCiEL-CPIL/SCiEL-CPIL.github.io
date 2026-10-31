import time
import zlib

_start_time = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill

def priority(vector, n):
    """
    Score vectors in F_3^n for greedy cap set construction.
    
    Strategy: Use a structural measure based on coordinate pair interactions.
    In F_3^n, lines are formed when three points are collinear. The probability
    that two points "constrain" each other into a line depends on their difference
    being a "thin" direction. Vectors with more "independent" coordinate patterns
    are less likely to form lines with other vectors.
    
    We use:
    1. Count of 2s (vectors with 2s are more constrained in F_3)
    2. Coordinate pair diversity: for each pair of coordinates (i,j), the pair
       (v_i, v_j) can be one of 9 values. More diverse pairs = more "spread out"
    3. A deterministic hash for tie-breaking
    """
    if time.time() - _start_time > _DEADLINE:
        # Ultra-fast fallback
        s = 0
        for v in vector:
            s += v * v
        return float(-s)
    
    # Count 0s, 1s, 2s
    count_0 = 0
    count_1 = 0
    count_2 = 0
    for v in vector:
        if v == 0:
            count_0 += 1
        elif v == 1:
            count_1 += 1
        else:
            count_2 += 1
    
    # Primary: penalize 2s strongly (they create more line constraints in F_3)
    # In F_3, 2 = -1, so vectors with many 2s are "negatives" of vectors with many 1s
    # but the line structure is asymmetric in a way that 2s are more problematic
    base_score = -10.0 * count_2 + 3.0 * count_1 - 1.0 * count_0
    
    # Secondary: coordinate pair diversity (capture 2nd-order structure)
    # For each pair (i,j) with i < j, encode the pair value
    # Use a compact representation: for each pair, add a small weight based on
    # whether the pair is "diverse" (mix of 0,1,2) vs "degenerate" (all same)
    pair_diversity = 0
    if n <= 12:  # Only compute pairs for small n to stay fast
        vec = list(vector)
        for i in range(n):
            vi = vec[i]
            for j in range(i + 1, n):
                vj = vec[j]
                # Degenerate pairs (0,0), (1,1), (2,2) are more likely to form lines
                # Mixed pairs (0,1), (1,2), (0,2), etc. are less constraining
                if vi == vj:
                    pair_diversity -= 1
                else:
                    pair_diversity += 1
    else:
        # For large n, approximate with a simpler measure
        # Variance of coordinates
        sum_v = count_1 + 2 * count_2
        mean = sum_v / n if n > 0 else 0
        variance = 0.0
        for v in vector:
            d = v - mean
            variance += d * d
        pair_diversity = variance
    
    # Tertiary: deterministic hash for tie-breaking
    h = 0
    for idx, v in enumerate(vector):
        h = (h * 31 + v + idx * 7) & 0xFFFFFFFF
    hash_score = (h / 0xFFFFFFFF) * 0.1
    
    return base_score + 0.5 * pair_diversity + hash_score
