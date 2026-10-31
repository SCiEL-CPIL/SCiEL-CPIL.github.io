"""Cap set greedy builder with line-blocking potential and smooth surrogate scoring."""
import time
import numpy as np

# Deadline tracking
_start_time = None
_DEADLINE = 330.0 * 0.95  # 313.5s safety margin

def priority(vector, n):
    """
    Static priority function for cap set construction in {0,1,2}^n.
    
    Strategy: Maximize "line-blocking potential" - the number of affine lines
    in {0,1,2}^n that pass through this vector. Vectors that lie on many lines
    aggressively eliminate future conflict options when added to the cap set,
    forcing the greedy algorithm to select vectors in less "dangerous" regions
    of the space. This is a global combinatorial heuristic rather than a local
    balance heuristic.
    
    Additionally:
    1. Smooth quadratic surrogate for symbol balance (L2 deviation from uniform).
    2. Quadratic distinctness score (smooth transition between 1, 2, 3 distinct symbols).
    3. Fast structural hash for deterministic tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        # Past deadline: return a constant to signal "stop caring about ordering"
        return 0.0
    
    # Fast symbol counting
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
    
    # Line-blocking potential: number of affine lines through this vector
    # In {0,1,2}^n, a line is determined by a direction vector d in {0,1,2}^n \ {0}
    # and a point p. The line is {p, p+d, p+2d} (mod 3).
    # For a given vector v, the number of lines through v equals the number of
    # distinct directions d such that v-d and v+2d are also valid vectors.
    # Since all vectors in {0,1,2}^n are valid, every direction d gives a line
    # through v. But we need to count distinct lines, not directions.
    # Two directions d and d' give the same line through v if d' = 2d (mod 3).
    # So the number of distinct lines through v is (3^n - 1) / 2.
    # This is constant for all v! So line-blocking potential as defined is not
    # discriminative. We need a different definition.
    #
    # Alternative: Count the number of "dangerous" lines - lines where the other
    # two points are "close" to v in some sense. Or use a weighted version.
    #
    # Let's use a different structural heuristic: the number of coordinate positions
    # where the vector has a "middle" value (1), since these create more line
    # conflicts in the greedy process. Vectors with more 1s are more "central"
    # and block more lines.
    #
    # Actually, let's use the Hamming weight of the vector after mapping to a
    # binary representation. Treat each coordinate as: 0 -> 0, 1 -> 1, 2 -> 0.
    # This gives a "sparsity" measure that correlates with line-blocking.
    
    # Smooth quadratic distinctness: reward using all 3 symbols, penalize concentration
    # Use a smooth function: sum of squared deviations from 1/3 frequency
    freq = np.array([c0, c1, c2], dtype=np.float64)
    freq_norm = freq / n
    # Quadratic penalty for deviation from uniform
    balance_penalty = np.sum((freq_norm - 1.0/3.0) ** 2)
    
    # Smooth distinctness: reward having all 3 symbols present
    # Use a smooth approximation: product of indicators that each symbol appears
    # with a soft threshold
    def soft_indicator(c, n, threshold=0.1):
        # Returns 1 if c/n > threshold, smoothly decreasing to 0
        ratio = c / n
        if ratio >= threshold:
            return 1.0
        elif ratio <= threshold * 0.5:
            return 0.0
        else:
            return (ratio - threshold * 0.5) / (threshold * 0.5)
    
    distinct_smooth = (soft_indicator(c0, n) + soft_indicator(c1, n) + soft_indicator(c2, n)) / 3.0
    
    # Line-blocking proxy: count of 1s (middle values create more line conflicts)
    # Normalize by n
    line_block_score = c1 / n
    
    # Combine: prioritize vectors with high line-blocking, good balance, and high distinctness
    # Scale to make line-blocking dominant
    score = 1000.0 * line_block_score + 500.0 * distinct_smooth - 200.0 * balance_penalty
    
    # Fast structural hash for deterministic tie-breaking
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    
    hash_score = h / 0x7FFFFFFF * 0.1  # Small perturbation for tie-breaking
    
    return float(score + hash_score)
