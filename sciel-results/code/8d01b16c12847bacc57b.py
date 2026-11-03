"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import numpy as np

# EVOLVE-BLOCK-START
def _score_set(U):
    """Compute the lower bound score for a finite set U of nonnegative integers."""
    U = list(set(U))
    if 0 not in U:
        U.append(0)
        U.sort()
    
    n = len(U)
    if n < 2:
        return 0.0
    
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Compute |U+U| and |U-U|
    # For efficiency, use sets of sums and differences
    sum_set = set()
    diff_set = set()
    
    U_arr = np.array(U, dtype=np.int64)
    
    # Vectorized computation for sum set
    # U+U: all pairwise sums
    sums = U_arr[:, None] + U_arr[None, :]
    sum_set = set(sums.flatten().tolist())
    
    # U-U: all pairwise differences
    diffs = U_arr[:, None] - U_arr[None, :]
    diff_set = set(diffs.flatten().tolist())
    
    sum_size = len(sum_set)
    diff_size = len(diff_set)
    
    if sum_size == 0:
        return 0.0
    
    ratio = diff_size / sum_size
    if ratio <= 0:
        return 0.0
    
    # Score: 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    # Wait, the task says: 1 + log(|U-U|/|U+U|)/log(2 max U + 1) on C6
    # Let me re-read: "score is the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1) on C6"
    # Hmm, this is ambiguous. Let me interpret it as:
    # score = 1 + log2(|U-U|/|U+U|) / log2(2*max(U) + 1)
    # or perhaps:
    # score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    # where log is natural log or log2?
    
    # Let me think about this more carefully. The "exponent lower bound" suggests:
    # We want to maximize the exponent such that |U-U| / |U+U| >= (2*max(U)+1)^exponent
    # So exponent = log(|U-U|/|U+U|) / log(2*max(U)+1)
    # And the score is 1 + that exponent? Or just the exponent?
    
    # The task says "score is the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)"
    # This reads as: score = 1 + (log(|U-U|/|U+U|) / log(2*max(U) + 1))
    
    # Let's use log base 2 for consistency, or natural log. Since it's a ratio,
    # the base doesn't matter for the ratio of logs.
    
    log_ratio = math.log(ratio)
    log_max = math.log(2 * max_u + 1)
    
    if log_max == 0:
        return 0.0
    
    score = 1 + log_ratio / log_max
    
    return score


def _construct_geometric(b, m):
    """Construct U = {floor(b^k) for k=0,...,m}"""
    U = set()
    U.add(0)
    for k in range(m + 1):
        val = int(math.floor(b ** k))
        U.add(val)
    return sorted(U)


def construct():
    """
    Construct a finite set U of nonnegative integers containing 0 to maximize
    the sumset/difference-set exponent lower bound.
    
    Uses geometric progression U = {floor(b^k) | k=0,...,m} with tunable base b
    and length m, searching over b and m to maximize the score.
    """
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Search over different values of m (number of elements minus 1)
    # and b (base of geometric progression)
    
    # For small sets, we can do a more thorough search
    # For larger sets, we need to be careful about computation time
    
    # Let's try various m values and for each, sweep over b
    m_values = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50]
    
    for m in m_values:
        # For each m, sweep over b in a reasonable range
        # b should be > 1 for the set to grow
        # Try b from 1.1 to maybe 10 or so
        
        # Use a log-spaced sweep for b
        b_values = np.linspace(1.01, 5.0, 50)
        
        for b in b_values:
            if b <= 1.0:
                continue
            U = _construct_geometric(b, m)
            if len(U) < 2:
                continue
            
            # Check if computation is feasible
            n = len(U)
            if n > 200:
                # Too large for full pairwise computation
                continue
            
            score = _score_set(U)
            if score > best_score:
                best_score = score
                best_set = U
    
    # Also try some hand-picked sets that are known to be good
    # The seed set [0, 1, 3]
    test_sets = [
        [0, 1, 3],
        [0, 1, 2, 4, 8],
        [0, 1, 3, 9, 27],
        [0, 1, 4, 16, 64],
        [0, 1, 5, 25, 125],
    ]
    
    for U in test_sets:
        score = _score_set(U)
        if score > best_score:
            best_score = score
            best_set = U
    
    # Try some structured sets with larger m
    # Geometric with integer base
    for base in [2, 3, 4, 5, 6, 7, 8, 10]:
        for m in range(2, 15):
            U = _construct_geometric(float(base), m)
            n = len(U)
            if n > 100:
                continue
            score = _score_set(U)
            if score > best_score:
                best_score = score
                best_set = U
    
    return best_set
# EVOLVE-BLOCK-END
