"""Cap set construction via greedy selection with geometric-center and hash priorities."""
import time

# EVOLVE-BLOCK-START

_start_time = time.time()
_DEADLINE = 325  # seconds, safety margin before 330s kill

# Precomputed primes for fast hashing
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
           73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
           157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
           239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
           331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419,
           421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503]

def _fast_hash(vector, n, seed=0):
    """Fast deterministic pseudo-random priority via linear prime-based hash."""
    h = 0
    for i, v in enumerate(vector):
        if i < len(_PRIMES):
            h += v * _PRIMES[i]
        else:
            h += v * (i * 31 + seed + 17)
    # Mix with seed and reduce to [0, 1)
    h = (h * 1000003 + seed * 999983) % (1 << 60)
    return h / (1 << 60)

def _line_blocking_score(vector):
    """
    Score a vector by its "line-blocking potential": the number of other vectors
    in {0,1,2}^n that would form a 3-term affine line (cap set violation) with
    this candidate if both were in the set.
    
    A 3-term AP in F_3^n: a, b, c where a + c = 2b (mod 3), i.e., b = (a+c)/2.
    For a given vector v, the number of pairs (a,c) such that v = (a+c)/2 mod 3
    equals the number of "lines" through v. We want vectors that are on many lines
    (high blocking potential) so that selecting them early eliminates many conflicts.
    
    For each coordinate where v[i] = 0: pairs are (1,2) or (2,1) -> 2 choices per coord
    For each coordinate where v[i] = 1: pairs are (0,2), (2,0), (1,1) -> 3 choices per coord  
    For each coordinate where v[i] = 2: pairs are (0,1), (1,0) -> 2 choices per coord
    
    Total lines through v = product over coordinates of choices[i]
    This is 2^(n0+n2) * 3^(n1) where n0,n1,n2 are counts of 0,1,2 in v.
    
    We want to MAXIMIZE this (select high-blocking vectors first) to greedily
    eliminate the most future conflict options.
    """
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
    
    # Number of affine lines passing through this vector
    # = 2^(count_0 + count_2) * 3^(count_1)
    # Use log to avoid overflow for large n
    import math
    log_lines = (count_0 + count_2) * math.log(2) + count_1 * math.log(3)
    
    return log_lines

def _structured_priority(vector, n):
    """
    Combined priority: line-blocking potential dominates (select vectors that
    lie on many affine lines to aggressively eliminate future conflicts),
    with hash-based diversity for tie-breaking.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        return -1e18
    
    # Line-blocking score (dominant signal): more lines through v = better
    line_score = _line_blocking_score(vector)
    
    # Count components for secondary heuristics
    n1 = 0
    n2 = 0
    n0 = 0
    for v in vector:
        if v == 0:
            n0 += 1
        elif v == 1:
            n1 += 1
        elif v == 2:
            n2 += 1
    
    # Balance bonus: prefer balanced distribution of 0s and 2s (symmetry)
    balance = abs(n0 - n2) / max(1, n)
    balance_score = (1.0 - balance) * 0.1
    
    # Multiple hash layers for robustness (fast)
    h1 = _fast_hash(vector, n, seed=1)
    h2 = _fast_hash(vector, n, seed=2)
    h3 = _fast_hash(vector, n, seed=3)
    
    # Hash component provides diversity and tie-breaking
    hash_score = (h1 * 0.4 + h2 * 0.35 + h3 * 0.25) * 0.3
    
    # Combine: line-blocking dominates, balance and hash provide refinement
    return line_score + balance_score + hash_score

def priority(vector, n):
    """
    Priority function for greedy cap set construction.
    Higher priority = selected earlier.
    """
    return _structured_priority(vector, n)

# EVOLVE-BLOCK-END
