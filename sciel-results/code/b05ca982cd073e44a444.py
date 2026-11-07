"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from functools import lru_cache

_DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_N = len(_DIVISORS)
_MAX_VAL = 1000000

# Precompute LCM of all divisors for periodicity
def _lcm(a, b):
    return a * b // math.gcd(a, b)

_LCM = 1
for d in _DIVISORS:
    _LCM = _lcm(_LCM, d)

# For each divisor index i, and for each residue class r in [0, _LCM),
# determine if r % _DIVISORS[i] == 0. This tells us which positions in
# a window of length _LCM are "available" for divisor i.
# Actually, we need: for a window starting at L, position p (0-indexed)
# has value L+p. We need (L+p) % d_i == 0, i.e., p ≡ -L (mod d_i).
# Since p is in [0, R-1] for window length R, and R <= _LCM,
# p ≡ -L (mod d_i) has at most one solution in [0, R-1] if R < d_i,
# or possibly more if R >= d_i. But R will be small (~30-500), and
# d_i are at least 6, so there could be multiple candidates per divisor.

# Better approach: For a given window [L, L+R-1], for each divisor d_i,
# the valid multiples are those m such that L <= m <= L+R-1 and m % d_i == 0.
# These are m = k*d_i where k*d_i is in [L, L+R-1].

# Strategy: Use backtracking search over divisors, trying to assign distinct
# multiples within a window. We search over window start L and window size R.
# For efficiency, we iterate L over a range and use the LCM periodicity.

# Actually, let's use a different structural approach:
# The problem is equivalent to: find L and R such that in the interval [L, L+R-1],
# there exist distinct multiples m_i of d_i for each i, with m_i in [L, L+R-1].
# Minimize R.

# Key insight: The pattern of which multiples are available repeats with period LCM.
# So we only need to check L mod LCM. There are LCM = 32730 distinct residue classes.
# For each residue class, we can compute the best R.

# For a fixed L, the candidates for divisor d_i are the multiples of d_i in [L, L+R-1].
# Let's define for each i, the set of candidate offsets: {k : L <= k*d_i <= L+R-1}
# which means k is in [ceil(L/d_i), floor((L+R-1)/d_i)].

# For a given L and R, we need to assign to each i a distinct value from its candidate set.
# This is a bipartite matching / exact cover problem.

# Efficient approach: Iterate R from small to large. For each R, iterate L over
# one full period [0, LCM-1]. For each (L, R), check if a valid assignment exists.
# Stop when found.

# But R could be up to a few hundred, and LCM is 32730, so that's ~3.2M checks.
# Each check involves a matching of size 10, which is fast.

# Let's optimize: For a given R, and for each L in [0, LCM), compute the candidate
# sets and check matchability. We can prune early.

def _can_assign(L, R, divisors):
    """Check if we can assign distinct multiples in [L, L+R-1] for each divisor."""
    n = len(divisors)
    # For each divisor i, find all multiples of divisors[i] in [L, L+R-1]
    # Represent as a list of candidate values
    candidates = []
    for i in range(n):
        d = divisors[i]
        # First multiple >= L
        k_start = (L + d - 1) // d
        # Last multiple <= L+R-1
        k_end = (L + R - 1) // d
        if k_start > k_end:
            return None
        cand = [k_start * d, (k_start + 1) * d, ..., k_end * d]
        # Actually just build the list
        cand = list(range(k_start * d, k_end * d + 1, d))
        if not cand:
            return None
        candidates.append(cand)
    
    # Now we need to pick one from each candidates[i] such that all picks are distinct.
    # Use backtracking with MRV (minimum remaining values) heuristic.
    # Sort by candidate count
    order = sorted(range(n), key=lambda i: len(candidates[i]))
    assigned = [None] * n
    used = set()
    
    def backtrack(idx):
        if idx == n:
            return True
        i = order[idx]
        for val in candidates[i]:
            if val not in used:
                used.add(val)
                assigned[i] = val
                if backtrack(idx + 1):
                    return True
                used.remove(val)
                assigned[i] = None
        return False
    
    if backtrack(0):
        return assigned
    return None

def construct():
    deadline = time.time() + 85  # 85 second deadline for 90s limit
    divisors = _DIVISORS
    n = len(divisors)
    
    best_result = None
    best_range = _MAX_VAL
    
    # Search over R from 1 upward
    for R in range(1, 5000):
        if time.time() > deadline:
            break
        if best_range is not None and R >= best_range:
            break
        
        # For this R, iterate L over one LCM period
        # L can be any positive integer, but due to LCM periodicity,
        # the pattern repeats every LCM. However, L must be >= 1.
        # Also, the values must be <= 1000000, so L+R-1 <= 1000000.
        # For small R, this is not a constraint.
        
        # We need L such that all assigned values are in [1, 1000000].
        # Since L >= 1 and L+R-1 <= 1000000, we have L in [1, 1000000-R+1].
        # Due to LCM periodicity, it suffices to check L in [1, LCM] (or [0, LCM-1] shifted).
        # But we also need the actual values to be positive, so L >= 1.
        
        # Let's check L in range(1, _LCM + 1)
        for L in range(1, _LCM + 1):
            if time.time() > deadline:
                break
            
            assigned = _can_assign(L, R, divisors)
            if assigned is not None:
                # Verify all values are <= 1000000
                if max(assigned) <= _MAX_VAL:
                    best_range = R
                    best_result = [int(assigned[i]) for i in range(n)]
                    return best_result
    
    if best_result is not None:
        return best_result
    
    # Fallback: return the divisors themselves (may not be optimal but valid)
    return [int(d) for d in divisors]

# EVOLVE-BLOCK-END
