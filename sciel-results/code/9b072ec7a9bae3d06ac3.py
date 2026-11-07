"""Rewritten program: exploit LCM-periodic structure for a much tighter search."""
import time
import math
from itertools import permutations

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
N = len(DIVISORS)

# Precompute LCM
def _lcm(a, b):
    return a * b // math.gcd(a, b)

LCM = 1
for d in DIVISORS:
    LCM = _lcm(LCM, d)

# Precompute, for each offset o in [0, LCM), the residue r_i = o mod d_i
# The multiples of d_i in [L, L+LCM) are at positions L + ((r_i - L) mod d_i)
# Equivalently, for a window starting at L with period LCM, the offset of the
# smallest multiple of d_i >= L is: off_i = (r_i - (L mod d_i)) mod d_i
# where r_i = 0 (since L mod d_i determines the phase).
#
# Actually, let's think differently. For a window [L, L+W], the smallest multiple
# of d_i >= L is: L + ((-L) mod d_i) = L + (d_i - L % d_i) % d_i
# Let's define: for each d_i, the "phase" is L mod d_i. The offset is (d_i - L % d_i) % d_i.
#
# Key structural insight: since LCM is the period, L mod d_i = (L mod LCM) mod d_i.
# So if we let p = L mod LCM (0 <= p < LCM), then for each i:
#   offset_i(p) = (DIVISORS[i] - p % DIVISORS[i]) % DIVISORS[i]
# This gives the distance from L to the first multiple of d_i >= L.
#
# The 10 values are: L + offset_i(p) for i=0..9, but we may need to add multiples
# of d_i to resolve collisions.
#
# The span is determined by: max(offset_i(p) + k_i * d_i) - min(offset_i(p) + k_i * d_i)
# where k_i >= 0 are integers chosen to make all values distinct.
#
# For a fixed p, the "base" offsets are fixed. We need to assign distinct values.
# The minimum span for a fixed p is achieved by a greedy assignment: sort by base
# offset, and for ties or conflicts, bump the later one up by d_i.
#
# But we also need all values <= MAX_VAL. Since L + offset_i + k_i*d_i <= MAX_VAL,
# and L can be any value with L mod LCM = p, the constraint is on the absolute values.
# For the best window, we want L as small as possible (to allow room) or as large
# as possible? Actually, for a fixed p and fixed set of offsets, the span is the same
# regardless of L (it's the difference). The constraint is that max_value <= MAX_VAL.
# So L + max(offset_i + k_i * d_i) <= MAX_VAL, i.e., L <= MAX_VAL - max_offset.
# Since L >= 1 and L mod LCM = p, we need p + LCM * t >= 1 for some t >= 0, and
# p + LCM * t <= MAX_VAL - max_offset. This is satisfiable as long as MAX_VAL - max_offset >= 1.
#
# So the problem reduces to: find p in [0, LCM) that minimizes the minimum span
# over distinct assignments of offsets + k_i * d_i.

# Precompute offsets for all p in [0, LCM)
# LCM = 2,00,230 which is manageable.

def construct():
    start = time.time()
    deadline = start + 80.0
    
    # Precompute for each p in [0, LCM): the base offsets
    # offsets[p][i] = (DIVISORS[i] - p % DIVISORS[i]) % DIVISORS[i]
    
    # For efficiency, note that offsets[p][i] depends only on p % DIVISORS[i].
    # We can precompute for each divisor d and each residue r in [0, d): 
    #   off[d][r] = (d - r) % d
    # Then offsets[p][i] = off[DIVISORS[i]][p % DIVISORS[i]]
    
    # Precompute off tables
    off_tables = []
    for d in DIVISORS:
        table = [(d - r) % d for r in range(d)]
        off_tables.append(table)
    
    best_span = MAX_VAL
    best_p = 0
    best_assignment = None
    
    # For each p, compute the base offsets, then find the minimum span assignment
    # Greedy: sort by (base_offset, index), assign in order, bumping if needed.
    
    # But we need to be careful: bumping one might cause another collision.
    # A better approach: for a fixed p, the problem is to assign distinct values
    # v_i = base_i + k_i * d_i, minimizing max(v_i) - min(v_i).
    # This is equivalent to finding the minimum W such that there exist k_i >= 0
    # with all v_i in [m, m+W] for some m, and all v_i distinct.
    #
    # For a fixed p, let's just do a greedy: sort by base offset, assign in order,
    # and for each, pick the smallest value >= previous assigned that is a multiple
    # of d_i (relative to L). Actually, the values are L + base_i + k_i * d_i.
    # The span is (max - min) of (base_i + k_i * d_i).
    #
    # Greedy algorithm for fixed p:
    # 1. Compute base_i for each i.
    # 2. Sort indices by base_i.
    # 3. Assign values in sorted order: for the first, value = base_i.
    #    For subsequent, value = smallest multiple of d_i (i.e., base_i + k*d_i) 
    #    that is >= last assigned value + 1 (to ensure distinctness and ordering).
    #    Wait, they don't need to be in order, just distinct. But sorting and
    #    assigning in increasing order is optimal for minimizing the max.
    #
    # Actually, the optimal assignment for a fixed set of "base" positions is:
    # sort by base, then greedily assign each to the smallest available value 
    # that is a multiple of its divisor and >= the current lower bound.
    
    for p in range(LCM):
        if time.time() > deadline:
            break
        
        # Compute base offsets
        bases = [off_tables[i][p % DIVISORS[i]] for i in range(N)]
        
        # Sort indices by base offset
        order = sorted(range(N), key=lambda i: bases[i])
        
        # Greedy assignment
        assigned = [0] * N
        last = -1
        valid = True
        for idx in order:
            d = DIVISORS[idx]
            base = bases[idx]
            # Smallest value >= last+1 that is of form base + k*d, k>=0
            if base >= last + 1:
                val = base
            else:
                # Need base + k*d >= last + 1
                # k*d >= last + 1 - base
                # k >= ceil((last + 1 - base) / d)
                k = (last + 1 - base + d - 1) // d
                if k < 0:
                    k = 0
                val = base + k * d
            assigned[idx] = val
            last = val
        
        span = last - assigned[order[0]]
        
        if span < best_span:
            best_span = span
            best_p = p
            best_assignment = assigned[:]
            
            # The minimum possible span for 10 distinct integers is 9
            if best_span <= 9:
                break
    
    # Now construct the actual values. We need L such that L mod LCM = best_p
    # and L + max(assigned) <= MAX_VAL.
    # L = best_p + LCM * t, for t >= 0.
    # We need L >= 1, so if best_p == 0, L = LCM * t for t >= 1, else L = best_p + LCM * t for t >= 0.
    # And L + max(assigned) <= MAX_VAL.
    
    max_assigned = max(best_assignment)
    
    # Find smallest valid L
    if best_p == 0:
        L = LCM  # t=1
    else:
        L = best_p  # t=0
    
    # Check if L + max_assigned <= MAX_VAL
    if L + max_assigned > MAX_VAL:
        # Try larger t
        # L + LCM*t + max_assigned <= MAX_VAL
        # LCM*t <= MAX_VAL - L - max_assigned
        # t <= (MAX_VAL - L - max_assigned) // LCM
        # But we need L >= 1, and we want the smallest L that works.
        # Actually, we want ANY valid L. Let's just use the smallest valid L.
        # If best_p == 0, L = LCM (t=1). If that doesn't work, no solution with this p.
        # But it should work since MAX_VAL is large.
        pass
    
    # Construct the result
    result = []
    for i in range(N):
        val = L + best_assignment[i]
        result.append(int(val))
    
    # Verify distinctness
    if len(set(result)) != N:
        # Fallback: use the divisors themselves
        result = [int(d) for d in DIVISORS]
    
    # Verify all <= MAX_VAL
    if any(v > MAX_VAL for v in result):
        # This shouldn't happen, but fallback
        result = [int(d) for d in DIVISORS]
    
    return result
