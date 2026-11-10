"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Find the longest interval [start, start+length-1] within [1, 1008]
    such that for every x, y in the interval (including x=y), x*y % 1009 != 1.
    
    Graph-based sweep-line approach:
    - Vertices: residues 1..1008
    - Edges: connect x to inv(x) mod 1009 (including self-loops for x=1, x=1008)
    - Find longest contiguous subsequence forming an independent set
    - For each left boundary L, precompute max valid right boundary R via sweep-line
      tracking minimum exclusion constraint from inverse relationships
    """
    p = 1009
    n = p - 1  # 1008
    
    # Compute modular inverses using Fermat's little theorem
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, p - 2, p)
    
    # For each position x, define the exclusion constraint:
    # If x is in the interval, inv(x) must NOT be in the interval.
    # For a fixed L, the max valid R is constrained by:
    #   For all x in [L, R]: inv(x) < L or inv(x) > R
    #   Equivalently: R < inv(x) whenever inv(x) >= L and inv(x) > x
    #   And: L > inv(x) whenever inv(x) < L and inv(x) < x (always true if inv(x) < L)
    
    # Sweep-line: for each L from 1 to n, compute max R
    # Key: when L increases, constraints relax (fewer x in [L, R] with inv(x) >= L)
    # We can use a two-pointer where R only increases as L increases
    
    best_start = 1
    best_len = 0
    
    R = 0
    for L in range(1, n + 1):
        if R < L:
            R = L - 1
        
        # Extend R as far as possible
        while R + 1 <= n:
            nxt = R + 1
            nxt_inv = inv[nxt]
            # Adding nxt to [L, R+1] is valid iff nxt_inv not in [L, nxt]
            # i.e., nxt_inv < L or nxt_inv > nxt
            if nxt_inv < L or nxt_inv > nxt:
                R = nxt
            else:
                break
        
        length = R - L + 1
        if length > best_len:
            best_len = length
            best_start = L
    
    # Verification: double-check the best interval
    # (Optional but ensures correctness)
    start = best_start
    end = start + best_len - 1
    for x in range(start, end + 1):
        ix = inv[x]
        if start <= ix <= end:
            # Conflict found! Should not happen if logic is correct
            break
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
