# EVOLVE-BLOCK-START
def construct():
    """
    Find the longest contiguous interval [start, start+length-1] within [1, 1008]
    such that for all x, y in the interval (including x=y), x*y % 1009 != 1.
    
    We use a brute-force approach with early termination for each starting point.
    Since N=1008 is small, O(N^2) is efficient and guarantees finding the global maximum.
    """
    p = 1009
    n = 1008
    
    # Precompute modular inverses for all x in [1, 1008]
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    best_len = 0
    best_start = 1
    
    # Iterate over all possible start positions A
    for A in range(1, n + 1):
        # If A is a self-inverse, it cannot be part of any valid interval
        if A == 1 or A == 1008:
            continue
        
        B = A
        # Extend B as far as possible while maintaining validity
        while B <= n:
            # If B is a self-inverse, it cannot be part of any valid interval
            if B == 1 or B == 1008:
                break
            
            # Check if adding B creates a conflict with any element in [A, B-1]
            # The only element that could conflict with B is inv[B].
            # If inv[B] is in [A, B-1], then we have a pair (inv[B], B) with product 1.
            # Note: inv[B] == B is handled by the self-inverse check above.
            if A <= inv[B] < B:
                break
            
            # If we reach here, [A, B] is a valid interval
            curr_len = B - A + 1
            if curr_len > best_len:
                best_len = curr_len
                best_start = A
            
            B += 1
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
