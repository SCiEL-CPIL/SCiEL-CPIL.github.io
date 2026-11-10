"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    p = 1009
    N = p - 1  # 1008
    
    # Precompute modular inverses using Fermat's little theorem (p is prime)
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # For each x, the conflict pair is (min(x, inv[x]), max(x, inv[x]))
    # An interval [s, e] is valid iff it does NOT contain both elements of any conflict pair.
    # For a given start s, the maximum valid end is:
    #   e_max(s) = min{ R | (L, R) is a conflict pair, L >= s } - 1
    # because if L >= s and e >= R, then both L and R are in [s, e].
    
    # Precompute min_R_from[s] = min{ R | (L, R) is a conflict pair, L >= s }
    # Using suffix minimum: iterate s from N down to 1.
    
    # First, for each possible L value, find the minimum R among pairs with that L.
    # Note: multiple x can have the same L = min(x, inv[x]).
    INF = float('inf')
    min_R_at_L = [INF] * (p + 2)  # index 1..1008
    
    for x in range(1, p):
        y = inv[x]
        L = min(x, y)
        R = max(x, y)
        if R < min_R_at_L[L]:
            min_R_at_L[L] = R
    
    # Suffix minimum: min_R_from[s] = min(min_R_at_L[s], min_R_from[s+1])
    min_R_from = [INF] * (p + 2)
    min_R_from[p] = INF  # no pairs with L >= 1009
    for s in range(p - 1, 0, -1):
        min_R_from[s] = min(min_R_at_L[s], min_R_from[s + 1])
    
    # Now find the longest valid interval
    best_len = 0
    best_start = 1
    
    for s in range(1, p):
        if time.time() > deadline - 2.0:
            break
        
        # Maximum valid end for start s
        max_e = min_R_from[s] - 1
        if max_e > N:
            max_e = N
        
        if max_e >= s:
            length = max_e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
    
    # Verify the found interval is valid (sanity check)
    s_final = best_start
    e_final = s_final + best_len - 1
    for x in range(s_final, e_final + 1):
        for y in range(x, e_final + 1):
            if (x * y) % p == 1:
                # Fallback: do a full brute-force search
                best_len_bf = 0
                best_start_bf = 1
                for s2 in range(1, p):
                    for e2 in range(s2, p):
                        valid = True
                        for x2 in range(s2, e2 + 1):
                            for y2 in range(x2, e2 + 1):
                                if (x2 * y2) % p == 1:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            len2 = e2 - s2 + 1
                            if len2 > best_len_bf:
                                best_len_bf = len2
                                best_start_bf = s2
                return [best_start_bf, best_len_bf]
    
    return [best_start, best_len]

# EVOLVE-BLOCK-END
