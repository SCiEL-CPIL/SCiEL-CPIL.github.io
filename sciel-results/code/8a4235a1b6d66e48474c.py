"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    p = 1009
    
    # Precompute modular inverses using Fermat's little theorem (p is prime)
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # For a fixed start s, the maximum valid end e is:
    # e_max(s) = min{v : v >= s, inv[v] >= s} - 1
    # Because if v >= s and inv[v] >= s, then both v and inv[v] are >= s.
    # If e >= v, then v is in [s, e]. Since inv[v] >= s and inv[v] <= e (if e >= inv[v]), conflict.
    # Actually, we need: for all x in [s, e], inv[x] not in [s, e].
    # This is equivalent to: for all x in [s, e], inv[x] < s or inv[x] > e.
    # 
    # Let m(s) = min{v : v >= s, inv[v] >= s}.
    # Then for any e < m(s), for all x in [s, e]: if inv[x] >= s, then inv[x] >= m(s) > e, so inv[x] > e. OK.
    # If inv[x] < s, OK.
    # So [s, m(s)-1] is valid.
    # For e >= m(s), let v = m(s). Then v >= s and inv[v] >= s. If e >= inv[v], then both v and inv[v] are in [s, e], conflict.
    # But if e < inv[v], it might still be valid. However, m(s) is the minimum v, so inv[m(s)] could be large.
    # 
    # Actually, the correct condition: [s, e] is valid iff for all x in [s, e], inv[x] > e or inv[x] < s.
    # This is equivalent to: e < min{inv[x] : x in [s, e], inv[x] >= s}.
    # 
    # Let f(s) = min{v : v >= s, inv[v] >= s}.
    # Claim: max valid e for start s is f(s) - 1.
    # Proof: 
    #   (1) [s, f(s)-1] is valid: For any x in [s, f(s)-1], if inv[x] >= s, then inv[x] >= f(s) > f(s)-1 >= e. So inv[x] > e. If inv[x] < s, OK.
    #   (2) [s, f(s)] is invalid: Let v = f(s). Then v >= s and inv[v] >= s. Since v = f(s) is the minimum such v, we have v <= inv[v] (not necessarily, but v >= s and inv[v] >= s). 
    #        Actually, we need to check if both v and inv[v] are in [s, f(s)]. v = f(s) is in [s, f(s)]. inv[v] >= s. Is inv[v] <= f(s)? Not necessarily.
    #        If inv[v] > f(s), then [s, f(s)] might still be valid.
    #
    # So the formula is not simply f(s) - 1. Let me reconsider.
    #
    # Better approach: for fixed s, the constraint is that for all x in [s, e], inv[x] > e or inv[x] < s.
    # This is equivalent to: e < min{inv[x] : x in [s, e], inv[x] >= s}.
    # Let g(s, e) = min{inv[x] : x in [s, e], inv[x] >= s}.
    # Then [s, e] is valid iff e < g(s, e).
    #
    # As e increases, g(s, e) is non-increasing (we're taking min over a larger set).
    # So once e >= g(s, e), it stays invalid for larger e.
    #
    # We can find the max valid e by binary search or linear scan.
    # Given p=1009, linear scan is fine.
    
    best_len = 0
    best_start = 1
    
    for s in range(1, p):
        if time.time() > deadline - 2.0:
            break
            
        # Find max valid e for this s
        max_valid_e = s - 1
        for e in range(s, p):
            # Check if [s, e] is valid
            # It's valid iff for all x in [s, e], inv[x] < s or inv[x] > e
            # Equivalently: no x in [s, e] with s <= inv[x] <= e
            # Which is: min{inv[x] : x in [s, e], inv[x] >= s} > e (or no such x)
            
            # Check if e itself causes a conflict:
            # inv[e] in [s, e] means s <= inv[e] <= e
            if inv[e] >= s and inv[e] <= e:
                # Conflict: e and inv[e] are both in [s, e] (if inv[e] != e, then inv[e] < e; if inv[e] == e, self-conflict)
                break
            else:
                # No new conflict introduced by e
                max_valid_e = e
                length = e - s + 1
                if length > best_len:
                    best_len = length
                    best_start = s
    
    return [best_start, best_len]

# EVOLVE-BLOCK-END
