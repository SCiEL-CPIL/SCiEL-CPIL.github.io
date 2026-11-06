"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    n = p - 1  # 1008
    
    # Compute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Self-inverse residues: x * x % p == 1
    # These are x = 1 and x = p-1 = 1008
    self_inv = set()
    for x in range(1, p):
        if (x * x) % p == 1:
            self_inv.add(x)
    
    # For a fixed start s, the maximum valid end e is found by incremental extension.
    # When adding candidate_e to [s, candidate_e-1], the only new constraint to check is:
    #   inv[candidate_e] must not be in [s, candidate_e-1]
    # (and candidate_e must not be self-inverse)
    #
    # Additionally, we need to ensure that for all x in [s, candidate_e-1],
    # inv[x] is not candidate_e. But inv[x] = candidate_e iff x = inv[candidate_e],
    # so this is the same condition: inv[candidate_e] not in [s, candidate_e-1].
    
    best_start = 2
    best_len = 0
    
    deadline = time.time() + 85  # 85 second deadline (5s safety margin)
    
    for s in range(1, n + 1):
        if time.time() > deadline:
            break
        if s in self_inv:
            continue
        
        e = s
        for candidate_e in range(s + 1, n + 1):
            if time.time() > deadline:
                break
            if candidate_e in self_inv:
                break
            ie = inv[candidate_e]
            # Check if ie is in [s, candidate_e - 1]
            if s <= ie < candidate_e:
                break
            e = candidate_e
        
        length = e - s + 1
        if length > best_len:
            best_len = length
            best_start = s
    
    # Verify the result
    s = best_start
    e = s + best_len - 1
    for x in range(s, e + 1):
        if (x * x) % p == 1:
            return [2, 1]  # Fallback
        for y in range(s, e + 1):
            if (x * y) % p == 1:
                return [2, 1]  # Fallback
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
