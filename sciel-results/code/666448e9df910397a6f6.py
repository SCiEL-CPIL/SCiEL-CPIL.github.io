"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    MOD = 1009
    N = 1008
    
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    # Precompute modular inverses: inv[x] = x^(-1) mod 1009 for x in [1, 1008]
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    best_start = 2
    best_len = 1
    
    # Two-pointer sliding window:
    # For each left pointer S, extend right pointer E as far as possible
    # such that [S, E] is valid (no x*y % MOD == 1 for any x,y in [S,E]).
    #
    # Key: when we extend E to new_E, the only new conflicts are pairs
    # (new_E, x) where x is already in [S, E-1] and new_E * x % MOD == 1,
    # i.e., x == inv[new_E]. So we just check if inv[new_E] is in [S, E-1].
    #
    # Also, if new_E is self-inverse (1 or 1008), we cannot include it.
    #
    # The window is monotone: if [S, E] is valid, then [S, E'] for E' < E is valid.
    # If [S, E] becomes invalid when adding E+1, then for S+1 we can try extending again.
    
    E = 0  # right pointer, starts before any valid position
    
    for S in range(2, MOD):  # S from 2 to 1007 (avoid self-inverses 1 and 1008)
        if time.time() > deadline:
            break
        
        # Ensure E >= S (window must contain at least S)
        if E < S - 1:
            E = S - 1
        
        # Extend E as far as possible
        while E + 1 <= 1007:  # E can go up to 1007 (avoid 1008 which is self-inverse)
            new_E = E + 1
            
            # Check if new_E is self-inverse (shouldn't happen for 2..1007, but be safe)
            if new_E * new_E % MOD == 1:
                break
            
            # Check if adding new_E creates a conflict with existing elements in [S, E]
            inv_new_E = inv[new_E]
            
            # Conflict if inv_new_E is in [S, E]
            if S <= inv_new_E <= E:
                break
            
            # Also need to check: for each x in [S, E], does x * new_E % MOD == 1?
            # That's equivalent to checking if inv[new_E] is in [S, E], which we just did.
            # But wait: we also need to check pairs among existing elements? No, because
            # [S, E] was already valid, so no conflicts among existing elements.
            # The only new pairs are (new_E, x) for x in [S, E].
            
            E = new_E
        
        if E >= S:
            current_len = E - S + 1
            if current_len > best_len:
                best_len = current_len
                best_start = S
    
    # Verification pass: confirm the best interval is truly valid
    if time.time() < deadline - 5:
        S = best_start
        E = best_start + best_len - 1
        valid = True
        for x in range(S, E + 1):
            for y in range(x, E + 1):  # check x<=y to cover all pairs including diagonal
                if x * y % MOD == 1:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback: brute-force search for longest valid interval
            best_start = 2
            best_len = 1
            for S in range(2, MOD):
                if time.time() > deadline:
                    break
                if S * S % MOD == 1:
                    continue
                E = S - 1
                while E + 1 <= 1007:
                    new_E = E + 1
                    if new_E * new_E % MOD == 1:
                        break
                    # Check all pairs in [S, new_E] involving new_E
                    conflict = False
                    for x in range(S, E + 1):
                        if x * new_E % MOD == 1:
                            conflict = True
                            break
                    if conflict:
                        break
                    E = new_E
                if E >= S:
                    current_len = E - S + 1
                    if current_len > best_len:
                        best_len = current_len
                        best_start = S
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
