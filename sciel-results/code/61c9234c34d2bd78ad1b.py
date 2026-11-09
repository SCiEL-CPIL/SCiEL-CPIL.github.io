"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 50  # safety margin before 60s limit
    p = 1009
    # Precompute modular inverses
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    # Identify self-inverse elements (x^2 ≡ 1 mod p): x=1 and x=p-1=1008
    self_inv = {1, p - 1}
    
    # For each element x, its inverse partner is inv[x]
    # Two elements x,y conflict if y == inv[x]
    # Note: inv[inv[x]] == x, so conflicts come in pairs (x, inv[x]) where x != inv[x]
    # Self-inverse elements conflict with themselves
    
    best_len = 0
    best_start = 1
    
    # Two-pointer sliding window
    e = 0
    for s in range(1, p):
        if time.time() > deadline:
            break
        # Optimization: if max possible length from s is <= best_len, break
        if p - s <= best_len:
            break
        
        # Ensure e >= s
        if e < s:
            e = s - 1
        
        # Extend e as far as possible
        while e + 1 < p:
            e_next = e + 1
            # Check if e_next is self-inverse (conflicts with itself)
            if e_next in self_inv:
                break
            # Check if inv[e_next] is already in [s, e_next-1]
            inv_e = inv[e_next]
            if s <= inv_e <= e_next - 1:
                break
            e = e_next
        
        current_len = e - s + 1
        if current_len > best_len:
            best_len = current_len
            best_start = s
    
    if best_len == 0:
        # Fallback: return a valid single-element interval
        best_start = 2
        best_len = 1
    
    # Verify the result is valid
    start = best_start
    length = best_len
    valid = True
    for x in range(start, start + length):
        if x in self_inv:
            valid = False
            break
        for y in range(start, start + length):
            if (x * y) % p == 1:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: find a valid interval by brute force
        for s in range(2, p - 1):
            for l in range(min(p - s, best_len), 0, -1):
                ok = True
                for x in range(s, s + l):
                    if x in self_inv:
                        ok = False
                        break
                    for y in range(s, s + l):
                        if (x * y) % p == 1:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    return [s, l]
        return [2, 1]
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
