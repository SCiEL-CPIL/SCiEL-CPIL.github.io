import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MAX_M = 128
    moduli = [m for m in range(2, MAX_M + 1) if m % 2 == 0]
    n = len(moduli)
    
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        gcd_tab[i][i] = moduli[i]
        for j in range(i + 1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_tab[i][j] = g
            gcd_tab[j][i] = g

    # Precompute masks for each pair (i, j) and residue r of i
    # mask[i][j][r] is a bitmask of residues for j that are FORBIDDEN by r_i = r
    # This is too much memory: 64 * 64 * 128 integers. 524k integers. Okay.
    # But we can compute on the fly or precompute per pair.
    
    # Let's use the bitmask domain reduction approach.
    # domains[j] is a bitmask of available residues for j.
    
    # Order variables by domain size (modulus)
    order = sorted(range(n), key=lambda i: moduli[i])
    # Reverse order? Most constrained first?
    # Smaller modulus -> smaller domain -> more constrained?
    # Actually, larger gcd with others -> more constraints.
    # Let's stick to ascending modulus.
    
    assigned = [-1] * n
    domains = [0] * n
    for i in range(n):
        domains[i] = (1 << moduli[i]) - 1

    def backtrack(idx):
        if time.time() > deadline - 2:
            return False
        if idx == n:
            return True
        
        i = order[idx]
        m = moduli[i]
        
        # Get available residues for i
        mask = domains[i]
        if mask == 0:
            return False
            
        # Iterate through set bits
        r = 0
        while mask:
            if mask & 1:
                # Try residue r
                assigned[i] = r
                # Update domains of future variables
                new_domains = [0] * n # Copy? No, use undo stack or just copy if n is small.
                # n=64, copying list of 64 ints is fast.
                new_domains = domains[:]
                
                success = True
                for next_idx in range(idx + 1, n):
                    j = order[next_idx]
                    g = gcd_tab[i][j]
                    # Residues r_j such that (r_j - r) % g == 0 are forbidden
                    # r_j = r + k*g
                    # Mask out these residues from new_domains[j]
                    
                    # Create mask of forbidden residues for j
                    forb_mask = 0
                    m_j = moduli[j]
                    # r_j in [0, m_j-1]
                    # r_j = r % g + k*g
                    start = r % g
                    for k in range(0, m_j, g):
                        val = start + k
                        if val < m_j:
                            forb_mask |= (1 << val)
                        else:
                            break
                    # Also need to handle if r % g is not the only start?
                    # Yes, r_j = r + t*g.
                    # r_j mod g == r mod g.
                    # So all r_j with same mod g are forbidden.
                    
                    new_domains[j] &= ~forb_mask
                    if new_domains[j] == 0:
                        success = False
                        break
                
                if success:
                    # Recurse
                    # Pass new_domains?
                    # To avoid passing large arrays, we can use a global array and undo.
                    # But for simplicity and n=64, copying is fine.
                    # However, recursion depth 64, copying 64 ints each time -> 64*64 operations.
                    # 4096 ops per level. 64 levels. 262k ops. Very fast.
                    
                    # But we need to restore domains on backtracking.
                    # So we must pass a copy or save/restore.
                    # Let's use a global domains array and save/restore.
                    
                    # Actually, passing a copy is safer and simple.
                    # But Python list copy is O(n).
                    
                    # Let's try a different structure:
                    # We will use a global `current_domains` and a stack of changes.
                    
                    # For now, let's just try the simple copy approach in the recursion call.
                    # It might be slow in Python due to overhead.
                    
                    # Alternative:
                    # Save old domains
                    old_domains = domains[:]
                    domains[:] = new_domains
                    if backtrack(idx + 1):
                        return True
                    domains[:] = old_domains
                    assigned[i] = -1
                else:
                    assigned[i] = -1
            
            mask >>= 1
            r += 1
            
        return False

    if backtrack(0):
        return [[assigned[i], moduli[i]] for i in range(n)]

    # Fallback: Greedy
    selected = []
    for i in order:
        m = moduli[i]
        for r in range(m):
            ok = True
            for (idx2, r2) in selected:
                g = gcd_tab[i][idx2]
                if (r - r2) % g == 0:
                    ok = False
                    break
            if ok:
                selected.append((i, r))
                break
    if selected:
        return [[r, moduli[idx]] for (idx, r) in selected]
    return [[0, 2]]
