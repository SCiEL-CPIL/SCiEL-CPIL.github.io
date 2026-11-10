import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s deadline, 5s safety margin for 60s limit
    
    MAX_MOD = 128
    all_moduli = list(range(2, MAX_MOD + 1))
    
    # Precompute gcd table
    gcd_tbl = {}
    for a in range(2, MAX_MOD + 1):
        for b in range(2, MAX_MOD + 1):
            gcd_tbl[(a, b)] = math.gcd(a, b)
    
    def solve_csp(moduli_list):
        """CSP greedy with MRV and forward checking."""
        valid_rs = {}
        for m in moduli_list:
            valid_rs[m] = list(range(m))
        
        assigned = {}
        remaining = set(moduli_list)
        
        while remaining:
            if time.time() > deadline - 2:
                break
            
            # MRV: find modulus with fewest valid residues
            best_m = None
            best_count = float('inf')
            for m in remaining:
                c = len(valid_rs[m])
                if c < best_count:
                    best_count = c
                    best_m = m
                    if c == 0:
                        break
            
            if best_m is None or best_count == 0:
                break
            
            # Pick first valid residue
            best_r = valid_rs[best_m][0]
            
            assigned[best_m] = best_r
            remaining.discard(best_m)
            
            # Forward checking: update valid_rs for remaining moduli
            for m2 in remaining:
                g = gcd_tbl[(best_m, m2)]
                target = best_r % g
                valid_rs[m2] = [r2 for r2 in valid_rs[m2] if r2 % g != target]
        
        return [[r, m] for (m, r) in assigned.items()]
    
    # Try with all moduli
    result = solve_csp(all_moduli)
    best_len = len(result)
    best_result = result
    
    # Try with even moduli
    if time.time() < deadline - 5:
        even = [m for m in all_moduli if m % 2 == 0]
        res = solve_csp(even)
        if len(res) > best_len:
            best_len = len(res)
            best_result = res
    
    # Try with multiples of 4
    if time.time() < deadline - 3:
        d4 = [m for m in all_moduli if m % 4 == 0]
        res = solve_csp(d4)
        if len(res) > best_len:
            best_len = len(res)
            best_result = res
    
    # Try with multiples of 8
    if time.time() < deadline - 2:
        d8 = [m for m in all_moduli if m % 8 == 0]
        res = solve_csp(d8)
        if len(res) > best_len:
            best_len = len(res)
            best_result = res
    
    # Try to extend: add unused moduli in descending order
    if time.time() < deadline - 1:
        used = set(m for (_, m) in best_result)
        extended = list(best_result)
        for m in sorted(all_moduli, reverse=True):
            if m in used:
                continue
            for r in range(m):
                ok = True
                for (r2, m2) in extended:
                    g = gcd_tbl[(m, m2)]
                    if r % g == r2 % g:
                        ok = False
                        break
                if ok:
                    extended.append([r, m])
                    used.add(m)
                    break
        if len(extended) > best_len:
            best_len = len(extended)
            best_result = extended
    
    return best_result
