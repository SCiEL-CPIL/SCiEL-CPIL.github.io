"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    moduli = list(range(2, 129))
    
    # Precompute gcd table
    gcd_table = [[0]*129 for _ in range(129)]
    for i in range(2, 129):
        for j in range(i, 129):
            gcd_table[i][j] = math.gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
            
    # Precompute masks for each m, g, r
    masks = {}
    for m in moduli:
        masks[m] = {}
        divs = [i for i in range(1, m+1) if m % i == 0]
        for g in divs:
            masks[m][g] = [0] * g
            for r in range(g):
                mask = 0
                for k in range(r, m, g):
                    mask |= (1 << k)
                masks[m][g][r] = mask
                
    def solve_exact(moduli_subset):
        domains = {}
        for m in moduli_subset:
            domains[m] = (1 << m) - 1
        assigned = {}
        
        def backtrack(idx):
            if time.time() > deadline:
                return None
            if idx == len(moduli_subset):
                return assigned.copy()
            
            m = moduli_subset[idx]
            dom = domains[m]
            if dom == 0:
                return None
            
            # Extract candidates
            candidates = []
            d = dom
            while d:
                lsb = d & -d
                r_val = lsb.bit_length() - 1
                candidates.append(r_val)
                d ^= lsb
                
            # Shuffle candidates for variety? No, deterministic is better for reproducibility
            # But maybe try 0, m-1, etc?
            
            for r in candidates:
                assigned[m] = r
                old_domains = {}
                valid = True
                for j in range(idx + 1, len(moduli_subset)):
                    m2 = moduli_subset[j]
                    g = gcd_table[m][m2]
                    mask = masks[m2][g][r % g]
                    new_dom = domains[m2] & ~mask
                    if new_dom == 0:
                        valid = False
                        break
                    old_domains[m2] = domains[m2]
                    domains[m2] = new_dom
                    
                if valid:
                    res = backtrack(idx + 1)
                    if res:
                        return res
                        
                for m2, old_dom in old_domains.items():
                    domains[m2] = old_dom
                del assigned[m]
                
            return None
            
        return backtrack(0)
        
    def solve_greedy(moduli_subset):
        # Greedy with lookahead
        domains = {}
        for m in moduli_subset:
            domains[m] = (1 << m) - 1
        assigned = {}
        remaining = moduli_subset[:]
        
        while remaining:
            if time.time() > deadline:
                break
            # Pick modulus with smallest domain
            m = min(remaining, key=lambda x: bin(domains[x]).count('1'))
            remaining.remove(m)
            
            dom = domains[m]
            if dom == 0:
                return None # Failed
            
            # Pick residue that minimizes domain reduction
            best_r = None
            min_loss = float('inf')
            
            d = dom
            candidates = []
            while d:
                lsb = d & -d
                r_val = lsb.bit_length() - 1
                candidates.append(r_val)
                d ^= lsb
                
            for r in candidates:
                loss = 0
                for m2 in remaining:
                    g = gcd_table[m][m2]
                    mask = masks[m2][g][r % g]
                    # Count bits removed
                    removed = domains[m2] & mask
                    loss += bin(removed).count('1')
                    
                if loss < min_loss:
                    min_loss = loss
                    best_r = r
                    
            assigned[m] = best_r
            # Update domains
            for m2 in remaining:
                g = gcd_table[m][m2]
                mask = masks[m2][g][best_r % g]
                domains[m2] &= ~mask
                
        return assigned
        
    best_sol = []
    
    # 1. Try exact solver on multiples of 4
    mult4 = [m for m in moduli if m % 4 == 0]
    sol = solve_exact(mult4)
    if sol:
        best_sol = sol
        
    # 2. Try exact solver on multiples of 6
    if len(best_sol) < 30:
        mult6 = [m for m in moduli if m % 6 == 0]
        sol = solve_exact(mult6)
        if sol and len(sol) > len(best_sol):
            best_sol = sol
            
    # 3. Try greedy on all even moduli
    if len(best_sol) < 60:
        even_moduli = [m for m in moduli if m % 2 == 0]
        sol = solve_greedy(even_moduli)
        if sol and len(sol) > len(best_sol):
            best_sol = sol
            
    # 4. Fallback
    if not best_sol:
        best_sol = {2: 1}
        
    return [[r, m] for m, r in best_sol.items()]
# EVOLVE-BLOCK-END
