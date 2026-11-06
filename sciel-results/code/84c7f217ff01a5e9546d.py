import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline with 5s margin
    
    # Moduli from 2 to 128
    all_moduli = list(range(2, 129))
    
    # Precompute gcd for all pairs
    n = len(all_moduli)
    gcd_table = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = gcd(all_moduli[i], all_moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Key insight: coprime moduli cannot both be selected
    # So we need a pairwise non-coprime subset
    # Start with all even numbers as a strong candidate
    even_moduli = [m for m in all_moduli if m % 2 == 0]
    
    # Try to solve CSP: assign residues to a subset of moduli
    # Use backtracking with MRV heuristic
    
    def solve(moduli_list):
        """Try to assign residues to all moduli in moduli_list such that all pairs are disjoint."""
        k = len(moduli_list)
        if k == 0:
            return []
        
        # Order moduli by most constrained first (highest degree / smallest domain)
        # For now, just use the given order
        residues = [None] * k
        
        def backtrack(idx):
            if time.time() > deadline:
                return None
            if idx == k:
                return residues[:]
            
            m = moduli_list[idx]
            # Try each possible residue
            for r in range(m):
                ok = True
                for j in range(idx):
                    g = gcd(moduli_list[idx], moduli_list[j])
                    if (r - residues[j]) % g == 0:
                        ok = False
                        break
                if ok:
                    residues[idx] = r
                    result = backtrack(idx + 1)
                    if result is not None:
                        return result
            residues[idx] = None
            return None
        
        return backtrack(0)
    
    # Strategy: start with a large candidate set and try to solve
    # Use even moduli first, ordered by ascending value
    candidate = sorted(even_moduli)
    
    # Try to solve with all even moduli
    result = solve(candidate)
    if result is not None:
        # Build output
        output = [[result[i], candidate[i]] for i in range(len(candidate))]
        return output
    
    # If full set fails, try removing moduli one by one
    # Use a greedy approach: try to build up the set
    selected = []
    residues = []
    
    for m in sorted(all_moduli):
        if time.time() > deadline:
            break
        # Check if m can be added
        # Must be non-coprime with all selected
        can_add = True
        for sm in selected:
            if gcd(m, sm) == 1:
                can_add = False
                break
        if not can_add:
            continue
        
        # Try to find a residue for m that works with all selected
        found = False
        for r in range(m):
            ok = True
            for j, sm in enumerate(selected):
                g = gcd(m, sm)
                if (r - residues[j]) % g == 0:
                    ok = False
                    break
            if ok:
                selected.append(m)
                residues.append(r)
                found = True
                break
        
        if not found:
            # Try removing some previously selected and re-adding
            # For now, skip
            pass
    
    if selected:
        output = [[residues[i], selected[i]] for i in range(len(selected))]
        return output
    
    # Fallback: just return the seed solution
    return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
