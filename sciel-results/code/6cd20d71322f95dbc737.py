import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    # Heuristic: Try to include moduli with many factors first? Or few?
    # Powers of 2 worked well.
    # Let's try a greedy approach.
    
    best_set = []
    
    # Try different orderings
    orderings = [
        sorted(moduli),
        sorted(moduli, reverse=True),
        sorted(moduli, key=lambda x: (bin(x).count('1'), x)), # Fewer bits first
        sorted(moduli, key=lambda x: (bin(x).count('1'), x), reverse=True)
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        current_set = []
        for m in order:
            if time.time() > deadline:
                break
            # Find a residue r for m such that it is disjoint from all in current_set
            valid_r = None
            # We can iterate r from 0 to m-1
            # Optimization: Only check residues that are not congruent to existing ones mod gcd
            found = False
            for r in range(m):
                ok = True
                for r_i, m_i in current_set:
                    g = __import__('math').gcd(m, m_i)
                    if r % g == r_i % g:
                        ok = False
                        break
                if ok:
                    valid_r = r
                    found = True
                    break
            if found:
                current_set.append([valid_r, m])
        
        if len(current_set) > len(best_set):
            best_set = current_set
            if time.time() > deadline:
                break
                
    # If best_set is empty or small, fallback to seed
    if len(best_set) < 7:
        return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
        
    return best_set
