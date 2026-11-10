import random
import time

def construct():
    # Moduli to consider: all even numbers from 2 to 128
    moduli = list(range(2, 129, 2))
    
    best_solution = []
    start_time = time.time()
    
    # Try multiple random orders to find a better packing
    # Limit time to 60s
    while time.time() - start_time < 55:
        random.shuffle(moduli)
        current = [] # list of [residue, modulus]
        
        for m in moduli:
            found = False
            # Try to find a residue r for m that is compatible with all in current
            # Optimize: check compatibility
            for r in range(m):
                compatible = True
                for r_j, m_j in current:
                    g = __import__('math').gcd(r, r_j) # Wait, gcd of residues? No.
                    # Condition: r != r_j mod gcd(m, m_j)
                    g_mod = __import__('math').gcd(m, m_j)
                    if r % g_mod == r_j % g_mod:
                        compatible = False
                        break
                if compatible:
                    current.append([r, m])
                    found = True
                    break
            
            if found:
                if len(current) > len(best_solution):
                    best_solution = current[:]
            else:
                # If no residue works for this m with current set, skip m
                # (In a greedy add, we just skip)
                pass
        
        # If we found a solution with all moduli, we can break
        if len(best_solution) == len(moduli):
            break

    return best_solution
