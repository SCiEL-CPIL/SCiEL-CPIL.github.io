import math

def construct():
    # Candidate moduli: all even numbers in [20, 200]
    # This ensures gcd(m_i, m_j) >= 2 for all pairs, allowing disjoint residues.
    # Odd numbers are coprime to evens, so we cannot mix them if we include any even.
    # Including any even (like 20) forces us to exclude all odds.
    # Sum of 1/m for evens is maximized by including all possible evens.
    
    moduli = list(range(20, 201, 2))
    
    def try_assign(mod_list):
        assigned = {} # m -> r
        # Sort by m ascending to prioritize high weights (small m)
        for m in mod_list:
            found = False
            for r in range(m):
                valid = True
                for m2, r2 in assigned.items():
                    g = math.gcd(m, m2)
                    if r % g == r2 % g:
                        valid = False
                        break
                if valid:
                    assigned[m] = r
                    found = True
                    break
            if not found:
                return None
        return assigned

    current_moduli = moduli
    result = try_assign(current_moduli)
    
    # If the full set fails, remove the largest modulus (lowest weight) and retry
    while result is None and current_moduli:
        current_moduli.pop() # Remove largest m
        result = try_assign(current_moduli)
        
    if result is None:
        # Fallback: just return empty or a single valid pair
        return [[20, 0]]
        
    return [[m, r] for m, r in result.items()]
