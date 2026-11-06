"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from functools import lru_cache

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _v2(n):
    """Calculate the exponent of 2 in the prime factorization of n."""
    count = 0
    while n % 2 == 0:
        count += 1
        n //= 2
    return count

def construct():
    MAX_M = 128
    
    # Generate all possible moduli
    moduli = list(range(2, MAX_M + 1))
    
    # Sort moduli by descending v2, then descending value
    # This prioritizes moduli with large powers of 2, leading to larger gcds
    moduli.sort(key=lambda m: (-_v2(m), -m))
    
    selected = []
    
    for m in moduli:
        # Try to find a residue r for this modulus
        found = False
        for r in range(m):
            valid = True
            for r_j, m_j in selected:
                g = _gcd(m, m_j)
                if r % g == r_j % g:
                    valid = False
                    break
            if valid:
                selected.append([r, m])
                found = True
                break
        if not found:
            # If no residue found, skip this modulus
            pass
            
    return selected
# EVOLVE-BLOCK-END
