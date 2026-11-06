"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s limit to be safe
    
    max_mod = 128
    
    # Find all primes up to max_mod
    primes = []
    for i in range(2, max_mod + 1):
        is_prime = True
        for j in range(2, int(math.isqrt(i)) + 1):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
            
    result = []
    
    for p in primes:
        # Find powers of p up to max_mod
        powers = []
        current = p
        while current <= max_mod:
            powers.append(current)
            current *= p
            
        if not powers:
            continue
            
        # Determine residues for these powers
        # r[j] corresponds to powers[j]
        residues = [0] * len(powers)
        
        for j in range(len(powers)):
            m_j = powers[j]
            # Find smallest r in [0, m_j) such that r not in forbidden classes
            # Forbidden classes are defined by previous residues r_i mod p^i
            found = False
            for r_candidate in range(m_j):
                valid = True
                for i in range(j):
                    m_i = powers[i]
                    r_i = residues[i]
                    if r_candidate % m_i == r_i % m_i:
                        valid = False
                        break
                if valid:
                    residues[j] = r_candidate
                    found = True
                    break
            if not found:
                # Should not happen based on theory, but fallback
                residues[j] = 0
                
        for idx, m in enumerate(powers):
            result.append([residues[idx], m])
            
        if time.time() > deadline:
            break
            
    return result
# EVOLVE-BLOCK-END
