"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

MAX_MOD = 128
DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _disjoint(r1, m1, r2, m2):
    """Two residue classes are disjoint iff gcd(m1,m2) does not divide (r2-r1)."""
    g = math.gcd(m1, m2)
    return (r2 - r1) % g != 0

def _can_add(r, m, pairs):
    """Check if (r, m) is disjoint from all pairs in the list."""
    for r2, m2 in pairs:
        if not _disjoint(r, m, r2, m2):
            return False
    return True

def construct():
    start_time = time.time()
    
    # Base set: Powers of 2 with specific residues known to be disjoint
    # m = 2^k, r = 2^(k-1) - 1
    # k=1: m=2, r=0? 
    # Inspiration used: k in range(7) -> m=2^(k+1), r=2^k-1
    # k=0: m=2, r=0 (2^0-1=0). 
    # k=1: m=4, r=1.
    # k=2: m=8, r=3.
    # k=3: m=16, r=7.
    # k=4: m=32, r=15.
    # k=5: m=64, r=31.
    # k=6: m=128, r=127.
    
    base_pairs = []
    for k in range(7):
        m = 1 << (k + 1)
        r = (1 << k) - 1
        base_pairs.append([r, m])
        
    best_pairs = base_pairs[:]
    
    # Candidate moduli: All even numbers from 2 to 128 not already in base_pairs
    base_moduli = {p[1] for p in base_pairs}
    candidates = [m for m in range(2, MAX_MOD + 1) if m % 2 == 0 and m not in base_moduli]
    
    # Sort candidates to prioritize those that might be easier to add
    # Smaller moduli have fewer residues to check, but also fewer "slots"
    # Let's try ascending order first
    candidates.sort()
    
    for m in candidates:
        if time.time() - start_time > DEADLINE - 5:
            break
            
        found = False
        # Try to find a residue r for this modulus m
        # We can iterate through all r in range(m)
        for r in range(m):
            if _can_add(r, m, best_pairs):
                best_pairs.append([r, m])
                found = True
                break
        
        # If we didn't find one, we skip this modulus
        # We don't backtrack, so we might miss a combination where
        # skipping m allows adding more later, but greedy is fast.
        
    # Try a second pass with different ordering or randomization if time permits
    # But the greedy pass above is likely quite good.
    # Let's try to add any remaining odd moduli? 
    # As analyzed, odd moduli will likely conflict with the power-of-2 base (gcd=1).
    # So we stick to even.
    
    return best_pairs
# EVOLVE-BLOCK-END
