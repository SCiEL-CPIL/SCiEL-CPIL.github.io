import math
import random
import time

MAX_MOD = 128
DEADLINE_MARGIN = 5.0

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r2 - r1) % g != 0

def construct():
    t0 = time.time()
    deadline = t0 + 60.0 - DEADLINE_MARGIN
    
    all_mods = list(range(2, MAX_MOD + 1))
    best_pairs = []
    
    # We will run multiple restarts of a greedy algorithm
    # Greedy: Pick moduli in a random order. For each, find a valid residue.
    # If no residue works, skip the modulus.
    
    num_restarts = 2000
    
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
            
        random.shuffle(all_mods)
        current_pairs = []
        
        for m in all_mods:
            # Try to find a residue r in [0, m-1] that is disjoint from all current pairs
            # We randomize the order of residues to escape local optima in the greedy choice
            residues = list(range(m))
            random.shuffle(residues)
            
            valid_r = None
            for r in residues:
                is_valid = True
                for r_k, m_k in current_pairs:
                    if not _disjoint(r, m, r_k, m_k):
                        is_valid = False
                        break
                if is_valid:
                    valid_r = r
                    break
            
            if valid_r is not None:
                current_pairs.append((valid_r, m))
            
        # Update best
        if len(current_pairs) > len(best_pairs):
            best_pairs = current_pairs[:]
            
            # Optional: If we hit a known high bound, we could stop, but we don't know the exact max.
            # 64 is the count of even moduli. If we get close to 64, it's good.
            # But we established that 2,4,6 conflict. So max is likely less than 64.
            # Let's just keep searching until time runs out.

    if not best_pairs:
        # Fallback: A single class is always valid
        best_pairs = [(0, 2)]
        
    return [[r, m] for r, m in best_pairs]
