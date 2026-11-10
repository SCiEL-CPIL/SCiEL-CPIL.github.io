"""Combined program: greedy maximization with seed-based improvement passes."""
# EVOLVE-BLOCK-START
import time
import math

def _is_disjoint(r1, m1, r2, m2):
    """Check if two residue classes are disjoint using exact gcd criterion.
    Classes {x: x ≡ r1 mod m1} and {x: x ≡ r2 mod m2} are disjoint iff
    gcd(|r1-r2|, m1, m2) > 1.
    """
    d = abs(r1 - r2)
    return math.gcd(math.gcd(d, m1), m2) > 1

def _can_add(r, m, chosen):
    """Check if (r, m) is disjoint from all classes in chosen."""
    for r2, m2 in chosen:
        if not _is_disjoint(r, m, r2, m2):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    MAX_MOD = 128
    
    best_count = 0
    best_result = None
    
    # Strategy 1: Start with the seed (7 classes with powers of 2) and improve
    seed_pairs = [(1, 2), (3, 4), (7, 8), (15, 16), (31, 32), (63, 64), (127, 128)]
    
    # Multiple improvement passes with different strategies
    for pass_num in range(10):
        if time.time() > deadline - 10:
            break
        
        # Try different starting points and orderings
        starting_points = [
            list(seed_pairs),
            [],  # empty start
        ]
        
        for start in starting_points:
            if time.time() > deadline - 10:
                break
            
            chosen = list(start)
            used_moduli = set(m for _, m in chosen)
            
            # Multiple rounds of adding moduli
            for round_num in range(5):
                if time.time() > deadline - 10:
                    break
                
                # Try different modulus orderings
                remaining_moduli = [m for m in range(2, MAX_MOD + 1) if m not in used_moduli]
                
                orderings = [
                    remaining_moduli,  # ascending
                    list(reversed(remaining_moduli)),  # descending
                    sorted(remaining_moduli, key=lambda m: m.bit_count()),  # by popcount
                    sorted(remaining_moduli, key=lambda m: -sum(1 for p in range(2, m) if m % p == 0)),  # composite-first
                ]
                
                for ordering in orderings:
                    if time.time() > deadline - 10:
                        break
                    
                    added = False
                    for m in ordering:
                        if m in used_moduli:
                            continue
                        
                        # Try different residue strategies
                        residue_strategies = [
                            list(range(m)),  # ascending
                            list(range(m - 1, -1, -1)),  # descending
                            sorted(range(m), key=lambda r: (math.gcd(r, m), r)),  # coprime first
                            sorted(range(m), key=lambda r: (m - r, r)),  # descending by value
                        ]
                        
                        for residues in residue_strategies:
                            if time.time() > deadline - 10:
                                break
                            
                            for r in residues:
                                if _can_add(r, m, chosen):
                                    chosen.append((r, m))
                                    used_moduli.add(m)
                                    added = True
                                    break
                            if added:
                                break
                        
                        if added:
                            break
                    
                    if not added:
                        break  # No more can be added in this round
            
            count = len(chosen)
            if count > best_count:
                best_count = count
                best_result = [[r, m] for r, m in chosen]
    
    # If we didn't find anything better, use the seed
    if best_result is None or len(best_result) == 0:
        best_result = [[r, m] for r, m in seed_pairs]
    
    return best_result
# EVOLVE-BLOCK-END
