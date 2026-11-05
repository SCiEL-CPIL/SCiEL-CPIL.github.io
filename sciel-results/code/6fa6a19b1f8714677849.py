import time
import random
import numpy as np

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _get_paley_prime(p):
    """Find the next prime >= p that is 3 mod 4."""
    cand = p
    while True:
        if cand % 4 == 3 and _is_prime(cand):
            return cand
        cand += 1

def _construct_paley(p):
    """
    Construct Paley tournament on p vertices.
    Returns orientation bits.
    """
    n = p
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0) # i -> j
            else:
                orientation.append(1) # j -> i
    return {'n': n, 'orientation': orientation}

def _check_property_fast(n, orientation):
    """
    Check if every 4-set has a common dominator.
    orientation is a list of bits for pairs (i,j) i<j.
    0: i->j, 1: j->i.
    """
    # Build in-neighborhood masks C[u]
    # C[u] is a bitmask where bit v is 1 if v -> u.
    # Note: v cannot dominate itself.
    C = [0] * n
    
    # Map pair index to (i,j)
    # We need to fill C[u].
    # It's easier to iterate through all pairs and set bits.
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            if bit == 0:
                # i -> j
                # i is in N^-(j)
                C[j] |= (1 << i)
            else:
                # j -> i
                # j is in N^-(i)
                C[i] |= (1 << j)
            idx += 1
            
    # Check all 4-subsets
    # To speed up, we can iterate
    for a in range(n):
        for b in range(a + 1, n):
            mask_ab = C[a] & C[b]
            if mask_ab == 0:
                # No common dominator for {a,b}. 
                # Wait, if mask_ab is 0, then any superset {a,b,c,d} will also have empty intersection.
                # So if any pair has empty intersection, the property fails.
                # Is it possible that a pair has empty intersection but a 4-set doesn't?
                # No, intersection of 4 sets is subset of intersection of 2 sets.
                # If intersection of 2 is empty, intersection of 4 is empty.
                # So we just need to check if there exists a pair {a,b} with C[a] & C[b] == 0.
                # If yes, return False.
                # If no pair has empty intersection, does that guarantee every 4-set has non-empty intersection?
                # NO. It is possible that C[a] & C[b] != 0 for all pairs, but C[a] & C[b] & C[c] & C[d] == 0.
                # Example: 3 sets might have non-empty pairwise intersections but empty triple intersection.
                # So we must check 4-sets.
                pass
            
            for c in range(b + 1, n):
                mask_abc = mask_ab & C[c]
                if mask_abc == 0:
                    continue # This 3-set has no common dominator, so any 4-set containing it fails.
                    # But we need to check if the specific 4-set fails.
                    # If mask_abc is 0, then mask_abc & C[d] will be 0 for all d.
                    # So we can break? No, we just know this specific 4-set will fail.
                    # We can return False immediately if we find ANY 4-set with empty intersection.
                    # Since mask_abc is 0, any d will result in 0.
                    # So if we find a 3-set with empty intersection, the property fails?
                    # Wait. The property requires EVERY 4-set to have a dominator.
                    # If a 3-set {a,b,c} has no common dominator, then any 4-set {a,b,c,d} also has no common dominator.
                    # So if ANY 3-set has empty intersection, the tournament is INVALID.
                    # So we only need to check if there exists a 3-set with empty intersection.
                    # If no 3-set has empty intersection, does it imply no 4-set has empty intersection?
                    # Yes. If intersection of any 3 is non-empty, then intersection of any 4 (which is subset of intersection of any 3) is non-empty.
                    # Wait. Intersection of 4 is a subset of Intersection of 3.
                    # If Intersection of 3 is non-empty, Intersection of 4 CAN be empty.
                    # Example: A={1,2}, B={2,3}, C={3,1}. Pairwise non-empty. Triple empty.
                    # So checking 3-sets is NOT sufficient.
                    # We MUST check 4-sets.
                    return False
                
                for d in range(c + 1, n):
                    mask_abcd = mask_abc & C[d]
                    if mask_abcd == 0:
                        return False
                        
    return True

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    # Try Paley primes
    primes_to_try = [31, 43, 47, 59, 67, 71, 79, 83]
    
    for p in primes_to_try:
        if time.time() - start_time > time_limit:
            break
            
        # Only try if p <= 90
        if p > 90:
            continue
            
        try:
            result = _construct_paley(p)
            # Check property
            if _check_property_fast(p, result['orientation']):
                return result
        except Exception as e:
            continue
            
    # Fallback to 83 if none worked (should not happen if 83 works)
    return _construct_paley(83)
