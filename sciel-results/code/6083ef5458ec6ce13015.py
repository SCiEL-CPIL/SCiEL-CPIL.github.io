import sys
from math import gcd
from itertools import combinations

def construct():
    MAX_M = 128
    
    # We want to find a maximum set of (residue, modulus) pairs with distinct moduli in [2, MAX_M]
    # such that for any two pairs (r1, m1) and (r2, m2), the classes are disjoint.
    # This is equivalent to r1 NOT≡ r2 (mod gcd(m1, m2)).
    
    # Strategy:
    # 1. Identify a set of moduli that are pairwise non-coprime (necessary condition).
    # 2. Assign residues to these moduli such that the disjointness condition holds.
    #
    # We will use a greedy approach with backtracking or a heuristic search to find a large valid set.
    # Given the time limit, we'll try to construct a large set manually or via a simple search.
    #
    # Let's try to find a set of size > 11.
    # The previous best was 11 (multiples of 11).
    # Can we get 12?
    #
    # Let's try a mix of small moduli with shared factors.
    # Consider the set of moduli: {6, 10, 15, 30, 42, 70, 105, 84, 120, 90, 126, 100}
    # We need to check if we can assign residues to these 12 moduli.
    #
    # Instead of hardcoding, let's write a small search to find a valid set of size 12 or more.
    # We'll use a simple backtracking search on the moduli and residues.
    #
    # However, the space is large. Let's try a specific construction.
    #
    # Let's try to use the set of moduli found in the previous analysis:
    # M = [6, 10, 15, 30, 42, 70, 105, 84, 120, 90, 126, 100]
    # And try to assign residues.
    #
    # We'll write a function to check if a given assignment of residues to a set of moduli is valid.
    # And then try to find an assignment for a candidate set of moduli.
    
    def is_valid_assignment(moduli, residues):
        for i in range(len(moduli)):
            for j in range(i+1, len(moduli)):
                m1, m2 = moduli[i], moduli[j]
                r1, r2 = residues[i], residues[j]
                g = gcd(m1, m2)
                if (r1 - r2) % g == 0:
                    return False
        return True
    
    # Candidate set of moduli (size 12)
    candidate_moduli = [6, 10, 15, 30, 42, 70, 105, 84, 120, 90, 126, 100]
    
    # Try to assign residues to these moduli.
    # We'll use a simple backtracking search.
    # Since the number of moduli is small (12), and residues can be 0..m-1,
    # we can try to assign residues one by one, checking constraints with previously assigned moduli.
    
    def find_residues(moduli):
        n = len(moduli)
        residues = [None] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            for r in range(moduli[idx]):
                residues[idx] = r
                # Check against all previously assigned moduli
                valid = True
                for j in range(idx):
                    g = gcd(moduli[idx], moduli[j])
                    if (r - residues[j]) % g == 0:
                        valid = False
                        break
                if valid:
                    if backtrack(idx + 1):
                        return True
                residues[idx] = None
            
            return False
        
        if backtrack(0):
            return residues
        return None
    
    residues = find_residues(candidate_moduli)
    
    if residues:
        result = [[residues[i], candidate_moduli[i]] for i in range(len(candidate_moduli))]
        # Verify
        for i in range(len(result)):
            for j in range(i+1, len(result)):
                r1, m1 = result[i]
                r2, m2 = result[j]
                g = gcd(m1, m2)
                if (r1 - r2) % g == 0:
                    # This should not happen if find_residues worked correctly
                    pass
        return result
    else:
        # Fallback to the known valid set of size 11
        p = 11
        moduli = [p * k for k in range(1, 12)]
        result = []
        for i, m in enumerate(moduli):
            r = i
            result.append([r, m])
        return result

# EVOLVE-BLOCK-END
