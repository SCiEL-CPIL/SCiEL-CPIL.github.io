"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 75s safety margin within 90s limit
    
    n = 16
    k = 4
    universe = list(range(1, n+1))
    
    # Generate all possible 4-subsets
    all_sets = list(combinations(universe, k))
    all_sets = [frozenset(s) for s in all_sets]
    num_sets = len(all_sets)  # 1820
    
    # Precompute pairwise intersections
    # For each pair (i, j), store the intersection frozenset
    # We need to detect: for any three distinct sets a, b, c,
    # a∩b == a∩c == b∩c (this is the sunflower condition)
    
    # Strategy: Build a conflict graph where two sets conflict if they
    # could be part of a sunflower with some third set.
    # Actually, the condition is: no three DISTINCT sets have all three
    # pairwise intersections equal.
    
    # Let's think about this differently. For three sets A, B, C to form
    # a sunflower, we need A∩B = A∩C = B∩C = core.
    
    # Approach: Greedy with backtracking / local search
    # 1. Start with a random ordering
    # 2. Greedily add sets that don't create a sunflower
    # 3. Try to improve via swaps
    
    # First, let's build the conflict structure efficiently
    # Two sets A and B are "conflicting" with set C if A∩B = A∩C = B∩C
    # This is complex. Let's use a different approach.
    
    # For a set S, and a core C (subset of S), the sets that have S∩C as their
    # intersection with S and contain C... this is getting complex.
    
    # Simpler approach: For each pair (A, B), compute their intersection I = A∩B.
    # Then A and B conflict with any C such that A∩C = I and B∩C = I.
    # i.e., C must contain I, and C must not contain any element of A\I or B\I... 
    # wait, A∩C = I means C contains all of I and none of A\I.
    # Similarly B∩C = I means C contains all of I and none of B\I.
    # So C must contain I and be disjoint from (A\I) ∪ (B\I).
    # Also C must be a 4-element set.
    
    # Let's precompute for each pair (i,j) the set of k that conflict
    # Actually, let's just do a direct check during construction.
    
    # For efficiency, let's precompute intersections between all pairs
    # inter[i][j] = all_sets[i] & all_sets[j]
    
    # This is 1820^2 / 2 ≈ 1.6M pairs, each intersection is small.
    # Let's do it.
    
    inter = [[None]*num_sets for _ in range(num_sets)]
    for i in range(num_sets):
        inter[i][i] = all_sets[i]
        for j in range(i+1, num_sets):
            inter[i][j] = all_sets[i] & all_sets[j]
            inter[j][i] = inter[i][j]
    
    # Now, for a candidate set c to be added to the family F,
    # we need to check that for every pair (a, b) in F, 
    # not (a∩b == a∩c == b∩c)
    
    # This is O(|F|^2) per addition, which is too slow for large F.
    
    # Alternative: For each new set c, for each existing set a in F,
    # compute a∩c. Then for each pair of existing sets (a, b) with the same
    # intersection with c, check if a∩b equals that intersection.
    
    # Better: For each existing set a, let I_a = a∩c.
    # Group existing sets by their intersection with c.
    # For each group with intersection I, check if any pair in the group has
    # intersection equal to I.
    
    # Precompute: for each set c, and each possible intersection value I,
    # which existing sets have intersection I with c?
    
    # This is still complex. Let's use a different strategy:
    # Maintain for each pair of sets in the family, their intersection.
    # When adding c, for each a in family, compute a∩c = I_a.
    # For each b in family, compute b∩c = I_b.
    # We need to check: does there exist a, b in family such that
    # a∩b = a∩c = b∩c?
    # i.e., a∩b = I_a = I_b.
    # So I_a = I_b and a∩b = I_a.
    
    # So: group family members by their intersection with c.
    # For each group G with common intersection I with c,
    # check if any pair in G has intersection equal to I.
    
    # We can precompute for each set c, the "conflict groups" - but that's
    # expensive. Let's just do it on the fly with some optimizations.
    
    # Let's try a greedy approach first, then local search.
    
    def check_can_add(c_idx, family_indices, inter):
        """Check if set c_idx can be added to family (list of indices)."""
        c = all_sets[c_idx]
        # Group family members by their intersection with c
        groups = {}
        for a_idx in family_indices:
            a = all_sets[a_idx]
            inter_ac = inter[a_idx][c_idx]
            if inter_ac not in groups:
                groups[inter_ac] = []
            groups[inter_ac].append(a_idx)
        
        # For each group, check if any pair has intersection equal to the group key
        for I, members in groups.items():
            if len(members) < 2:
                continue
            # Check all pairs in members
            for i in range(len(members)):
                for j in range(i+1, len(members)):
                    a_idx = members[i]
                    b_idx = members[j]
                    if inter[a_idx][b_idx] == I:
                        return False
        return True
    
    # Greedy construction with random restarts
    best_family = []
    
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        # Random ordering
        order = list(range(num_sets))
        random.shuffle(order)
        
        family = []
        family_set = set()
        
        for c_idx in order:
            if time.time() > deadline - 5:
                break
            if c_idx in family_set:
                continue
            if check_can_add(c_idx, family, inter):
                family.append(c_idx)
                family_set.add(c_idx)
        
        if len(family) > len(best_family):
            best_family = family[:]
        
        # Local search: try to swap out a set for two others
        # This is complex, so let's try a simpler improvement
        # Try adding sets that were skipped
        if time.time() > deadline - 3:
            break
        
        # Try to add more sets
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for c_idx in range(num_sets):
                if c_idx in family_set:
                    continue
                if check_can_add(c_idx, family, inter):
                    family.append(c_idx)
                    family_set.add(c_idx)
                    improved = True
                    if len(family) > len(best_family):
                        best_family = family[:]
        
        if time.time() > deadline - 2:
            break
    
    # Convert to list of sorted lists
    result = [sorted(all_sets[i]) for i in best_family]
    
    # Final safety check
    if time.time() > deadline:
        # Return what we have
        pass
    
    return result

# EVOLVE-BLOCK-END
