"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_sunflower_free(family):
    """Check if no three distinct sets in family have all three pairwise intersections equal."""
    n = len(family)
    if n < 3:
        return True
    # For each pair of sets, compute their intersection
    # Then for each triple, check if all three pairwise intersections are equal
    # This is O(n^3) but we can optimize by grouping pairs by intersection
    # Two sets A,B have intersection I. If we have three sets A,B,C where:
    #   A∩B = A∩C = B∩C = I, then it's a sunflower with core I.
    # 
    # Key insight: For a fixed intersection set I, if we have 3+ sets that all contain I
    # and whose pairwise intersections are exactly I, that's a sunflower.
    #
    # More precisely: three sets A, B, C form a sunflower if A∩B = A∩C = B∩C.
    # Let I = A∩B. Then A∩C = I and B∩C = I.
    # This means C contains I, and C shares no elements with A\I or B\I.
    
    # Efficient check: For each pair (A,B), compute I = A∩B.
    # Then any other set C such that A∩C = I and B∩C = I would form a sunflower.
    # A∩C = I means C contains I and C ∩ (A\I) = empty.
    # B∩C = I means C contains I and C ∩ (B\I) = empty.
    # So C must contain I, and C ∩ (A\I ∪ B\I) = empty.
    
    # Let's use a different approach: group sets by which elements they contain.
    # For each possible intersection I (subset of [16]), find all sets that contain I.
    # Among those sets, two sets A,B have A∩B ⊇ I. For them to have A∩B = I,
    # they must share exactly I and no more.
    
    # Simpler: just check all triples. With n up to 250, n^3 = 15.6M which might be too slow.
    # Let's be smarter.
    
    # For each pair (i,j), compute I_ij = family[i] & family[j]
    # Store in a dict: key = frozenset(I_ij), value = list of (i,j) pairs
    # For a sunflower, we need three sets a,b,c with:
    #   a∩b = a∩c = b∩c = I
    # So pairs (a,b), (a,c), (b,c) all have intersection I.
    # 
    # For each I, let P_I be the set of pairs with intersection I.
    # We need to find if there exist three sets a,b,c such that all three pairs are in P_I.
    # This is equivalent to: in the graph where vertices are sets and edges are pairs in P_I,
    # is there a triangle?
    
    # But building this for all I might be expensive. Let's try a different angle.
    
    # Actually, for the greedy construction, we can check incrementally:
    # When adding a new set S, check if it forms a sunflower with any two existing sets.
    # For each pair (A,B) in existing family, check if A∩B = A∩S = B∩S.
    # This is O(n^2) per addition, total O(n^3) for n additions.
    
    # For n=250, that's about 250^3/6 ≈ 2.6M checks, each check is set operations.
    # Might be feasible within time limits.
    
    return True  # Placeholder, we'll do incremental check in construct

def _forms_sunflower(new_set, existing):
    """Check if new_set forms a sunflower with any two sets in existing."""
    for i in range(len(existing)):
        for j in range(i+1, len(existing)):
            a = existing[i]
            b = existing[j]
            ab = a & b
            a_new = a & new_set
            b_new = b & new_set
            if ab == a_new == b_new:
                return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    # Generate all 4-element subsets of [1,16]
    all_sets = [set(c) for c in combinations(range(1, 17), 4)]
    
    # Try multiple random orderings and greedy constructions
    best_family = []
    
    # First, try a deterministic greedy approach with a specific ordering
    # Sort by some heuristic - maybe by frequency of elements or just lexicographic
    for attempt in range(10):
        if time.time() > deadline:
            break
        
        if attempt == 0:
            # Lexicographic order
            candidates = list(all_sets)
        else:
            # Random order
            candidates = list(all_sets)
            random.shuffle(candidates)
        
        family = []
        for s in candidates:
            if time.time() > deadline:
                break
            if not _forms_sunflower(s, family):
                family.append(s)
        
        if len(family) > len(best_family):
            best_family = family[:]
        
        # Early termination if we've hit the maximum
        if len(best_family) >= 250:
            break
    
    return [sorted(s) for s in best_family]
# EVOLVE-BLOCK-END
