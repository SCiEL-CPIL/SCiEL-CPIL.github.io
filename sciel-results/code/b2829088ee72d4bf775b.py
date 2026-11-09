"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def _is_sunflower_triple(a, b, c):
    """Check if three sets form a sunflower (all pairwise intersections equal)."""
    return (a & b) == (a & c) == (b & c)

def _build_conflict_graph(all_sets):
    """Build conflict graph where edge (i,j) exists if sets i and j form a sunflower with some third set."""
    n = len(all_sets)
    # For each pair of sets, check if they form a sunflower with any third set
    # Two sets A, B form a sunflower with C iff (A&B) == (A&C) == (B&C)
    # This means A&C = B&C = A&B, so C must contain A&B and C\A = C\B (symmetric difference with respect to A&B)
    # More precisely: let I = A & B. Then C must satisfy C & A = I and C & B = I.
    # This means C must contain I, and C can contain any elements from (A\I) ∪ (B\I) as long as
    # C & A = I (so C cannot contain elements of A\I) and C & B = I (so C cannot contain elements of B\I).
    # Wait, that's not right. Let me think again.
    # A&C = I means C contains exactly the elements of I from A. So C ∩ (A\I) = ∅.
    # B&C = I means C contains exactly the elements of I from B. So C ∩ (B\I) = ∅.
    # So C must be a subset of [1,16] \ (A\I) \ (B\I) = [1,16] \ (A∪B\I) that contains I.
    # Actually, C can contain elements outside A∪B freely.
    # So C must satisfy: I ⊆ C, C ∩ (A\I) = ∅, C ∩ (B\I) = ∅.
    # The elements C can contain are: I ∪ ([1,16] \ (A ∪ B))
    # And C must be a 4-element set containing I.
    # Let |I| = k. Then C needs 4-k more elements from [1,16] \ (A ∪ B).
    # Let S = [1,16] \ (A ∪ B). |S| = 16 - |A∪B| = 16 - (|A| + |B| - |I|) = 16 - (4 + 4 - k) = 8 + k.
    # So we need to choose 4-k elements from S.
    
    # For efficiency, we'll use a different approach:
    # For each pair (A, B), compute I = A & B.
    # Then check if there exists a 4-set C such that C & A = I and C & B = I.
    # This is equivalent to: C contains I, C is disjoint from (A\I) and (B\I), and |C| = 4.
    # The available elements for C are I ∪ S where S = [1,16] \ (A ∪ B).
    # We need to pick 4-|I| elements from S.
    # If |S| < 4-|I|, no such C exists.
    # Otherwise, there exists such a C.
    
    # But we also need C to be distinct from A and B.
    
    # Actually, for the conflict graph, we just need to know if A and B can be part of a sunflower triple.
    # So for each pair (i, j), we check if there exists k ≠ i, j such that (A_i, A_j, A_k) is a sunflower.
    
    # A faster approach: for each pair (A, B), compute I = A & B.
    # Then any C that forms a sunflower with A and B must satisfy C & A = I and C & B = I.
    # We can enumerate all possible C that satisfy this and check if any is in all_sets.
    
    # Let's precompute a dictionary mapping each set to its index.
    set_to_idx = {frozenset(s): i for i, s in enumerate(all_sets)}
    
    conflict = [set() for _ in range(n)]
    
    for i in range(n):
        if time.time() > time.time():  # placeholder
            pass
        for j in range(i + 1, n):
            A = all_sets[i]
            B = all_sets[j]
            I = A & B
            k = len(I)
            # C must contain I, be disjoint from (A\I) and (B\I), and have size 4.
            # Available elements for the remaining 4-k positions: S = [1,16] \ (A ∪ B)
            forbidden = (A - I) | (B - I)
            available = set(range(1, 17)) - forbidden
            # C = I ∪ T where T ⊆ available, |T| = 4 - k
            if 4 - k < 0:
                continue
            if len(available) < 4 - k:
                continue
            # Check if there exists a valid C in all_sets
            # We need to check if any 4-set C with I ⊆ C ⊆ I ∪ available exists in all_sets
            # and C ≠ A, C ≠ B.
            # Since |I ∪ available| = k + |available|, and we need to choose 4-k from available,
            # the number of candidates is C(|available|, 4-k).
            # For small k, this is manageable.
            
            # For k=0: need 4 elements from available. |available| = 16 - |A∪B| = 16 - 8 = 8. C(8,4)=70.
            # For k=1: need 3 elements from available. |available| = 16 - 7 = 9. C(9,3)=84.
            # For k=2: need 2 elements from available. |available| = 16 - 6 = 10. C(10,2)=45.
            # For k=3: need 1 element from available. |available| = 16 - 5 = 11. C(11,1)=11.
            # For k=4: need 0 elements. Only C=I=A=B, but A≠B so this case doesn't arise for distinct sets.
            
            found = False
            if k == 0:
                for T in combinations(available, 4):
                    C = set(I) | set(T)
                    if frozenset(C) in set_to_idx:
                        idx = set_to_idx[frozenset(C)]
                        if idx != i and idx != j:
                            found = True
                            break
            elif k == 1:
                for T in combinations(available, 3):
                    C = set(I) | set(T)
                    if frozenset(C) in set_to_idx:
                        idx = set_to_idx[frozenset(C)]
                        if idx != i and idx != j:
                            found = True
                            break
            elif k == 2:
                for T in combinations(available, 2):
                    C = set(I) | set(T)
                    if frozenset(C) in set_to_idx:
                        idx = set_to_idx[frozenset(C)]
                        if idx != i and idx != j:
                            found = True
                            break
            elif k == 3:
                for T in combinations(available, 1):
                    C = set(I) | set(T)
                    if frozenset(C) in set_to_idx:
                        idx = set_to_idx[frozenset(C)]
                        if idx != i and idx != j:
                            found = True
                            break
            
            if found:
                conflict[i].add(j)
                conflict[j].add(i)
    
    return conflict

def _greedy_construct(order, deadline):
    """Greedy construction given an ordering of all 4-subsets."""
    chosen = []
    chosen_list = []
    for s in order:
        if time.time() > deadline:
            break
        # Check against all pairs in chosen
        conflict = False
        for i in range(len(chosen_list)):
            for j in range(i+1, len(chosen_list)):
                if _is_sunflower_triple(chosen_list[i], chosen_list[j], s):
                    conflict = True
                    break
            if conflict:
                break
        if not conflict:
            chosen_list.append(s)
            chosen.append(s)
    return chosen_list

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # Precompute all 4-element subsets
    all_sets = [set(row) for row in combinations(range(1, 17), 4)]
    n_sets = len(all_sets)
    
    best_family = []
    
    # First: lexicographic order (baseline)
    if time.time() < deadline:
        family = _greedy_construct(all_sets, deadline)
        if len(family) > len(best_family):
            best_family = family
    
    # Then: multiple random orderings
    rng = random.Random(42)
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        shuffled = all_sets[:]
        rng.shuffle(shuffled)
        family = _greedy_construct(shuffled, deadline)
        if len(family) > len(best_family):
            best_family = family
            # If we hit the maximum, stop early
            if len(best_family) >= 250:
                break
    
    return [sorted(s) for s in best_family]
# EVOLVE-BLOCK-END
