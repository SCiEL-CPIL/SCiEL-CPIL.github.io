"""
Strategy: Model as maximum independent set in a conflict graph.
Two 4-sets A, B "conflict" if there exists a third 4-set C such that
A∩B = A∩C = B∩C (sunflower triple). We find the maximum independent set
using branch-and-bound with greedy coloring for upper bounds.
"""

import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Generate all 4-subsets of [1,16] as bitmasks
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_sets.append(mask)

    n_total = len(all_sets)  # 1820

    # Precompute for each set, what intersections it has with other sets
    # and build the conflict graph
    # Two sets A, B conflict if there exists C != A, B such that
    # A∩B = A∩C = B∩C
    #
    # For a given pair (A, B) with core S = A∩B:
    # C must contain S, and C∩(A\S) = empty, C∩(B\S) = empty
    # So C = S ∪ T where T is a (4-|S|)-subset of [16]\(A∪B)
    #
    # Precompute: for each set index, store the set
    # For efficiency, use a dict mapping (core_mask, excluded_mask) -> list of set indices

    # Build index: for each possible core (subset of [16]), which sets contain it
    # But cores can be any subset... let's be smarter.

    # Alternative: For each pair (i, j), compute if they conflict.
    # This is O(n^2 * something). n=1820, so n^2 = 3.3M pairs.
    # For each pair, check if there exists a conflicting third set.

    # Precompute: for each set, its bitmask
    # For pair (A, B): core = A & B
    # A_third must satisfy: core ⊆ A_third, (A_third & (A^core)) == 0, (A_third & (B^core)) == 0
    # i.e., A_third & (A | B) == core
    # And |A_third| = 4

    # Let forbidden = (A | B) ^ core  -- elements in A∪B but not in core
    # A_third must avoid forbidden, contain core, and have 4 total elements.
    # Available elements: [16] \ forbidden
    # Need to pick (4 - |core|) elements from available \ core

    # Precompute for each possible (core, forbidden) pair whether such a C exists
    # Actually, just check: is there a 4-set C in all_sets such that
    # (C & core) == core and (C & forbidden) == 0 and C != A and C != B?

    # For speed: precompute for each mask, which sets have that mask as subset
    # and which sets are disjoint from a given mask

    # Let's precompute: subset_index[core_mask] = list of set indices that contain core_mask
    # And disjoint_index[forbidden_mask] = list of set indices disjoint from forbidden_mask

    # But core_mask and forbidden_mask can be many values. Let's just do direct check.

    # Actually, for the conflict graph, we need: for pair (i,j), does there exist k != i,j
    # such that all_sets[k] & (all_sets[i] | all_sets[j]) == (all_sets[i] & all_sets[j])
    # and all_sets[k] != all_sets[i] and all_sets[k] != all_sets[j]?

    # Let me precompute a lookup: for each set, store its mask
    # For the conflict check, use bit operations

    # Build conflict graph as adjacency list
    # conflict[i] = set of j such that i and j conflict

    # To speed up: for each pair (i, j), compute core = sets[i] & sets[j]
    # forbidden = (sets[i] | sets[j]) ^ core
    # Need: exists k != i, j such that (sets[k] & core) == core and (sets[k] & forbidden) == 0

    # Precompute: for each possible core_mask (16-bit), list of indices k where (sets[k] & core_mask) == core_mask
    # That's subset_contains[core_mask]

    # And for each possible forbidden_mask, list of indices k where (sets[k] & forbidden_mask) == 0
    # That's disjoint_from[forbidden_mask]

    # But there are 2^16 = 65536 possible masks, and 1820 sets.
    # For each mask, we can precompute which sets contain it or are disjoint from it.

    # Let's precompute:
    # contain[mask] = frozenset of indices k where (sets[k] & mask) == mask
    # disjoint[mask] = frozenset of indices k where (sets[k] & mask) == 0

    # This is 65536 * 1820 checks = 119M, which might be too slow in Python.

    # Alternative: just do the pairwise check directly with early termination.
    # For each pair (i, j), compute core and forbidden, then scan all k to check.
    # That's 1820^2 * 1820 = 6 billion, way too slow.

    # Better: Precompute for each set k, its mask. Then for a given (core, forbidden),
    # we need to check if any set k satisfies the condition.

    # Let me precompute a dictionary: for each set k, store its mask.
    # Then for the conflict check, iterate over all k and check the condition.
    # But that's still O(n) per pair.

    # Alternative approach: Don't build the full conflict graph. Instead,
    # use a different strategy:
    # 1. Greedily build a large family
    # 2. Use local search to improve

    # But the directive says to try a structural change. Let me try the
    # branch-and-bound approach but with a smarter conflict check.

    # Key insight: For the conflict check, instead of scanning all k,
    # we can precompute for each (core, forbidden) pair the count of valid k.
    # But there are too many (core, forbidden) pairs.

    # Let me try a different angle: Use the fact that for k=4, the number
    # of possible cores is limited. |core| can be 0, 1, 2, 3, or 4.

    # For |core| = 4: A = B, which can't happen for distinct sets.
    # For |core| = 3: A and B share 3 elements. Then C must contain those 3,
    #   and the 4th element of C must be from [16]\(A∪B). |A∪B| = 5, so
    #   11 choices for the 4th element. So there are 11 possible C's.
    # For |core| = 2: A and B share 2 elements. C must contain those 2,
    #   and pick 2 more from [16]\(A∪B). |A∪B| = 6, so 10 choices, C(10,2)=45.
    # For |core| = 1: A and B share 1 element. C must contain that 1,
    #   and pick 3 more from [16]\(A∪B). |A∪B| = 7, so 9 choices, C(9,3)=84.
    # For |core| = 0: A and B are disjoint. C must pick 4 from [16]\(A∪B).
    #   |A∪B| = 8, so 8 choices, C(8,4)=70.

    # So for each pair, the number of potential conflicting C's is at most 84.

    # Let me precompute: for each set, store its elements as a frozenset for quick lookup.
    # And build an index: for each 4-element subset (as a frozenset), its index.

    set_to_index = {}
    for idx, mask in enumerate(all_sets):
        elements = frozenset(i for i in range(16) if mask & (1 << i))
        set_to_index[elements] = idx

    # Now, for the conflict check between sets i and j:
    # core = all_sets[i] & all_sets[j]
    # forbidden = (all_sets[i] | all_sets[j]) ^ core
    # We need to check if there exists a 4-set C (index k != i, j) such that:
    #   C contains core, C is disjoint from forbidden

    # Let's precompute for each set, the set of elements it contains (as a set for quick intersection)
    set_elements = [set(i for i in range(16) if all_sets[idx] & (1 << i)) for idx in range(n_total)]

    # For the branch-and-bound, I'll use a simpler approach:
    # 1. Greedily build a good initial solution
    # 2. Use local search (add/remove/swap) to improve
    # 3. The conflict check is done on-the-fly

    # Let me implement an efficient conflict check and local search.

    def check_conflict(family_indices, candidate_idx):
        """Check if adding candidate to family creates a sunflower triple."""
        cand_mask = all_sets[candidate_idx]
        cand_elems = set_elements[candidate_idx]
        for i in family_indices:
            a_mask = all_sets[i]
            ab = a_mask & cand_mask
            for j in family_indices:
                if j == i:
                    continue
                b_mask = all_sets[j]
                if (a_mask & b_mask) == ab and (b_mask & cand_mask) == ab:
                    return True
        return False

    # Greedy construction
    def greedy_construct(order, deadline):
        family = []
        family_set = set()
        for idx in order:
            if time.time() > deadline:
                break
            if not check_conflict(family, idx):
                family.append(idx)
                family_set.add(idx)
        return family

    # Local improvement: try to swap out one set for another
    def local_improve(family, all_indices, deadline, max_swaps=500):
        family = list(family)
        family_set = set(family)
        improved = True
        swaps_made = 0

        while improved and swaps_made < max_swaps and time.time() < deadline:
            improved = False
            for i in range(len(family)):
                if time.time() > deadline:
                    break
                removed = family[i]
                # Try to find a replacement
                for candidate in all_indices:
                    if candidate in family_set:
                        continue
                    # Check if candidate can replace removed
                    temp_family = [f for j, f in enumerate(family) if j != i]
                    if not check_conflict(temp_family, candidate):
                        family[i] = candidate
                        family_set.discard(removed)
                        family_set.add(candidate)
                        improved = True
                        swaps_made += 1
                        break
                if improved:
                    break

        return family

    # Run multiple greedy passes with different orderings
    rng = random.Random(42)
    best_family = []
    best_size = 0

    # Pass 1: Natural order
    order1 = list(range(n_total))
    family1 = greedy_construct(order1, deadline)
    if len(family1) > best_size:
        best_size = len(family1)
        best_family = family1

    # Pass 2: Reverse order
    if time.time() < deadline - 10:
        order2 = list(range(n_total - 1, -1, -1))
        family2 = greedy_construct(order2, deadline)
        if len(family2) > best_size:
            best_size = len(family2)
            best_family = family2

    # Pass 3-10: Random orderings
    for trial in range(7):
        if time.time() > deadline - 5:
            break
        order = list(range(n_total))
        rng.shuffle(order)
        family = greedy_construct(order, deadline)
        if len(family) > best_size:
            best_size = len(family)
            best_family = family

    # Local improvement on best family
    if time.time() < deadline - 10 and best_family:
        improved_family = local_improve(best_family, list(range(n_total)), deadline, max_swaps=200)
        if len(improved_family) > best_size:
            best_size = len(improved_family)
            best_family = improved_family

    # Try more local improvement rounds
    for round_num in range(3):
        if time.time() > deadline - 5 or not best_family:
            break
        improved_family = local_improve(best_family, list(range(n_total)), deadline, max_swaps=100)
        if len(improved_family) > best_size:
            best_size = len(improved_family)
            best_family = improved_family

    # Convert bitmasks back to sorted lists
    result = []
    for idx in best_family:
        mask = all_sets[idx]
        elements = [i + 1 for i in range(16) if mask & (1 << i)]
        result.append(sorted(elements))

    return result
