import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 42.0  # Safety margin before 90s kill

    n = 16
    k = 4

    # Generate all k-subsets as bitmasks
    all_sets = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
    
    num_sets = len(all_sets)  # C(16,4) = 1820
    set_to_idx = {s: i for i, s in enumerate(all_sets)}

    # Precompute: for each pair (i, j), find all partners k that form a sunflower
    # A sunflower: A, B, C with A∩B = A∩C = B∩C
    # For pair (A, B), core K = A∩B
    # C must satisfy: C ⊇ K, C ∩ (A\K) = ∅, C ∩ (B\K) = ∅
    
    # Precompute partner lists for each pair
    # partners[i][j] = list of indices k such that (all_sets[i], all_sets[j], all_sets[k]) form a sunflower
    # But storing this for all ~3.3M pairs is expensive. Instead, compute on the fly but cache.
    
    # Alternative: For each set X, precompute which pairs (A,B) in the family block X.
    # Better: Precompute for each pair (i,j) the bitmask of "forbidden" sets.
    # Actually, let's precompute for each pair (i,j) the list of partner indices.
    
    # Number of pairs: C(1820, 2) ≈ 1.65M. Each partner list could be up to ~1820.
    # That's too much memory. Let's compute partners on the fly but use a smarter approach.
    
    # Key insight: For a given set X and a family F, X can be added iff for all pairs (A,B) in F,
    # X is NOT a sunflower partner of (A,B).
    
    # Let's precompute: for each set X (index x), and for each pair (i,j), whether x is a partner.
    # That's 1820 * C(1820,2) ≈ 3.3 billion entries. Too much.
    
    # Instead: For each set X, precompute the set of "core patterns" that block it.
    # A pair (A,B) blocks X if: K = A∩B ⊆ X, A\K ∩ X = ∅, B\K ∩ X = ∅.
    # Equivalently: A ∩ X = B ∩ X = K.
    
    # For a fixed X, a pair (A,B) blocks X iff A∩X = B∩X.
    # So for each X, group all sets by their intersection with X.
    # Any two sets in the same group (with the same intersection with X) form a blocking pair.
    
    # This is the key structural insight! For each X, compute the partition of all_sets by their 
    # intersection with X. If any group has size ≥ 2 and both members are in the family, X is blocked.
    
    # Precompute for each X (index x): a mapping from intersection-mask to list of set indices
    # But the intersection mask is a subset of X, so at most 2^4 = 16 possible values per X.
    
    # Let's precompute: for each set index x, and for each possible intersection value v (subset of all_sets[x]),
    # the list of set indices i such that all_sets[i] & all_sets[x] == v.
    
    # Actually, the intersection is a bitmask that is a subset of all_sets[x].
    # all_sets[x] has 4 bits set, so there are 2^4 = 16 possible intersection values.
    
    # Precompute: intersect_groups[x][v] = list of indices i where all_sets[i] & all_sets[x] == v
    # But v ranges over subsets of all_sets[x]. Let's use a dict per x.
    
    intersect_groups = [None] * num_sets
    for x in range(num_sets):
        X = all_sets[x]
        groups = {}
        for i in range(num_sets):
            inter = all_sets[i] & X
            if inter not in groups:
                groups[inter] = []
            groups[inter].append(i)
        intersect_groups[x] = groups

    # Now, can_add(X_idx, family_set) where family_set is a set of indices:
    # For the given X_idx, look at intersect_groups[X_idx].
    # For each group (intersection value v), check if any two members of the group are in family_set.
    # If so, X is blocked.
    
    def can_add_fast(X_idx, family_set):
        groups = intersect_groups[X_idx]
        for v, members in groups.items():
            # Check if at least 2 members are in family_set
            count_in_family = 0
            for m in members:
                if m in family_set:
                    count_in_family += 1
                    if count_in_family >= 2:
                        return False
        return True

    # Greedy pass with degree-based ordering
    def greedy_pass(order):
        current_family = []
        current_family_set = set()
        for X_idx in order:
            if can_add_fast(X_idx, current_family_set):
                current_family.append(X_idx)
                current_family_set.add(X_idx)
        return current_family, current_family_set

    # Local search: try to add, then swap
    def local_search(family, family_set, time_limit):
        candidates = [i for i in range(num_sets) if i not in family_set]
        random.shuffle(candidates)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Phase 1: Try to add any valid candidate
            for idx in range(len(candidates) - 1, -1, -1):
                X_idx = candidates[idx]
                if can_add_fast(X_idx, family_set):
                    family.append(X_idx)
                    family_set.add(X_idx)
                    candidates.pop(idx)
                    improved = True
                    break

            if improved:
                continue

            # Phase 2: No direct additions. Try swaps.
            if not candidates:
                break
            
            # Pick a random candidate
            X_idx = random.choice(candidates)
            
            # Find all pairs in family that block X_idx
            # A pair (A, B) blocks X if A∩X = B∩X
            groups = intersect_groups[X_idx]
            blockers = []  # list of (i, j) pairs where both i,j in family and A∩X = B∩X
            for v, members in groups.items():
                in_family = [m for m in members if m in family_set]
                if len(in_family) >= 2:
                    # All pairs within in_family block X
                    for a in range(len(in_family)):
                        for b in range(a + 1, len(in_family)):
                            blockers.append((in_family[a], in_family[b]))
            
            if not blockers:
                # Should have been added in Phase 1
                family.append(X_idx)
                family_set.add(X_idx)
                candidates.remove(X_idx)
                improved = True
                break

            # Try to resolve by removing one member from a blocking pair
            random.shuffle(blockers)
            swap_success = False
            for i, j in blockers:
                # Try removing family[i]
                removed = family[i]
                family.remove(removed)
                family_set.discard(removed)
                
                if can_add_fast(X_idx, family_set):
                    family.append(X_idx)
                    family_set.add(X_idx)
                    candidates.remove(X_idx)
                    improved = True
                    swap_success = True
                    break
                else:
                    family.append(removed)
                    family_set.add(removed)
                    
                    # Try removing family[j]
                    removed = family[j]
                    family.remove(removed)
                    family_set.discard(removed)
                    
                    if can_add_fast(X_idx, family_set):
                        family.append(X_idx)
                        family_set.add(X_idx)
                        candidates.remove(X_idx)
                        improved = True
                        swap_success = True
                        break
                    else:
                        family.append(removed)
                        family_set.add(removed)
            
            if not swap_success:
                break

        return family, family_set

    # Compute degree for ordering: for each set, how many other sets share a "conflict"
    # Use a simpler heuristic: order by random, but also try a few different orderings
    
    best_family = []
    best_family_set = set()
    best_size = 0

    # Run multiple restarts
    restart = 0
    while time.time() < deadline:
        restart += 1
        
        # Try different ordering strategies
        if restart % 3 == 1:
            # Random order
            order = list(range(num_sets))
            random.shuffle(order)
        elif restart % 3 == 2:
            # Sort by popcount of intersection with a random pivot
            pivot = random.choice(all_sets)
            order = sorted(range(num_sets), key=lambda i: bin(all_sets[i] & pivot).count('1'))
            random.shuffle(order)  # Add some randomness
        else:
            # Random order with slight bias
            order = list(range(num_sets))
            random.shuffle(order)
        
        current_family, current_family_set = greedy_pass(order)
        
        # Apply local search
        remaining_time = deadline - time.time()
        if remaining_time > 2.0:
            current_family, current_family_set = local_search(
                current_family, current_family_set, time.time() + min(remaining_time - 1.0, 10.0)
            )
        
        if len(current_family) > best_size:
            best_size = len(current_family)
            best_family = current_family[:]
            best_family_set = current_family_set.copy()
            
        if best_size >= 250:
            break

    # Convert best_family indices back to sorted lists of integers
    result = []
    for idx in best_family:
        mask = all_sets[idx]
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
        
    return result
