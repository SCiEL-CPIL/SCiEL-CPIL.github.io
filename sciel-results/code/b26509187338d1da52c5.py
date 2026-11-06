import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin

    n = 16
    k = 4

    # Generate all k-subsets as bitmasks
    all_sets = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
    
    num_sets = len(all_sets)
    set_to_idx = {s: i for i, s in enumerate(all_sets)}

    # Helper to check if X is a sunflower partner for pair (A, B)
    def is_sunflower_partner(A, B, X):
        K = A & B
        if (X & K) != K:
            return False
        A_diff = A & ~K
        if (X & A_diff) != 0:
            return False
        B_diff = B & ~K
        if (X & B_diff) != 0:
            return False
        return True

    def can_add(X, family):
        L = len(family)
        for i in range(L):
            A = family[i]
            for j in range(i + 1, L):
                B = family[j]
                if is_sunflower_partner(A, B, X):
                    return False
        return True

    best_family = []
    best_size = 0

    def greedy_pass(order):
        current_family = []
        for X in order:
            if can_add(X, current_family):
                current_family.append(X)
        return current_family

    def local_search(family):
        family_set = set(family)
        candidates = [X for X in all_sets if X not in family_set]
        
        if not candidates:
            return family

        random.shuffle(candidates)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Try to add any valid candidate
            # We iterate through candidates. If we add one, we restart the scan.
            # To avoid O(N^2) scanning every time, we just pick a random candidate first?
            # No, deterministic scan is fine for small N.
            for idx in range(len(candidates) - 1, -1, -1):
                X = candidates[idx]
                if can_add(X, family):
                    family.append(X)
                    family_set.add(X)
                    candidates.pop(idx)
                    improved = True
                    break # Restart outer loop to re-check all candidates

            if improved:
                continue

            # Phase 2: No direct additions possible. Try swaps.
            if not candidates:
                break
            
            # Pick a random candidate to try to swap in
            X = random.choice(candidates)
            
            # Find all pairs (A, B) in family that block X
            # i.e., is_sunflower_partner(A, B, X) is True
            L = len(family)
            blockers = []
            for i in range(L):
                A = family[i]
                for j in range(i + 1, L):
                    B = family[j]
                    if is_sunflower_partner(A, B, X):
                        blockers.append((i, j))
            
            if not blockers:
                # If no blockers, it should have been added in Phase 1.
                # This can happen if can_add was expensive and state changed? 
                # Or just a logical safety net.
                family.append(X)
                family_set.add(X)
                candidates.remove(X)
                improved = True
                break

            # Try to resolve one blocker by removing one of its members
            random.shuffle(blockers)
            swap_success = False
            for i, j in blockers:
                # Try removing family[i]
                A = family[i]
                # Construct new family without A
                # Since family is a list, we can just remove by value or index
                # We need to check if X can be added to family \ {A}
                
                # Optimization: Instead of building a new list and running full can_add (O(L^2)),
                # we can just check if removing A resolves the specific blocker (i,j) 
                # and doesn't create new blockers with other pairs.
                # However, removing A might create new sunflowers with X and other pairs?
                # No, X is fixed. The condition is that for ALL pairs (U, V) in family\{A},
                # is_sunflower_partner(U, V, X) is False.
                
                # The only pairs that were blocking X are in `blockers`.
                # If we remove A, the pair (i, j) is gone.
                # Are there other pairs involving A that blocked X?
                # Yes, potentially. We need to ensure that removing A doesn't leave other blockers.
                # Actually, `blockers` contains ALL pairs that block X.
                # So if we remove A, we must ensure that no other pair in `blockers` remains.
                # This is only true if A is involved in ALL blocking pairs.
                
                # Simpler approach: Just try the swap and verify with can_add on the modified list.
                # Given L <= 250, O(L^2) check is ~60k ops. 
                # We might try a few swaps. 60k * few is fine.
                
                # Remove A
                family.remove(A)
                family_set.discard(A)
                
                if can_add(X, family):
                    family.append(X)
                    family_set.add(X)
                    candidates.remove(X)
                    improved = True
                    swap_success = True
                    break
                else:
                    # Restore A
                    family.append(A)
                    family_set.add(A)
                    
                    # Try removing B
                    B = family[j]
                    family.remove(B)
                    family_set.discard(B)
                    
                    if can_add(X, family):
                        family.append(X)
                        family_set.add(X)
                        candidates.remove(X)
                        improved = True
                        swap_success = True
                        break
                    else:
                        # Restore B
                        family.append(B)
                        family_set.add(B)
            
            if not swap_success:
                # If we couldn't swap in this X, we might try another X.
                # To avoid getting stuck in an infinite loop of trying the same failing X,
                # we can remove X from candidates or just break and let the main loop restart.
                # Breaking is safer.
                break

        return family

    # Run multiple restarts
    while time.time() < deadline:
        random.shuffle(all_sets)
        current_family = greedy_pass(all_sets)
        
        # Apply local search
        current_family = local_search(current_family)
        
        if len(current_family) > best_size:
            best_size = len(current_family)
            best_family = current_family[:]
            
        if best_size >= 250:
            break

    # Convert best_family masks back to sorted lists of integers
    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
        
    return result
