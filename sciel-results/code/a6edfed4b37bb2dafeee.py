# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    Uses bitmask-based simulated annealing for speed.
    """
    m = 205
    deadline = time.time() + 620.0  # safety margin before 630s limit

    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)

    # Build conflict masks: conflict_mask[i] has bit j set if i and j conflict
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d1 = (i - j) % m
            if d1 in forbidden:
                mask |= (1 << j)
        conflict_mask[i] = mask

    def popcount(x):
        return x.bit_count()

    def count_conflicts(S):
        """Count number of conflicting pairs in the bitmask set S."""
        cnt = 0
        temp = S
        while temp:
            # Extract lowest set bit
            lsb = temp & -temp
            i = lsb.bit_length() - 1
            # Count conflicts of i with the rest of S
            cnt += popcount(S & conflict_mask[i])
            temp ^= lsb
        return cnt // 2

    def try_add(S, elem):
        """Check if elem can be added to S without creating conflicts."""
        return (S & conflict_mask[elem]) == 0

    # Greedy initialization using bitmasks
    def greedy_construct(order=None):
        if order is None:
            order = list(range(m))
        S = 0
        for candidate in order:
            if try_add(S, candidate):
                S |= (1 << candidate)
        return S

    # Try multiple greedy orders to find a good starting point
    best_S = 0
    best_size = 0

    # Standard greedy
    S1 = greedy_construct()
    best_S = S1
    best_size = popcount(S1)

    # Randomized greedy attempts
    rng = random.Random(42)
    for attempt in range(100):
        order = list(range(m))
        rng.shuffle(order)
        S = greedy_construct(order)
        size = popcount(S)
        if size > best_size:
            best_size = size
            best_S = S

    # Simulated Annealing
    current_S = best_S
    best_sa_S = best_S
    best_sa_size = best_size
    current_conflicts = count_conflicts(current_S)

    T_start = 1.0
    T_end = 0.001
    alpha = 0.9998

    T = T_start
    step = 0
    
    while time.time() < deadline:
        step += 1

        # Choose a random element to toggle
        elem = rng.randint(0, m - 1)
        bit = 1 << elem

        if current_S & bit:
            # Try removing
            new_S = current_S ^ bit
            # Conflicts can only decrease or stay same when removing
            # We don't strictly need to count if we know it decreases, 
            # but for SA we might want to accept non-improving moves? 
            # Usually removing is always good for conflict minimization, 
            # but bad for size. 
            # In this problem, we want MAX SIZE with ZERO CONFLICTS.
            # So we should only accept removals if they allow us to add more later?
            # Or just use it to escape local optima where we are stuck?
            # Actually, if we have 0 conflicts, removing an element is never beneficial for the final goal 
            # unless it allows adding 2+ elements. 
            # So we should be careful. 
            # Let's stick to the standard SA: minimize a cost function.
            # Cost = -size * large_weight + conflicts * weight?
            # Or simply: Accept if conflicts decrease. If conflicts same, accept if size increases.
            # If conflicts increase, accept with prob exp(-delta_conf/T).
            
            new_conflicts = count_conflicts(new_S)
            delta_conflicts = new_conflicts - current_conflicts
            delta_size = -1

            # We want to minimize conflicts primarily. 
            # If delta_conflicts < 0, accept (removing helps clear conflicts).
            # If delta_conflicts == 0, removing reduces size, so don't accept unless exploring.
            # If delta_conflicts > 0, very unlikely.
            
            if delta_conflicts < 0:
                current_S = new_S
                current_conflicts = new_conflicts
            elif delta_conflicts == 0:
                # Rarely accept removals if no conflict change, to explore
                if rng.random() < 0.01:
                    current_S = new_S
            # else: don't accept
        else:
            # Try adding
            if try_add(current_S, elem):
                new_S = current_S | bit
                # Conflicts remain 0 if we were at 0, or might increase if we had conflicts?
                # Wait, if current_S has conflicts, adding might add more.
                # But try_add checks if elem conflicts with ANY in S.
                # So if try_add is True, new conflicts = current_conflicts.
                # If try_add is False, new conflicts > current_conflicts.
                
                # Since try_add returned True, delta_conflicts is 0.
                # Size increases by 1. Always accept.
                current_S = new_S
                # current_conflicts unchanged
                if popcount(current_S) > best_sa_size:
                    best_sa_size = popcount(current_S)
                    best_sa_S = current_S
            else:
                # Adding creates conflicts
                new_S = current_S | bit
                new_conflicts = count_conflicts(new_S)
                delta_conflicts = new_conflicts - current_conflicts
                # delta_size = 1
                
                if delta_conflicts < 0:
                    # Impossible if try_add failed? 
                    # If try_add failed, elem conflicts with some in S.
                    # So new_S has at least those conflicts.
                    # current_S might have other conflicts.
                    # It is possible that current_S has many conflicts, and adding elem 
                    # somehow... no, adding a vertex cannot reduce the number of conflict pairs 
                    # in the existing subgraph. It only adds new edges.
                    # So delta_conflicts >= 0.
                    pass
                else:
                    prob = np.exp(-delta_conflicts / T)
                    if rng.random() < prob:
                        current_S = new_S
                        current_conflicts = new_conflicts

        # Occasionally try swap moves: remove one, add another
        if step % 5 == 0 and popcount(current_S) > 0:
            # Pick a random element from current_S to remove
            # Find a random set bit
            temp = current_S
            # Randomly select a bit from temp
            # Simple way: pick random index, check if set. If not, retry.
            # Or: convert to list? Too slow.
            # Let's just pick a random elem in 0..204 and check if it's in S.
            remove_elem = rng.randint(0, m - 1)
            if not (current_S & (1 << remove_elem)):
                continue
            
            add_elem = rng.randint(0, m - 1)
            if add_elem == remove_elem or (current_S & (1 << add_elem)):
                continue

            # Try swap: remove remove_elem, add add_elem
            new_S = (current_S & ~(1 << remove_elem)) | (1 << add_elem)
            
            # Check if add_elem conflicts with (S \ {remove_elem})
            # This is equivalent to: (new_S & conflict_mask[add_elem]) == 0 
            # because remove_elem is not in new_S, so it won't be in the intersection.
            # Wait, if remove_elem was the ONLY conflict for add_elem, then this works.
            # If add_elem conflicted with other elements in S, it will still conflict.
            
            if (new_S & conflict_mask[add_elem]) == 0:
                # No new conflicts introduced by add_elem relative to new_S
                # But we removed remove_elem, which might have reduced conflicts.
                # So new_conflicts <= current_conflicts?
                # Actually, we need to calculate new_conflicts to be sure, 
                # or reason about it.
                # Removing a node removes its incident edges. Adding a node adds its incident edges.
                # Since add_elem has 0 edges to new_S, it adds 0 conflicts.
                # So new_conflicts = current_conflicts - (conflicts involving remove_elem).
                # Thus new_conflicts <= current_conflicts.
                
                new_conflicts = count_conflicts(new_S)
                delta_conflicts = new_conflicts - current_conflicts
                
                if delta_conflicts <= 0:
                    current_S = new_S
                    current_conflicts = new_conflicts
                    if popcount(current_S) > best_sa_size:
                        best_sa_size = popcount(current_S)
                        best_sa_S = current_S
                else:
                    prob = np.exp(-delta_conflicts / T)
                    if rng.random() < prob:
                        current_S = new_S
                        current_conflicts = new_conflicts

        # Cool down
        T *= alpha
        if T < T_end:
            T = T_end

    # Final local search polish
    def local_search_polish(S):
        """Try to improve the set by adding elements or swapping."""
        improved = True
        while improved:
            improved = False
            # Try adding any element that doesn't create conflicts
            for elem in range(m):
                if not (S & (1 << elem)):
                    if try_add(S, elem):
                        S |= (1 << elem)
                        improved = True
            # Try swapping: remove one, add another
            # Iterate over all pairs (rem, add) where rem in S, add not in S
            # This is O(N^2) per pass, but N is small (<= 205).
            for rem in range(m):
                if not (S & (1 << rem)):
                    continue
                for add in range(m):
                    if (S & (1 << add)) or add == rem:
                        continue
                    new_S = (S & ~(1 << rem)) | (1 << add)
                    if (new_S & conflict_mask[add]) == 0:
                        # Check if this swap reduces conflicts or keeps them same
                        # Since add has no conflicts with new_S, 
                        # new_conflicts = current_conflicts - conflicts(rem, S\{rem})
                        # So it's always <= current_conflicts.
                        # We accept if it doesn't increase conflicts (it won't) 
                        # and we prefer to do it if it allows further additions?
                        # Actually, just doing it if it's valid is good for exploration.
                        # But we want to maximize size. Swaps keep size same.
                        # They are useful to reach a state where more additions are possible.
                        S = new_S
                        improved = True
                        break
                if improved:
                    break
        return S

    best_sa_S = local_search_polish(best_sa_S)
    if popcount(best_sa_S) > best_size:
        best_size = popcount(best_sa_S)
        best_S = best_sa_S

    # Also polish the original greedy best
    polished_greedy = local_search_polish(best_S)
    if popcount(polished_greedy) > popcount(best_S):
        best_S = polished_greedy
        best_size = popcount(best_S)

    # Convert bitmask to array
    result = []
    temp = best_S
    while temp:
        lsb = temp & -temp
        i = lsb.bit_length() - 1
        result.append(i)
        temp ^= lsb
    
    return np.array(sorted(result), dtype=np.int64)


# EVOLVE-BLOCK-END
