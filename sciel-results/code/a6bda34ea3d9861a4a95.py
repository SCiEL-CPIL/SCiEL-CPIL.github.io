# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    Uses bitmask-based simulated annealing and local search for speed.
    """
    m = 205
    # Use a safety margin to ensure we return before the 630s limit
    deadline = time.time() + 600 

    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    forbidden_set = set(forbidden)

    # Precompute conflict masks
    # conflict_mask[i] is an integer bitmask where bit j is 1 if i and j conflict
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d = (i - j) % m
            if d in forbidden_set:
                mask |= (1 << j)
        conflict_mask[i] = mask

    def mask_to_list(mask):
        """Convert a bitmask to a list of set bits."""
        res = []
        i = 0
        while mask:
            if mask & 1:
                res.append(i)
            mask >>= 1
            i += 1
        return res

    def list_to_mask(lst):
        """Convert a list of integers to a bitmask."""
        mask = 0
        for x in lst:
            mask |= (1 << x)
        return mask

    def is_valid_mask(mask):
        """Check if the set represented by mask is an independent set."""
        # For each i in mask, check if it conflicts with any other in mask
        # (mask & conflict_mask[i]) should be 0 for all i in mask?
        # No, conflict_mask[i] contains j if i~j. 
        # If mask has i and j, and i~j, then bit j is set in conflict_mask[i].
        # So (mask & conflict_mask[i]) will have bit j set.
        # Thus, for all i in mask, (mask & conflict_mask[i]) must be 0?
        # Wait, if i is in mask, and j is in mask, and i~j, then bit j is in conflict_mask[i].
        # So mask & conflict_mask[i] will be non-zero.
        # Therefore, a set is valid iff for all i in mask, (mask & conflict_mask[i]) == 0.
        # This is O(N^2) if we iterate, but we can do it smarter?
        # Actually, just checking all i is fine for N=205 if done rarely.
        # But in SA, we do swaps.
        # For a swap (remove r, add a):
        # new_mask = (mask & ~(1<<r)) | (1<<a)
        # We know old mask was valid.
        # We only need to check if 'a' conflicts with any other element in new_mask.
        # The other elements are (mask & ~(1<<r)).
        # So we check if (new_mask & conflict_mask[a]) == 0.
        # This is O(1) bitwise op.
        pass

    def greedy_seed():
        """Greedy construction using bitmasks."""
        best_mask = 0
        best_size = 0
        for trial in range(10):
            order = list(range(m))
            if trial > 0:
                random.shuffle(order)
            current_mask = 0
            for candidate in order:
                # Check if candidate can be added
                # candidate is in current_mask? No, we iterate unique candidates.
                # Check conflict with current_mask
                if (current_mask & conflict_mask[candidate]) == 0:
                    current_mask |= (1 << candidate)
            
            size = bin(current_mask).count('1')
            if size > best_size:
                best_size = size
                best_mask = current_mask
        return mask_to_list(best_mask)

    def local_search_improve(mask):
        """
        Try to improve by adding elements or swapping to increase size.
        Returns the improved mask.
        """
        current_mask = mask
        current_size = bin(current_mask).count('1')
        improved = True
        start_time = time.time()
        
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            
            # 1. Try adding any element that fits
            # Iterate through all candidates not in current_mask
            # We can just iterate 0..m-1 and check if bit is not set
            for cand in range(m):
                if not (current_mask & (1 << cand)):
                    # Check if cand can be added
                    if (current_mask & conflict_mask[cand]) == 0:
                        current_mask |= (1 << cand)
                        current_size += 1
                        improved = True
                        break # Restart the loop to check for more additions
            
            if improved:
                continue

            # 2. Try swaps: remove one, add one
            # To be efficient, we only try swaps that increase size? 
            # No, swaps keep size same but might allow future additions.
            # However, simple swaps (size 0 delta) are hard to accept in SA unless we have a good heuristic.
            # For local search, we usually only accept improvements (size +1).
            # Since we already tried all additions, any swap that results in a valid set
            # of the same size doesn't immediately help unless it enables an addition.
            # A 2-opt style move: remove r, add a.
            # If we remove r and add a, size is same. 
            # We should check if this new configuration allows an addition.
            # This is complex. 
            # Alternative: Random restarts are more effective than complex local swaps for this small N.
            # Let's stick to greedy + simple SA.
            
            # If no addition found, we are stuck. 
            # We can try a random swap to escape local minimum, but that's SA's job.
            # Local search here is just "greedy completion".
            break
            
        return current_mask

    def simulated_annealing(initial_mask, max_steps=100000):
        """
        Simulated annealing using bitmasks.
        """
        current_mask = initial_mask
        current_size = bin(current_mask).count('1')
        best_mask = current_mask
        best_size = current_size

        T = 1.0
        T_min = 0.01
        cooling = 0.9999

        steps = 0
        all_indices = list(range(m))
        
        while T > T_min and steps < max_steps:
            if time.time() > deadline - 5:
                break

            # Choose a move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.3:
                # Try to add an element
                # Pick a random candidate not in current
                # To avoid generating a list every time, we can pick a random index and check
                # But if it's in current, we retry. 
                # Better: pick from available.
                # For speed, just pick random 0..m-1 and check if it's in current.
                # If it is, skip (or try again).
                # Let's just pick a random one.
                cand = random.randint(0, m - 1)
                if not (current_mask & (1 << cand)):
                    # Check if can add
                    if (current_mask & conflict_mask[cand]) == 0:
                        new_mask = current_mask | (1 << cand)
                        delta = 1
                        # Always accept improvement
                        current_mask = new_mask
                        current_size += 1
                        if current_size > best_size:
                            best_mask = current_mask
                            best_size = current_size
                    else:
                        # Can't add. Maybe swap?
                        # Find a conflicting element to remove
                        # conflicts = current_mask & conflict_mask[cand]
                        # Pick one randomly
                        conflicts = current_mask & conflict_mask[cand]
                        if conflicts:
                            # Pick random bit from conflicts
                            # We need to pick a random set bit from an integer.
                            # This is tricky in pure Python without loops.
                            # Loop is fine for small N.
                            r = random.randint(0, m - 1)
                            # Ensure r is in conflicts
                            # This might take a few tries
                            found = False
                            for _ in range(10):
                                r = random.randint(0, m - 1)
                                if (conflicts & (1 << r)):
                                    found = True
                                    break
                            if found:
                                new_mask = (current_mask & ~(1 << r)) | (1 << cand)
                                # Check validity of new_mask regarding cand
                                # We removed r which was conflicting. 
                                # We need to ensure cand doesn't conflict with others.
                                # (new_mask & conflict_mask[cand]) should be 0.
                                if (new_mask & conflict_mask[cand]) == 0:
                                    delta = 0
                                    if random.random() < np.exp(delta / T):
                                        current_mask = new_mask
                                        # Size unchanged
                                        if current_size > best_size:
                                            best_mask = current_mask
                                            best_size = current_size

            elif move_type < 0.5:
                # Try to remove an element
                if current_size > 0:
                    r = random.randint(0, m - 1)
                    if (current_mask & (1 << r)):
                        new_mask = current_mask & ~(1 << r)
                        delta = -1
                        if random.random() < np.exp(delta / T):
                            current_mask = new_mask
                            current_size -= 1

            else:
                # Swap: remove one, add one
                if current_size > 0:
                    r = random.randint(0, m - 1)
                    if (current_mask & (1 << r)):
                        cand = random.randint(0, m - 1)
                        if not (current_mask & (1 << cand)):
                            new_mask = (current_mask & ~(1 << r)) | (1 << cand)
                            # Check validity
                            if (new_mask & conflict_mask[cand]) == 0:
                                delta = 0
                                if random.random() < np.exp(delta / T):
                                    current_mask = new_mask
                                    # Size unchanged
                                    if current_size > best_size:
                                        best_mask = current_mask
                                        best_size = current_size

            T *= cooling
            steps += 1

        return best_mask

    # Main search
    best_overall_mask = 0
    best_overall_size = 0

    # Start with greedy seeds
    for trial in range(20):
        if time.time() > deadline - 10:
            break
        seed_list = greedy_seed()
        seed_mask = list_to_mask(seed_list)
        # Improve with local search (greedy completion)
        improved_mask = local_search_improve(seed_mask)
        size = bin(improved_mask).count('1')
        if size > best_overall_size:
            best_overall_size = size
            best_overall_mask = improved_mask

    # Simulated annealing from multiple starts
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        seed_list = greedy_seed()
        seed_mask = list_to_mask(seed_list)
        # Run SA
        result_mask = simulated_annealing(seed_mask, max_steps=20000)
        size = bin(result_mask).count('1')
        if size > best_overall_size:
            best_overall_size = size
            best_overall_mask = result_mask

    # Final local search on the best
    if best_overall_mask:
        final_mask = local_search_improve(best_overall_mask)
        size = bin(final_mask).count('1')
        if size > best_overall_size:
            best_overall_size = size
            best_overall_mask = final_mask

    # Ensure we have at least one element
    if not best_overall_mask:
        best_overall_mask = 1 << 0

    best_list = mask_to_list(best_overall_mask)
    
    # Final sanity check
    # Verify no conflicts
    valid = True
    for i in range(len(best_list)):
        for j in range(i+1, len(best_list)):
            a = best_list[i]
            b = best_list[j]
            if (1 << b) & conflict_mask[a]:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to a simple valid set if something went wrong (shouldn't happen)
        best_list = [0]

    return np.array(best_list, dtype=np.int64)

# EVOLVE-BLOCK-END
