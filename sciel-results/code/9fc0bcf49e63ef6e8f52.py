import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    max_val = 256
    max_size = 60
    
    def _try_greedy(order, deadline):
        """
        Attempts to build a valid set by adding elements in the given order.
        Uses bitset DP for fast validity checks.
        Returns the constructed list of integers.
        """
        # masks[k] is a bitset where bit s is 1 if sum s is reachable with k elements
        # Max sum approx 60 * 256 = 15360. Python ints handle this easily.
        masks = [0] * (max_size + 1)
        masks[0] = 1  # Sum 0 with 0 elements
        
        current_set = []
        
        for x in order:
            if len(current_set) >= max_size:
                break
            if time.time() > deadline:
                break
            
            # Check if adding x is valid
            # new_sums for cardinality k are (masks[k-1] << x)
            # These new sums must not collide with existing sums of different cardinalities.
            # Existing sums for cardinality j are in masks[j].
            # Collision if (masks[k-1] << x) & masks[j] != 0 for any j != k-1.
            
            valid = True
            # We only need to check k such that masks[k-1] is not 0
            # max current cardinality is len(current_set)
            # So k can go up to len(current_set) + 1
            
            # Pre-compute the shifted masks for the current element
            # shift_masks[k] = masks[k-1] << x
            # We need to check if shift_masks[k] intersects masks[j] for j != k-1
            
            # Optimization: 
            # Union of all masks[j] for j != k-1 is (all_masks ^ masks[k-1])
            # But we need to check per k.
            
            # Let's just iterate k
            for k in range(1, len(current_set) + 2):
                if masks[k-1] == 0:
                    continue
                new_sums = masks[k-1] << x
                # Check collision with all other cardinalities
                # We can check against all masks[j] where j != k-1
                # To be efficient, we can check against the union of all masks except masks[k-1]
                # union_others = 0
                # for j in range(len(current_set)+1):
                #    if j != k-1: union_others |= masks[j]
                # if new_sums & union_others: valid = False; break
                
                # Simpler: check against each j
                # But number of cardinalities is small (< 60)
                # Bitwise AND is fast.
                
                # Let's use the union trick if we precompute it?
                # No, k changes, so union_others changes.
                
                # Just loop j
                for j in range(len(current_set) + 1):
                    if j == k - 1:
                        continue
                    if masks[j] != 0 and (new_sums & masks[j]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Update masks
                # Update in reverse order to avoid using x multiple times
                for k in range(len(current_set) + 1, 0, -1):
                    masks[k] |= (masks[k-1] << x)
                current_set.append(x)
        
        return current_set

    best_set = []
    
    # Strategy 1: Descending order (spread sums)
    order_desc = list(range(max_val, 0, -1))
    res = _try_greedy(order_desc, deadline)
    if len(res) > len(best_set):
        best_set = res
        
    # Strategy 2: Ascending order
    order_asc = list(range(1, max_val + 1))
    if time.time() < deadline:
        res = _try_greedy(order_asc, deadline)
        if len(res) > len(best_set):
            best_set = res
            
    # Strategy 3: Random orders
    # If we have time, try several random permutations
    # Since the bitset check is very fast, we can try many.
    attempts = 0
    while time.time() < deadline - 5:
        random_order = list(range(1, max_val + 1))
        random.shuffle(random_order)
        res = _try_greedy(random_order, deadline)
        if len(res) > len(best_set):
            best_set = res
            # If we reached max size, we can stop
            if len(best_set) >= max_size:
                break
        attempts += 1
        # Limit attempts to avoid infinite loop if something is wrong, though time check handles it
        if attempts > 10000:
            break

    # Final validation and return
    if not best_set:
        # Fallback to superincreasing
        best_set = [1, 2, 4, 8, 16, 32, 64, 128]
        
    # Convert to list of ints
    return list(best_set)

# EVOLVE-BLOCK-END
