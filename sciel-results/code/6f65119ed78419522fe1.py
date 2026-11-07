import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    max_val = 256
    max_size = 60
    
    def try_order(order):
        """
        Greedy construction with bitset DP for fast validity checks.
        """
        masks = [0] * (max_size + 1)
        masks[0] = 1  # Sum 0 with 0 elements
        
        current_set = []
        
        for x in order:
            if len(current_set) >= max_size:
                break
            if time.time() > deadline:
                break
            
            valid = True
            for i in range(len(current_set) + 1):
                if masks[i] == 0:
                    continue
                
                new_sums = masks[i] << x
                target_k = i + 1
                
                for j in range(len(current_set) + 1):
                    if j == target_k:
                        continue
                    if masks[j] != 0 and (new_sums & masks[j]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                for k in range(len(current_set) + 1, 0, -1):
                    masks[k] |= (masks[k-1] << x)
                current_set.append(x)
        
        return current_set, masks

    def can_add(x, current_set, masks):
        """Check if x can be added to current_set."""
        for i in range(len(current_set) + 1):
            if masks[i] == 0:
                continue
            new_sums = masks[i] << x
            target_k = i + 1
            for j in range(len(current_set) + 1):
                if j == target_k:
                    continue
                if masks[j] != 0 and (new_sums & masks[j]):
                    return False
        return True

    def add_element(x, current_set, masks):
        """Add x to current_set and update masks."""
        for k in range(len(current_set) + 1, 0, -1):
            masks[k] |= (masks[k-1] << x)
        current_set.append(x)

    def remove_element(x, current_set, masks):
        """Remove x from current_set and rebuild masks."""
        current_set.remove(x)
        new_masks = [0] * (max_size + 1)
        new_masks[0] = 1
        for y in current_set:
            for k in range(len(current_set), 0, -1):
                if new_masks[k-1]:
                    new_masks[k] |= (new_masks[k-1] << y)
        return new_masks

    def local_search(initial_set, time_limit, max_swaps=50):
        """
        Attempt to improve the set by swapping elements or adding new ones.
        """
        if time.time() > time_limit:
            return initial_set
        
        current_set = list(initial_set)
        current_size = len(current_set)
        
        masks = [0] * (max_size + 1)
        masks[0] = 1
        for x in current_set:
            for k in range(current_size, 0, -1):
                if masks[k-1]:
                    masks[k] |= (masks[k-1] << x)
        
        used = set(current_set)
        candidates = [x for x in range(1, max_val + 1) if x not in used]
        
        improved = True
        swap_count = 0
        while improved and time.time() < time_limit and swap_count < max_swaps:
            improved = False
            
            # Try adding a new element
            if current_size < max_size:
                for x in candidates:
                    if time.time() > time_limit:
                        break
                    
                    if can_add(x, current_set, masks):
                        add_element(x, current_set, masks)
                        current_size += 1
                        used.add(x)
                        candidates.remove(x)
                        improved = True
                        swap_count += 1
                        break
            
            if not improved and current_size > 0:
                # Try swapping: remove one element, try to add a different one
                # Focus on removing smaller elements first (they're more likely to cause collisions)
                for idx in range(len(current_set)):
                    if time.time() > time_limit:
                        break
                    
                    x_remove = current_set[idx]
                    # Try removing this element and see if we can add a better one
                    test_set = current_set[:idx] + current_set[idx+1:]
                    test_masks = [0] * (max_size + 1)
                    test_masks[0] = 1
                    for y in test_set:
                        for k in range(len(test_set), 0, -1):
                            if test_masks[k-1]:
                                test_masks[k] |= (test_masks[k-1] << y)
                    
                    # Try to add any candidate that works with the reduced set
                    for x_add in candidates:
                        if time.time() > time_limit:
                            break
                        if can_add(x_add, test_set, test_masks):
                            # Check if x_add is "better" than x_remove (larger value generally better)
                            if x_add > x_remove:
                                # Do the swap
                                current_set = test_set + [x_add]
                                current_size = len(current_set)
                                used.discard(x_remove)
                                used.add(x_add)
                                if x_remove in candidates:
                                    candidates.remove(x_remove)
                                if x_add in candidates:
                                    candidates.remove(x_add)
                                else:
                                    candidates.append(x_remove)
                                improved = True
                                swap_count += 1
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        # Rebuild final masks
        final_masks = [0] * (max_size + 1)
        final_masks[0] = 1
        for x in current_set:
            for k in range(len(current_set), 0, -1):
                if final_masks[k-1]:
                    final_masks[k] |= (final_masks[k-1] << x)
        
        return current_set, final_masks

    best_set = []
    best_size = 0
    
    # Strategy 1: Descending order
    order_desc = list(range(max_val, 0, -1))
    res, _ = try_order(order_desc)
    if len(res) > best_size:
        best_set = res
        best_size = len(res)
    
    # Strategy 2: Ascending order
    if time.time() < deadline - 5:
        order_asc = list(range(1, max_val + 1))
        res, _ = try_order(order_asc)
        if len(res) > best_size:
            best_set = res
            best_size = len(res)
    
    # Strategy 3: Many randomized orders with local search
    rng = random.Random(42)
    
    for attempt in range(500):
        if time.time() > deadline - 3.0:
            break
        
        # Create diverse random orders
        candidates = list(range(1, max_val + 1))
        
        # Different randomization strategies
        strategy = attempt % 5
        
        if strategy == 0:
            # Pure random shuffle
            rng.shuffle(candidates)
        elif strategy == 1:
            # Biased: larger numbers first
            candidates.sort(key=lambda x: -x + rng.random() * 50)
        elif strategy == 2:
            # Biased: smaller numbers first
            candidates.sort(key=lambda x: x + rng.random() * 50)
        elif strategy == 3:
            # Interleaved: alternate large and small
            top = list(range(max_val // 2 + 1, max_val + 1))
            bot = list(range(1, max_val // 2 + 1))
            rng.shuffle(top)
            rng.shuffle(bot)
            interleaved = []
            for i in range(max_val // 2):
                if i < len(top):
                    interleaved.append(top[i])
                if i < len(bot):
                    interleaved.append(bot[i])
            candidates = interleaved
        else:
            # Cluster-based: group by ranges and shuffle within
            clusters = []
            for start in range(1, max_val + 1, 32):
                end = min(start + 32, max_val + 1)
                cluster = list(range(start, end))
                rng.shuffle(cluster)
                clusters.append(cluster)
            rng.shuffle(clusters)
            candidates = [x for cluster in clusters for x in cluster]
        
        res, masks = try_order(candidates)
        current_size = len(res)
        
        if current_size > best_size:
            best_set = res
            best_size = current_size
        
        # Apply local search if we have time and the set isn't too small
        if current_size >= 15 and time.time() < deadline - 5.0:
            improved_set, improved_masks = local_search(res, deadline - 3.0, max_swaps=30)
            if len(improved_set) > best_size:
                best_set = improved_set
                best_size = len(improved_set)
        
        # Early termination if we found a very large set
        if best_size >= 35:
            # Try a few more with local search
            for _ in range(5):
                if time.time() > deadline - 3.0:
                    break
                candidates2 = list(range(1, max_val + 1))
                rng.shuffle(candidates2)
                res2, _ = try_order(candidates2)
                if len(res2) >= best_size - 2:
                    improved_set2, _ = local_search(res2, deadline - 3.0, max_swaps=20)
                    if len(improved_set2) > best_size:
                        best_set = improved_set2
                        best_size = len(improved_set2)
                if best_size >= 40:
                    break
    
    # Final local search on the best set found
    if best_size > 0 and time.time() < deadline - 2.0:
        final_set, _ = local_search(best_set, deadline - 1.0, max_swaps=50)
        if len(final_set) > best_size:
            best_set = final_set
            best_size = len(final_set)
    
    # Ensure we return a valid list of ints
    if not best_set:
        best_set = [2**i for i in range(9) if 2**i <= 256]
    
    return [int(x) for x in best_set]
