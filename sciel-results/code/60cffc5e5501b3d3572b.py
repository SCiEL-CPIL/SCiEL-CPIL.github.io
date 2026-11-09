import time
import random

START_TIME = time.time()
DEADLINE = 55.0  # Safety margin before 60s kill

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def check_validity_bitset(sums, n):
    """
    Check if the bitsets in 'sums' represent a valid cardinality-separated set.
    sums[k] is an integer where bit s is 1 if sum s is achievable with k elements.
    n is the current size of the set.
    Valid iff sums[i] & sums[j] == 0 for all i != j.
    """
    for i in range(n + 1):
        for j in range(i + 1, n + 2):
            if sums[i] & sums[j]:
                return False
    return True

def try_greedy_order(order, max_val=256):
    """
    Greedily build a set by trying elements in the given order.
    Returns the largest valid set found.
    """
    # sums[k] is a bitset: bit s set means sum s is achievable with exactly k elements
    # Start: empty subset gives sum 0 with cardinality 0
    sums = [0] * 62
    sums[0] = 1  # bit 0 set: sum 0 achievable with 0 elements
    
    chosen = []
    max_sum = 0  # track max possible sum to bound bitset operations
    
    for candidate in order:
        if candidate < 1 or candidate > max_val:
            continue
        if candidate in chosen:
            continue
        
        # Try adding candidate: new sums at cardinality k = old sums[k] | (old sums[k-1] shifted by candidate)
        # We need to check if any new sum at cardinality k collides with existing sums at cardinality j != k
        # Efficient check: for each k, new_sums[k] = sums[k] | ((sums[k-1] << candidate) if k > 0 else 0)
        # Collision check: new_sums[k] & sums[j] for j != k must be 0
        # But actually we just need: for all k != j, (sums[k] | ((sums[k-1] << candidate) if k>0 else 0)) & (sums[j] | ((sums[j-1] << candidate) if j>0 else 0)) == 0
        # Simpler: compute all new_sums, then check pairwise
        
        new_sums = sums[:]  # copy
        current_max_card = len(chosen)
        
        # Update sums in reverse order to avoid using updated values
        # Actually, let's compute the new sums properly
        # new_sums[k] = sums[k] | (sums[k-1] << candidate) for k >= 1
        # new_sums[0] = sums[0] (unchanged, still just bit 0)
        
        # Let's build new_sums from scratch
        ns = [0] * 62
        ns[0] = 1  # sum 0 always achievable with 0 elements
        
        for k in range(1, current_max_card + 2):
            # Without candidate: sums[k]
            # With candidate: sums[k-1] shifted by candidate
            val = sums[k]
            if k - 1 >= 0 and k - 1 <= current_max_card:
                val |= (sums[k-1] << candidate)
            ns[k] = val
        
        # Check validity: for all i != j, ns[i] & ns[j] == 0
        valid = True
        for i in range(current_max_card + 2):
            if ns[i] == 0:
                continue
            for j in range(i + 1, current_max_card + 3):
                if ns[j] == 0:
                    continue
                if ns[i] & ns[j]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            sums = ns
            chosen.append(candidate)
            max_sum += candidate
    
    return chosen

def limited_backtrack(candidates, sums, chosen, depth=0, max_depth=8, max_val=256):
    """
    Limited-depth backtracking search with forward checking.
    Returns the best extension found within the depth limit.
    """
    if is_done() or depth >= max_depth:
        return list(chosen)
    
    best = list(chosen)
    
    for i, candidate in enumerate(candidates):
        if candidate < 1 or candidate > max_val:
            continue
        if candidate in chosen:
            continue
        
        # Try adding candidate
        current_max_card = len(chosen)
        ns = [0] * 62
        ns[0] = 1
        
        for k in range(1, current_max_card + 2):
            val = sums[k]
            if k - 1 >= 0 and k - 1 <= current_max_card:
                val |= (sums[k-1] << candidate)
            ns[k] = val
        
        # Check validity
        valid = True
        for a in range(current_max_card + 2):
            if ns[a] == 0:
                continue
            for b in range(a + 1, current_max_card + 3):
                if ns[b] == 0:
                    continue
                if ns[a] & ns[b]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Recurse with remaining candidates
            remaining = candidates[i+1:]
            result = limited_backtrack(remaining, ns, chosen + [candidate], depth + 1, max_depth, max_val)
            if len(result) > len(best):
                best = result
            
            # Early exit if we found a very good result
            if len(best) >= max_depth:
                break
    
    return best

def construct():
    best_set = []
    best_len = 0
    
    # Strategy 1: Try ascending order
    order_asc = list(range(1, 257))
    result = try_greedy_order(order_asc)
    if len(result) > best_len:
        best_set = result
        best_len = len(result)
    
    # Strategy 2: Try descending order
    order_desc = list(range(256, 0, -1))
    result = try_greedy_order(order_desc)
    if len(result) > best_len:
        best_set = result
        best_len = len(result)
    
    # Strategy 3: Random restarts with different orderings
    random.seed(42)
    attempts = 0
    max_attempts = 2000
    
    while not is_done() and attempts < max_attempts:
        attempts += 1
        
        # Create a random ordering with some bias
        r = random.random()
        if r < 0.3:
            # Mostly ascending with some randomization
            order = list(range(1, 257))
            random.shuffle(order)
            # Re-sort but with some random swaps
            for _ in range(random.randint(0, 20)):
                i = random.randint(0, 255)
                j = random.randint(0, 255)
                order[i], order[j] = order[j], order[i]
        elif r < 0.6:
            # Random ordering
            order = list(range(1, 257))
            random.shuffle(order)
        else:
            # Descending with randomization
            order = list(range(256, 0, -1))
            for _ in range(random.randint(0, 20)):
                i = random.randint(0, 255)
                j = random.randint(0, 255)
                order[i], order[j] = order[j], order[i]
        
        result = try_greedy_order(order)
        if len(result) > best_len:
            best_set = result
            best_len = len(result)
    
    # Strategy 4: Limited backtracking from a good starting point
    if best_set and not is_done():
        # Use the best set as a guide, try to extend it with backtracking
        used = set(best_set)
        remaining = [x for x in range(1, 257) if x not in used]
        remaining.sort()
        
        # Compute sums for the current best set
        n = len(best_set)
        sums = [0] * (n + 2)
        sums[0] = 1
        for elem in best_set:
            ns = [0] * (n + 2)
            ns[0] = 1
            for k in range(1, n + 1):
                val = sums[k]
                if k - 1 >= 0:
                    val |= (sums[k-1] << elem)
                ns[k] = val
            sums = ns
        
        # Try to extend with limited backtracking
        if not is_done():
            # Limit candidates to keep search tractable
            search_candidates = remaining[:30]
            result = limited_backtrack(search_candidates, sums, best_set, 0, 5, 256)
            if len(result) > best_len:
                best_set = result
                best_len = len(result)
    
    # Strategy 5: Fresh backtracking from scratch with smart ordering
    if not is_done():
        # Try a fresh backtracking search with a good initial candidate order
        # Use a mix of small and large values
        smart_order = []
        # Add some small values
        smart_order.extend(range(1, 20))
        # Add some medium values
        smart_order.extend(range(50, 80))
        # Add some large values
        smart_order.extend(range(200, 257))
        # Shuffle slightly
        random.shuffle(smart_order)
        
        init_sums = [0] * 62
        init_sums[0] = 1
        
        result = limited_backtrack(smart_order, init_sums, [], 0, 10, 256)
        if len(result) > best_len:
            best_set = result
            best_len = len(result)
    
    # Final verification of best_set
    if best_set:
        # Verify validity
        sums = [0] * (len(best_set) + 2)
        sums[0] = 1
        for elem in best_set:
            ns = [0] * (len(best_set) + 2)
            ns[0] = 1
            for k in range(1, len(best_set) + 1):
                val = sums[k]
                if k - 1 >= 0:
                    val |= (sums[k-1] << elem)
                ns[k] = val
            sums = ns
        
        # Check pairwise
        valid = True
        n = len(best_set)
        for i in range(n + 1):
            for j in range(i + 1, n + 2):
                if sums[i] & sums[j]:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to a known valid set
            best_set = list(range(128, 136))
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best_set]
