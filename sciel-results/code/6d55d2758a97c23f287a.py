"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # 1. Precompute Primes
    primes = _primes(N + 1)
    prime_set = set(primes)
    
    # 2. Initial Construction: Greedy Removal
    # Start with all numbers. Remove least populated residue class for each prime p <= k.
    # This ensures a valid starting set.
    points = set(range(N + 1))
    
    # We need to update k as we remove.
    # However, the constraint is p <= len(points).
    # As we remove points, len(points) decreases, so the set of relevant primes shrinks.
    # This makes the greedy removal very fast and safe.
    
    # Optimization: Instead of iterating all points for each prime, 
    # we can just pick a residue to remove. 
    # Actually, for the initial seed, a simpler approach is to just remove 
    # specific classes. But let's stick to the robust greedy removal for the seed.
    
    # To make it faster, we can precompute residues.
    # But N=5000 is small.
    
    # A faster seed: 
    # It is known that for admissible tuples, we can remove specific residues.
    # Let's just use the simple greedy:
    for p in primes:
        if p > len(points):
            break
        # Count residues
        counts = [0] * p
        for x in points:
            counts[x % p] += 1
        # Find min
        min_count = min(counts)
        # Remove all x in points where x % p == argmin
        target_res = counts.index(min_count)
        points = {x for x in points if x % p != target_res}
        
    # 3. State Representation
    # mask: boolean array of size N+1
    mask = np.zeros(N + 1, dtype=np.int8)
    for x in points:
        mask[x] = 1
        
    k = int(np.sum(mask))
    
    # Data structures for fast validity checks and cost calculation
    # counts[p][r] = number of elements in mask with x % p == r
    # num_empty[p] = number of residues r with counts[p][r] == 0
    # last_empty[p] = the unique empty residue if num_empty[p] == 1, else -1
    
    relevant_primes = [p for p in primes if p <= N + 1]
    
    counts = {}
    num_empty = {}
    last_empty = {}
    
    # Initialize state
    for p in relevant_primes:
        c = np.zeros(p, dtype=np.int32)
        for x in range(N + 1):
            if mask[x]:
                c[x % p] += 1
        counts[p] = c
        empties = np.where(c == 0)[0]
        num_empty[p] = len(empties)
        if len(empties) == 1:
            last_empty[p] = int(empties[0])
        else:
            last_empty[p] = -1
            
    # Helper: Update state for adding x
    def apply_add(x):
        nonlocal k
        k += 1
        mask[x] = 1
        for p in relevant_primes:
            if p > k:
                break
            r = x % p
            counts[p][r] += 1
            if counts[p][r] == 1:
                num_empty[p] -= 1
                if num_empty[p] == 1:
                    # Find the remaining empty
                    # Optimization: We can track empties more efficiently, but N is small
                    empties = np.where(counts[p] == 0)[0]
                    last_empty[p] = int(empties[0]) if len(empties) == 1 else -1
                else:
                    last_empty[p] = -1
                    
    # Helper: Update state for removing x
    def apply_remove(x):
        nonlocal k
        k -= 1
        mask[x] = 0
        for p in relevant_primes:
            if p > k:
                break
            r = x % p
            counts[p][r] -= 1
            if counts[p][r] == 0:
                num_empty[p] += 1
                if num_empty[p] == 1:
                    empties = np.where(counts[p] == 0)[0]
                    last_empty[p] = int(empties[0]) if len(empties) == 1 else -1
                else:
                    last_empty[p] = -1

    # Helper: Check if adding x is valid
    def is_valid_add(x):
        new_k = k + 1
        for p in relevant_primes:
            if p > new_k:
                break
            if num_empty[p] == 1:
                if x % p == last_empty[p]:
                    return False
        return True

    # Helper: Calculate "Cost" of removing x
    # Cost = number of primes p <= k where x is the ONLY element in its residue class?
    # No, removing x is always valid. 
    # The "value" of keeping x is how many "bottlenecks" it creates.
    # If we remove x, we might allow adding other elements.
    # A good heuristic for "which x to remove" is:
    # Remove x that covers the most "unique" residue slots.
    # i.e., for how many primes p is counts[p][x%p] == 1?
    # If we remove x, these counts become 0, creating new empty slots.
    # This makes it easier to add other elements that fall into these slots.
    
    def removal_value(x):
        val = 0
        for p in relevant_primes:
            if p > k:
                break
            r = x % p
            if counts[p][r] == 1:
                val += 1
        return val

    # Helper: Calculate "Gain" of adding x
    # If we add x, it is valid only if it doesn't fill the last empty slot for any p.
    # The "gain" is 1 (size increase). 
    # But we also care about "future potential".
    # A simple gain is just 1. But to guide the search, we might prefer adding x
    # that doesn't reduce num_empty for many primes?
    # Actually, adding x increases counts. If counts[p][r] was 0, it becomes 1, reducing num_empty.
    # We want to minimize the reduction of num_empty to keep options open.
    # So, "Cost of Adding" = number of primes p where x % p is currently empty.
    # We prefer adding x with LOW cost (fewer new empty slots filled).
    
    def add_cost(x):
        cost = 0
        new_k = k + 1
        for p in relevant_primes:
            if p > new_k:
                break
            r = x % p
            if counts[p][r] == 0:
                cost += 1
        return cost

    # 4. Local Search Loop (Hill Climbing with Swaps)
    # Strategy:
    # 1. Try to add any valid x. If found, add the one with lowest cost.
    # 2. If no valid x, find a swap: remove x1, add x2.
    #    We want to maximize net gain. Since we remove 1 and add 1, net size is 0.
    #    Wait, the goal is to INCREASE size.
    #    If no single add is valid, we are stuck in a local maximum for single additions.
    #    To escape, we must remove one element to make room for TWO elements? 
    #    Or remove one to make room for one different element that allows further growth?
    #    
    #    Actually, if no single add is valid, it means for every x not in mask, 
    #    there is some prime p <= k+1 such that num_empty[p]==1 and x%p == last_empty[p].
    #    
    #    If we remove y, k becomes k-1. 
    #    The condition for adding x becomes: for all p <= k, num_empty'[p] > 0 or (num_empty'[p]==1 and x%p != last_empty'[p]).
    #    Note: primes p where k < p <= k+1 are no longer checked (since new k is k-1, wait. 
    #    If we remove y, new size is k-1. The constraint applies to p <= k-1.
    #    So primes p in {k, k-1, ...} are checked.
    #    Previously, we checked p <= k. Now we check p <= k-1.
    #    So the constraint for p=k is GONE. This is a huge relaxation!
    #    
    #    Therefore, if we are stuck at size k, it is likely due to the prime p=k (if k is prime) 
    #    or primes close to k.
    #    Removing ANY element y relaxes the constraint for p=k (if k is prime) and potentially others.
    #    
    #    So, if no add is valid:
    #    1. Identify the "blocking" primes. These are primes p <= k+1 with num_empty[p]==1.
    #    2. We must remove an element y.
    #    3. Which y? We want to remove y such that we can then add MANY elements.
    #    4. Heuristic: Remove y that is "most constrained" or "least useful".
    #       Use removal_value(y) as a proxy. Higher value = more useful? 
    #       Actually, if counts[p][y%p]==1, y is the sole occupant. Removing it creates an empty slot.
    #       We want to create empty slots in primes that are currently blocking.
    #       
    #    Let's try a simple loop:
    #    While time remains:
    #       1. Try to add best x (lowest cost). If valid, do it.
    #       2. If not, try swaps.
    #          For a few candidate y (low removal_value? or random?):
    #             Remove y.
    #             Try to add best x.
    #             If added, keep the swap (size unchanged, but state changed).
    #             If not, revert y.
    #          This is a plateau walk.
    #          
    #    Better: 
    #    If no add is valid, we are at a local max.
    #    We should try to remove y and add x such that the new state allows FURTHER additions.
    #    This is complex.
    #    
    #    Simpler Robust Strategy:
    #    Iterate:
    #      - Find all valid adds. If any, add the one with min cost.
    #      - Else, find a y to remove. 
    #        To choose y: 
    #        Look at the blocking primes (p with num_empty[p]==1).
    #        For each such p, the "last_empty" residue is blocked.
    #        If we remove an element y, does it help?
    #        Removing y increases num_empty for p if counts[p][y%p] was 1.
    #        So, we should remove y that is the sole occupant of a residue for a blocking prime.
    #        
    #    Let's implement:
    #    1. Check adds.
    #    2. If none, find blocking primes.
    #    3. Find y that is sole occupant for at least one blocking prime.
    #       If multiple, pick the one with highest "impact" (sole occupant for most blocking primes).
    #    4. Remove y.
    #    5. Go to 1.
    
    best_k = k
    best_mask = mask.copy()
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add
        best_x = -1
        min_cost = float('inf')
        
        # Optimization: Only check x not in mask.
        # To speed up, we can precompute list of not_in_mask? 
        # Or just loop 0..N. N=5000 is small.
        
        for x in range(N + 1):
            if mask[x] == 0:
                if is_valid_add(x):
                    cost = add_cost(x)
                    if cost < min_cost:
                        min_cost = cost
                        best_x = x
                        # Can't get better than 0 cost
                        if cost == 0:
                            break
        
        if best_x != -1:
            apply_add(best_x)
            improved = True
            if k > best_k:
                best_k = k
                best_mask = mask.copy()
            continue # Restart loop with new state
        
        # Phase 2: No valid add. Try to remove to unblock.
        # Find blocking primes
        blocking_primes = []
        for p in relevant_primes:
            if p > k + 1:
                break
            if num_empty[p] == 1:
                blocking_primes.append(p)
                
        if not blocking_primes:
            # Should not happen if is_valid_add failed, but safety check
            break
            
        # Find best y to remove
        # y should be sole occupant (counts[p][y%p]==1) for at least one blocking prime.
        best_y = -1
        max_impact = 0
        
        # We only need to consider y in mask.
        # To speed up, we can iterate through mask elements?
        # Or just 0..N and check mask[x]==1.
        
        for y in range(N + 1):
            if mask[y] == 1:
                impact = 0
                for p in blocking_primes:
                    r = y % p
                    if counts[p][r] == 1:
                        impact += 1
                if impact > max_impact:
                    max_impact = impact
                    best_y = y
                    if max_impact == len(blocking_primes):
                        break
        
        if best_y == -1:
            # No element is a sole occupant for any blocking prime.
            # This means removing any element will NOT increase num_empty for any blocking prime.
            # Wait, if counts[p][r] > 1 for all y in mask for all blocking p, 
            # then removing y reduces counts but doesn't create 0.
            # So num_empty stays 1.
            # Then we are stuck?
            # Actually, if we remove y, k decreases.
            # The prime p=k (if prime) is no longer checked.
            # If k is prime and was blocking, removing y solves it immediately.
            # If k is not prime, the blocking primes are < k.
            # Removing y doesn't change num_empty for them if no sole occupants.
            # But it reduces k.
            # So the set of checked primes shrinks.
            # So removing ANY y is valid and might unblock if the blocker was p=k.
            
            # Let's just remove a random y or the one with lowest removal_value (least useful).
            min_val = float('inf')
            for y in range(N + 1):
                if mask[y] == 1:
                    val = removal_value(y)
                    if val < min_val:
                        min_val = val
                        best_y = y
            
            if best_y == -1:
                break
                
            apply_remove(best_y)
            # We don't count this as "improved" in size, but it's a step.
            # We hope next iteration allows adds.
            # To avoid infinite loops, we can track state or just let time run out.
            # Since we are deterministic, if we cycle, we waste time.
            # But usually, removing an element opens up space.
            continue
            
        else:
            apply_remove(best_y)
            continue

    # Final Greedy Polish
    # Rebuild state for best_mask to ensure consistency
    # Actually, best_mask is a copy. We should return it.
    # But maybe we can do a final linear scan add on best_mask?
    # The loop above already does this.
    
    return list(np.nonzero(best_mask)[0])
# EVOLVE-BLOCK-END
