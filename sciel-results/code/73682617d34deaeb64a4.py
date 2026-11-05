import time
import random
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute bad masks
    # bad_mask[a-1][b-1] is a bitmask of c (1..128) such that (a,b,c) is a bad triple
    # We can also precompute a list of bad triples for each pair, or just use the mask.
    # Using bitmask allows fast "is there any c in current set?" check.
    # But for "which c's conflict?", we need the mask.
    
    # Actually, for the local search "try add x", we need to check if x is in the union of bad masks of all pairs in S.
    # This is O(|S|^2) to compute the union if done from scratch, but we can maintain it.
    
    bad_mask = [[0] * N for _ in range(N)]
    
    # Precompute LCM table for speed? Or just use math.lcm.
    # math.lcm is C-implemented and fast.
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm(a, b)
            mask = 0
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if lab == lcm(a, c) == lcm(b, c):
                    mask |= (1 << (c - 1))
            bad_mask[a - 1][b - 1] = mask
            bad_mask[b - 1][a - 1] = mask
            
    # Helper to check if a set is valid (should be fast if we maintain conflict mask)
    # But for initialization, we might just build greedily.
    
    def greedy_build(order):
        current_set = set()
        # We can maintain the "forbidden" mask.
        # forbidden_mask has bit i set if adding (i+1) would create a bad triple with current_set.
        forbidden_mask = 0
        
        for x in order:
            # Check if x is forbidden
            if (forbidden_mask >> (x - 1)) & 1:
                continue
            
            # Add x
            current_set.add(x)
            
            # Update forbidden_mask
            # New conflicts arise from pairs (x, y) for y in current_set (excluding x)
            # Actually, we just added x. The new pairs are (x, y) for all y already in current_set.
            # Wait, current_set includes x now. So we iterate over y in current_set where y != x.
            for y in current_set:
                if y == x:
                    continue
                forbidden_mask |= bad_mask[x - 1][y - 1]
                
        return current_set

    best_set = set()
    best_size = 0
    
    # Generate diverse orders
    def num_distinct_primes(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    def divisor_count(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: (-num_distinct_primes(x), x)),
        sorted(range(1, N + 1), key=lambda x: (num_distinct_primes(x), x)),
        sorted(range(1, N + 1), key=lambda x: (-divisor_count(x), x)),
        sorted(range(1, N + 1), key=lambda x: (divisor_count(x), x)),
    ]
    
    # Add some random orders
    for _ in range(10):
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        orders.append(random_order)
        
    # Run greedy for each order
    for order in orders:
        if time.time() > deadline:
            break
        s = greedy_build(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
            
    # Local Search on the best set found
    current_set = set(best_set)
    
    # Reconstruct forbidden_mask for current_set
    def calc_forbidden_mask(S):
        mask = 0
        S_list = list(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                mask |= bad_mask[a - 1][b - 1]
        return mask

    forbidden_mask = calc_forbidden_mask(current_set)
    
    # Local Search Loop
    # We want to try to add elements or swap elements to increase size.
    # Since we maximize cardinality, we only care about adding.
    # If we can't add, we might try to remove one and add two? (1-for-2 swap)
    # Or just remove one and add one (swap) if it allows future additions?
    # But for now, let's stick to "try to add". If stuck, try a 1-for-2 swap.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to add any element
        added_any = False
        for x in range(1, N + 1):
            if x in current_set:
                continue
            if (forbidden_mask >> (x - 1)) & 1:
                continue
            
            # Try adding x
            current_set.add(x)
            # Update forbidden_mask
            for y in current_set:
                if y == x:
                    continue
                forbidden_mask |= bad_mask[x - 1][y - 1]
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            improved = True
            added_any = True
            break # Restart loop with new state
            
        if added_any:
            continue
            
        # 2. If no single addition works, try 1-for-2 swaps
        # Remove y, add x1 and x2.
        # This is O(N^3) roughly.
        # To speed up:
        # For each y in current_set:
        #   Remove y temporarily.
        #   Calculate new forbidden mask.
        #   Try to add two elements greedily?
        #   If size increases, keep it.
        
        # A simpler heuristic:
        # Pick a random y from current_set.
        # Remove it.
        # Try to add elements greedily until stuck.
        # If final size > current size, accept. Else revert.
        
        # Let's do a few random restarts of the "remove and refill" strategy.
        
        candidates_to_remove = list(current_set)
        random.shuffle(candidates_to_remove)
        
        for y in candidates_to_remove:
            if time.time() > deadline:
                break
            
            # Temporarily remove y
            current_set.remove(y)
            
            # Recalculate forbidden mask
            # To be efficient, we could just re-calculate from scratch since N=128 is small.
            # O(|S|^2) is ~ 128^2 = 16k ops. Very fast.
            old_forbidden = forbidden_mask
            forbidden_mask = calc_forbidden_mask(current_set)
            
            # Try to add elements
            local_added = 0
            for x in range(1, N + 1):
                if x in current_set:
                    continue
                if (forbidden_mask >> (x - 1)) & 1:
                    continue
                
                # Add x
                current_set.add(x)
                for z in current_set:
                    if z == x:
                        continue
                    forbidden_mask |= bad_mask[x - 1][z - 1]
                local_added += 1
                
                # We want to add 2 to break even on the removal of 1.
                # If we add 2, net gain +1.
                # Let's try to add as many as possible.
                
            if len(current_set) > len(best_set):
                best_size = len(current_set)
                best_set = set(current_set)
                improved = True
                # Keep this state and continue loop
                break
            else:
                # Revert
                current_set.add(y)
                forbidden_mask = old_forbidden
                
    return sorted(best_set)
