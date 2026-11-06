import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    max_n = 512
    max_size = 48

    def can_add(x, S, dp):
        """
        Checks if x can be added to S without violating constraints.
        dp[k] is a bitset of achievable sums using k elements from S.
        Returns True if x can be added, False otherwise.
        """
        n_s = len(S)
        
        # Check 1: x is not the mean of any subset T of S with |T| >= 2
        # sum(T) = x * |T|
        for k in range(2, n_s + 1):
            target_sum = x * k
            if (dp[k] >> target_sum) & 1:
                return False
        
        # Check 2: No y in S becomes the mean of a subset involving x
        # y is mean of T' = T U {x}, T subset S \ {y}, |T| = j >= 1
        # sum(T) + x = y * (j + 1) => sum(T) = y * (j + 1) - x
        # We check if sum(T) is achievable by j elements from S.
        # Conservative: If achievable from S, assume it's achievable from S\{y} (reject).
        # This ensures validity.
        for y in S:
            for j in range(1, n_s + 1):
                target_sum = y * (j + 1) - x
                if target_sum < 0:
                    continue
                if (dp[j] >> target_sum) & 1:
                    return False
                    
        return True

    def greedy_build(order):
        """
        Builds a set using the given order of candidates.
        Returns the set of integers.
        """
        S = []
        dp = [0] * (max_size + 1)
        dp[0] = 1
        
        for x in order:
            if len(S) >= max_size:
                break
            if time.time() > deadline - 5.0:
                break
                
            if can_add(x, S, dp):
                S.append(x)
                # Update DP
                # Iterate backwards to avoid using x multiple times
                for k in range(len(S) - 1, -1, -1):
                    dp[k + 1] |= (dp[k] << x)
                    
        return S

    def try_fill(S):
        """
        Attempts to add more elements to S by scanning all candidates.
        """
        if time.time() > deadline - 5.0:
            return S
            
        S = list(S)
        dp = [0] * (max_size + 1)
        dp[0] = 1
        for x in S:
            for k in range(len(S) - 1, -1, -1):
                dp[k + 1] |= (dp[k] << x)
                
        changed = True
        while changed and len(S) < max_size:
            changed = False
            for x in range(1, max_n + 1):
                if x in S:
                    continue
                if time.time() > deadline - 5.0:
                    break
                if can_add(x, S, dp):
                    S.append(x)
                    for k in range(len(S) - 1, -1, -1):
                        dp[k + 1] |= (dp[k] << x)
                    changed = True
                    break # Restart scan to ensure consistency
        return S

    def local_search(S, max_iters=2000):
        """
        Structural change: Local search with swaps to escape local minima.
        Tries to swap elements in S with elements not in S to find a valid configuration
        that allows adding more elements, or directly finds a valid set of size |S|.
        Since we want to MAXIMIZE size, and we start with a valid set, this function
        primarily tries to find a "better" valid set of the same size that might be easier
        to extend, or if we allow removals/additions, we can try to grow.
        
        However, the current `can_add` is strict. A swap (remove x, add y) keeps size constant.
        If the new set is valid, we keep it if it has "more potential".
        Potential is hard to measure. 
        Instead, let's try a different structural approach:
        Randomized Restart with Conflict-Driven Heuristic.
        
        Let's implement a simple conflict-driven local search:
        1. Start with a valid set S.
        2. Try to add a random element y.
        3. If it fails, identify the conflicting elements.
        4. Try to swap one of the conflicting elements with another random element.
        
        But computing conflicts exactly is expensive.
        
        Alternative Structural Change:
        Use the "conflict score" idea from Inspiration 1.
        Sort candidates by how "central" they are (likely to be means).
        Numbers in the middle of the range [1, 512] are more likely to be means of other numbers.
        Numbers near the edges (1, 2, 511, 512) are less likely.
        So, a greedy that prioritizes edge numbers might build a larger set.
        """
        # We will implement a heuristic ordering based on "centrality"
        # Score(x) = number of pairs (u, v) such that x = (u+v)/2.
        # This is roughly proportional to min(x-1, 512-x).
        
        pass # Placeholder, we will use the heuristic in the main loop

    best_set = []
    
    # 1. Deterministic Greedy with different orderings
    
    # Descending
    order_desc = range(max_n, 0, -1)
    S_desc = greedy_build(order_desc)
    best_set = S_desc
    best_size = len(best_set)
    
    # Ascending
    order_asc = range(1, max_n + 1)
    S_asc = greedy_build(order_asc)
    if len(S_asc) > best_size:
        best_set = S_asc
        best_size = len(S_asc)
        
    # 2. Heuristic: "Edge-First"
    # Numbers far from the center are less likely to be the mean of two other numbers.
    # Center is 256.5. Distance = abs(x - 256.5)
    # Sort by distance descending (edges first)
    order_edge = sorted(range(1, max_n + 1), key=lambda x: -abs(x - 256.5))
    S_edge = greedy_build(order_edge)
    if len(S_edge) > best_size:
        best_set = S_edge
        best_size = len(S_edge)
        
    # 3. Heuristic: "Safe" order based on pre-computed 3AP conflict counts
    # Score(x) = number of y in [1, 512]\{x} such that 2x-y is in [1, 512]\{x,y}
    # This is the number of 3-term APs centered at x.
    # Lower score = safer to include.
    scores = {}
    for x in range(1, max_n + 1):
        count = 0
        # u < x < v, u+v = 2x
        # u ranges from max(1, 2x-512) to x-1
        # v = 2x - u
        # v <= 512 => 2x - u <= 512 => u >= 2x - 512
        # v > x => 2x - u > x => x > u (always true for u < x)
        # u >= 1
        start_u = max(1, 2*x - 512)
        end_u = x - 1
        if start_u <= end_u:
            count = end_u - start_u + 1
        scores[x] = count
        
    # Sort by score ascending (safest first)
    order_safe = sorted(range(1, max_n + 1), key=lambda x: scores[x])
    S_safe = greedy_build(order_safe)
    if len(S_safe) > best_size:
        best_set = S_safe
        best_size = len(S_safe)
        
    # 4. Randomized Greedy with Heuristic Bias
    # Instead of pure random, bias towards "safe" numbers.
    random.seed(42)
    
    # We can run multiple random shuffles, but also try "noisy" heuristic orders
    # Mix of safe and random
    
    while time.time() < deadline - 15.0:
        # Create a biased order:
        # With probability 0.5, pick from the top 50% safest numbers.
        # With probability 0.5, pick from all numbers.
        # This is complex to implement efficiently in a single pass.
        
        # Simpler: Just try more random restarts.
        # But also try the "safe" order with some noise.
        
        # Let's try a few specific heuristic variations if time permits
        if len(best_set) < max_size:
            # Try a "mixed" order: first half safe, second half random
            safe_half = order_safe[:max_n//2]
            rest = [x for x in range(1, max_n+1) if x not in set(safe_half)]
            random.shuffle(rest)
            mixed_order = safe_half + rest
            S_mixed = greedy_build(mixed_order)
            if len(S_mixed) > best_size:
                best_set = S_mixed
                best_size = len(S_mixed)
                if best_size >= max_size:
                    break

        # Standard random restart
        order = list(range(1, max_n + 1))
        random.shuffle(order)
        S_rand = greedy_build(order)
        if len(S_rand) > best_size:
            best_set = S_rand
            best_size = len(best_set)
            if best_size >= max_size:
                break
                
    # 5. Try to fill gaps in the best set found
    if time.time() < deadline - 5.0:
        filled_set = try_fill(best_set)
        if len(filled_set) > best_size:
            best_set = filled_set
            best_size = len(filled_set)
            
    # 6. Local Search: Try to swap elements to allow filling
    # If we are stuck at size K < 48, maybe we can swap an element to make room for 2 others?
    # This is complex. The current `try_fill` is pretty good.
    # Let's add a simple random swap local search to the best set.
    
    if len(best_set) < max_size and time.time() < deadline - 2.0:
        S_current = best_set[:]
        S_set = set(S_current)
        complement = [x for x in range(1, max_n + 1) if x not in S_set]
        
        # We want to see if we can find a valid set of size len(S_current) + 1
        # by swapping one element.
        # Actually, swapping keeps size constant.
        # We need to find a valid set of size K that can be extended to K+1.
        
        # Strategy:
        # 1. Take current best set S (size K).
        # 2. Try to add a random element y from complement.
        # 3. If it fails, it means y is the mean of some subset of S, or some element of S is the mean of a subset involving y.
        # 4. This doesn't directly help with swapping.
        
        # Let's stick to the robust greedy + fill approach which has proven effective.
        # The "Edge-First" and "Safe-First" heuristics are the main structural additions.
        
    return best_set
