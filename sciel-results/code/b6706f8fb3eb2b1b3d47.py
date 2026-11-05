import time
import random

def construct():
    start_time = time.time()
    deadline = 55.0  # 5s margin
    
    best_set = []
    best_size = 0
    
    def is_valid(S):
        # Bitset DP
        reachable = 1  # bit 0 is set
        for x in S:
            reachable |= (reachable << x)
            if (reachable >> 200) & 1:
                return False
        return True

    def check_and_update(current_set):
        nonlocal best_set, best_size
        if len(current_set) > best_size:
            if is_valid(current_set):
                best_set = list(current_set)
                best_size = len(current_set)
                
    def find_witness(S, target):
        # Finds a subset of S summing to target
        # S is a list of ints
        n = len(S)
        if target < 0 or target > sum(S):
            return None
        # dp[j] = list of indices in S used to form sum j
        # We only care about sums up to target
        dp = [None] * (target + 1)
        dp[0] = []
        
        for i, x in enumerate(S):
            # Iterate backwards
            for j in range(target, x - 1, -1):
                if dp[j] is None and dp[j-x] is not None:
                    # dp[j] = dp[j-x] + [i]
                    # To save memory/time, we can store tuples or just indices
                    # Since target is small (200), lists are fine.
                    dp[j] = dp[j-x] + [i]
                    
        if dp[target] is not None:
            return [S[i] for i in dp[target]]
        return None

    def local_search_v2(start_set, max_iters=50000):
        current = set(start_set)
        all_elems = set(range(1, 101))
        excluded = list(all_elems - current)
        best_local = list(current)
        best_local_size = len(current)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            if not excluded:
                break
            
            x = random.choice(excluded)
            new_set = current | {x}
            if is_valid(new_set):
                current = new_set
                excluded.remove(x)
                if len(current) > best_local_size:
                    best_local = list(current)
                    best_local_size = len(current)
                    check_and_update(current)
            else:
                # Try swap
                # Find witness in current for sum 200 - x
                # Note: 200 - x must be >= 0
                if x < 200:
                    witness = find_witness(list(current), 200 - x)
                    if witness:
                        # Try to remove a random element from witness
                        # We can try a few candidates if the first fails
                        # But witness is small usually.
                        # Let's pick a random one.
                        y = random.choice(witness)
                        candidate = (current - {y}) | {x}
                        if is_valid(candidate):
                            current = candidate
                            # Update excluded
                            # y is now out, x is now in
                            excluded.append(y)
                            if x in excluded:
                                excluded.remove(x)
                            
        return best_local

    # 1. Deterministic Greedies
    # Large to Small
    s = []
    reachable = 1
    for x in range(100, 0, -1):
        new_reach = reachable | (reachable << x)
        if not ((new_reach >> 200) & 1):
            s.append(x)
            reachable = new_reach
    check_and_update(s)
    
    # Small to Large
    s = []
    reachable = 1
    for x in range(1, 101):
        new_reach = reachable | (reachable << x)
        if not ((new_reach >> 200) & 1):
            s.append(x)
            reachable = new_reach
    check_and_update(s)
    
    # 2. Local search: start from a solution, try to add back excluded elements
    # Use the new local_search_v2
    
    # Run randomized greedy + local search
    for _ in range(50):
        if time.time() > deadline:
            break
        
        order = list(range(1, 101))
        random.shuffle(order)
        s = []
        reachable = 1
        for x in order:
            new_reach = reachable | (reachable << x)
            if not ((new_reach >> 200) & 1):
                s.append(x)
                reachable = new_reach
        check_and_update(s)
        
        # Local search from this starting point
        improved = local_search_v2(s, max_iters=10000)
        check_and_update(improved)
    
    # 3. Final polish: local search from best found
    if best_set:
        for _ in range(10):
            if time.time() > deadline:
                break
            improved = local_search_v2(best_set, max_iters=50000)
            check_and_update(improved)
    
    return best_set
