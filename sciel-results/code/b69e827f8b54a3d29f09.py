import time
import random

def construct():
    """
    Returns a list of up to 48 distinct positive integers <= 512 such that no element 
    is the exact arithmetic mean of any subset of at least two other elements.
    
    Strategy:
    1. Use a fast pair-check as a filter.
    2. Use a global validity check with bitset DP for full verification.
    3. Multi-restart greedy with diverse orderings.
    4. Local search (add/swap) to maximize cardinality.
    """
    deadline = time.time() + 85  # Safety margin for 90s limit
    N = 512
    target_size = 48

    def is_pair_valid(S, x):
        """Check if x creates any pair-mean violation with S."""
        S_set = set(S)
        # x = (a+b)/2 => a+b = 2x
        for a in S:
            b = 2*x - a
            if b in S_set and b != a:
                return False
        # y = (a+x)/2 => a = 2y - x
        for y in S:
            a = 2*y - x
            if a in S_set and a != y:
                return False
        return True

    def is_valid_global(S):
        """Full validity check using bitset DP for each element."""
        m = len(S)
        if m < 3:
            return True
        
        S_sorted = sorted(S)
        max_k = m - 1
        max_sum = 512 * 48  # 24576
        
        for idx, a in enumerate(S_sorted):
            # Build DP for S without a
            dp = [0] * (max_k + 1)
            dp[0] = 1
            
            count = 0
            for v in S_sorted:
                if v == a and count == 0:
                    count = 1
                    continue
                
                for k in range(min(count, max_k - 1), -1, -1):
                    if dp[k]:
                        dp[k + 1] |= dp[k] << v
                count += 1
            
            for k in range(2, max_k + 1):
                target_sum = a * k
                if target_sum > max_sum:
                    break
                if (dp[k] >> target_sum) & 1:
                    return False
        
        return True

    def greedy_build(order, use_pair_filter=True):
        """Build a set greedily following the given order."""
        current_set = []
        current_set_set = set()
        
        for c in order:
            if len(current_set) >= target_size:
                break
            if c in current_set_set:
                continue
            
            if use_pair_filter and not is_pair_valid(current_set, c):
                continue
            
            test_set = current_set + [c]
            if is_valid_global(test_set):
                current_set.append(c)
                current_set_set.add(c)
                
        return current_set

    best_set = []
    best_len = 0
    rng = random.Random(42)

    # Phase 1: Deterministic orderings
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        list(range(1, N + 1, 2)),
        list(range(2, N + 1, 2)),
        list(range(N, 0, -2)),
        list(range(2, N + 1, 4)),
        list(range(1, N + 1, 4)),
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        s = greedy_build(order)
        if len(s) > best_len:
            best_len = len(s)
            best_set = s[:]
            if best_len == target_size:
                return best_set

    # Phase 2: Randomized restarts
    for i in range(50):
        if time.time() > deadline:
            break
        p = list(range(1, N + 1))
        rng.shuffle(p)
        s = greedy_build(p)
        if len(s) > best_len:
            best_len = len(s)
            best_set = s[:]
            if best_len == target_size:
                return best_set

    # Phase 3: Local search to improve
    if best_len < target_size:
        for _ in range(200):
            if time.time() > deadline:
                break
            
            current_set = best_set[:]
            current_set_set = set(current_set)
            
            # Try to add random elements
            candidates = [x for x in range(1, N + 1) if x not in current_set_set]
            if not candidates:
                break
            
            for x in rng.sample(candidates, min(len(candidates), 20)):
                if not is_pair_valid(current_set, x):
                    continue
                
                test_set = current_set + [x]
                if is_valid_global(test_set):
                    current_set = test_set
                    current_set_set.add(x)
                    
                    # Try to fill more
                    remaining = [y for y in range(1, N + 1) if y not in current_set_set]
                    rng.shuffle(remaining)
                    for y in remaining:
                        if len(current_set) >= target_size:
                            break
                        if not is_pair_valid(current_set, y):
                            continue
                        test_set2 = current_set + [y]
                        if is_valid_global(test_set2):
                            current_set.append(y)
                            current_set_set.add(y)
                    
                    if len(current_set) > best_len:
                        best_len = len(current_set)
                        best_set = current_set[:]
                        if best_len == target_size:
                            return best_set
                    break
            
            # Try swap
            if current_set and candidates:
                a = rng.choice(current_set)
                x = rng.choice(candidates)
                if a != x:
                    new_set = [v for v in current_set if v != a]
                    new_set.append(x)
                    new_set_set = set(new_set)
                    
                    if is_valid_global(new_set):
                        # Greedy fill
                        remaining = [y for y in range(1, N + 1) if y not in new_set_set]
                        rng.shuffle(remaining)
                        for y in remaining:
                            if len(new_set) >= target_size:
                                break
                            if not is_pair_valid(new_set, y):
                                continue
                            test_set3 = new_set + [y]
                            if is_valid_global(test_set3):
                                new_set.append(y)
                                new_set_set.add(y)
                        
                        if len(new_set) > best_len:
                            best_len = len(new_set)
                            best_set = new_set[:]
                            if best_len == target_size:
                                return best_set

    return best_set
