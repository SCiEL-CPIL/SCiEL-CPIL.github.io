import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill
    
    TARGET = 200
    MASK = (1 << (TARGET + 1)) - 1  # bits 0..200
    TARGET_BIT = 1 << TARGET
    
    def try_order(order):
        """Greedy: add elements in given order if it doesn't create sum=target.
        Returns (selected_list, reachable_bitset)."""
        reachable = 1  # bit 0 set
        selected = []
        for x in order:
            new_sums = (reachable << x) & MASK
            if not (new_sums & TARGET_BIT):
                selected.append(x)
                reachable |= new_sums
        return selected, reachable
    
    def recompute_reach(selected):
        """Recompute reach bitset from a list/set of selected elements."""
        reachable = 1
        for x in selected:
            reachable |= (reachable << x) & MASK
        return reachable
    
    def greedy_add(selected_set, reachable, candidates):
        """Add as many candidates as possible to selected_set.
        Returns (selected_set, reachable)."""
        for x in candidates:
            if x in selected_set:
                continue
            new_sums = (reachable << x) & MASK
            if not (new_sums & TARGET_BIT):
                selected_set.add(x)
                reachable |= new_sums
        return selected_set, reachable
    
    def conflict_count(x, reachable):
        """Count how many reachable sums s would make s+x = 200.
        Actually just check if (200-x) is reachable. For heuristic ordering,
        we can estimate 'danger' by how many subsets of already-selected elements
        sum to 200-x. But for simplicity, use: if 200-x is reachable, conflict=1, else 0.
        Better heuristic: elements closer to 100 have fewer pairs that sum to 200 with small numbers.
        For greedy ordering, prioritize elements where (TARGET-x) is NOT reachable."""
        return 1 if (reachable & (1 << (TARGET - x))) else 0
    
    best_S = set()
    best_size = 0
    best_reach = 1
    
    # Strategy 1: Deterministic orderings (fast, no randomness)
    deterministic_orders = [
        list(range(100, 0, -1)),  # descending
        list(range(1, 101)),      # ascending
        list(range(100, 50, -1)) + list(range(50, 0, -1)),  # large then medium
        list(range(1, 51)) + list(range(100, 50, -1)),      # small then large
        list(range(50, 101)) + list(range(49, 0, -1)),      # middle-out
    ]
    
    # Alternating from ends
    alt_order = []
    lo, hi = 1, 100
    while lo <= hi:
        alt_order.append(hi)
        if lo != hi:
            alt_order.append(lo)
        lo += 1
        hi -= 1
    deterministic_orders.append(alt_order)
    
    for order in deterministic_orders:
        if time.time() > deadline - 5:
            break
        selected, reachable = try_order(order)
        if len(selected) > best_size:
            best_size = len(selected)
            best_S = set(selected)
            best_reach = reachable
    
    # Strategy 2: Start with safe large numbers, then greedily add small
    # Numbers > 100 can't be in [1,100], but numbers 101-200 don't exist.
    # Key: if x > 100, x alone exceeds 100 but we're bounded by [1,100].
    # Actually all numbers are <= 100. The safe set {67,...,100}: any two sum to at most 200,
    # but we need NO subset summing to exactly 200.
    # {100, 100} not possible (each once). {100, 99, ...} - check if any subset sums to 200.
    # 100 + 100 = 200 but we only have one 100. So {67,...,100} might be safe?
    # 100+99=199, 100+98=198... 67+68=135. Max pair: 100+99=199 < 200.
    # Three elements: 100+99+98=297 > 200. So no subset of {67..100} sums to 200? 
    # Wait: 100+99=199, need 1 more. No 1 in set. 100+98=198, need 2. No 2.
    # Actually 67+68+65=200? 67+68=135, +65=200. But 65 not in {67..100}.
    # Let's just verify computationally.
    safe_large = set(range(67, 101))
    safe_large_reach = recompute_reach(safe_large)
    if not (safe_large_reach & TARGET_BIT):
        # Safe! Now try to add smaller numbers
        small_numbers = list(range(1, 67))
        random.shuffle(small_numbers)
        S_temp = set(safe_large)
        reach_temp = safe_large_reach
        S_temp, reach_temp = greedy_add(S_temp, reach_temp, small_numbers)
        if len(S_temp) > best_size:
            best_size = len(S_temp)
            best_S = set(S_temp)
            best_reach = reach_temp
    
    # Strategy 3: Randomized greedy with "fewest conflicts" ordering
    for iteration in range(200):
        if time.time() > deadline - 10:
            break
        
        candidates = list(range(1, 101))
        random.shuffle(candidates)
        
        S_temp = set()
        reach_temp = 1
        
        # Greedy with dynamic conflict-aware ordering
        remaining = list(candidates)
        while remaining:
            # Pick the element with fewest conflicts (i.e., (200-x) not in reach)
            # Among those, pick randomly for diversity
            safe_candidates = [x for x in remaining if not (reach_temp & (1 << (TARGET - x)))]
            if not safe_candidates:
                break
            x = random.choice(safe_candidates)
            remaining.remove(x)
            S_temp.add(x)
            reach_temp |= (reach_temp << x) & MASK
        
        if len(S_temp) > best_size:
            best_size = len(S_temp)
            best_S = set(S_temp)
            best_reach = reach_temp
    
    # Strategy 4: Local search - 1-swap moves
    if time.time() < deadline - 10 and best_S:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            S_list = list(best_S)
            random.shuffle(S_list)
            
            for y in S_list:
                if time.time() > deadline - 5:
                    break
                
                S_temp = set(best_S)
                S_temp.discard(y)
                reach_temp = recompute_reach(S_temp)
                
                # Re-greedy: add all valid remaining elements
                candidates = [x for x in range(1, 101) if x not in S_temp]
                random.shuffle(candidates)
                S_temp, reach_temp = greedy_add(S_temp, reach_temp, candidates)
                
                if len(S_temp) > best_size:
                    best_S = S_temp
                    best_reach = reach_temp
                    best_size = len(best_S)
                    improved = True
                    break
    
    # Final polish: try to add any remaining valid elements
    if best_S:
        best_reach = recompute_reach(best_S)
        remaining = [x for x in range(1, 101) if x not in best_S]
        random.shuffle(remaining)
        best_S, best_reach = greedy_add(best_S, best_reach, remaining)
        best_size = len(best_S)
    
    return list(best_S) if best_S else [1]
