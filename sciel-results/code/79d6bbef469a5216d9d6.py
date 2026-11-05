import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s margin before 90s kill
    
    best_set = []
    best_size = 0
    
    def is_valid(S):
        # Bitset DP - mask to keep only bits 0..200 for speed
        reachable = 1  # bit 0 is set
        for x in S:
            reachable |= (reachable << x)
            reachable &= (1 << 201) - 1  # mask to 201 bits
            if (reachable >> 200) & 1:
                return False
        return True
    
    def check_and_update(current_set):
        nonlocal best_set, best_size
        if len(current_set) > best_size:
            if is_valid(current_set):
                best_set = list(current_set)
                best_size = len(current_set)
    
    # 1. Deterministic Greedies
    # Large to Small
    s = []
    reachable = 1
    for x in range(100, 0, -1):
        new_reach = (reachable | (reachable << x)) & ((1 << 201) - 1)
        if not ((new_reach >> 200) & 1):
            s.append(x)
            reachable = new_reach
    check_and_update(s)
    
    # Small to Large
    s = []
    reachable = 1
    for x in range(1, 101):
        new_reach = (reachable | (reachable << x)) & ((1 << 201) - 1)
        if not ((new_reach >> 200) & 1):
            s.append(x)
            reachable = new_reach
    check_and_update(s)
    
    # 2. Local search with add and swap moves
    def local_search(start_set, max_iters=50000):
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
            # Try to add x
            if is_valid(current | {x}):
                current.add(x)
                excluded.remove(x)
                if len(current) > best_local_size:
                    best_local = list(current)
                    best_local_size = len(current)
                    check_and_update(current)
            else:
                # Try swap: remove y from current, add x
                # Try a few random y's
                for _ in range(3):
                    y = random.choice(list(current))
                    candidate = (current - {y}) | {x}
                    if is_valid(candidate):
                        current = candidate
                        excluded.remove(x)
                        excluded.append(y)
                        break
        
        return best_local
    
    # Run randomized greedy + local search
    for _ in range(50):
        if time.time() > deadline:
            break
        
        order = list(range(1, 101))
        random.shuffle(order)
        s = []
        reachable = 1
        for x in order:
            new_reach = (reachable | (reachable << x)) & ((1 << 201) - 1)
            if not ((new_reach >> 200) & 1):
                s.append(x)
                reachable = new_reach
        check_and_update(s)
        
        # Local search from this starting point
        improved = local_search(s, max_iters=10000)
        check_and_update(improved)
    
    # 3. Final polish: local search from best found
    if best_set:
        for _ in range(20):
            if time.time() > deadline:
                break
            improved = local_search(best_set, max_iters=50000)
            check_and_update(improved)
    
    return best_set
