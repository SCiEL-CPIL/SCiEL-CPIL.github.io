import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    n = 256

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def is_valid_set(S):
        """Check if set S satisfies the property."""
        s = sorted(S)
        for i in range(len(s)):
            a = s[i]
            for j in range(i + 1, len(s)):
                b = s[j]
                for k in range(j, len(s)):
                    c = s[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add(S, x):
        """Check if x can be added to set S while maintaining validity."""
        s = S # Assume S is a list, we iterate over it
        
        # Case 1: x is 'a' (smallest). x < b <= c.
        # b, c must be in S and > x
        for b in s:
            if b <= x:
                continue
            for c in s:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is 'b' (middle). a < x <= c.
        # a must be in S and < x
        # c must be in S and >= x
        for a in s:
            if a >= x:
                continue
            for c in s:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is 'c' (largest). a < b <= x.
        # a, b must be in S and < x
        for a in s:
            if a >= x:
                continue
            for b in s:
                if b <= a or b >= x:
                    continue
                if (b + x) % a == 0:
                    return False
        
        return True

    def local_search(S, max_iterations=2000):
        S = list(S)
        S_set = set(S)
        available = [x for x in range(1, n + 1) if x not in S_set]
        
        for _ in range(max_iterations):
            if time_left() < 2.0:
                break
            improved = False
            
            # Try adding a single element
            for i in range(len(available)):
                x = available[i]
                if can_add(S, x):
                    S.append(x)
                    S_set.add(x)
                    available.pop(i)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps: remove one, add one
            if not S:
                break
                
            remove_idx = random.randint(0, len(S) - 1)
            removed = S.pop(remove_idx)
            S_set.discard(removed)
            available.append(removed)
            
            found_add = False
            for i in range(len(available)):
                x = available[i]
                if can_add(S, x):
                    S.append(x)
                    S_set.add(x)
                    available.pop(i)
                    found_add = True
                    break
            
            if not found_add:
                S.insert(remove_idx, removed)
                S_set.add(removed)
                available.remove(removed)
            else:
                # If we found an add, 'removed' is still in available?
                # No, we added 'removed' to available.
                # If we found x, we popped x from available.
                # 'removed' is still in available.
                # We should NOT add 'removed' back to S.
                # So 'removed' remains in available.
                # This is correct.
                pass
                
        return S

    best_set = []
    best_len = 0

    # Strategy 1: Upper Tail + Local Search
    for start in range(120, 135):
        if time_left() < 5.0:
            break
        S = list(range(start, n + 1))
        S = local_search(S, max_iterations=1000)
        if len(S) > best_len:
            best_len = len(S)
            best_set = S

    # Strategy 2: Random Greedy + Local Search
    for _ in range(20):
        if time_left() < 5.0:
            break
        S = []
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        for x in candidates:
            if can_add(S, x):
                S.append(x)
        S = local_search(S, max_iterations=1000)
        if len(S) > best_len:
            best_len = len(S)
            best_set = S

    # Strategy 3: Greedy Small-to-Large + Local Search
    if time_left() > 10.0:
        S = []
        for x in range(1, n + 1):
            if can_add(S, x):
                S.append(x)
        S = local_search(S, max_iterations=2000)
        if len(S) > best_len:
            best_len = len(S)
            best_set = S

    # Strategy 4: Greedy Large-to-Small + Local Search
    if time_left() > 10.0:
        S = []
        for x in range(n, 0, -1):
            if can_add(S, x):
                S.append(x)
        S = local_search(S, max_iterations=2000)
        if len(S) > best_len:
            best_len = len(S)
            best_set = S

    # Final verification
    if best_set and not is_valid_set(best_set):
        best_set = list(range(129, n + 1))

    if not best_set:
        best_set = list(range(129, n + 1))

    return best_set
