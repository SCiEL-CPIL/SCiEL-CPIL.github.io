import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 256
    
    def is_valid_set(S):
        """Check if set S satisfies the property."""
        S_list = sorted(S)
        for i, a in enumerate(S_list):
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if a < b <= c and (b + c) % a == 0:
                        return False
        return True
    
    def can_add(a, S):
        """Check if element a can be added to set S without violating property."""
        S_list = sorted(S)
        for b in S_list:
            if a < b:
                for c in S_list:
                    if b <= c and (b + c) % a == 0:
                        return False
        return True
    
    # Strategy 1: Greedy from above (add largest first)
    def greedy_from_above():
        S = set()
        for a in range(n, 0, -1):
            if can_add(a, S):
                S.add(a)
        return S
    
    # Strategy 2: Greedy from below (add smallest first)
    def greedy_from_below():
        S = set()
        for a in range(1, n + 1):
            if can_add(a, S):
                S.add(a)
        return S
    
    # Strategy 3: Start with upper tail, then try to add smaller elements
    def greedy_with_tail(base_start):
        S = set(range(base_start, n + 1))
        # Try to add elements from below
        for a in range(1, base_start):
            if can_add(a, S):
                S.add(a)
        return S
    
    # Local search: try to swap elements to improve
    def local_search(S, max_iters=1000):
        S = set(S)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            S_list = sorted(S)
            for i, a in enumerate(S_list):
                if time.time() > deadline:
                    return S
                for b in range(1, n + 1):
                    if b in S:
                        continue
                    # Try removing a and adding b
                    new_S = S - {a}
                    if can_add(b, new_S):
                        new_S.add(b)
                        # Check if this improves (same size, but maybe enables more additions)
                        # For now, just accept if valid
                        # Actually, we want to maximize cardinality, so swapping same-size doesn't help directly
                        # But it might enable future additions. Let's do a quick check.
                        pass
        return S
    
    # More aggressive local search: remove one, try to add multiple
    def aggressive_local_search(S, max_iters=500):
        S = set(S)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            S_list = sorted(S)
            for a in S_list:
                if time.time() > deadline:
                    return S
                # Try removing a
                new_S = S - {a}
                # Try to add elements that were previously blocked
                for b in range(1, n + 1):
                    if b in S:
                        continue
                    if can_add(b, new_S):
                        new_S.add(b)
                        if len(new_S) > len(S):
                            S = new_S
                            improved = True
                            break
        return S
    
    # Collect candidate sets
    candidates = []
    
    # Base strategies
    candidates.append(greedy_from_above())
    candidates.append(greferred_from_below() if False else greedy_from_below())
    
    # Various tail starts
    for base in [128, 144, 160, 171, 192, 200, 210, 220, 230, 240]:
        candidates.append(greedy_with_tail(base))
    
    # Find best candidate
    best_set = max(candidates, key=len)
    
    # Apply aggressive local search
    best_set = aggressive_local_search(best_set)
    
    # Try local search from multiple starting points
    for S in candidates:
        if time.time() > deadline:
            break
        if len(S) >= len(best_set):
            improved = aggressive_local_search(S, max_iters=200)
            if len(improved) > len(best_set):
                best_set = improved
    
    # Final validation
    if not is_valid_set(best_set):
        # Fallback to simple upper tail
        best_set = set(range(192, n + 1))
    
    return sorted(best_set)
