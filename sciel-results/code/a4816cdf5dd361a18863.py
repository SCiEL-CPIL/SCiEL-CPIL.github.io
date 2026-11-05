import time
import itertools

def construct(k, g):
    if k != 4:
        return _construct_fallback(k, g)
        
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Try Cyclic Groups Z_n for n < 320
    # Moore bound is 161.
    for n in range(161, 320):
        if time.time() > deadline:
            break
        # Try generators a, b
        # a, b should be coprime to n? Not necessarily, but for 4-regular we need distinct neighbors.
        # Neighbors: a, -a, b, -b.
        # Must be distinct and non-zero.
        # a != 0, -a != 0 => a != n/2 if n even.
        # a != b, a != -b, -a != b, -a != -b.
        for a in range(1, n//2 + 1):
            if 2*a == n: continue
            for b in range(a+1, n//2 + 1):
                if 2*b == n: continue
                if a == b or a == n-b or -a == b or -a == n-b: continue
                # Check girth
                if _check_cyclic_girth(n, a, b, g):
                    edges = []
                    for i in range(n):
                        for s in [a, b]:
                            j = (i + s) % n
                            if i < j:
                                edges.append((i, j))
                    return n, edges
        if time.time() > deadline:
            break

    # 2. Try A6 (360) - likely too slow or yields 360
    # Skip if time is low
    if time.time() < deadline - 5:
        # ... A6 logic ...
        pass

    # 3. Fallback
    return _construct_fallback(4, g)

def _check_cyclic_girth(n, a, b, g):
    # Check for non-backtracking closed walks of length < g
    # Walk: s1, s2, ..., sL. sum(si) = 0 mod n.
    # si in {a, -a, b, -b}.
    # si != -s_{i-1}.
    # L < g.
    S = [a, n-a, b, n-b]
    # Map to indices for fast comparison?
    # Just use values.
    
    for L in range(4, g): # Min girth is 4 for 4-regular? No, 3-regular min 3. 4-regular min 4.
        # Enumerate
        # Use BFS or recursion
        # Depth L-1
        def dfs(step, current_sum, prev):
            if step == L:
                return current_sum % n == 0
            for s in S:
                if s == (n - prev) % n: # Backtracking
                    continue
                if dfs(step + 1, (current_sum + s) % n, s):
                    return True
            return False
        
        for s in S:
            if dfs(1, s % n, s):
                return False
    return True
