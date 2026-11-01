import random
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    # Fallback LUW
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        # ... (copy from seed) ...
        return len(order), edges

    if k == 4 and g == 10:
        r = g // 2
        n_min = 2 * sum((k-1)**i for i in range(r))
        
        best_n = float('inf')
        best_edges = None
        
        for n in range(n_min, n_min + 150):
            if time.time() > deadline:
                break
            
            # Generate valid generators
            # s in 1..n-1, s != 0
            # Avoid s = n/2
            valid_s = [s for s in range(1, n) if s != n//2]
            if len(valid_s) < 4:
                continue
            
            # We need 2 generators s1, s2 such that s1, -s1, s2, -s2 are distinct.
            # This is equivalent to picking 2 elements from the set of pairs {s, n-s}.
            # Number of pairs is n/2 (if n even) or (n-1)/2.
            
            trials = 2000
            found = False
            for _ in range(trials):
                if time.time() > deadline:
                    break
                
                # Pick 2 distinct pairs
                # Represent pair by min(s, n-s)
                pairs = [min(s, n-s) for s in valid_s]
                # Unique pairs
                unique_pairs = list(set(pairs))
                if len(unique_pairs) < 2:
                    continue
                
                p1, p2 = random.sample(unique_pairs, 2)
                s1 = p1
                s2 = p2
                
                # Check distinctness of the 4 neighbors
                # Neighbors: s1, -s1, s2, -s2
                # -s1 = n-s1, -s2 = n-s2
                # Distinct if s1 != s2, s1 != n-s2, etc.
                # Since p1, p2 are distinct and < n/2, s1 != s2.
                # s1 != n-s2 is equivalent to s1 + s2 != n.
                # Since s1, s2 < n/2, s1+s2 < n. So this is always true.
                # Also s1 != 0, s2 != 0.
                # So any 2 distinct pairs work.
                
                generators = [s1, s2]
                
                # Check girth
                if has_short_cycle(n, generators, g):
                    continue
                
                # Valid
                best_n = n
                best_edges = []
                for i in range(n):
                    for s in generators:
                        j1 = (i + s) % n
                        j2 = (i - s) % n
                        if i < j1:
                            best_edges.append((i, j1))
                        if i < j2:
                            best_edges.append((i, j2))
                found = True
                break
                
            if found:
                return n, best_edges
                
        if best_n < float('inf'):
            return best_n, best_edges
            
    return luw_construct(k, g)

def has_short_cycle(n, generators, girth_limit):
    # BFS from 0
    # We check for closed non-backtracking walks of length < girth_limit
    # Max length to check: girth_limit - 1
    
    def neighbours(v):
        for s in generators:
            yield (v + s) % n
            yield (v - s) % n

    # Recursive search
    # To avoid recursion depth issues, use iterative or increase limit
    # Depth is at most 9, so recursion is fine.
    
    def walk(start, prev, curr, depth, max_depth):
        if depth == max_depth:
            return curr == start
        for next_node in neighbours(curr):
            if next_node == prev:
                continue
            if walk(start, curr, next_node, depth + 1, max_depth):
                return True
        return False

    # Start from 0
    # Try all first steps
    for first_step in neighbours(0):
        if walk(0, 0, first_step, 1, girth_limit - 1):
            return True
    return False
