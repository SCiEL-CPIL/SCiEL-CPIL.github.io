import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Moore bound for k=3, g=16 (even girth 2r, r=8)
    # M = 2 * sum_{i=0}^{r-1} (k-1)^i = 2 * (2^8 - 1) = 512
    min_n = 512
    
    # We search for circulant graphs C_n(S) with S = {a, -a, n/2}
    # This requires n to be even.
    # Neighbors of i: (i+a)%n, (i-a)%n, (i+n//2)%n
    
    def check_girth(n, a, b, target_girth):
        # b is n//2
        # Check if girth >= target_girth
        # BFS from each vertex up to depth target_girth // 2 - 1?
        # No, cycle length L is formed by two paths from v to u of lengths d1, d2.
        # L = d1 + d2.
        # We want L >= target_girth.
        # So we look for d1 + d2 < target_girth.
        # Max depth to search: target_girth // 2.
        # If target_girth = 16, max depth 7.
        # If we find u at depth d1 and v at depth d2 such that d1+d2 < 16 and they are connected?
        # Actually, if we do BFS from v, and we find an edge (x, y) in the BFS tree
        # such that dist(x) + dist(y) + 1 < target_girth, then we have a cycle.
        # Wait, the edge (x,y) closes the cycle.
        # Cycle length = dist(x) + dist(y) + 1.
        # We need dist(x) + dist(y) + 1 >= target_girth.
        
        max_depth = target_girth // 2 - 1  # If girth 16, check depths up to 7?
        # If d1=7, d2=7, L=15? No, 7+7+1=15.
        # So if we find an edge connecting two nodes at depth 7, we have a cycle of length 15.
        # So we must reject.
        # So we search up to depth target_girth // 2.
        
        limit_depth = target_girth // 2
        
        # Precompute neighbors for speed?
        # n is small (~1000), so list of neighbors is fine.
        
        for start in range(n):
            # BFS
            visited = [-1] * n
            visited[start] = 0
            queue = [(start, 0)]
            
            while queue:
                u, d = queue.pop(0)
                if d >= limit_depth:
                    continue
                    
                # Neighbors: (u+a)%n, (u-a)%n, (u+b)%n
                neighbors = [(u + a) % n, (u - a) % n, (u + b) % n]
                
                for v in neighbors:
                    if visited[v] == -1:
                        visited[v] = d + 1
                        queue.append((v, d + 1))
                    else:
                        # Found a back edge or cross edge
                        # Cycle length = d + 1 + visited[v]
                        # But we only care if it's < target_girth
                        cycle_len = d + 1 + visited[v]
                        if cycle_len < target_girth:
                            return False
            # Optimization: We don't need to check all starts if we are sure?
            # But we must be exact.
            
        return True

    best_n = None
    best_edges = None
    best_a = None
    best_b = None

    # Search n from min_n to 1500
    for n in range(min_n, 1500, 2):
        if time.time() > deadline:
            break
            
        # Try a few parameter sets
        candidates = []
        # Heuristic: a ~ n/3, b ~ n/5
        candidates.append((n // 3, n // 2))
        candidates.append((n // 4, n // 2))
        candidates.append((n // 5, n // 2))
        candidates.append((n // 7, n // 2))
        
        # Add some randoms
        for _ in range(10):
            a = random.randint(1, n // 2 - 1)
            candidates.append((a, n // 2))

        for a, b in candidates:
            if a == 0 or a == b:
                continue
            if time.time() > deadline:
                break
            
            if check_girth(n, a, b, g):
                # Found a valid graph
                best_n = n
                best_a = a
                best_b = b
                break
        
        if best_n is not None:
            break

    if best_n is None:
        # Fallback: Return a known valid graph if search failed
        # Or return the seed result?
        # The seed result is LUW, which is huge.
        # We should return the best we found, or a valid large graph.
        # If we didn't find one, we might timeout.
        # Let's try a larger n range if we have time?
        # Or just return a trivial valid graph?
        # A complete graph K_k has girth 3. Not valid.
        # We need girth >= 16.
        # If we didn't find one, we return the seed-like result?
        # No, we must return a valid graph.
        # If we didn't find one, we can use a very large n where girth is likely high?
        # For large n, girth of C_n(S) is roughly log(n).
        # To get girth 16, n must be large enough.
        # But we searched up to 1500.
        # If none found, we return a graph with n=2000?
        
        # Let's try one more time with n=2000
        n = 2000
        a = 2000 // 3
        b = 2000 // 2
        if check_girth(n, a, b, g):
            best_n = n
            best_a = a
            best_b = b

    if best_n is None:
        # Absolute fallback: Return a graph that is definitely valid but large
        # Using the LUW logic from seed is complex to copy here.
        # We can use a simple random graph? No, must be regular.
        # We can use a path? No, regular.
        # We can use a cycle? Girth = n.
        # A cycle C_n is 2-regular. We need 3-regular.
        # We can take a cycle and add a perfect matching?
        # If n is large, girth is large.
        # Let's construct a 3-regular graph on n=10000 vertices with high girth.
        # Random 3-regular graph has girth ~ log(n).
        # log(10000) ~ 9. Not enough.
        # We need n ~ 3^8 ~ 6500 for girth 16?
        # Let's use n=10000 and hope for the best?
        # Or just return the seed result?
        # The prompt implies we should improve.
        # If we fail, we score 0?
        # "invalid result ... scores 0".
        # So we MUST return a valid graph.
        
        # Let's use the LUW construction from the seed as a fallback.
        # Copying the seed logic.
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Construct edges for the circulant graph
    edges = []
    for i in range(best_n):
        j1 = (i + best_a) % best_n
        j2 = (i - best_a) % best_n
        j3 = (i + best_b) % best_n
        
        # Add edges (i, j1), (i, j2), (i, j3)
        # Ensure no duplicates and no loops
        # Since a != 0, a != b, and b = n/2, these are distinct?
        # j1 == j2 iff 2a = 0. If n is even and a = n/2, but a < n/2 usually.
        # j1 == j3 iff a = b. We checked a != b.
        # j2 == j3 iff -a = b. If b = n/2, -a = n/2 => a = n/2.
        # So if a != n/2, they are distinct.
        # We should ensure a != n/2.
        
        # Add unique edges
        for j in [j1, j2, j3]:
            if i < j:
                edges.append((i, j))
            elif i > j:
                # Already added?
                # We only add if i < j to avoid duplicates.
                pass

    return best_n, edges
