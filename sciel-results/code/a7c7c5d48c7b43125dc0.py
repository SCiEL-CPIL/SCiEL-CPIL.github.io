import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Calculate Moore Bound
    # n >= k * sum_{i=0}^{(g-3)/2} (k-1)^i
    # For odd g
    moore_bound = 0
    term = 1
    for i in range((g - 1) // 2):
        moore_bound += term
        term *= (k - 1)
    moore_bound *= k
    
    # For k=5, g=9, moore_bound = 425
    # n must be even for k=5 circulant
    if moore_bound % 2 != 0:
        moore_bound += 1
        
    # Search for Circulant Graph
    # k=5 => S = {n/2, s1, -s1, s2, -s2}
    # n must be even
    
    max_n_search = 3000
    if k != 5 or g != 9:
        # For other parameters, just return LUW or simple construction
        # But task specifies k=5, g=9
        pass
        
    # Optimized Girth Check for Circulant Graph
    def check_girth_circulant(n, steps, girth_limit):
        # BFS from 0
        # steps is a list of 5 steps
        dist = [-1] * n
        dist[0] = 0
        queue = [0]
        head = 0
        min_cycle = float('inf')
        
        while head < len(queue):
            u = queue[head]
            head += 1
            du = dist[u]
            
            if du * 2 + 1 >= min_cycle:
                # Optimization: if current depth is too large, we might stop?
                # No, we need to find ANY cycle < girth_limit
                # If du >= girth_limit, we can stop?
                # If du >= (girth_limit-1)//2, any cycle found from here will be >= 2*du+1?
                # Not necessarily, but likely.
                pass
                
            for s in steps:
                v = (u + s) % n
                if dist[v] == -1:
                    dist[v] = du + 1
                    queue.append(v)
                else:
                    # Found a cycle
                    # If v is parent of u, skip? 
                    # In BFS, if dist[v] == du - 1, it's the tree edge.
                    # But in undirected graph, we traverse both ways.
                    # We should only count cycles if dist[v] <= dist[u]?
                    # Actually, if dist[v] < dist[u], the cycle was already counted when v was processed?
                    # Let's just take min(dist[u], dist[v])
                    cycle_len = du + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                    if min_cycle < girth_limit:
                        return False
        
        return min_cycle >= girth_limit

    # Try Circulants
    # To save time, we sample n and steps
    # We prioritize smaller n
    
    # Deterministic set of "good" steps first?
    # Random is fine.
    
    # Limit attempts per n
    attempts_per_n = 100
    
    for n in range(moore_bound, max_n_search, 2):
        if time.time() > deadline:
            break
            
        # Try a few specific structured steps first?
        # e.g. s1 = n//5, s2 = n//3
        candidates = []
        # Structured
        candidates.append((n//5, n//3))
        candidates.append((n//4, n//3))
        candidates.append((n//6, n//4))
        
        # Random
        for _ in range(attempts_per_n):
            s1 = random.randint(1, n//2 - 1)
            s2 = random.randint(1, n//2 - 1)
            if s1 != s2:
                candidates.append((s1, s2))
                
        for s1, s2 in candidates:
            if time.time() > deadline:
                break
            if s1 == s2: continue
            if s1 == 0 or s2 == 0: continue
            if s1 == n//2 or s2 == n//2: continue
            
            steps = [n//2, s1, -s1, s2, -s2]
            # Ensure no duplicates in steps (mod n)
            # n//2 is unique. s1, -s1 unique if s1 != 0, n/2.
            # s2, -s2 unique if s2 != 0, n/2.
            # s1 != s2, s1 != -s2 (mod n) => 2*s1 != n and 2*s2 != n (already checked)
            # s1 != -s2 => s1 + s2 != 0 mod n => s1 + s2 != n
            
            if s1 + s2 == n: continue
            if s1 == s2: continue
            
            if check_girth_circulant(n, steps, g):
                edges = []
                for i in range(n):
                    for s in steps:
                        j = (i + s) % n
                        if i < j:
                            edges.append((i, j))
                return n, edges

    # Fallback: LUW Graph
    # Re-implement LUW for k=5, g=9
    # q = 5, d = 4
    return construct_luw(5, 9)

def construct_luw(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # For g=9, d=4
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
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
