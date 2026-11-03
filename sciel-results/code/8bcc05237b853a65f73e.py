import time
import random
import math
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses Simulated Annealing to find a small graph, falling back to 
    Lazebnik-Ustimenko-Woldar construction if time runs out.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Moore Bound for k=3, g=16 (even g)
    # n >= 2 * sum((k-1)^i, i=0, g/2 - 1)
    # For k=3, g=16: 2 * (1 + 2 + 4 + 8 + 16 + 32 + 64 + 128) = 2 * 255 = 510.
    # Smallest even n >= 510 is 510? No, 510 is even.
    # But 3-regular graph requires n*k to be even. 3*n even => n even.
    # So n must be even.
    
    # Try a range of n values starting from the lower bound
    # We want to minimize n, so we try smaller n first.
    # However, smaller n is harder to achieve high girth.
    # Let's try n = 510, 512, ..., up to 1000.
    
    target_n_list = []
    # Start from 510, step by 2
    for n in range(510, 1200, 2):
        if n % 2 != 0: continue
        target_n_list.append(n)
        
    # If we have lots of time, we can be more aggressive with search time per n.
    # If not, we should try fewer n values.
    
    best_result = None
    
    # Strategy:
    # 1. Try to find a graph with girth >= g for increasing n.
    # 2. If found, return immediately.
    # 3. If time runs out, return the best valid graph found so far (largest girth for smallest n? No, must be valid girth >= g).
    #    If no valid graph found, fallback to LUW.
    
    # We will allocate time slots to each n.
    # Simpler: Just loop through n, and for each n, run SA until time is up or success.
    # But if n=510 fails, we move to n=512.
    
    # Let's reserve 10 seconds for the final LUW fallback generation if needed.
    # So we have 100 seconds for SA.
    
    # To ensure we don't spend all time on n=510, we limit time per n.
    # E.g., 5 seconds per n.
    # 100s / 5s = 20 attempts.
    # n=510 to n=548.
    
    # Actually, finding girth 16 on n=510 is extremely difficult.
    # The known smallest is 936.
    # It is highly likely we will fail for n < 900.
    # So we should jump to n=900 or 936 directly?
    # Or start at 510 but give it very little time?
    
    # Let's try n = 936 first, as it's a known achievable size.
    # If we can find a graph of size 936, that's a great score.
    # If we can find one of size 900, even better.
    
    # Let's prioritize n values close to the known best (936) but slightly smaller.
    # Try 900, 902, ..., 936, 940...
    # And also try smaller ones if time permits?
    # Given the difficulty, let's focus on the 900-1000 range.
    
    candidate_ns = list(range(900, 1000, 2))
    # Add some smaller ones at the end if we have time?
    # No, let's stick to the promising range.
    
    # Shuffle to avoid bias? No, we want smallest n.
    # So we try 900, 902, ...
    
    # Time allocation:
    # We have ~100s.
    # Let's give each n a small budget, say 2s.
    # 50 candidates * 2s = 100s.
    
    for n in candidate_ns:
        if time.time() > deadline - 15: # Reserve 15s for fallback
            break
        
        # Run SA for this n
        result = search_for_graph(n, k, g, time_budget=2.0)
        if result is not None:
            return result
            
    # If we reach here, we didn't find a graph in the 900-1000 range.
    # Try a few smaller ones? Or just fallback?
    # Let's try a couple of smaller ones quickly.
    for n in [512, 600, 700, 800]:
        if time.time() > deadline - 10:
            break
        result = search_for_graph(n, k, g, time_budget=1.0)
        if result is not None:
            return result
            
    # Fallback to LUW
    return luw_construct(k, g)


def luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    """
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


def generate_random_3_regular(n):
    """
    Generates a random 3-regular graph on n vertices using configuration model.
    Returns adjacency list (list of sets) or None if failed.
    """
    if n % 2 != 0:
        return None
    if n < 4:
        return None
        
    stubs = [i for i in range(n) for _ in range(3)]
    random.shuffle(stubs)
    
    edges = set()
    adj = [set() for _ in range(n)]
    
    # Pair up stubs
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            # Loop, retry pairing
            # Simple retry: swap with a random other stub
            j = random.randint(0, len(stubs)-1)
            if j == i: j = (j+1) % len(stubs)
            stubs[i], stubs[j] = stubs[j], stubs[i]
            v = stubs[i+1]
            if u == v:
                # Still loop, give up and restart? Or just accept invalid?
                # Better to restart generation
                return generate_random_3_regular(n)
        if (u, v) in edges or (v, u) in edges:
            # Multi-edge, retry
            j = random.randint(0, len(stubs)-1)
            if j in [i, i+1]: j = (j+1) % len(stubs)
            stubs[i+1], stubs[j] = stubs[j], stubs[i+1]
            v = stubs[i+1]
            if (u, v) in edges or (v, u) in edges or u == v:
                return generate_random_3_regular(n)
                
        edges.add((min(u, v), max(u, v)))
        adj[u].add(v)
        adj[v].add(u)
        
    # Verify degrees
    for i in range(n):
        if len(adj[i]) != 3:
            return generate_random_3_regular(n)
            
    return adj


def compute_girth(adj, n, max_depth=7):
    """
    Computes the girth of the graph.
    Returns the length of the shortest cycle, or max_depth*2+1 if none found up to that size.
    """
    min_girth = max_depth * 2 + 2 # e.g., 16 for depth 7
    
    for u in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[u] = 0
        q = deque([(u, 0)])
        
        while q:
            c, d = q.popleft()
            if d == max_depth:
                continue
                
            for nb in adj[c]:
                if dist[nb] == -1:
                    dist[nb] = d + 1
                    parent[nb] = c
                    q.append((nb, d + 1))
                elif nb != parent[c]:
                    # Found a cycle
                    # The cycle length is dist[c] + dist[nb] + 1
                    # But this is only valid if the path from u to nb and u to c are disjoint except u?
                    # In BFS tree, if we find a cross edge (c, nb), the cycle is dist[c] + dist[nb] + 1.
                    # This holds if the LCA is u? Not necessarily.
                    # However, for girth calculation, this is the standard method.
                    cycle_len = dist[c] + dist[nb] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth <= 4:
                            return min_girth
        if min_girth <= 4:
            return min_girth
            
    return min_girth


def search_for_graph(n, k, g, time_budget):
    """
    Uses Simulated Annealing to find a k-regular graph with girth >= g.
    """
    if k != 3:
        return None
        
    # Initialize
    adj = generate_random_3_regular(n)
    if adj is None:
        return None
        
    current_girth = compute_girth(adj, n, max_depth=g//2 - 1)
    if current_girth >= g:
        return n, list(set((min(u,v), max(u,v)) for u in range(n) for v in adj[u]))
        
    best_girth = current_girth
    best_adj = [set(s) for s in adj]
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.99
    
    start_time = time.time()
    moves = 0
    
    while time.time() - start_time < time_budget:
        # Pick random edges to rewire
        # Select a random vertex u, and two of its neighbors v1, v2
        # Select a random vertex x, and two of its neighbors y1, y2
        # Ensure u != x
        
        u = random.randint(0, n-1)
        v1, v2 = random.sample(list(adj[u]), 2)
        
        x = random.randint(0, n-1)
        if x == u:
            continue
            
        # x must not be connected to u, v1, or v2 to avoid multi-edges/loops easily
        # Check if x is in adj[u]
        if x in adj[u]:
            continue
            
        # Pick neighbors of x that are not u, v1, v2
        valid_xs = [nb for nb in adj[x] if nb not in [u, v1, v2]]
        if len(valid_xs) < 2:
            continue
            
        y1, y2 = random.sample(valid_xs, 2)
        
        # Proposed rewiring:
        # Remove (u,v1), (x,y1)
        # Add (u,y1), (x,v1)
        # Check if (u,y1) and (x,v1) are already edges
        if (u, y1) in set(adj[u]) or (x, v1) in set(adj[x]):
            continue
        if (y1, u) in set(adj[y1]) or (v1, x) in set(adj[v1]):
            continue
            
        # Perform rewiring
        adj[u].remove(v1)
        adj[v1].remove(u)
        adj[x].remove(y1)
        adj[y1].remove(x)
        
        adj[u].add(y1)
        adj[y1].add(u)
        adj[x].add(v1)
        adj[v1].add(x)
        
        # Compute new girth
        new_girth = compute_girth(adj, n, max_depth=g//2 - 1)
        moves += 1
        
        if new_girth >= g:
            # Success!
            edges = set()
            for i in range(n):
                for j in adj[i]:
                    edges.add((min(i,j), max(i,j)))
            return n, list(edges)
            
        # Acceptance
        if new_girth > best_girth:
            best_girth = new_girth
            best_adj = [set(s) for s in adj]
            
        delta = new_girth - current_girth
        if delta > 0 or random.random() < math.exp(delta / T):
            current_girth = new_girth
        else:
            # Revert
            adj[u].remove(y1)
            adj[y1].remove(u)
            adj[x].remove(v1)
            adj[v1].remove(x)
            
            adj[u].add(v1)
            adj[v1].add(u)
            adj[x].add(y1)
            adj[y1].add(x)
            
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Restart with higher T?
            T = 1.0
            
    # If we didn't succeed, return None
    # We could return best_adj if it was valid? No, must be girth >= g.
    return None
