import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    time_limit = 110  # Leave buffer for evaluation overhead
    
    # Parameters for k=5, g=9
    # Best known: 1116 (Exoo et al. 2026)
    # Moore bound: 428
    # We aim for the smallest N >= Moore bound such that 5N is even (always true for k=5)
    # and we can construct a valid graph.
    
    # The seed program returns ~1250. We want to beat that.
    # Let's try to construct graphs of size N.
    # For k=5, N must be even.
    
    # Strategy:
    # 1. Try random regular graph construction with girth checking.
    # 2. Use a faster girth check (BFS up to depth g-2 from each node).
    # 3. If we find a valid graph, return it.
    # 4. Start from a small N (e.g., Moore bound + even adjustment) and increase.
    
    # Moore bound for k=5, g=9:
    # n >= 1 + k * sum_{i=0}^{(g/2)-1} (k-1)^i
    # g=9, so radius = 4.
    # n >= 1 + 5 * (1 + 4 + 16 + 64) = 1 + 5 * 85 = 1 + 425 = 426.
    # Actually, the formula is 1 + k * (1 + (k-1) + ... + (k-1)^{floor(g/2)-1})
    # For odd girth g=2r+1, r=4.
    # n >= 1 + k * sum_{i=0}^{r-1} (k-1)^i = 1 + 5 * (1 + 4 + 16 + 64) = 426.
    # But the bound is 428 in the problem statement? Let's check.
    # Moore bound for (5,9): 1 + 5(1+4+16+64) = 426.
    # Wait, the problem says "best lower bound is 428".
    # Let's try N from 426 upwards (even only).
    
    # However, constructing a girth-9 graph is hard.
    # The known smallest is 1116.
    # It is unlikely we can find a graph much smaller than 1116 with random construction.
    # But we should try a range of N values.
    
    # Let's define a function to check if a graph has girth >= g.
    def check_girth(adj, n, g):
        # BFS from each node up to depth g-2
        # If we reach a neighbor of the start node at depth d < g-1, we have a cycle of length d+1 < g.
        # Actually, if we reach a node v at depth d, and v is a neighbor of start u,
        # then we have a cycle of length d+1.
        # We need d+1 >= g => d >= g-1.
        # So if we reach a neighbor of u at depth d < g-1, it's bad.
        
        for u in range(n):
            # BFS from u
            dist = [-1] * n
            dist[u] = 0
            q = deque([u])
            while q:
                curr = q.popleft()
                d = dist[curr]
                if d >= g - 1:
                    continue
                for v in adj[curr]:
                    if dist[v] == -1:
                        dist[v] = d + 1
                        q.append(v)
                    else:
                        # If v is already visited, and v is a neighbor of u?
                        # No, we need to check if the path creates a short cycle.
                        # Actually, standard BFS finds shortest paths.
                        # If the shortest path from u to v is d, and (u,v) is an edge,
                        # then we have a cycle of length d+1.
                        # We need to check all neighbors of u.
                        # This logic is slightly off for checking girth directly in one BFS per node.
                        # Better: Run BFS from u. For each neighbor v of u, check dist[u][v].
                        # If dist[u][v] < g-1, then there is a cycle of length dist[u][v]+1 < g.
                        # Wait, dist[u][v] is the shortest path from u to v.
                        # If (u,v) is an edge, then the cycle length is dist[u][v] + 1.
                        # We need dist[u][v] + 1 >= g => dist[u][v] >= g-1.
                        # So if any neighbor v of u has dist[u][v] < g-1, girth < g.
                        pass
        
        # Let's re-implement check_girth more carefully.
        # For each node u, run BFS to find shortest distances to all other nodes.
        # For each neighbor v of u, if dist[u][v] < g-1, return False.
        # Note: dist[u][v] for neighbor v is at least 1.
        # If dist[u][v] = 1, that's the edge itself, cycle length 2? No, simple graph, no self loops.
        # Actually, if v is a neighbor, the path u->...->v of length d plus edge (v,u) forms a cycle of length d+1.
        # We need d+1 >= g => d >= g-1.
        
        for u in range(n):
            dist = [-1] * n
            dist[u] = 0
            q = deque([u])
            while q:
                curr = q.popleft()
                d = dist[curr]
                if d >= g - 1:
                    continue
                for v in adj[curr]:
                    if dist[v] == -1:
                        dist[v] = d + 1
                        q.append(v)
            # Check neighbors of u
            for v in adj[u]:
                if dist[v] < g - 1:
                    return False
        return True

    # Function to generate a random k-regular graph on N vertices.
    # We can use the configuration model or greedy.
    # Configuration model: create N*k stubs, pair them randomly.
    # Then check for multi-edges and loops. If valid, check girth.
    
    def generate_random_regular(N, k, seed):
        random.seed(seed)
        stubs = []
        for i in range(N):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        
        adj = [[] for _ in range(N)]
        edges = []
        valid = True
        for i in range(0, N * k, 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].append(v)
            adj[v].append(u)
            edges.append((u, v))
        
        if not valid:
            return None, None
        
        # Check if simple (no multi-edges already checked by 'v in adj[u]' during construction? 
        # Actually, 'v in adj[u]' is O(k). Since k=5, it's fast.
        # But we need to ensure no multi-edges. The check above handles it.
        
        # Now check girth
        if check_girth(adj, N, g):
            return N, edges
        
        return None, None

    # We will try N values starting from a reasonable lower bound.
    # The seed returns ~1250.
    # Let's try N = 1116 (known best), then 1118, 1120, etc.
    # If we fail, we go up.
    # If we find a valid graph, we return it.
    
    # To speed up, we can try multiple seeds for each N.
    
    # Start from N=1116. If we can't find one quickly, try larger.
    # But 1116 is the known minimum. It might be hard to find randomly.
    # Maybe we should start from a larger N where random graphs are more likely to have high girth?
    # No, as N increases, it's easier to avoid short cycles?
    # Actually, for fixed k and g, as N increases, the probability of a random regular graph having girth >= g increases.
    # So starting from a larger N might be easier to find a valid graph quickly.
    # But we want to MINIMIZE N.
    # So we should try small N first.
    
    # However, if N is too small, it's impossible.
    # Let's try N from 1116 to 1300.
    # For each N, try up to 50 seeds.
    # If we find a valid graph, return it.
    # If we run out of time, return the best (smallest N) we found?
    # The task says "n must lie between the Moore bound and max_vertices".
    # It doesn't say we must find the minimum. It says "Score: n, minimized".
    # So we should return the smallest N for which we found a valid graph.
    
    best_n = None
    best_edges = None
    
    # Try N from 1116 to 1400 (step 2)
    # But we need to be careful with time.
    # check_girth is O(N * E) = O(N^2 k) ~ 1116^2 * 5 ~ 6 million ops.
    # In Python, this might take ~1-2 seconds per graph.
    # 50 seeds * 2 seconds = 100 seconds. Too slow for many N.
    
    # We need a faster girth check or fewer seeds.
    # Let's optimize check_girth.
    # We can stop early if we find a short cycle.
    # In check_girth, if we find dist[v] < g-1 for a neighbor v, we return False immediately.
    # This is already done.
    
    # Let's try fewer seeds for small N, and more for larger N?
    # Or just try a few seeds for each N.
    
    # Let's try N = 1116 with 10 seeds.
    # Then N = 1120 with 10 seeds.
    # Etc.
    
    # Actually, the known graph with 1116 vertices is constructed using a specific method (voltage graphs?).
    # Random construction might not find it.
    # But we might find a graph with 1118 or 1120 vertices.
    
    # Let's start from N=1116 and go up.
    
    max_N_to_try = 1500  # Upper limit to avoid infinite loop
    seeds_per_N = 20
    
    current_N = 1116
    if current_N % 2 != 0:
        current_N += 1
        
    while current_N <= max_N_to_try:
        if time.time() - start_time > time_limit:
            break
            
        found = False
        for seed in range(seeds_per_N):
            if time.time() - start_time > time_limit:
                break
            n, edges = generate_random_regular(current_N, k, seed)
            if n is not None:
                best_n = n
                best_edges = edges
                found = True
                break
        
        if found:
            break
            
        current_N += 2
        
    if best_n is not None:
        return best_n, best_edges
        
    # Fallback: use the seed's construction method (voltage graph / Cayley-like)
    # Re-implement the seed's fallback logic which is guaranteed to work.
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
