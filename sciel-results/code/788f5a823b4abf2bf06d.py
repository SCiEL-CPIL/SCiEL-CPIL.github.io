import time
import random
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    For k=3, uses cyclic Cayley graphs Z_N with generators {a, -a, b}.
    Searches for the smallest N and generators that satisfy the girth constraint.
    """
    if k != 3:
        # Fallback to seed logic for other k, though task specifies k=3 context
        # For robustness, if k != 3, we might fail or use a generic search.
        # But the task is specifically about n(3,16).
        # We will implement a generic random regular graph search for k != 3 if needed,
        # but focus on k=3.
        pass

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Moore bound for k=3, g=16 is 512.
    # We search for N starting from 512.
    
    def check_girth(edges, n, target_girth):
        """
        Returns the girth of the graph.
        Uses BFS from each node to find the shortest cycle.
        Optimization: If we find a cycle of length < target_girth, return early.
        """
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # To be efficient, we only need to know if girth >= target_girth.
        # We can run BFS from each node up to depth target_girth // 2 - 1?
        # Actually, a cycle of length L is found if we traverse L/2 steps and meet.
        # Simpler: Just run BFS from each node and find shortest cycle back to start.
        # Since N is small (~1000), this is fast.
        
        min_cycle = float('inf')
        
        for i in range(n):
            # BFS from i
            dist = [-1] * n
            parent = [-1] * n
            dist[i] = 0
            q = deque([i])
            
            while q:
                u = q.popleft()
                if dist[u] > min_cycle // 2: 
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        # Found a cycle
                        # Length is dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                            if min_cycle < target_girth:
                                return min_cycle
            if min_cycle < target_girth:
                return min_cycle
                
        return min_cycle

    def generate_cyclic_graph(N, a, b):
        """
        Generates edges for Z_N with generators a, -a, b.
        Returns (n, edges) or None if invalid (loops/multi-edges).
        """
        # Check for simple graph conditions
        # 1. No loops: a != 0, -a != 0, b != 0 mod N
        # 2. No multi-edges: a != b, a != -b, -a != b, -a != -a (impossible), etc.
        # Generators are a, N-a, b.
        # Pairs of edges from node i:
        # (i, i+a), (i, i-a), (i, i+b)
        
        if a % N == 0 or (N-a) % N == 0 or b % N == 0:
            return None
        if a % N == (N-a) % N: # 2a = 0
            return None
        if b % N == (N-b) % N: # 2b = 0
            return None
        if a % N == b % N:
            return None
        if a % N == (N-b) % N:
            return None
        if (N-a) % N == b % N:
            return None
        if (N-a) % N == (N-b) % N:
            return None

        edges = []
        for i in range(N):
            j1 = (i + a) % N
            j2 = (i - a) % N
            j3 = (i + b) % N
            
            # Add edges, ensuring we don't add duplicates (undirected)
            # We add all 3, then sort and unique later?
            # Or just add (min, max)
            edges.append((i, j1))
            edges.append((i, j2))
            edges.append((i, j3))
            
        # Remove duplicates
        unique_edges = set()
        for u, v in edges:
            if u == v:
                return None # Loop found
            if u > v:
                u, v = v, u
            unique_edges.add((u, v))
            
        # Check if any node has degree != 3
        # In this construction, every node has exactly 3 neighbors unless there are collisions
        # which we checked above. But let's be sure.
        deg = [0] * N
        for u, v in unique_edges:
            deg[u] += 1
            deg[v] += 1
        for d in deg:
            if d != 3:
                return None
                
        return N, list(unique_edges)

    # Search strategy
    # Start N from Moore bound
    # For k=3, g=16, Moore bound is 512.
    # We try N from 512 upwards.
    
    # To speed up, we can pre-filter N.
    # N should be large enough.
    
    max_N = 2000 # Upper limit for search to ensure we finish in time
    if time.time() > deadline:
        # Fallback to seed if search fails immediately?
        # We should ensure we return *something* valid.
        # The seed is valid but large.
        # If we don't find one, we return the seed's result?
        # No, we must implement the seed logic or a valid fallback.
        # Let's just let the loop run.
        pass

    random.seed(42)
    
    for N in range(512, max_N + 1):
        if time.time() > deadline - 10:
            break
            
        # Try random (a, b)
        # a, b should be "small" to keep the graph "local" and increase girth?
        # Actually, for cyclic graphs, the girth is related to the smallest solution to x*a + y*b = 0.
        # Random a, b usually give decent girth.
        
        attempts = 0
        max_attempts = 500 # Limit attempts per N to avoid getting stuck
        
        while attempts < max_attempts and time.time() < deadline - 10:
            attempts += 1
            # Choose a, b from 1 to N-1
            # Ensure a < N/2 to avoid redundancy? Not strictly necessary but good.
            a = random.randint(1, N - 1)
            b = random.randint(1, N - 1)
            
            res = generate_cyclic_graph(N, a, b)
            if res is None:
                continue
                
            n, edges = res
            # Check girth
            # Optimization: Check girth only up to g-1
            # If girth >= g, we are good.
            
            # Fast girth check
            # We can check for cycles of length 3, 4, ... g-1
            # BFS is O(N+E).
            girth = check_girth(edges, n, g)
            
            if girth >= g:
                return n, edges

    # If we didn't find one in the range, we might need to go higher or use a different method.
    # For k=3, g=16, it is known that such graphs exist for N ~ 500-1000.
    # If we fail, we should return a valid graph, even if larger.
    # We can fallback to a known construction or increase N.
    
    # Fallback: Increase N range or use a different generator set.
    # Let's try a few more Ns with more attempts if we have time.
    if time.time() < deadline - 5:
        for N in range(max_N, max_N + 500):
            if time.time() > deadline - 5:
                break
            for _ in range(100):
                a = random.randint(1, N - 1)
                b = random.randint(1, N - 1)
                res = generate_cyclic_graph(N, a, b)
                if res is None: continue
                n, edges = res
                girth = check_girth(edges, n, g)
                if girth >= g:
                    return n, edges

    # Absolute Fallback: Return a simple valid 3-regular graph with large girth?
    # The seed is the safest fallback.
    # We will duplicate the seed logic here to ensure we always return a valid graph.
    
    # Seed Logic: Lazebnik-Ustimenko-Woldar
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
