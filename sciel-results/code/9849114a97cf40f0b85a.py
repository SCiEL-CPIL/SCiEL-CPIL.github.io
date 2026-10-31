def construct(k, g):
    # For k=3, g=16: use a voltage graph construction to lift a small base graph
    # to a 3-regular graph with girth at least 16.
    #
    # Strategy: Use a base graph (small) and assign voltages from a cyclic group Z_m
    # so that the resulting lift has large girth. The lift of a tree-like base with
    # appropriate voltages can produce large-girth regular graphs.
    #
    # A classic approach: take a "bipartite" base with vertices arranged in layers,
    # and use a cyclic group of order m to create the lift. The girth of the lift
    # is related to the shortest non-backtracking walk in the base whose voltage
    # sum is zero mod m.
    #
    # For 3-regular, girth 16: we need the shortest cycle in the lift to be >= 16.
    # The Moore bound for n(3,16) is 2*(3*(3^(7)-1)/(3-1)) = 2*(3*(2187-1)/2) = 2*3*1093 = 6558? 
    # Wait, let me recompute: Moore bound for k-regular girth g:
    #   If g is even: n >= 2 * (k*(k-1)^((g/2)-1) - 1) / (k-2)
    #   For k=3, g=16: g/2 = 8, so (g/2)-1 = 7
    #   n >= 2 * (3*(2^7) - 1) / (3-2) = 2 * (3*128 - 1) = 2 * 383 = 766
    # Hmm, that doesn't match the stated 512 lower bound. Let me recheck.
    # Actually the Moore bound formula for even girth g:
    #   n >= 2 * (k-1)^((g/2)-1) * k / 2 ... let me use the standard one.
    # Standard Moore bound for k-regular graph of girth g (g even):
    #   n >= 2 * sum_{i=0}^{g/2 - 1} (k-1)^i for k=3? No.
    # The correct Moore bound for a k-regular graph with girth g:
    #   If g is odd: n >= (k*(k-1)^((g-1)/2) - (k-1)) / (k-2)
    #   If g is even: n >= 2*(k*(k-1)^((g/2)-1) - 1) / (k-2)
    # For k=3, g=16 (even): (g/2)-1 = 7
    #   n >= 2*(3*(2^7) - 1)/(3-2) = 2*(3*128 - 1)/1 = 2*383 = 766
    # But the problem says lower bound is 512. Maybe I'm using the wrong formula.
    # Regardless, the task says the smallest known is 936.
    #
    # Let's use a different approach: direct construction via random 3-regular graph
    # generation and girth checking, or use a known explicit construction.
    #
    # Given the time constraint (120s), let's try a practical approach:
    # 1. Generate random 3-regular graphs of increasing size near the Moore bound.
    # 2. Check girth by BFS from each vertex.
    # 3. Return the first valid one found.
    #
    # However, this may be too slow. Instead, let's use a voltage graph construction
    # with a carefully chosen base graph.
    
    if k != 3:
        # Fallback to Lazebnik-Ustimenko-Woldar for other k values
        return _luw_construct(k, g)
    
    # For k=3, g=16: try voltage graph construction
    # Base graph: a path or small tree with voltages in Z_m
    
    # Let's try a systematic search over small base graphs and cyclic voltages.
    # The base graph should be a tree or have very few cycles to control girth.
    
    # Approach: Use a "ladder" or "chain" base graph with 3 vertices per layer,
    # and assign voltages so that the lift has large girth.
    
    # Actually, let's try a simpler approach: use the fact that for 3-regular graphs,
    # we can use a circulant-like construction. 
    
    # Let's try to construct using a known method: 
    # Take a cycle of length m, and add chords to make it 3-regular, then check girth.
    
    # Given the complexity, let's use a random approach with girth verification.
    
    def generate_random_3reg(n):
        """Generate a random 3-regular graph on n vertices."""
        if n < 4 or n % 2 == 1:
            return None
        # Use configuration model
        random.seed(42)  # Fixed seed for reproducibility
        stubs = []
        for i in range(n):
            for _ in range(3):
                stubs.append(i)
        random.shuffle(stubs)
        edges = set()
        for i in range(0, n * 3, 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                continue
            if (min(u,v), max(u,v)) in edges:
                continue
            edges.add((min(u,v), max(u,v)))
        # Check if all vertices have degree 3
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(d != 3 for d in deg):
            return None
        return list(edges)
    
    def compute_girth(n, edges):
        """Compute the girth of a graph using BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        min_cycle = float('inf')
        
        for start in range(n):
            # BFS with parent tracking
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        min_cycle = min(min_cycle, cycle_len)
        
        return min_cycle if min_cycle != float('inf') else 0
    
    # Try to find a 3-regular graph with girth >= 16
    # Start near the Moore bound and increase
    
    # Moore bound for k=3, g=16: let's use 512 as the stated lower bound
    # Try sizes: 512, 528, 544, ... up to some limit
    
    # For efficiency, try a few sizes with random restarts
    
    target_girth = g
    min_n = 512
    max_n = 1000
    
    # Try to find a valid graph
    for n in range(min_n, max_n + 1, 8):  # Step by 8 (even numbers)
        if n % 2 == 1:
            continue
        for attempt in range(5):
            edges = generate_random_3reg(n)
            if edges is None:
                continue
            girth = compute_girth(n, edges)
            if girth >= target_girth:
                return n, edges
    
    # If we couldn't find one in the range, fall back to a larger construction
    # Use the Lazebnik-Ustimenko-Woldar construction as a fallback
    return _luw_construct(k, g)
