"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(k, g):
    deadline = time.time() + 130.0  # safety margin before 150s kill
    random.seed(42)
    
    # Find smallest prime >= k
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    def smallest_prime_ge(n):
        if n <= 2:
            return 2
        p = n if n % 2 != 0 else n + 1
        while not is_prime(p):
            p += 2
        return p
    
    def build_luw(q, g):
        """Build LUW graph CD(d,q) with girth >= g. Returns (n, edges, adjacency)."""
        # Choose minimal d such that girth >= g
        # LUW graph CD(d,q) has girth d+5 if d odd, d+4 if d even
        if g <= 5:
            d = 3
        else:
            d = 3
            while True:
                girth_est = d + 5 if d % 2 == 1 else d + 4
                if girth_est >= g:
                    break
                d += 1
        
        # GF(q) operations
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add_op = lambda x, y: x ^ y
            sub_op = lambda x, y: x ^ y
            mul_op = lambda x, y: table[x][y]
        else:
            add_op = lambda x, y: (x + y) % q
            sub_op = lambda x, y: (x - y) % q
            mul_op = lambda x, y: x * y % q
        
        # Precompute tables
        add_t = [[add_op(i, j) for j in range(q)] for i in range(q)]
        sub_t = [[sub_op(i, j) for j in range(q)] for i in range(q)]
        mul_t = [[mul_op(i, j) for j in range(q)] for i in range(q)]
        
        # Rule
        rule_a = [0] * (d + 1)
        rule_b = [0] * (d + 1)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]
        for c in range(2, d + 1):
            rule_a[c], rule_b[c] = rule[c]
        
        # Build graph via BFS from zero point
        def compute_neighbors_point(x):
            result = []
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = add_t[x[c]][mul_t[y[a]][x[b]]]
                result.append((1, tuple(y)))
            return result
        
        def compute_neighbors_line(x):
            result = []
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = sub_t[x[c]][mul_t[x[a]][y[b]]]
                result.append((0, tuple(y)))
            return result
        
        start = (0, (0,) * (d + 1))
        index = {start: 0}
        order = [start]
        edges = []
        adj = []
        
        for vertex in order:
            if time.time() > deadline:
                break
            side, x = vertex
            if side == 0:
                neighbors = compute_neighbors_point(x)
            else:
                neighbors = compute_neighbors_line(x)
            for other in neighbors:
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                    adj.append(set())
                if side == 0:
                    edges.append((index[vertex], index[other]))
        
        n = len(order)
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        return n, edges, adj
    
    def compute_girth(adj, n, max_girth_check):
        """Compute girth using BFS from each vertex. Returns actual girth."""
        girth = max_girth_check + 1
        for start in range(n):
            if time.time() > deadline:
                return girth
            # BFS tracking (node, parent, depth)
            # Use visited with parent to detect cycles
            visited = {}  # node -> (parent, depth)
            queue = [(start, -1, 0)]
            found = False
            while queue and not found:
                node, parent, depth = queue.pop(0)
                if node in visited:
                    # Found a cycle
                    cycle_len = depth + visited[node][1] + 1
                    if cycle_len < girth:
                        girth = cycle_len
                    if cycle_len <= max_girth_check:
                        found = True
                    continue
                visited[node] = (parent, depth)
                if depth >= max_girth_check:
                    continue
                for nb in adj[node]:
                    if nb != parent:
                        queue.append((nb, node, depth + 1))
        return girth
    
    def check_girth_bf(adj, n, g):
        """Check if graph has girth >= g using non-backtracking BFS."""
        # For each vertex, do BFS up to depth g//2 - 1, check for collisions
        max_depth = g // 2 - 1
        for start in range(n):
            if time.time() > deadline:
                return False
            # BFS: (node, parent, depth)
            # Track visited to detect if two different paths reach same node
            visited = {}  # node -> depth
            queue = [(start, -1, 0)]
            while queue:
                node, parent, depth = queue.pop(0)
                if depth > max_depth:
                    continue
                if node in visited:
                    if visited[node] != depth:
                        # Two different paths, potential cycle
                        cycle_len = visited[node] + depth + 1
                        if cycle_len < g:
                            return False
                    continue
                visited[node] = depth
                for nb in adj[node]:
                    if nb != parent:
                        queue.append((nb, node, depth + 1))
        return True
    
    def local_search_reduce(n, adj, k, g):
        """Try to reduce degree from current to k via edge swaps."""
        # Work with a copy of adjacency
        adj_list = [list(s) for s in adj]
        degree = [len(a) for a in adj_list]
        
        # We need to remove (degree[i] - k) edges from each vertex
        # Use edge swaps: remove edge (u,v) and add edge (u,v') and (v,u')
        # where v' is adjacent to u and u' is adjacent to v
        # This preserves degrees
        
        # First, try to find edges to remove that don't create short cycles
        # Strategy: for each vertex, try removing one edge and adding another
        # to maintain regularity
        
        # Actually, let's use a different approach:
        # Find a k-regular subgraph by removing edges
        # This is NP-hard in general, but for small excess degree, we can try
        
        excess = [degree[i] - k for i in range(n)]
        if max(excess) <= 0:
            return adj_list, True
        
        # For q=5, k=4: excess is 1 for each vertex
        # We need to find a 1-factor (perfect matching) in the graph to remove
        # such that the remaining graph has girth >= g
        
        # Find a perfect matching whose removal preserves girth
        # This is complex. Alternative: use simulated annealing
        
        # Simpler approach: try random edge removals with girth checking
        # For excess=1, we need to find a perfect matching to remove
        
        # Let's try: randomly pick edges to remove, checking girth
        # This is expensive. Instead, let's just try a few random matchings.
        
        # For now, let's use a greedy approach:
        # Find edges whose removal doesn't create short cycles
        # and forms a matching
        
        removed_edges = set()
        matched = [False] * n
        
        # Try multiple random orderings
        for attempt in range(200):
            if time.time() > deadline:
                break
            removed = set()
            matched_local = [False] * n
            success = True
            
            # Random permutation of vertices
            order = list(range(n))
            random.shuffle(order)
            
            for v in order:
                if matched_local[v]:
                    continue
                # Try to find an edge (v, u) to remove
                # such that u is not yet matched
                candidates = [u for u in adj_list[v] if not matched_local[u]]
                random.shuffle(candidates)
                
                found = False
                for u in candidates:
                    # Check if removing (v, u) creates a short cycle
                    # Temporarily remove and check
                    adj_list[v].remove(u)
                    adj_list[u].remove(v)
                    
                    # Quick check: does the graph still have girth >= g?
                    # This is expensive. For now, just check locally.
                    # A cycle of length < g must pass through v or u or nearby.
                    # For g=10, check cycles of length 4, 6, 8 through v or u.
                    
                    # Simplified: just accept for now, verify at end
                    matched_local[v] = True
                    matched_local[u] = True
                    removed.add((min(v,u), max(v,u)))
                    found = True
                    break
                
                if not found:
                    # Restore
                    if candidates:
                        u = candidates[0]
                        # u was not removed from adj_list[v] in this iteration
                        # Actually we need to be careful
                    success = False
                    break
            
            if success and len(removed) == n // 2:
                # Verify girth
                adj_check = [list(s) for s in adj_list]
                if check_girth_bf(adj_check, n, g):
                    # Rebuild
                    adj_final = [set() for _ in range(n)]
                    for i in range(n):
                        for j in adj_list[i]:
                            if j > i:
                                adj_final[i].add(j)
                                adj_final[j].add(i)
                    return [list(s) for s in adj_final], True
        
        # Restore original
        for i in range(n):
            adj_list[i] = list(adj[i])
        return adj_list, False
    
    # Step 1: Build LUW with q = smallest prime >= k
    q = smallest_prime_ge(k)
    
    # If q == k, just use the standard construction
    if q == k:
        n, edges, adj = build_luw(q, g)
        return n, edges
    
    # Step 2: Build LUW with q > k and try to reduce
    n, edges, adj = build_luw(q, g)
    
    # Verify initial girth
    if not check_girth_bf(adj, n, g):
        # Fallback to q=k
        n, edges, adj = build_luw(k, g)
        return n, edges
    
    # Step 3: Try local search to reduce degree
    adj_reduced, success = local_search_reduce(n, adj, k, g)
    
    if success:
        # Build edge list from reduced adjacency
        edges_new = []
        for i in range(n):
            for j in adj_reduced[i]:
                if j > i:
                    edges_new.append((i, j))
        return n, edges_new
    
    # Fallback: return the q-regular graph (invalid for k-regular requirement)
    # Actually, we must return a k-regular graph. Let's fall back to q=k.
    n, edges, adj = build_luw(k, g)
    return n, edges
# EVOLVE-BLOCK-END
