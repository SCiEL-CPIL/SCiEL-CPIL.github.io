import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin (330s limit, 40s buffer)
    
    def time_ok():
        return time.time() < deadline

    def find_violation_fast(adj, n):
        """
        Fast check for K5 or independent 5-set using bitmasks.
        Returns (type, vertices) or None.
        """
        # Check for K5
        for u in range(n):
            au = adj[u]
            for v in range(u + 1, n):
                if au & (1 << v):
                    common_uv = au & adj[v]
                    # Need 3 more vertices in common_uv, each pair connected
                    w = 0
                    while common_uv:
                        w_bit = common_uv & -common_uv
                        w = w_bit.bit_length() - 1
                        common_uv ^= w_bit
                        if w <= v:
                            continue
                        common_uvw = common_uv & adj[w]
                        x = 0
                        while common_uvw:
                            x_bit = common_uvw & -common_uvw
                            x = x_bit.bit_length() - 1
                            common_uvw ^= x_bit
                            if x <= w:
                                continue
                            common_uvwx = common_uvw & adj[x]
                            if common_uvwx:
                                y_bit = common_uvwx & -common_uvwx
                                y = y_bit.bit_length() - 1
                                return ('clique', [u, v, w, x, y])
        
        # Check for Independent Set of size 5
        # non_adj[i] = all j != i not connected to i
        non_adj = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            non_adj[i] = all_mask & ~(1 << i) & ~adj[i]
        
        for u in range(n):
            nu = non_adj[u]
            for v in range(u + 1, n):
                if nu & (1 << v):
                    common_uv = nu & non_adj[v]
                    w = 0
                    while common_uv:
                        w_bit = common_uv & -common_uv
                        w = w_bit.bit_length() - 1
                        common_uv ^= w_bit
                        if w <= v:
                            continue
                        common_uvw = common_uv & non_adj[w]
                        x = 0
                        while common_uvw:
                            x_bit = common_uvw & -common_uvw
                            x = x_bit.bit_length() - 1
                            common_uvw ^= x_bit
                            if x <= w:
                                continue
                            common_uvwx = common_uvw & non_adj[x]
                            if common_uvwx:
                                y_bit = common_uvwx & -common_uvwx
                                y = y_bit.bit_length() - 1
                                return ('independent', [u, v, w, x, y])
        
        return None

    def get_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append((i, j))
        return edges

    # Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    current_n = q
    best_n = q
    best_adj = adj[:]

    def local_search_fix(adj_list, n, max_time=3.0):
        """
        Perform local search to fix violations in the graph.
        Returns (new_adj, valid) where valid is True if no violations remain.
        """
        start = time.time()
        stuck_count = 0
        last_viol = None
        
        while time.time() - start < max_time and time.time() < deadline - 10:
            viol = find_violation_fast(adj_list, n)
            if viol is None:
                return adj_list, True
            
            vtype, verts = viol
            
            # Check if stuck (same violation repeatedly)
            viol_key = (vtype, tuple(sorted(verts)))
            if last_viol is not None and last_viol == viol_key:
                stuck_count += 1
                if stuck_count > 30:
                    # Try a different random pair from the violating set
                    u, v = random.sample(verts, 2)
                    if vtype == 'clique':
                        adj_list[u] &= ~(1 << v)
                        adj_list[v] &= ~(1 << u)
                    else:
                        adj_list[u] |= (1 << v)
                        adj_list[v] |= (1 << u)
                    stuck_count = 0
                    last_viol = None
                    continue
            else:
                stuck_count = 0
            
            last_viol = viol_key
            
            # Pick a pair to fix
            if vtype == 'clique':
                # Remove an edge from the clique
                # Strategy: try to remove an edge that is "least useful" or random
                u, v = random.sample(verts, 2)
                adj_list[u] &= ~(1 << v)
                adj_list[v] &= ~(1 << u)
            else:
                # Add an edge to break independence
                u, v = random.sample(verts, 2)
                adj_list[u] |= (1 << v)
                adj_list[v] |= (1 << u)
        
        # Final check
        if find_violation_fast(adj_list, n) is None:
            return adj_list, True
        return adj_list, False

    def try_grow_random(adj_current, current_n, target_n, max_time=5.0):
        """
        Try to grow graph from current_n to target_n vertices using random initialization.
        """
        start = time.time()
        
        for attempt in range(50):
            if time.time() - start > max_time or time.time() > deadline - 15:
                break
            
            new_adj = [0] * target_n
            for i in range(current_n):
                new_adj[i] = adj_current[i]
            
            new_idx = target_n - 1
            # Random connection pattern
            for i in range(current_n):
                if random.random() < 0.5:
                    new_adj[i] |= (1 << new_idx)
                    new_adj[new_idx] |= (1 << i)
            
            new_adj, valid = local_search_fix(new_adj, target_n, max_time=2.0)
            if valid:
                return new_adj
        
        return None

    def try_grow_all_edges(adj_current, current_n, target_n, max_time=5.0):
        """
        Try to grow graph from current_n to target_n vertices starting with all edges to new vertex.
        """
        start = time.time()
        
        for attempt in range(20):
            if time.time() - start > max_time or time.time() > deadline - 15:
                break
            
            new_adj = [0] * target_n
            for i in range(current_n):
                new_adj[i] = adj_current[i]
            
            new_idx = target_n - 1
            # Start with all edges to new vertex
            for i in range(current_n):
                new_adj[i] |= (1 << new_idx)
                new_adj[new_idx] |= (1 << i)
            
            new_adj, valid = local_search_fix(new_adj, target_n, max_time=2.0)
            if valid:
                return new_adj
        
        return None

    def try_grow_no_edges(adj_current, current_n, target_n, max_time=5.0):
        """
        Try to grow graph from current_n to target_n vertices starting with no edges to new vertex.
        """
        start = time.time()
        
        for attempt in range(20):
            if time.time() - start > max_time or time.time() > deadline - 15:
                break
            
            new_adj = [0] * target_n
            for i in range(current_n):
                new_adj[i] = adj_current[i]
            
            # new_idx = target_n - 1, already 0
            
            new_adj, valid = local_search_fix(new_adj, target_n, max_time=2.0)
            if valid:
                return new_adj
        
        return None

    def try_grow_optimal(adj_current, current_n, target_n, max_time=5.0):
        """
        Try to grow graph from current_n to target_n vertices using an optimized connection strategy.
        We try to connect the new vertex to a subset of vertices that minimizes the risk of creating
        a K5 or independent set. We do this by checking the neighborhood structure.
        """
        start = time.time()
        
        # For each potential connection pattern, we can estimate the "risk".
        # A simpler heuristic: try to connect to vertices that have low degree or high degree
        # depending on whether we are more likely to form cliques or independent sets.
        
        # Let's try a few specific patterns:
        # 1. Connect to vertices with lowest degree
        # 2. Connect to vertices with highest degree
        # 3. Connect to a random subset of size k
        
        new_idx = target_n - 1
        
        # Calculate degrees
        degrees = [bin(adj_current[i]).count('1') for i in range(current_n)]
        
        # Pattern 1: Connect to lowest degree vertices
        sorted_by_degree = sorted(range(current_n), key=lambda x: degrees[x])
        for k in range(0, current_n + 1, 2):  # Try various sizes
            if time.time() - start > max_time or time.time() > deadline - 15:
                break
            
            new_adj = [0] * target_n
            for i in range(current_n):
                new_adj[i] = adj_current[i]
            
            # Connect to the k lowest degree vertices
            for i in range(min(k, current_n)):
                v = sorted_by_degree[i]
                new_adj[i] |= (1 << new_idx)
                new_adj[new_idx] |= (1 << i)
            
            new_adj, valid = local_search_fix(new_adj, target_n, max_time=1.0)
            if valid:
                return new_adj
        
        # Pattern 2: Connect to highest degree vertices
        sorted_by_degree_desc = sorted(range(current_n), key=lambda x: -degrees[x])
        for k in range(0, current_n + 1, 2):
            if time.time() - start > max_time or time.time() > deadline - 15:
                break
            
            new_adj = [0] * target_n
            for i in range(current_n):
                new_adj[i] = adj_current[i]
            
            # Connect to the k highest degree vertices
            for i in range(min(k, current_n)):
                v = sorted_by_degree_desc[i]
                new_adj[i] |= (1 << new_idx)
                new_adj[new_idx] |= (1 << i)
            
            new_adj, valid = local_search_fix(new_adj, target_n, max_time=1.0)
            if valid:
                return new_adj
        
        return None

    # Try to grow step by step
    for target_n in range(current_n + 1, 47):
        if not time_ok():
            break
        
        success = False
        
        # Try multiple strategies
        strategies = [
            ("random", try_grow_random),
            ("all_edges", try_grow_all_edges),
            ("no_edges", try_grow_no_edges),
            ("optimal", try_grow_optimal)
        ]
        
        # Shuffle strategies to add randomness
        random.shuffle(strategies)
        
        for strategy_name, strategy_func in strategies:
            if not time_ok():
                break
            
            new_adj = strategy_func(adj, current_n, target_n, max_time=5.0)
            if new_adj is not None:
                current_n = target_n
                adj = new_adj
                best_n = current_n
                best_adj = adj[:]
                success = True
                break
        
        if not success:
            # Try one more time with a different approach: 
            # add vertex with all edges, then remove to fix cliques
            if time_ok():
                new_adj = try_grow_all_edges(adj, current_n, target_n, max_time=10.0)
                if new_adj is not None:
                    current_n = target_n
                    adj = new_adj
                    best_n = current_n
                    best_adj = adj[:]
                    success = True
            
            if not success:
                # Try another approach: start with no edges to new vertex
                if time_ok():
                    new_adj = try_grow_no_edges(adj, current_n, target_n, max_time=10.0)
                    if new_adj is not None:
                        current_n = target_n
                        adj = new_adj
                        best_n = current_n
                        best_adj = adj[:]
                        success = True
            
            if not success:
                break
    
    # Final verification
    if find_violation_fast(best_adj, best_n) is not None:
        # Fallback to Paley
        best_n = 37
        best_adj = [0] * 37
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    best_adj[i] |= (1 << j)
                    best_adj[j] |= (1 << i)

    return best_n, get_edges(best_adj, best_n)
