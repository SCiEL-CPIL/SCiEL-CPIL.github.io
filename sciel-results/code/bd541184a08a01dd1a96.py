import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill
    
    # Start with 8 disjoint K4s: n=32, alpha=8, C5-free
    n = 32
    edges = []
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    best_n = n
    best_edges = edges[:]
    
    def check_c5_fast(adj, n):
        """Check if graph has a 5-cycle using path enumeration."""
        # For each pair of vertices (a, d) at distance 2, check if they share
        # a common neighbor that also has a common neighbor with another vertex
        # More efficient: check all 5-tuples with ordering constraint
        for v0 in range(n):
            neighbors_v0 = list(adj[v0])
            for v1 in neighbors_v0:
                if v1 <= v0:
                    continue
                for v2 in adj[v1]:
                    if v2 <= v1 or v2 == v0:
                        continue
                    for v3 in adj[v2]:
                        if v3 <= v2 or v3 == v0 or v3 == v1:
                            continue
                        for v4 in adj[v3]:
                            if v4 == v0 and v4 > v3 and v4 != v1 and v4 != v2:
                                return True
        return False
    
    def max_independent_set_size(adj, n, upper_bound=None):
        """Find maximum independent set size using branch and bound on complement."""
        # Build complement adjacency
        comp_adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if v not in adj[u]:
                    comp_adj[u].add(v)
                    comp_adj[v].add(u)
        
        max_clique = 0
        if upper_bound is not None:
            max_clique = upper_bound  # We only care if it exceeds this
        
        def expand(clique_size, candidates):
            nonlocal max_clique
            if not candidates:
                if clique_size > max_clique:
                    max_clique = clique_size
                return
            # Bound
            if clique_size + len(candidates) <= max_clique:
                return
            # Pick vertex with max degree in candidates
            v = max(candidates, key=lambda x: len(comp_adj[x] & candidates))
            # Include v
            new_candidates = candidates & comp_adj[v]
            expand(clique_size + 1, new_candidates)
            # Exclude v
            candidates.discard(v)
            expand(clique_size, candidates)
            candidates.add(v)
        
        expand(0, set(range(n)))
        return max_clique
    
    def has_independent_set_of_size_k(adj, n, k):
        """Check if there's an independent set of size exactly k."""
        # Equivalent to checking if complement has a clique of size k
        comp_adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if v not in adj[u]:
                    comp_adj[u].add(v)
                    comp_adj[v].add(u)
        
        found = [False]
        
        def search(clique, candidates, depth):
            nonlocal found
            if found[0]:
                return
            if depth == k:
                found[0] = True
                return
            if not candidates:
                return
            if len(candidates) < k - depth:
                return
            # Pick vertex
            v = max(candidates, key=lambda x: len(comp_adj[x] & candidates))
            # Include v
            new_candidates = candidates & comp_adj[v]
            if len(new_candidates) >= k - depth - 1:
                search(clique | {v}, new_candidates, depth + 1)
            if found[0]:
                return
            # Exclude v
            candidates.discard(v)
            search(clique, candidates, depth)
            candidates.add(v)
        
        search(set(), set(range(n)), 0)
        return found[0]
    
    # Try to add vertices 32, 33, 34
    # For each new vertex, try different connection patterns
    
    def try_add_vertex(adj, n, edges, new_v, conn_set, deadline):
        """Try adding a new vertex with given connections. Returns (success, new_adj, new_edges)."""
        if time.time() > deadline - 3:
            return False, None, None
        
        # Check C5-freedom
        adj_temp = [set(s) for s in adj]
        adj_temp.append(set())
        for u in conn_set:
            adj_temp[u].add(new_v)
            adj_temp[new_v].add(u)
        
        if check_c5_fast(adj_temp, n + 1):
            return False, None, None
        
        # Check alpha < 9 (i.e., no independent set of size 9)
        if has_independent_set_of_size_k(adj_temp, n + 1, 9):
            return False, None, None
        
        return True, adj_temp, edges + [[u, new_v] for u in conn_set]
    
    # Generate candidate connection sets for new vertices
    def generate_candidates(adj, n, new_v, num_candidates=50):
        """Generate diverse candidate connection sets."""
        candidates = []
        
        # Pattern 1: Connect to 1 vertex from each K4
        for pattern in range(4):
            conn = set()
            for b in range(8):
                base = b * 4
                conn.add(base + pattern)
            candidates.append(conn)
        
        # Pattern 2: Connect to 2 vertices from each K4
        for i in range(4):
            for j in range(i + 1, 4):
                conn = set()
                for b in range(8):
                    base = b * 4
                    conn.add(base + i)
                    conn.add(base + j)
                candidates.append(conn)
        
        # Pattern 3: Random connections
        for _ in range(num_candidates):
            conn = set()
            for b in range(8):
                base = b * 4
                k = random.randint(1, 3)  # 1-3 vertices from each K4
                chosen = random.sample(range(4), k)
                for c in chosen:
                    conn.add(base + c)
            candidates.append(conn)
        
        # Pattern 4: Connect to all but 1 from each K4 (to reduce alpha)
        for missing in range(4):
            conn = set()
            for b in range(8):
                base = b * 4
                for c in range(4):
                    if c != missing:
                        conn.add(base + c)
            candidates.append(conn)
        
        return candidates
    
    # Try adding vertices one by one
    current_adj = [set(s) for s in adj]
    current_edges = edges[:]
    current_n = n
    
    for new_v in range(32, 35):
        if time.time() > deadline - 5:
            break
        
        candidates = generate_candidates(current_adj, current_n, new_v, num_candidates=100)
        
        # Shuffle for diversity
        random.shuffle(candidates)
        
        success = False
        for conn in candidates:
            if time.time() > deadline - 3:
                break
            
            ok, new_adj, new_edges = try_add_vertex(
                current_adj, current_n, current_edges, new_v, conn, deadline
            )
            
            if ok:
                current_adj = new_adj
                current_edges = new_edges
                current_n = new_v + 1
                best_n = current_n
                best_edges = current_edges[:]
                success = True
                break
        
        if not success:
            # Try more random candidates
            for _ in range(200):
                if time.time() > deadline - 3:
                    break
                
                conn = set()
                for b in range(8):
                    base = b * 4
                    k = random.randint(0, 3)
                    if k > 0:
                        chosen = random.sample(range(4), k)
                        for c in chosen:
                            conn.add(base + c)
                
                ok, new_adj, new_edges = try_add_vertex(
                    current_adj, current_n, current_edges, new_v, conn, deadline
                )
                
                if ok:
                    current_adj = new_adj
                    current_edges = new_edges
                    current_n = new_v + 1
                    best_n = current_n
                    best_edges = current_edges[:]
                    success = True
                    break
        
        if not success:
            break
    
    return {'n': best_n, 'edges': best_edges}
