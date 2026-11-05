import time
import random

def construct():
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    best_n = 0
    best_adj = None
    best_edges = None

    def _is_c4_free(n, adj):
        """Check if graph is C4-free: no two vertices share more than one common neighbor."""
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                # If common has more than 1 bit set, there's a C4
                if common and (common & (common - 1)):
                    return False
        return True

    def _greedy_independent_set(n, adj, max_size=11):
        """Try to find an independent set of size max_size using greedy randomized approach.
        Returns True if found, False otherwise."""
        # Try multiple random orderings
        for _ in range(50):
            order = list(range(n))
            random.shuffle(order)
            indep = []
            used = 0
            for v in order:
                if used & (1 << v):
                    continue
                # Check if v is non-adjacent to all in indep
                conflict = False
                for u in indep:
                    if adj[v] & (1 << u):
                        conflict = True
                        break
                if not conflict:
                    indep.append(v)
                    used |= (1 << v)
                    if len(indep) >= max_size:
                        return True
        return False

    def _greedy_alpha_upper(n, adj):
        """Estimate independence number using greedy coloring on complement."""
        # Build complement adjacency
        comp = [0] * n
        all_bits = (1 << n) - 1
        for i in range(n):
            comp[i] = all_bits & ~(adj[i] | (1 << i))
        
        # Greedy coloring of complement = clique cover of original
        # The chromatic number of complement >= alpha(original)
        uncolored = (1 << n) - 1
        colors = 0
        while uncolored:
            colors += 1
            color_class = 0
            candidate = uncolored
            while candidate:
                lb = candidate & (-candidate)
                v = lb.bit_length() - 1
                # v can join color_class if no edge to existing members in complement
                if comp[v] & color_class:
                    candidate ^= lb
                    continue
                color_class |= lb
                candidate ^= lb
            if color_class:
                uncolored &= ~color_class
        return colors  # This is an upper bound on alpha

    def _try_add_vertex(n, adj, deadline):
        """Try to add a new vertex connected to existing vertices while maintaining C4-free.
        Returns (new_n, new_adj) or (None, None) if no valid addition found."""
        # Try to add vertex n connected to various subsets of existing vertices
        # For the new vertex to not create a C4:
        # For any two neighbors u, v of the new vertex, u and v must share at most 1 common neighbor
        # But since the new vertex is a common neighbor of u and v, u and v must share at most 0 other common neighbors
        # i.e., u and v must be non-adjacent or share exactly the new vertex as common neighbor
        # Actually: if u and v are both neighbors of new vertex w, then for C4-free:
        # u and v can have at most 1 common neighbor. w is one common neighbor.
        # So u and v must have no OTHER common neighbor, i.e., N(u) ∩ N(v) ⊆ {w}
        # Since w is new, N(u) ∩ N(v) in old graph must be empty (they must be non-adjacent with no common neighbors)
        # Wait, that's too strict. Let me reconsider.
        # C4: a-b-c-d-a. If w is new, connected to u and v.
        # A C4 involving w: w-u-x-v-w where u-x and x-v are edges, and u,v,w,x distinct.
        # So if u and v are both neighbors of w, and there exists x such that x is neighbor of both u and v,
        # then w-u-x-v-w is a C4 (if all four distinct and edges exist).
        # So for the graph to remain C4-free after adding w with neighbors S:
        # For any u, v in S, N(u) ∩ N(v) must be empty (in the old graph).
        # Because if x in N(u) ∩ N(v), then w-u-x-v-w is a C4.
        
        # So we need to find a subset S of vertices where no two vertices in S share a common neighbor.
        # This means S is an independent set in the "common neighbor" graph, or equivalently,
        # the vertices in S form a set where pairwise they have no common neighbors.
        
        # Let's try: pick a random vertex u, then find all vertices v such that N(u) ∩ N(v) = empty.
        # These v can potentially be in S along with u.
        # Then from those, pick v, find w such that N(u) ∩ N(w) = empty AND N(v) ∩ N(w) = empty.
        # etc.
        
        # Simpler: just try small connections.
        # Option 1: Add vertex with degree 0 (isolated). Always C4-safe.
        # But isolated vertices increase alpha, so we need to be careful.
        
        # Option 2: Add vertex connected to one existing vertex.
        # If w connected only to u, then for any v != u, w and v share no common neighbors (since w has only u as neighbor).
        # And u and any other vertex: adding w as neighbor of u doesn't create C4 unless...
        # Actually if w is connected only to u, then w has degree 1, so it can't be part of any C4.
        # So always safe.
        
        # Let's try adding vertex connected to 1 or 2 vertices.
        random.shuffle(range(n))
        for u in range(n):
            # Try connecting new vertex to just u
            new_adj = adj[:] + [0]
            new_adj[u] |= (1 << n)
            new_adj[n] = (1 << u)
            if _is_c4_free(n + 1, new_adj):
                return n + 1, new_adj
        
        # Try connecting to 2 vertices u, v where N(u) ∩ N(v) = empty
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] & adj[v]) == 0:
                    new_adj = adj[:] + [0]
                    new_adj[u] |= (1 << n)
                    new_adj[v] |= (1 << n)
                    new_adj[n] = (1 << u) | (1 << v)
                    if _is_c4_free(n + 1, new_adj):
                        return n + 1, new_adj
        
        return None, None

    def _local_search(n, adj, deadline):
        """Try to improve the graph by adding vertices via local search."""
        current_n = n
        current_adj = adj[:]
        
        while time.time() < deadline - 5:
            new_n, new_adj = _try_add_vertex(current_n, current_adj, deadline)
            if new_n is None:
                break
            current_n = new_n
            current_adj = new_adj
        
        return current_n, current_adj

    def _evaluate(n, adj):
        """Check if graph is valid (C4-free and alpha < 11)."""
        if not _is_c4_free(n, adj):
            return False
        # Use greedy independent set search to check for independent 11-set
        if _greedy_independent_set(n, adj, max_size=11):
            return False
        return True

    # Strategy: Start with a base graph and grow it
    # Base: polarity graph of PG(2,5) has 31 vertices and is C4-free
    
    def _polarity_graph_q5():
        """Construct polarity graph of PG(2,5)."""
        q = 5
        # Points of PG(2,5): (x:y:z) with not all zero, up to scalar
        pts = []
        # x=1: (1, y, z) for y,z in 0..4 -> 25 points
        for y in range(q):
            for z in range(q):
                pts.append((1, y, z))
        # x=0, y=1: (0, 1, z) for z in 0..4 -> 5 points
        for z in range(q):
            pts.append((0, 1, z))
        # x=0, y=0: (0, 0, 1) -> 1 point
        pts.append((0, 0, 1))
        n = len(pts)  # 31
        adj = [0] * n
        for i in range(n):
            a1, b1, c1 = pts[i]
            for j in range(i + 1, n):
                a2, b2, c2 = pts[j]
                # Orthogonal if a1*a2 + b1*b2 + c1*c2 = 0 mod 5
                if (a1 * a2 + b1 * b2 + c1 * c2) % q == 0:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return n, adj

    # Try multiple random restarts with different initial graphs
    for restart in range(200):
        if time.time() > deadline - 10:
            break
        
        random.seed(42 + restart)
        
        if restart < 100:
            # Start with polarity graph
            n, adj = _polarity_graph_q5()
        else:
            # Start with a smaller random C4-free graph
            n = random.randint(5, 15)
            adj = [0] * n
            # Add random edges, checking C4-free
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.3:
                        # Check if adding edge creates C4
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        if not _is_c4_free(n, adj):
                            adj[i] &= ~(1 << j)
                            adj[j] &= ~(1 << i)
        
        # Try to grow the graph
        n, adj = _local_search(n, adj, deadline - 5)
        
        # Evaluate
        if n > best_n and _evaluate(n, adj):
            best_n = n
            best_adj = adj[:]
            # Build edge list
            edges = []
            for i in range(n):
                m = adj[i]
                while m:
                    lb = m & (-m)
                    j = lb.bit_length() - 1
                    if i < j:
                        edges.append([i, j])
                    m ^= lb
            best_edges = edges
    
    # Fallback: ensure we have at least a valid graph
    if best_n < 30:
        # Use a simple construction: 10 disjoint triangles (n=30, alpha=10)
        best_n = 30
        best_adj = [0] * best_n
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            best_adj[a] |= (1 << b) | (1 << c)
            best_adj[b] |= (1 << a) | (1 << c)
            best_adj[c] |= (1 << a) | (1 << b)
        best_edges = []
        for i in range(best_n):
            m = best_adj[i]
            while m:
                lb = m & (-m)
                j = lb.bit_length() - 1
                if i < j:
                    best_edges.append([i, j])
                m ^= lb
    
    return {'n': best_n, 'edges': best_edges}
