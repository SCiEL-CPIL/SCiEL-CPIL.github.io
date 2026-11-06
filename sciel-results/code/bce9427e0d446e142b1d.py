import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    def elapsed():
        return time.time() - start_time
    
    def check_c4_free(adj, n):
        """Check if graph is C4-free: no two vertices share 2+ common neighbors"""
        for u in range(n):
            for v in range(u+1, n):
                common = adj[u] & adj[v]
                if len(common) >= 2:
                    return False
        return True
    
    def compute_alpha(adj, n):
        """Compute independence number via branch and bound"""
        # Use a simple but effective approach
        nodes = list(range(n))
        # Order by degree descending
        nodes.sort(key=lambda x: -len(adj[x]))
        
        best = 0
        best_set = []
        
        def branch(remaining, current_set, current_size):
            nonlocal best, best_set
            if current_size + len(remaining) <= best:
                return
            if not remaining:
                if current_size > best:
                    best = current_size
                    best_set = current_set[:]
                return
            
            # Pick vertex with max degree in remaining subgraph
            v = remaining[0]
            
            # Option 1: don't include v
            branch(remaining[1:], current_set, current_size)
            
            # Option 2: include v
            current_set.append(v)
            new_remaining = [u for u in remaining[1:] if u not in adj[v]]
            branch(new_remaining, current_set, current_size + 1)
            current_set.pop()
        
        # For large n, use a heuristic upper bound
        if n > 35:
            # Use greedy coloring as upper bound on alpha
            # Actually, use a simpler bound: alpha <= n / (avg_degree + 1) roughly
            # Use greedy independent set as lower bound
            greedy_set = []
            remaining = set(nodes)
            while remaining:
                # Pick vertex with min degree in remaining
                v = min(remaining, key=lambda x: len(adj[x] & remaining))
                greedy_set.append(v)
                remaining -= adj[v]
                remaining.discard(v)
            best = len(greedy_set)
            best_set = greedy_set
            return best, best_set
        
        branch(nodes, [], 0)
        return best, best_set
    
    def compute_alpha_fast(adj, n, max_nodes=2000000):
        """Faster alpha computation with node cap"""
        nodes = list(range(n))
        nodes.sort(key=lambda x: -len(adj[x]))
        
        best = 0
        best_set = []
        node_count = [0]
        
        def branch(remaining, current_set, current_size):
            nonlocal best, best_set
            node_count[0] += 1
            if node_count[0] > max_nodes:
                return
            if current_size + len(remaining) <= best:
                return
            if not remaining:
                if current_size > best:
                    best = current_size
                    best_set = current_set[:]
                return
            
            v = remaining[0]
            
            # Don't include v
            branch(remaining[1:], current_set, current_size)
            
            # Include v
            current_set.append(v)
            new_remaining = [u for u in remaining[1:] if u not in adj[v]]
            branch(new_remaining, current_set, current_size + 1)
            current_set.pop()
        
        branch(nodes, [], 0)
        return best, best_set
    
    def polarity_graph_pq(q):
        """
        Construct polarity graph of PG(2,q).
        Vertices: 0..q^2+q (points of projective plane)
        Two vertices adjacent if point i is on polar line of point j.
        This is C4-free.
        """
        n = q * q + q + 1
        # Points of PG(2,q): (x, y, 1) for x,y in 0..q-1, plus (1,0,0), (0,1,0), (1,1,0)
        # Actually, standard representation:
        # Affine points: (x, y, 1) for x,y in GF(q)
        # Line at infinity: (1, t, 0) for t in GF(q), plus (0, 1, 0)
        # Total: q^2 + q + 1
        
        # For simplicity, use a known polarity. 
        # Let's use the standard polarity for odd q: (x,y,z) -> (x, y, z) with some form
        # Actually, let's use a simpler approach.
        
        # For q=6, let's just build it computationally.
        # Points: represent as (x, y, z) homogeneous coordinates over GF(q)
        # A point is a 1-dim subspace of GF(q)^3
        
        # Build all points
        points = []
        seen = set()
        for x in range(q):
            for y in range(q):
                p = (x, y, 1)
                points.append(p)
        # Points at infinity: (1, t, 0) for t in 0..q-1, and (0, 1, 0)
        for t in range(q):
            points.append((1, t, 0))
        points.append((0, 1, 0))
        
        n = len(points)
        assert n == q * q + q + 1
        
        # Define a polarity. For simplicity, use the standard one:
        # For point (x, y, z), the polar line has equation: x*X + y*Y + z*Z = 0 (for some form)
        # Actually, a polarity in PG(2,q) maps points to lines.
        # A simple polarity: (x, y, z) -> line x*X + y*Y - z*Z = 0 (for q odd)
        # Or use: (x, y, z) -> line x*X + y*Y + z*(x+y) = 0 ... this is getting complicated.
        
        # Let's use a different approach: use the known fact that for the standard polarity
        # on PG(2,q), the graph is C4-free. Let me just hardcode for small q or use a 
        # simpler construction.
        
        # Alternative: use a block design directly.
        # For a 2-(v, k, lambda) design, the incidence graph is bipartite and C4-free if lambda=1.
        # A projective plane is a 2-(q^2+q+1, q+1, 1) design.
        # The incidence graph has 2*(q^2+q+1) vertices, which is too many.
        
        # Let me use the "point graph" approach: 
        # Actually, the simplest C4-free construction: take a set of points and connect
        # two points if they lie on a common line in a block design with lambda=1.
        # But in a projective plane, every pair of points lies on exactly one line, so
        # the "point graph" would be complete! That's not right.
        
        # The correct construction: the POLARITY graph.
        # In a polarity, each point P has a polar line P*. 
        # Define graph: P ~ Q if P lies on Q* (equivalently Q lies on P*).
        # This graph is C4-free.
        
        # For PG(2,q) with q odd, use the standard polarity:
        # Point (x, y, z) has polar line: x*X + y*Y + z*Z = 0
        # Wait, that's not a polarity, that's just a correlation.
        # A polarity requires that if P is on line l*, then l is on P*.
        
        # Let me use a concrete polarity for PG(2, q):
        # For q odd, the standard polarity is: (x, y, z) -> line x*X + y*Y - z*Z = 0
        # Check: point (x,y,z) is on line a*X + b*Y - c*Z = 0 iff ax + by - cz = 0
        # Line (a,b,c) is on point (x,y,z)* iff... 
        # The polar of line (a,b,c) is point (a, b, -c)
        # Point (x,y,z) is on line (a,b,c)* = (a, b, -c) iff... 
        # Hmm, this is getting confused.
        
        # Let me just use a well-known small case and verify.
        # For q=2: PG(2,2) has 7 points. The polarity graph is the Fano plane complement... 
        # Actually, for q=2, the polarity graph is C7 (7-cycle), which is C4-free. Alpha=3.
        
        # For our purposes, let's just try a random C4-free construction approach.
        
        return None  # Skip this complex construction for now
    
    # Simpler approach: use a structured random construction with C4-avoidance
    # and alpha reduction
    
    def try_construction(n, seed, max_alpha=10):
        """Try to construct a C4-free graph on n vertices with alpha <= max_alpha"""
        if elapsed() > deadline:
            return None
        
        rng = random.Random(seed)
        
        # Start with empty graph
        adj = [set() for _ in range(n)]
        
        # Try to add edges greedily, prioritizing edges that reduce independence
        # First, create a base structure with high minimum degree
        
        # Strategy: start with a random graph, then remove C4s, then add edges to reduce alpha
        # Or: build up edges ensuring C4-freedom at each step
        
        # Better: use a "block" structure
        # Divide vertices into groups, create intra-group edges (triangles/cliques)
        # and inter-group edges carefully
        
        # Approach 1: Start with disjoint cliques, then add inter-clique edges
        # Each clique of size k contributes k vertices and reduces alpha by k-1 compared to independent
        
        # Let's try: m triangles (3m vertices) + some extra vertices
        # With m triangles, alpha = m (one from each triangle) if no inter-triangle edges
        # We need alpha <= 10, so m <= 10 gives 30 vertices
        # To get more vertices, we need to add more and reduce alpha
        
        # Approach 2: Use a "near-complete" structure
        # Start with a random graph at density p, remove C4s, then repair alpha
        
        # Let's try: generate random edges, filter C4s, check alpha
        # For n vertices, C4-free max edges is about n^(3/2)/2
        # So density p ~ n^(-1/2)
        
        p = 1.0 / (n ** 0.5) * 1.5  # slightly above threshold
        
        # Generate candidate edges
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        
        # Remove edges that create C4s
        # Process edges, keep if adding doesn't create a C4
        adj = [set() for _ in range(n)]
        kept_edges = []
        
        for u, v in edges:
            # Check if adding (u,v) creates a C4
            # A C4 is created if there exist a,b such that a~b, a~u, b~v or a~v, b~u
            # i.e., u and v have a common neighbor that also has a common neighbor with the other
            # More precisely: adding (u,v) creates a C4 iff there exist a != b such that
            # a in N(u), b in N(v), a in N(v), b in N(u) -- no that's wrong
            # A C4: u-a-v-b-u. So we need a in N(u)∩N(v) and b in N(u)∩N(v) with a!=b? No.
            # C4: u-a-v-b-u means a adjacent to u and v, b adjacent to u and v, a!=b.
            # So adding (u,v) creates a C4 iff |N(u) ∩ N(v)| >= 1 AND there's another common neighbor?
            # No: if u and v already have a common neighbor w, then u-w-v-?-u requires ? adjacent to both.
            # Actually: adding edge (u,v) creates a C4 iff there exist distinct a,b with 
            # a~u, a~v, b~u, b~v. I.e., |N(u)∩N(v)| >= 2.
            # Wait no. If u and v have a common neighbor w, that gives a triangle u-w-v, not C4.
            # C4: u-a-v-b-u. So a and b are distinct, a~u, a~v, b~u, b~v.
            # So yes, adding (u,v) creates a C4 iff |N(u) ∩ N(v)| >= 2.
            
            common = adj[u] & adj[v]
            if len(common) >= 2:
                continue  # Would create C4
            
            # Also need to check: does adding (u,v) create a C4 through other paths?
            # A C4 containing edge (u,v): u-v-a-b-u where a~v, a~b, b~u, b!=a
            # This requires a in N(v)\{u} and b in N(u)\{v} with a~b and a!=b
            # But this is equivalent to: there exist a in N(v), b in N(u), a!=b, a~b
            # which is the same as |N(u) ∩ N(v)| >= 2 if we consider the symmetric case.
            # Actually no, let me reconsider.
            # After adding (u,v), a C4 exists iff there are 4 distinct vertices forming a 4-cycle.
            # The new edge (u,v) can be part of a C4: u-v-a-b-u
            # This requires: v~a, a~b, b~u, and u,v,a,b distinct.
            # So a in N(v), b in N(u), a~b, a!=b, a!=u, b!=v.
            # This is NOT the same as |N(u)∩N(v)|>=2.
            # 
            # Let me just check: for each a in N(v), for each b in N(u), if a!=b and a~b,
            # then we have a C4. But this is O(deg(u)*deg(v)) per edge, which could be slow.
            #
            # For efficiency, let's use the condition: adding (u,v) is safe iff
            # there do not exist distinct a,b with a~u, a~v, b~u, b~v.
            # This IS |N(u)∩N(v)| >= 2.
            # 
            # Wait, I think I'm confusing myself. Let me think again.
            # 
            # Current graph G. Add edge (u,v). When does G+(u,v) contain a C4?
            # Case 1: G already contains a C4. (We maintain invariant that G is C4-free)
            # Case 2: The C4 uses the new edge (u,v). Then the C4 is u-v-a-b-u or u-v-b-a-u.
            #   This requires: v~a, a~b, b~u with a,b distinct from u,v and from each other.
            #   So a in N(v), b in N(u), a~b, a!=b.
            #   This is NOT simply |N(u)∩N(v)|>=2.
            #
            # Example: u~w, v~w (so w in N(u)∩N(v)). Add (u,v). 
            #   Do we have a C4? u-v-w-?-u. Need ?~v and ?~u and ?~w. 
            #   If no other common neighbors, no C4. Just a triangle u-v-w.
            #
            # So the condition for C4 is: there exist a in N(v), b in N(u), a!=b, a~b.
            #
            # For efficiency, we can check: for each a in N(v), check if a has a neighbor in N(u)\{a}.
            # This is O(deg(v) * deg(a)) in worst case.
            
            # For now, use the simpler (but slightly conservative) check:
            # If |N(u) ∩ N(v)| >= 2, definitely creates C4.
            # But we might miss some C4s. Let's do a more thorough check for small degrees.
            
            if len(common) >= 2:
                continue
            
            # Additional check for C4s not captured by common neighbors
            # A C4 using (u,v): u-v-a-b-u where a~v, a~b, b~u
            # Check: for a in N(v), is there b in N(u) with b!=a and a~b?
            creates_c4 = False
            for a in adj[v]:
                if a == u:
                    continue
                # Check if a has a neighbor in N(u) other than itself
                for b in adj[a]:
                    if b in adj[u] and b != a and b != v:
                        creates_c4 = True
                        break
                if creates_c4:
                    break
            
            if creates_c4:
                continue
            
            # Safe to add
            adj[u].add(v)
            adj[v].add(u)
            kept_edges.append((u, v))
        
        # Check alpha
        alpha, _ = compute_alpha_fast(adj, n, max_nodes=1000000)
        
        if alpha <= max_alpha:
            return kept_edges, alpha
        
        # If alpha too large, try to reduce by adding more edges
        # Find an independent set and add edges within it
        if alpha > max_alpha:
            # Get the independent set
            # Use greedy to find a large independent set
            indep = []
            remaining = set(range(n))
            while remaining:
                v = min(remaining, key=lambda x: len(adj[x] & remaining))
                indep.append(v)
                remaining -= adj[v]
                remaining.discard(v)
            
            # Try to add edges within the independent set to reduce alpha
            for i in range(len(indep)):
                for j in range(i + 1, len(indep)):
                    u, v = indep[i], indep[j]
                    if v in adj[u]:
                        continue
                    # Check if adding (u,v) creates C4
                    common = adj[u] & adj[v]
                    if len(common) >= 2:
                        continue
                    creates_c4 = False
                    for a in adj[v]:
                        if a == u:
                            continue
                        for b in adj[a]:
                            if b in adj[u] and b != a and b != v:
                                creates_c4 = True
                                break
                        if creates_c4:
                            break
                    if creates_c4:
                        continue
                    adj[u].add(v)
                    adj[v].add(u)
                    kept_edges.append((u, v))
            
            # Recheck alpha
            alpha, _ = compute_alpha_fast(adj, n, max_nodes=1000000)
            if alpha <= max_alpha:
                return kept_edges, alpha
        
        return None
    
    # Try various n from high to low
    best_result = {'n': 30, 'edges': []}
    # Seed: 10 disjoint triangles
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        best_result['edges'] += [[a, b], [a, c], [b, c]]
    
    for n in range(43, 30, -1):
        if elapsed() > deadline:
            break
        
        for seed in range(50):
            if elapsed() > deadline:
                break
            
            result = try_construction(n, seed, max_alpha=10)
            if result is not None:
                edges, alpha = result
                best_result = {'n': n, 'edges': [[u, v] for u, v in edges]}
                break  # Found valid graph for this n, try higher n
    
    return best_result
