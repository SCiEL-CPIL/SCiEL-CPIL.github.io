"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
from itertools import combinations

def construct():
    n = 16
    # Precompute all Q3 embeddings
    # Q3 vertices: 000, 001, 010, 011, 100, 101, 110, 111
    # Edges in Q3: pairs differing in exactly 1 bit
    q3_vertices = [0, 1, 2, 3, 4, 5, 6, 7]
    q3_edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if (i ^ j).bit_count() == 1:
                q3_edges.append((i, j))
    
    # All possible embeddings of Q3 into 16 vertices
    # An embedding is a bijection from {0..7} to {0..15}
    # We need to check if the edge structure matches
    # For efficiency, we'll precompute all valid embeddings
    
    # Actually, let's think differently. 
    # A Q3 subgraph is defined by 8 distinct vertices and 12 specific edges among them.
    # The 8 vertices form a Q3 if and only if there exists a labeling 
    # v0..v7 such that vi and vj are adjacent in G iff (i^j).bit_count() == 1.
    
    # For each set of 8 vertices, we can check if it forms a Q3.
    # But that's C(16,8) = 12870 sets, and checking each is expensive.
    
    # Better approach: enumerate all permutations of 8 vertices from 16.
    # That's 16! / 8! = 518918400 which is too many.
    
    # Alternative: use the fact that Q3 has specific structure.
    # Let's precompute all "Q3 patterns" as edge sets on 8 labeled vertices.
    # Then for each assignment of 8 distinct vertices from 0..15 to labels 0..7,
    # check if all 12 edges are present.
    
    # But 16P8 is still large. Let's think of a smarter way.
    
    # Actually, for the SA approach, we don't need to precompute all embeddings.
    # We can compute violations incrementally.
    
    # Let's use a different strategy:
    # 1. Start with a dense random graph
    # 2. Use SA to minimize the number of Q3 subgraphs
    # 3. The objective is: minimize (# Q3s) + lambda * (max_edges - #edges)
    #    Or better: maximize (#edges) subject to (#Q3s) == 0
    #    Using SA: maximize (#edges - penalty * #Q3s)
    
    adj = [0] * n  # Bitmask adjacency
    
    def has_edge(u, v):
        return (adj[u] >> v) & 1
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def count_edges():
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                if has_edge(u, v):
                    total += 1
        return total
    
    # Count Q3s using a given edge (u,v) - assuming (u,v) is present
    # This counts the number of Q3 subgraphs that include edge (u,v)
    def count_q3s_with_edge(u, v):
        """
        Count Q3 subgraphs that use edge (u,v).
        In Q3, each edge is in exactly 2 Q3 faces (4-cycles), but for the whole cube,
        each edge is in exactly 2 different Q3 embeddings if we fix the edge.
        
        Actually, let's think: In Q3, an edge connects two vertices. 
        The remaining 6 vertices must complete the cube.
        
        Let u and v be adjacent in Q3. WLOG, u=000, v=001.
        The other vertices are: 010, 011, 100, 101, 110, 111.
        
        The structure: 
        - u (000) connects to 010, 001, 100
        - v (001) connects to 000, 011, 101
        - 010 connects to 000, 011, 110
        - 100 connects to 000, 101, 110
        - 011 connects to 001, 010, 111
        - 101 connects to 001, 100, 111
        - 110 connects to 010, 100, 111
        - 111 connects to 011, 101, 110
        
        So given u and v (edge), we need to find:
        - a, b: other neighbors of u (besides v)
        - c, d: other neighbors of v (besides u)
        - e, f: the remaining two vertices
        
        With the constraint that the edge structure matches.
        
        This is complex. Let's use a brute-force check for a given set of 8 vertices.
        """
        # For now, let's use a simpler approach: 
        # Precompute all Q3 embeddings as tuples of 8 vertex indices
        # But this is memory intensive.
        
        # Alternative: use the fact that we can check if adding/removing an edge
        # creates/destroys Q3s by checking all possible "completions".
        
        # Let's implement a direct check:
        # Given that (u,v) is an edge, find all ways to complete to a Q3.
        
        # Neighbors of u (excluding v)
        neigh_u_mask = adj[u] & ~(1 << v)
        neigh_v_mask = adj[v] & ~(1 << u)
        
        neigh_u = []
        mask = neigh_u_mask
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            neigh_u.append(idx)
            mask -= lsb
            
        neigh_v = []
        mask = neigh_v_mask
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            neigh_v.append(idx)
            mask -= lsb
            
        count = 0
        
        # We need to pick 2 neighbors a, b of u and 2 neighbors c, d of v
        # such that the 8 vertices {u, v, a, b, c, d, e, f} form a Q3.
        
        # In Q3, u and v are adjacent. 
        # u has degree 3, v has degree 3.
        # The other 6 vertices have degree 3 each.
        
        # Key insight: In Q3, if u=000 and v=001, then:
        # - The neighbors of u are: 001, 010, 100
        # - The neighbors of v are: 000, 011, 101
        # - 010 and 011 are adjacent
        # - 100 and 101 are adjacent
        # - 010 and 100 are NOT adjacent
        # - 011 and 101 are NOT adjacent
        # - 010 and 101 are NOT adjacent
        # - 011 and 100 are NOT adjacent
        # - 110 is adjacent to 010, 100, 111
        # - 111 is adjacent to 011, 101, 110
        
        # So the structure is determined.
        
        # Let's enumerate:
        for i in range(len(neigh_u)):
            a = neigh_u[i]
            for j in range(i + 1, len(neigh_u)):
                b = neigh_u[j]
                for k in range(len(neigh_v)):
                    c = neigh_v[k]
                    for l in range(k + 1, len(neigh_v)):
                        d = neigh_v[l]
                        
                        # Check: a and c must be adjacent OR a and d must be adjacent
                        # In Q3: a=010, c=011 are adjacent. a=010, d=101 are not.
                        # So exactly one of (a,c) or (a,d) should be an edge?
                        # Actually, in Q3, 010-011 is an edge, 010-101 is not.
                        # Similarly, 100-101 is an edge, 100-011 is not.
                        
                        # So we need: (a,c) and (b,d) are edges, OR (a,d) and (b,c) are edges.
                        # But NOT both, and NOT neither.
                        
                        ac = has_edge(a, c)
                        ad = has_edge(a, d)
                        bc = has_edge(b, c)
                        bd = has_edge(b, d)
                        
                        # Also, a and b are NOT adjacent in Q3 (010-100 dist 2)
                        # c and d are NOT adjacent in Q3 (011-101 dist 2)
                        if has_edge(a, b) or has_edge(c, d):
                            continue
                            
                        if (ac and bd) or (ad and bc):
                            # Now find e and f
                            # e is common neighbor of a and b (but not u, v, c, d)
                            # In Q3: 110 is common neighbor of 010 and 100.
                            common_ab = adj[a] & adj[b]
                            forbidden = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d)
                            candidates_e = common_ab & ~forbidden
                            
                            if not candidates_e:
                                continue
                                
                            for e in [x for x in range(n) if (candidates_e >> x) & 1]:
                                # f is common neighbor of c, d, e
                                # In Q3: 111 is common neighbor of 011, 101, 110.
                                common_cde = adj[c] & adj[d] & adj[e]
                                forbidden_f = forbidden | (1 << e)
                                candidates_f = common_cde & ~forbidden_f
                                
                                if candidates_f:
                                    # Found a Q3
                                    # Check that f is valid
                                    for f in [x for x in range(n) if (candidates_f >> x) & 1]:
                                        # Verify the full structure
                                        # Check all 12 edges
                                        if (has_edge(u, v) and has_edge(u, a) and has_edge(u, b) and
                                            has_edge(v, c) and has_edge(v, d) and
                                            has_edge(a, c) if ac else True and
                                            has_edge(b, d) if bd else True and
                                            has_edge(a, d) if ad else True and
                                            has_edge(b, c) if bc else True and
                                            has_edge(a, e) and has_edge(b, e) and
                                            has_edge(c, f) and has_edge(d, f) and
                                            has_edge(e, f)):
                                            count += 1
                                        
        return count
    
    # Better approach: precompute all Q3 embeddings
    # Since n=16, we can enumerate all 8-subsets and check if they form a Q3.
    # C(16,8) = 12870, which is manageable.
    
    def is_q3(vertices):
        """Check if the 8 vertices form a Q3 subgraph."""
        if len(vertices) != 8:
            return False
        
        # Try all 8! = 40320 permutations? Too many.
        # Better: use the structure of Q3.
        
        # In Q3, each vertex has degree 3 within the subgraph.
        # First check: each vertex must have exactly 3 neighbors in the set.
        vset = set(vertices)
        for v in vertices:
            deg = 0
            mask = adj[v]
            for u in vertices:
                if u != v and (mask >> u) & 1:
                    deg += 1
            if deg != 3:
                return False
        
        # Now check the structure.
        # Pick a vertex, say v0. Its 3 neighbors in the set are v1, v2, v3.
        # In Q3, v1, v2, v3 are pairwise non-adjacent.
        v0 = vertices[0]
        neighbors = [u for u in vertices if u != v0 and has_edge(v0, u)]
        if len(neighbors) != 3:
            return False
        
        # Check that neighbors are pairwise non-adjacent
        for i in range(3):
            for j in range(i + 1, 3):
                if has_edge(neighbors[i], neighbors[j]):
                    return False
        
        # Now, the remaining 4 vertices must form the "opposite" structure.
        # In Q3, if v0=000, neighbors are 001, 010, 100.
        # The other vertices are 011, 101, 110, 111.
        # 011 is adjacent to 001 and 010 (but not 100)
        # 101 is adjacent to 001 and 100 (but not 010)
        # 110 is adjacent to 010 and 100 (but not 001)
        # 111 is adjacent to 011, 101, 110
        
        # Let's assign:
        # n0, n1, n2 = neighbors
        # For each pair of neighbors, find their common neighbor in the remaining set.
        
        remaining = [v for v in vertices if v != v0 and v not in neighbors]
        
        # Find the vertex adjacent to both n0 and n1
        c01 = None
        for v in remaining:
            if has_edge(n0, v) and has_edge(n1, v):
                c01 = v
                break
        if c01 is None:
            return False
        
        # Find the vertex adjacent to both n0 and n2
        c02 = None
        for v in remaining:
            if has_edge(n0, v) and has_edge(n2, v):
                c02 = v
                break
        if c02 is None:
            return False
        
        # Find the vertex adjacent to both n1 and n2
        c12 = None
        for v in remaining:
            if has_edge(n1, v) and has_edge(n2, v):
                c12 = v
                break
        if c12 is None:
            return False
        
        # The 4th vertex should be adjacent to c01, c02, c12
        if len(remaining) != 4:
            return False
        
        c012 = None
        for v in remaining:
            if v not in [c01, c02, c12]:
                c012 = v
                break
        if c012 is None:
            return False
        
        # Check c012 is adjacent to c01, c02, c12
        if not (has_edge(c012, c01) and has_edge(c012, c02) and has_edge(c012, c12)):
            return False
        
        # Check that c01, c02, c12 are pairwise non-adjacent
        if has_edge(c01, c02) or has_edge(c01, c12) or has_edge(c02, c12):
            return False
        
        return True
    
    # Precompute all Q3 embeddings
    q3_embeddings = []
    for combo in combinations(range(n), 8):
        if is_q3(combo):
            q3_embeddings.append(combo)
    
    # Now use SA to maximize edges while minimizing Q3s
    random.seed(42)
    
    # Start with a moderately dense random graph
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < 0.6:
                add_edge(u, v)
    
    def count_q3s():
        """Count total Q3 subgraphs."""
        count = 0
        for combo in q3_embeddings:
            # Check if all 12 edges are present
            vlist = list(combo)
            # We need to check if the subgraph induced by vlist is a Q3
            # But we already know the structure from is_q3.
            # Actually, we need to check if ALL edges of the Q3 are present.
            # The 12 edges of Q3 on vertices 0-7.
            # But we don't know the labeling.
            
            # Let's just check if is_q3 returns True for the current adjacency.
            # But is_q3 uses the global adj, so it's correct.
            if is_q3(vlist):
                count += 1
        return count
    
    # This is too slow. Let's use incremental counting.
    
    # Better: for each edge, count how many Q3s it's in.
    # When we add/remove an edge, we update the count.
    
    # Let's precompute, for each edge (u,v), the list of Q3 embeddings that contain it.
    edge_to_q3s = {i: [] for i in range(n * n)}
    
    for idx, combo in enumerate(q3_embeddings):
        vlist = list(combo)
        # Find all edges in this Q3
        # We need to determine which pairs are edges in the Q3 structure.
        # Use the is_q3 logic to find the structure.
        
        # Actually, let's just check all pairs in the combo.
        # For each pair (a,b) in combo, if has_edge(a,b) in the current graph,
        # then this Q3 is "active" if all 12 Q3 edges are present.
        
        # This is getting complicated. Let's simplify.
        
        # For each Q3 embedding, store the 12 edges that must be present.
        # But the edges depend on the labeling.
        
        # Alternative approach: just use count_q3s() directly, but optimize it.
        pass
    
    # Let's use a different SA strategy:
    # Objective: maximize (edges - penalty * q3_count)
    # Use penalty that increases over time.
    
    def get_objective(penalty):
        e_count = count_edges()
        q3_count = count_q3s()
        return e_count - penalty * q3_count
    
    # This is too slow for SA. Let's use incremental updates.
    
    # Precompute for each edge, which Q3s it belongs to.
    # A Q3 is identified by its 8 vertices. An edge (u,v) is in a Q3 if
    # u and v are in the same Q3 AND they are adjacent in the Q3 structure.
    
    # Let's rebuild: for each Q3 embedding, determine the 12 edges.
    
    def get_q3_edges(combo):
        """Return the 12 edges of the Q3 on the given 8 vertices."""
        vlist = list(combo)
        # Use the is_q3 logic to find the structure
        v0 = vlist[0]
        neighbors = [u for u in vlist if u != v0 and has_edge(v0, u)]
        if len(neighbors) != 3:
            return None
        
        n0, n1, n2 = neighbors
        
        remaining = [v for v in vlist if v != v0 and v not in neighbors]
        
        c01 = None
        for v in remaining:
            if has_edge(n0, v) and has_edge(n1, v):
                c01 = v
                break
        if c01 is None:
            return None
        
        c02 = None
        for v in remaining:
            if has_edge(n0, v) and has_edge(n2, v):
                c02 = v
                break
        if c02 is None:
            return None
        
        c12 = None
        for v in remaining:
            if has_edge(n1, v) and has_edge(n2, v):
                c12 = v
                break
        if c12 is None:
            return None
        
        c012 = None
        for v in remaining:
            if v not in [c01, c02, c12]:
                c012 = v
                break
        if c012 is None:
            return None
        
        # The 12 edges:
        edges = [
            (v0, n0), (v0, n1), (v0, n2),
            (n0, c01), (n0, c02),
            (n1, c01), (n1, c12),
            (n2, c02), (n2, c12),
            (c01, c012), (c02, c012), (c12, c012)
        ]
        return edges
    
    # Precompute edge-to-Q3 mapping
    edge_q3_count = {}
    for u in range(n):
        for v in range(u + 1, n):
            edge_q3_count[(u, v)] = 0
    
    q3_active = []  # List of (combo, edges) for active Q3s
    
    def update_q3s():
        """Update the list of active Q3s."""
        nonlocal q3_active
        q3_active = []
        for combo in q3_embeddings:
            edges = get_q3_edges(combo)
            if edges is not None:
                # Check if all edges are present
                all_present = True
                for a, b in edges:
                    if not has_edge(a, b):
                        all_present = False
                        break
                if all_present:
                    q3_active.append((combo, edges))
    
    # This is still slow. Let's just use a simpler approach:
    # Start with a known good graph and improve.
    
    # Let's try a different initialization: start with all edges, then remove Q3s.
    
    # Initialize with complete graph
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            add_edge(u, v)
    
    # Now remove edges to eliminate Q3s
    # Use greedy: while there's a Q3, remove an edge from it.
    
    # But we want to maximize edges, so we should be strategic.
    
    # Let's use SA with a fast violation counter.
    
    # Precompute all 8-subsets that could potentially be Q3s
    # (i.e., where each vertex has degree >= 3 within the subset)
    candidate_q3s = []
    for combo in combinations(range(n), 8):
        vset = set(combo)
        valid = True
        for v in combo:
            deg = 0
            for u in combo:
                if u != v and has_edge(u, v):
                    deg += 1
            if deg < 3:
                valid = False
                break
        if valid:
            candidate_q3s.append(combo)
    
    # Now use SA
    random.seed(42)
    
    # Start with a random dense graph
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < 0.7:
                add_edge(u, v)
    
    def count_violations():
        """Count Q3 subgraphs."""
        count = 0
        for combo in candidate_q3s:
            if is_q3(combo):
                count += 1
        return count
    
    # SA parameters
    T = 5.0
    T_min = 0.01
    alpha = 0.9995
    steps = 100000
    
    best_edges = count_edges()
    best_adj = adj[:]
    best_violations = count_violations()
    
    for step in range(steps):
        T *= alpha
        if T < T_min:
            T = T_min
        
        # Random edge flip
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        
        old_edges = count_edges()
        old_violations = count_violations()
        
        if has_edge(u, v):
            remove_edge(u, v)
        else:
            add_edge(u, v)
        
        new_edges = count_edges()
        new_violations = count_violations()
        
        # Objective: maximize edges, minimize violations
        # Use: score = edges - penalty * violations
        # Penalty should be large enough that violations are avoided.
        penalty = 100.0
        
        old_score = old_edges - penalty * old_violations
        new_score = new_edges - penalty * new_violations
        
        delta = new_score - old_score
        
        if delta >= 0:
            accept = True
        else:
            accept = random.random() < np.exp(delta / T)
        
        if not accept:
            # Revert
            if has_edge(u, v):
                remove_edge(u, v)
            else:
                add_edge(u, v)
        else:
            # Update best
            if new_violations < best_violations or (new_violations == best_violations and new_edges > best_edges):
                best_violations = new_violations
                best_edges = new_edges
                best_adj = adj[:]
    
    # Extract edges from best_adj
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges

# EVOLVE-BLOCK-END
