"""Q3-free graph on 16 vertices - maximize edges using bitset detection + SA."""
import time
import random

# Precompute all Q3 embeddings as frozensets of 12 edge-pair indices
# Edge index for (u,v) with u<v: u*16 + v (using 16-bit encoding per vertex)
# We use a 128-bit bitmask per vertex for adjacency

def _edge_idx(u, v):
    if u > v:
        u, v = v, u
    return u * 16 + v

def _build_q3_embeddings():
    """Build all distinct Q3 subgraphs (as sets of 12 edge indices)."""
    # Q3 vertices are 3-bit values 0..7. Edges connect vertices differing in 1 bit.
    # We need all injective mappings from {0..7} to {0..15} that preserve the Q3 structure.
    # For each permutation of 8 distinct vertices from 16, check if they form a Q3.
    # Too many (16P8 = 518,918,400). Instead, enumerate by structure.
    
    # Q3 has 8 vertices, 12 edges. The 12 edges in Q3 (with 3-bit labels):
    q3_edges_local = []
    for i in range(8):
        for b in range(3):
            j = i ^ (1 << b)
            if i < j:
                q3_edges_local.append((i, j))
    # q3_edges_local has 12 edges
    
    # Enumerate: pick 8 vertices, then check if the induced subgraph is Q3.
    # But we need all *subgraphs* isomorphic to Q3, not just induced.
    # A subgraph isomorphic to Q3 means: there exist 8 distinct vertices v0..v7 and
    # 12 edges among them forming the Q3 pattern.
    
    # Better approach: enumerate all ways to map the 8 Q3 vertices to 16 vertices.
    # For each injection f: {0..7} -> {0..15}, the edges f(i)-f(j) for (i,j) in Q3
    # give a Q3 subgraph. Two injections give the same edge set if they differ by
    # an automorphism of Q3 (which is the hyperoctahedral group of size 384).
    
    # To avoid duplicates, we can just collect all edge sets (as frozensets) in a set.
    # Number of injections: 16*15*14*13*12*11*10*9 = 518,918,400. That's too many.
    
    # Alternative: use the fact that Q3 is bipartite with parts of size 4.
    # In Q3, vertices with even parity form one part, odd parity the other.
    # Each vertex has degree 3.
    
    # Smarter enumeration: 
    # 1. Choose 4 vertices for part A (even parity in Q3): C(16,4) = 1820
    # 2. Choose 4 vertices for part B (odd parity in Q3): C(12,4) = 495
    # 3. For each such pair of 4-sets, check if there's a perfect matching structure
    #    forming Q3.
    
    # Actually, let me just enumerate injections but with a smarter check.
    # 16P8 is ~519M which is too slow in Python.
    
    # Alternative: enumerate all 8-subsets (C(16,8) = 12870), and for each,
    # check all 8! = 40320 labelings to see if any gives Q3.
    # 12870 * 40320 = ~519M. Still too many.
    
    # Better: for each 8-subset, the number of Q3 embeddings is at most 8!/384 ≈ 105.
    # But checking each labeling is expensive.
    
    # Most efficient: enumerate all 12-edge subsets that could be Q3.
    # A Q3 has 8 vertices, 12 edges, each vertex degree 3.
    # Enumerate all 8-vertex subsets (12870), then within each, find all
    # 12-edge subgraphs where each vertex has degree exactly 3.
    
    embeddings = set()
    from itertools import combinations
    
    for verts in combinations(range(16), 8):
        vset = set(verts)
        # All possible edges within these 8 vertices
        all_edges = []
        for i in range(8):
            for j in range(i+1, 8):
                all_edges.append((verts[i], verts[j]))
        
        # We need to find subsets of 12 edges where each vertex has degree 3.
        # This is equivalent to finding 3-regular subgraphs on these 8 vertices with 12 edges.
        # A 3-regular graph on 8 vertices has exactly 12 edges (8*3/2 = 12).
        # So we need to check if the induced subgraph contains a 3-regular spanning subgraph
        # that is isomorphic to Q3.
        
        # Build adjacency for the 8 vertices
        # For each vertex, it can connect to 7 others. We need degree 3.
        # Enumerate all 3-regular graphs on these 8 vertices and check if any is Q3.
        
        # Use the fact that Q3 is the unique 3-regular bipartite graph on 8 vertices
        # with parts of size 4? No, there are multiple.
        # Q3 is characterized by: 3-regular, bipartite, girth 4, diameter 3.
        
        # Simpler: just check if the 8-vertex induced subgraph has a Q3 as a subgraph.
        # For each permutation of the 8 vertices (assigning them to Q3 positions 0..7),
        # check if all 12 Q3 edges are present.
        
        # 8! = 40320 per 8-subset, times 12870 subsets = 519M. Too slow.
        
        # Optimization: fix vertex 0 to Q3 position 0 (symmetry of Q3 under vertex relabeling
        # that fixes one vertex has size 384/8 = 48... actually the stabilizer of a vertex in
        # Aut(Q3) has size 48). So we fix one vertex and try 7! / 48 = 840 permutations.
        # 12870 * 840 = ~10.8M. More feasible.
        
        # Actually, let's just do it the direct way but with early termination.
        # For each 8-subset, try to find a Q3 embedding.
        
        # Map: local vertex i (0..7) -> global vertex verts[i]
        # Q3 edges: pairs (i,j) where i^j is a power of 2
        q3_e = []
        for i in range(8):
            for b in range(3):
                j = i ^ (1 << b)
                if i < j:
                    q3_e.append((i, j))
        
        # Try all permutations, but fix verts[0] to local vertex 0
        # to reduce by factor 8 (symmetry of choosing which global vertex maps to local 0)
        # Then we have 7! = 5040 permutations of the remaining 7.
        # For each, check if all 12 Q3 edges are present in the graph.
        # But we don't have the graph yet - we're enumerating possible Q3 subgraphs.
        # The Q3 subgraph is defined by its 12 edges. For a given permutation, the 12 edges
        # are determined. We just need to collect them.
        
        # Since we want ALL Q3 subgraphs (not just those in a given graph), we just need
        # to enumerate all distinct 12-edge sets that form a Q3.
        
        # For a fixed 8-subset, the number of distinct Q3 embeddings (up to Q3 automorphism)
        # is at most 8!/384 = 105.
        
        # Let's enumerate all 8! permutations, compute the 12-edge set, and add to a global set.
        # But 8! * 12870 is too many.
        
        # Use the stabilizer: fix local vertex 0 to global vertex verts[0].
        # Then permute the remaining 7 local vertices among the remaining 7 global vertices.
        # That's 7! = 5040 per 8-subset. 12870 * 5040 = 64.9M. Still a lot but maybe OK
        # if each check is fast. Actually we just need to compute the edge set, no graph check.
        
        from itertools import permutations
        for perm in permutations(verts[1:], 7):
            mapping = [verts[0]] + list(perm)
            # mapping[i] = global vertex for local Q3 vertex i
            edge_set = frozenset(_edge_idx(mapping[i], mapping[j]) for i, j in q3_e)
            embeddings.add(edge_set)
    
    return list(embeddings)

# Precompute at module load (may take a few seconds)
_Q3_EMBEDDINGS = _build_q3_embeddings()
_NUM_Q3 = len(_Q3_EMBEDDINGS)
print(f"Found {_NUM_Q3} distinct Q3 embeddings")

# For fast checking: for each Q3 embedding, store the 12 edge indices
_Q3_EDGES = [list(e) for e in _Q3_EMBEDDINGS]

def _has_q3(adj_bits):
    """Check if the graph with adjacency bitmasks contains a Q3 subgraph.
    adj_bits[v] is a 16-bit integer with bit u set if edge (v,u) exists.
    Returns True if a Q3 is found."""
    for edges in _Q3_EDGES:
        all_present = True
        for e in edges:
            u = e // 16
            v = e % 16
            if not (adj_bits[u] >> v) & 1:
                all_present = False
                break
        if all_present:
            return True
    return False

def _count_q3_violations(adj_bits):
    """Count how many Q3 embeddings are present (for penalty in SA)."""
    count = 0
    for edges in _Q3_EDGES:
        all_present = True
        for e in edges:
            u = e // 16
            v = e % 16
            if not (adj_bits[u] >> v) & 1:
                all_present = False
                break
        if all_present:
            count += 1
    return count

def _random_q3_free_graph(target_density):
    """Generate a random graph with ~target_density fraction of edges, then remove Q3s."""
    adj = [0] * 16
    n_edges = 0
    # Add edges randomly
    for u in range(16):
        for v in range(u+1, 16):
            if random.random() < target_density:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                n_edges += 1
    # Remove edges to break Q3s (greedy)
    changed = True
    while changed:
        changed = False
        for edges in _Q3_EDGES:
            all_present = True
            present_edges = []
            for e in edges:
                u = e // 16
                v = e % 16
                if (adj[u] >> v) & 1:
                    present_edges.append(e)
                else:
                    all_present = False
                    break
            if all_present:
                # Remove a random edge from this Q3
                e = random.choice(present_edges)
                u = e // 16
                v = e % 16
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                n_edges -= 1
                changed = True
    return adj, n_edges

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Start with a good initial graph
    best_adj = None
    best_edges = 0
    
    # Try several random initializations
    for trial in range(20):
        if time.time() > deadline:
            break
        for density in [0.5, 0.55, 0.6, 0.65, 0.7]:
            if time.time() > deadline:
                break
            adj, n_edges = _random_q3_free_graph(density)
            if n_edges > best_edges:
                best_edges = n_edges
                best_adj = adj[:]
    
    if best_adj is None:
        # Fallback: K3,13
        adj = [0] * 16
        for a in range(3):
            for b in range(3, 16):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
        best_adj = adj
        best_edges = 39
    
    # Simulated annealing
    adj = best_adj[:]
    current_edges = best_edges
    temp = 50.0
    cooling = 0.9995
    
    # Precompute all possible edges
    all_possible_edges = [(u, v) for u in range(16) for v in range(u+1, 16)]
    edge_to_idx = {e: i for i, e in enumerate(all_possible_edges)}
    
    # For SA: try adding edges, and if Q3 is created, remove a random edge from that Q3
    # Or: swap operations - remove one edge, add another
    
    while time.time() < deadline:
        # Try to add a random missing edge
        # Find missing edges
        missing = []
        for u in range(16):
            for v in range(u+1, 16):
                if not (adj[u] >> v) & 1:
                    missing.append((u, v))
        
        if not missing:
            break
        
        u, v = random.choice(missing)
        
        # Try adding edge (u,v)
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        new_edges = current_edges + 1
        
        # Check for Q3 violations
        violations = _count_q3_violations(adj)
        
        if violations == 0:
            # Accept: improved
            current_edges = new_edges
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj[:]
        else:
            # Reject: undo
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        
        # Occasionally try a swap: remove one edge, add another
        if random.random() < 0.3:
            # Remove a random edge
            present = []
            for u in range(16):
                for v in range(u+1, 16):
                    if (adj[u] >> v) & 1:
                        present.append((u, v))
            if present:
                ru, rv = random.choice(present)
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
                current_edges -= 1
                
                # Add a random missing edge
                missing2 = []
                for u in range(16):
                    for v in range(u+1, 16):
                        if not (adj[u] >> v) & 1:
                            missing2.append((u, v))
                if missing2:
                    au, av = random.choice(missing2)
                    adj[au] |= (1 << av)
                    adj[av] |= (1 << au)
                    current_edges += 1
                    
                    if not _has_q3(adj):
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = adj[:]
        
        temp *= cooling
    
    # Convert best_adj to edge list
    edges = []
    for u in range(16):
        for v in range(u+1, 16):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
