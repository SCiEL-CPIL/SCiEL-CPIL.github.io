import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_idx = {e: i for i, e in enumerate(all_edges)}
    m_total = len(all_edges)
    
    # Precompute all Q3 embeddings
    # Each embedding is a set of 8 vertices and 12 edges
    # We use a bitset (int) to represent the set of edges in the embedding
    embeddings = []
    for v in range(n):
        # In a Q3, v is 000. Neighbors are 001, 010, 100.
        # We iterate over triples of neighbors (a, b, c) which are pairwise non-adjacent in Q3.
        # Then x=011 (adj to a,b), y=101 (adj to a,c), z=110 (adj to b,c), w=111 (adj to x,y,z).
        # We need to check if the graph contains these edges.
        # To precompute, we just list all possible sets of 8 vertices that form a Q3 structure.
        # A set of 8 vertices {v, a, b, c, x, y, z, w} forms a Q3 if the edges are exactly the 12 cube edges.
        # Since we are precomputing for the *complete* graph K16, any set of 8 vertices 
        # that can be mapped to Q3 vertices is a potential embedding.
        # Actually, any 8 distinct vertices can be mapped to Q3 vertices in 8! / 8 ways (automorphisms of Q3 is 48, so 8!/48 = 1680 mappings).
        # But we only need the set of edges. The set of edges in the embedding depends on the mapping.
        # For a fixed set of 8 vertices, there are multiple ways to map them to Q3.
        # We need to find ALL sets of 12 edges that form a Q3 subgraph.
        
        # Efficient way:
        # Iterate over all 8-subsets of vertices. For each, check if it induces a subgraph isomorphic to Q3.
        # There are C(16,8) = 12870 subsets.
        # For each subset, we can check if it contains a Q3.
        # Actually, we just need the list of edge-sets.
        
        pass # This loop is too slow or complex to precompute all edge sets directly in Python without optimization.
        
    # Alternative Precomputation:
    # Generate all possible Q3 edge sets by iterating over all injective maps from Q3 vertices to 0..15.
    # Q3 vertices: 0..7. Edges: 12.
    # Number of maps: P(16, 8) = 16*15*...*9 = 518,918,400. Too many.
    
    # Better Precomputation:
    # We can generate embeddings by picking v, a, b, c, x, y, z, w.
    # v: 16 choices.
    # a, b, c: distinct neighbors of v in Q3. In K16, any 3 distinct vertices from remaining 15.
    # x: neighbor of a, b. Must be distinct from v, a, b, c.
    # y: neighbor of a, c. Distinct from v, a, b, c, x.
    # z: neighbor of b, c. Distinct from v, a, b, c, x, y.
    # w: neighbor of x, y, z. Distinct from others.
    
    # This is still large but manageable if we prune?
    # Let's just use the on-the-fly detection for the initial phase, but optimize it heavily using NumPy.
    
    # Strategy:
    # 1. Start with a dense graph (e.g., two 8-cliques).
    # 2. Use a fast NumPy-based check to find any Q3.
    # 3. If found, remove an edge.
    # 4. Try to add edges back using a fast check.
    
    # Let's precompute the adjacency matrix of Q3 for fast checking.
    # Q3 adj matrix (8x8)
    q3_adj = np.zeros((8, 8), dtype=np.int8)
    for i in range(8):
        for j in range(8):
            if i != j and bin(i ^ j).count('1') == 1:
                q3_adj[i, j] = 1
                
    # Precompute all valid permutations of 8 vertices? No, that's 40320.
    # Instead, we can check if a specific set of 8 vertices induces a Q3 by checking all 48 automorphisms?
    # No, we need to check if the induced subgraph is ISOMORPHIC to Q3.
    # This is equivalent to checking if there exists a permutation of the 8 vertices such that the adjacency matches Q3.
    
    # Faster approach for the search:
    # Use the "Vertex-based" check from the seed but optimized with sets/NumPy.
    # The seed's score was 1.38 (approx 60-70 edges?).
    # Let's aim for a structured start.
    
    # Initialize: Two 8-cliques.
    # Vertices 0-7 form a clique, 8-15 form a clique.
    # Edges: C(8,2) * 2 = 28 * 2 = 56 edges.
    # Are there Q3s in two 8-cliques?
    # A Q3 needs 8 vertices. If all 8 are in one clique, the induced subgraph is K8, which is not Q3 (Q3 has degree 3, K8 has degree 7).
    # So we need vertices from both cliques.
    # Let k vertices be in Clique 1, 8-k in Clique 2.
    # Edges within C1 are all present. Edges within C2 are all present.
    # Edges between C1 and C2 are all ABSENT in our initial graph.
    # In Q3, there are edges between "levels".
    # If we have no edges between the two groups, the graph is disconnected.
    # Q3 is connected. So a Q3 cannot exist in a disconnected graph.
    # Thus, the initial graph of two 8-cliques is Q3-free!
    # Edge count: 56.
    
    # Can we do better?
    # Yes, we can add edges between the cliques.
    # We want to maximize edges.
    # Start with 56 edges. Try to add the 64 inter-clique edges.
    
    current_edges = set()
    # Add intra-clique edges
    for i in range(8):
        for j in range(i+1, 8):
            current_edges.add((i, j))
    for i in range(8):
        for j in range(i+1, 8):
            current_edges.add((i+8, j+8))
            
    # Build adjacency sets for fast lookup
    adj = {i: set() for i in range(n)}
    for u, v in current_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    best_edges = set(current_edges)
    best_count = len(current_edges)
    
    # List of inter-clique edges to try adding
    inter_edges = [(i, j+8) for i in range(8) for j in range(8)]
    random.seed(42)
    random.shuffle(inter_edges)
    
    # Fast Q3 check function
    # Optimized version of the seed's logic
    def find_q3_fast(adj):
        # Iterate over vertices v
        for v in range(n):
            neighbors = list(adj.get(v, set()))
            if len(neighbors) < 3:
                continue
            # Iterate over triples of neighbors
            # Optimization: Pre-filter triples that are pairwise non-adjacent
            # a, b, c must be pairwise non-adjacent
            n_len = len(neighbors)
            for i in range(n_len):
                a = neighbors[i]
                for j in range(i+1, n_len):
                    b = neighbors[j]
                    if b in adj[a]: continue # a,b adjacent
                    for k in range(j+1, n_len):
                        c = neighbors[k]
                        if c in adj[a]: continue # a,c adjacent
                        if c in adj[b]: continue # b,c adjacent
                        
                        # a, b, c are pairwise non-adjacent neighbors of v
                        # Candidates for x (adj to a, b), y (adj to a, c), z (adj to b, c)
                        # x, y, z must be distinct from v, a, b, c and each other
                        # x, y, z must be pairwise non-adjacent
                        
                        na = adj[a]
                        nb = adj[b]
                        nc = adj[c]
                        
                        # x is in na & nb
                        # y is in na & nc
                        # z is in nb & nc
                        
                        # To speed up, we can iterate over the smaller sets
                        # But sets are small (max 15)
                        
                        xs = na & nb
                        ys = na & nc
                        zs = nb & nc
                        
                        forbidden = {v, a, b, c}
                        xs = xs - forbidden
                        ys = ys - forbidden
                        zs = zs - forbidden
                        
                        # Iterate over x, y, z
                        # Further optimization: x, y, z must be pairwise non-adjacent
                        for x in xs:
                            # x must not be adjacent to y or z
                            # So y must be in ys - adj[x]
                            # And z must be in zs - adj[x]
                            ys_valid = ys - adj[x]
                            for y in ys_valid:
                                # y must not be adjacent to z
                                # z must be in zs - adj[x] - adj[y]
                                zs_valid = zs - adj[x] - adj[y]
                                for z in zs_valid:
                                    # Check x, y, z distinct (ensured by sets if we subtracted properly? 
                                    # xs, ys, zs might overlap. 
                                    # x in xs, y in ys. If x==y, then y in xs & ys.
                                    # We need x != y, y != z, x != z.
                                    if x == y or y == z or x == z:
                                        continue
                                    
                                    # Find w
                                    # w is neighbor of x, y, z
                                    nx = adj[x]
                                    ny = adj[y]
                                    nz = adj[z]
                                    
                                    ws = nx & ny & nz
                                    ws = ws - forbidden - {x, y, z}
                                    
                                    for w in ws:
                                        # w must NOT be adjacent to a, b, c
                                        if w in adj[a]: continue
                                        if w in adj[b]: continue
                                        if w in adj[c]: continue
                                        
                                        # Found Q3
                                        # Return edges to allow removal
                                        return [
                                            (v, a), (v, b), (v, c),
                                            (a, x), (b, x),
                                            (a, y), (c, y),
                                            (b, z), (c, z),
                                            (x, w), (y, w), (z, w)
                                        ]
        return None
    
    # Greedy addition of inter-clique edges
    for e in inter_edges:
        if time.time() > deadline - 5:
            break
            
        u, v = e
        # Try adding e
        current_edges.add(e)
        adj[u].add(v)
        adj[v].add(u)
        
        q3_edges = find_q3_fast(adj)
        
        if q3_edges is not None:
            # Check if e is in the Q3
            e_norm = (u, v) if u < v else (v, u)
            contains_e = False
            for qe in q3_edges:
                if (qe[0] == u and qe[1] == v) or (qe[0] == v and qe[1] == u):
                    contains_e = True
                    break
            
            if contains_e:
                # Cannot add e
                current_edges.remove(e)
                adj[u].remove(v)
                adj[v].remove(u)
            else:
                # This shouldn't happen if graph was Q3-free
                # But if it does, remove e for safety
                current_edges.remove(e)
                adj[u].remove(v)
                adj[v].remove(u)
        else:
            # e can be added
            pass
            
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)
            
    # Try to add intra-clique edges? No, they are already present.
    # Try to add any other missing edges? 
    # The only missing edges are the inter-clique ones we rejected.
    # We can try to re-add them in a different order or with some removals.
    # But simple greedy is likely sufficient for this structure.
    
    # Let's do a few rounds of "shuffle and try" for the rejected edges to see if order matters
    rejected = [e for e in inter_edges if e not in current_edges]
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        random.shuffle(rejected)
        for e in rejected:
            if time.time() > deadline - 5:
                break
            u, v = e
            if e in current_edges: continue
            
            current_edges.add(e)
            adj[u].add(v)
            adj[v].add(u)
            
            q3_edges = find_q3_fast(adj)
            
            if q3_edges is not None:
                e_norm = (u, v) if u < v else (v, u)
                contains_e = False
                for qe in q3_edges:
                    if (qe[0] == u and qe[1] == v) or (qe[0] == v and qe[1] == u):
                        contains_e = True
                        break
                if contains_e:
                    current_edges.remove(e)
                    adj[u].remove(v)
                    adj[v].remove(u)
                else:
                    current_edges.remove(e)
                    adj[u].remove(v)
                    adj[v].remove(u)
            
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)
            
    return [[u, v] for u, v in best_edges]
