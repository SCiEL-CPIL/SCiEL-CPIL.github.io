import random
import time
import itertools

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_edges = 0
    best_graph = None
    
    # Helper to check if adding edge (u, v) to graph G creates K3,3
    # G is a list of lists of neighbors (bitmasks)
    def is_k33_free_with_edge(G, u, v):
        # Temporarily add edge
        old_u = G[u]
        old_v = G[v]
        G[u] |= (1 << v)
        G[v] |= (1 << u)
        
        # Check if any K3,3 exists
        # We only need to check triples that involve u or v?
        # No, adding an edge can only create new K3,3s that use that edge.
        # So u and v must be in the bipartition.
        # Case 1: u in A, v in B.
        # We need 2 other in A (neighbors of v) and 2 other in B (neighbors of u).
        # Let A_candidates = neighbors(v) \ {u}
        # Let B_candidates = neighbors(u) \ {v}
        # We need to find a1, a2 in A_candidates and b1, b2 in B_candidates
        # such that a1, a2 are connected to b1, b2.
        # This is equivalent to checking if the bipartite graph between A_candidates and B_candidates
        # contains a K2,2.
        
        # Actually, it's easier to just run the full check for small n.
        # But we can optimize:
        # The new K3,3 must contain (u,v).
        # So u is in one part, v in the other.
        # Part A contains u. Part B contains v.
        # Let A = {u} + {x | x in N(v), x != u}
        # Let B = {v} + {y | y in N(u), y != v}
        # We need to pick 2 more from A\{u} and 2 more from B\{v} such that they are fully connected.
        
        # Let's just use the generic check for simplicity first, it's fast enough.
        # 1140 triples.
        for i, j, k in itertools.combinations(range(n), 3):
            mask = G[i] & G[j] & G[k]
            if mask.bit_count() >= 3:
                G[u] = old_u
                G[v] = old_v
                return False
        
        G[u] = old_u
        G[v] = old_v
        return True

    def find_k33(G):
        for i, j, k in itertools.combinations(range(n), 3):
            mask = G[i] & G[j] & G[k]
            if mask.bit_count() >= 3:
                # Find 3 bits
                bits = []
                m = mask
                while len(bits) < 3 and m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    bits.append(idx)
                    m ^= lsb
                return [(i, l) for l in bits] + [(j, l) for l in bits] + [(k, l) for l in bits]
        return None

    for seed in range(1000):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        # Initialize graph
        # Start with a random graph with p=0.5
        G = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    G[i] |= (1 << j)
                    G[j] |= (1 << i)
                    
        # Phase 1: Remove edges to make K3,3 free
        while True:
            edges_to_remove = find_k33(G)
            if edges_to_remove is None:
                break
            # Remove a random edge from the found K3,3
            # To be smarter, we could pick the edge that is in the most K3,3s, 
            # but random is fine for now.
            # Let's pick the first one for speed, or random.
            e = random.choice(edges_to_remove)
            u, v = e
            G[u] &= ~(1 << v)
            G[v] &= ~(1 << u)
            
        # Phase 2: Try to add back edges
        # Get list of missing edges
        missing = []
        for i in range(n):
            for j in range(i+1, n):
                if not (G[i] & (1 << j)):
                    missing.append((i, j))
                    
        random.shuffle(missing)
        added_any = True
        while added_any:
            added_any = False
            for u, v in missing:
                if not (G[u] & (1 << v)):
                    if is_k33_free_with_edge(G, u, v):
                        G[u] |= (1 << v)
                        G[v] |= (1 << u)
                        added_any = True
                        # Remove from missing list? 
                        # It's easier to just keep checking, but we can mark it.
                        # Since we are iterating a static list, we just skip if already added.
                        # The check `if not (G[u] & (1 << v))` handles it.

        # Count edges
        edge_count = sum(G[i].bit_count() for i in range(n)) // 2
        if edge_count > best_edges:
            best_edges = edge_count
            best_graph = G[:]
            
    # Convert best_graph to list of edges
    if best_graph is None:
        # Fallback to seed
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    edges = []
    for u in range(n):
        mask = best_graph[u]
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            m ^= lsb
    return edges
