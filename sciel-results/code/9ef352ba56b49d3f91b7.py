import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n = 16
    random.seed(42)

    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    total_edges = len(all_edges)  # 120

    # Map global edges to indices
    edge_index = [[-1] * n for _ in range(n)]
    edge_list = []
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_index[u][v] = idx
            edge_index[v][u] = idx
            edge_list.append((u, v))
            idx += 1

    # Generate all Q3 subgraphs (as bitmasks of edges)
    # A Q3 is a subgraph on 8 vertices with 12 edges.
    # We iterate over all 8-subsets of vertices.
    # For each, we find all subsets of 12 edges that form a Q3.
    
    # To speed up Q3 detection in K8, we can use the property that Q3 is bipartite and 3-regular.
    # Or just check isomorphism.
    # Given the small size (8 vertices, 28 edges), we can precompute the list of Q3 edge masks for a canonical K8.
    
    # Let's generate Q3s for the first 8 vertices (0..7)
    # Then map them to any 8-subset.
    
    # First, find all Q3s in K8 (vertices 0..7)
    # Edges in K8: 28 edges.
    # We need to find all 12-edge subsets that are isomorphic to Q3.
    
    # Helper to check if a set of edges on 8 vertices is Q3
    def is_q3(edge_set, vertices):
        # edge_set is a frozenset of (u,v) pairs
        # vertices is a tuple of 8 vertices
        # Check if it's 3-regular and bipartite and connected
        # Degrees
        deg = {v: 0 for v in vertices}
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
        if any(d != 3 for d in deg.values()):
            return False
        
        # Bipartite check
        # BFS
        visited = {}
        for v in vertices:
            if v not in visited:
                # Start BFS
                queue = [v]
                visited[v] = 0
                while queue:
                    curr = queue.pop(0)
                    # Find neighbors
                    for u, w in edge_set:
                        if u == curr:
                            nb = w
                        elif w == curr:
                            nb = u
                        else:
                            continue
                        if nb in visited:
                            if visited[nb] == visited[curr]:
                                return False # Not bipartite
                        else:
                            visited[nb] = 1 - visited[curr]
                            queue.append(nb)
        return True

    # Generate all Q3 masks for vertices 0..7
    # Edges are indexed 0..27 in the local K8
    local_edge_list = [(u, v) for u in range(8) for v in range(u + 1, 8)]
    local_edge_index = [[-1] * 8 for _ in range(8)]
    l_idx = 0
    for u in range(8):
        for v in range(u + 1, 8):
            local_edge_index[u][v] = l_idx
            local_edge_index[v][u] = l_idx
            l_idx += 1
            
    q3_local_masks = []
    # Iterate over all 12-subsets of 28 edges
    # This is C(28,12) which is large (~30 million). Too slow.
    
    # Better: Generate Q3s by their structure.
    # Q3 is a cube. It has 8 vertices.
    # We can generate all 3-regular bipartite graphs on 8 vertices?
    # Or simply:
    # A Q3 is defined by a bipartition (A,B) of size 4,4 and a 3-regular bipartite graph between them.
    # The number of 3-regular bipartite graphs on 4+4 vertices is small.
    # It's the number of 4x4 0-1 matrices with row/col sums 3.
    # This is equivalent to 4x4 permutation matrices complemented?
    # Sum 3 means 1 zero per row/col.
    # So it's a permutation matrix of zeros.
    # There are 4! = 24 such matrices.
    # So there are 24 distinct 3-regular bipartite graphs on 4+4 labeled vertices.
    # However, we need to map these to the 8 vertices of K8.
    
    # For each 8-subset S of vertices:
    # For each partition of S into A (4) and B (4):
    # For each of the 24 patterns:
    # Construct the edge set.
    # Check if it's a simple graph (it will be).
    # Add to list.
    
    # This will generate duplicates (isomorphic copies) but that's fine.
    # We just need the set of all possible Q3 edge masks.
    
    all_q3_masks = set()
    
    # Precompute the 24 patterns as sets of local edges (0..7) for A=0..3, B=4..7
    # Pattern is defined by a permutation p of 4..7 for the "missing" edges?
    # No, sum 3 means 1 missing edge per row/col.
    # Let the missing edges be (i, 4+p[i]) for i=0..3, where p is a permutation of 0..3.
    # Then the present edges are all (i, 4+j) where j != p[i].
    
    patterns = []
    for p in itertools.permutations(range(4)):
        edges = set()
        for i in range(4):
            for j in range(4):
                if j != p[i]:
                    edges.add((i, 4+j))
        patterns.append(frozenset(edges))
        
    # Now, for each 8-subset of vertices in K16
    all_subsets = list(itertools.combinations(range(n), 8))
    
    for S in all_subsets:
        S_set = set(S)
        # Generate all bipartitions of S into A, B of size 4
        # Choose 4 vertices for A
        for A in itertools.combinations(S, 4):
            A_set = set(A)
            B = tuple(v for v in S if v not in A_set)
            
            # Map local indices 0..3 to A, 4..7 to B
            # A is a tuple, B is a tuple
            # For each pattern:
            for pat in patterns:
                mask = 0
                for u_local, v_local in pat:
                    u_glob = A[u_local]
                    v_glob = B[v_local - 4]
                    g_idx = edge_index[u_glob][v_glob]
                    mask |= (1 << g_idx)
                all_q3_masks.add(mask)
                
    # Convert to list for faster iteration
    q3_masks = list(all_q3_masks)
    num_q3s = len(q3_masks)
    
    # Precompute for each edge, which Q3s contain it
    q3_by_edge = [[] for _ in range(total_edges)]
    for q_mask in q3_masks:
        # Iterate through bits
        m = q_mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            q3_by_edge[e_idx].append(q_mask)
            m ^= lsb

    # Helper to check if adding edge e_idx to G creates a Q3
    def is_q3_free_addition(G, e_idx):
        for q_mask in q3_by_edge[e_idx]:
            # Check if all other edges in q_mask are in G
            # q_mask has e_idx set.
            # We need to check if (G & (q_mask ^ (1 << e_idx))) == (q_mask ^ (1 << e_idx))
            other_edges = q_mask ^ (1 << e_idx)
            if (G & other_edges) == other_edges:
                return False
        return True

    def count_edges(G):
        return bin(G).count('1')

    best_graph = 0
    best_count = 0

    # Local Search with Restart
    # We will run multiple restarts.
    
    while time.time() < deadline - 5.0:
        # Greedy construction
        G = 0
        # Randomize edge order
        edge_order = list(range(total_edges))
        random.shuffle(edge_order)
        
        for e_idx in edge_order:
            if is_q3_free_addition(G, e_idx):
                G |= (1 << e_idx)
        
        current_count = count_edges(G)
        if current_count > best_count:
            best_count = current_count
            best_graph = G

        # Local Search (Hill Climbing)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Try to add missing edges
            missing_edges = [i for i in range(total_edges) if not (G >> i) & 1]
            random.shuffle(missing_edges)
            
            for e_add in missing_edges:
                if is_q3_free_addition(G, e_add):
                    G |= (1 << e_add)
                    current_count += 1
                    improved = True
                    break # Restart loop to re-shuffle or just continue? 
                    # Breaking restarts the inner while, which is good for fresh shuffling.
            
            if not improved:
                # Try swaps
                # Find an edge e_add that is blocked.
                # Find a blocking Q3.
                # Remove an edge from that Q3.
                # Add e_add.
                
                # Pick a random missing edge
                if missing_edges:
                    e_add = random.choice(missing_edges)
                    # Find a blocking Q3
                    blocker = None
                    for q_mask in q3_by_edge[e_add]:
                        other_edges = q_mask ^ (1 << e_add)
                        if (G & other_edges) == other_edges:
                            blocker = q_mask
                            break
                    
                    if blocker:
                        # Pick a random edge from blocker to remove
                        # It must be in G
                        # blocker contains e_add (not in G) and 11 edges in G.
                        # We pick one of the 11.
                        candidates = []
                        m = blocker ^ (1 << e_add)
                        while m:
                            lsb = m & -m
                            e_idx = lsb.bit_length() - 1
                            candidates.append(e_idx)
                            m ^= lsb
                            
                        if candidates:
                            e_remove = random.choice(candidates)
                            # Try the swap
                            new_G = (G & ~(1 << e_remove)) | (1 << e_add)
                            # Check if new_G is Q3-free
                            # We only need to check if e_add creates a Q3 in (G \ {e_remove})
                            # And if e_remove was critical for other Q3s? No, removing helps.
                            # So we just check is_q3_free_addition(G \ {e_remove}, e_add)
                            G_temp = G & ~(1 << e_remove)
                            if is_q3_free_addition(G_temp, e_add):
                                G = new_G
                                # Size is same, but structure changed.
                                # We mark improved to allow further moves
                                improved = True
                                # Do not break, try more swaps?
                                # Break to restart loop?
                                break

    # Final add-back pass
    for e_idx in range(total_edges):
        if not (best_graph >> e_idx) & 1:
            if is_q3_free_addition(best_graph, e_idx):
                best_graph |= (1 << e_idx)

    # Convert to edge list
    result = []
    for u, v in edge_list:
        e_idx = edge_index[u][v]
        if (best_graph >> e_idx) & 1:
            result.append([u, v])

    return result
