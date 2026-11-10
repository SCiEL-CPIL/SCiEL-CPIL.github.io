import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 115
    n = 16
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Precompute Q3 embeddings: list of all mappings from Q3 vertices (0-7) to graph vertices (0-15)
    # Q3 edges: (0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)
    q3_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    
    # To check if a graph contains Q3, we can iterate over all 16P8 injections? No, too many.
    # Instead, we can iterate over all subsets of 8 vertices from 16? C(16,8) = 12870.
    # For each subset, check if it induces a Q3.
    # A set of 8 vertices induces a Q3 if the induced subgraph is isomorphic to Q3.
    # Q3 has 12 edges.
    
    # Let's precompute all 8-vertex subsets.
    subsets = list(combinations(range(n), 8))
    
    # For each subset, we need to know if it can be mapped to Q3.
    # Actually, we can just check if the induced subgraph on the subset is isomorphic to Q3.
    # Since Q3 is a specific graph, we can precompute its canonical form or just check edge counts and degrees.
    # Q3 degrees are all 3. So the induced subgraph must be 3-regular on 8 vertices with 12 edges.
    # Is every 3-regular graph on 8 vertices with 12 edges a Q3? No, there are others (e.g. K_{4,4} minus a perfect matching is Q3? No. K_4 x K_2? No).
    # Actually, Q3 is the unique bipartite 3-regular graph on 8 vertices? No, K_{3,3} is 3-regular on 6.
    # Let's just do a direct isomorphism check for the small case, or precompute the adjacency matrix of Q3 and check if the induced subgraph matches it under some permutation.
    
    # Better approach:
    # Precompute all permutations of 8 vertices? 8! = 40320. Too many to do for every subset.
    # Instead, for each subset, build the induced adjacency matrix.
    # Check if it's isomorphic to Q3.
    # We can use a simple backtracking isomorphism check for 8 vertices.
    
    def is_isomorphic_to_q3(vertices, adj_mask):
        # vertices: tuple of 8 vertex indices
        # adj_mask: list of 16-bit integers for each of the 16 vertices (global)
        # Build local adjacency matrix 8x8
        local_adj = [[0]*8 for _ in range(8)]
        for i in range(8):
            for j in range(i+1, 8):
                u = vertices[i]
                v = vertices[j]
                if (adj_mask[u] >> v) & 1:
                    local_adj[i][j] = 1
                    local_adj[j][i] = 1
        
        # Check if local_adj is isomorphic to Q3
        # Q3 adjacency (0-indexed):
        # 0: 1,2,4
        # 1: 0,3,5
        # 2: 0,3,6
        # 3: 1,2,7
        # 4: 0,5,6
        # 5: 1,4,7
        # 6: 2,4,7
        # 7: 3,5,6
        
        # We can try to map Q3 vertices to local vertices.
        # Fix mapping of Q3 vertex 0 to local vertex 0.
        # Then Q3 vertex 1 must be a neighbor of 0 in local.
        # Q3 vertex 2 must be a neighbor of 0 in local.
        # Q3 vertex 4 must be a neighbor of 0 in local.
        
        q3_adj = [
            [0,1,1,0,1,0,0,0],
            [1,0,0,1,0,1,0,0],
            [1,0,0,1,0,0,1,0],
            [0,1,1,0,0,0,0,1],
            [1,0,0,0,0,1,1,0],
            [0,1,0,0,1,0,0,1],
            [0,0,1,0,1,0,0,1],
            [0,0,0,1,0,1,1,0]
        ]
        
        # Backtracking search for isomorphism
        mapping = [-1]*8 # maps q3_vertex -> local_vertex
        inv_mapping = [-1]*8 # maps local_vertex -> q3_vertex
        
        def backtrack(q3_v):
            if q3_v == 8:
                return True
            for l_v in range(8):
                if inv_mapping[l_v] != -1:
                    continue
                # Check consistency with previous mappings
                valid = True
                for prev_q3 in range(q3_v):
                    if q3_adj[prev_q3][q3_v] != local_adj[inv_mapping[l_v] if inv_mapping[l_v] != -1 else -1][l_v]:
                        # Wait, inv_mapping[l_v] is the q3 vertex mapped to l_v.
                        # We need to check if the edge between prev_q3 and q3_v matches the edge between mapping[prev_q3] and l_v.
                        pass
                
                # Let's do it properly
                ok = True
                for prev_q3 in range(q3_v):
                    m_prev = mapping[prev_q3]
                    if m_prev == -1:
                        continue
                    edge_in_q3 = q3_adj[prev_q3][q3_v]
                    edge_in_local = local_adj[m_prev][l_v]
                    if edge_in_q3 != edge_in_local:
                        ok = False
                        break
                if not ok:
                    continue
                
                mapping[q3_v] = l_v
                inv_mapping[l_v] = q3_v
                if backtrack(q3_v + 1):
                    return True
                mapping[q3_v] = -1
                inv_mapping[l_v] = -1
            return False
        
        # Optimization: Fix mapping of Q3 vertex 0 to local vertex 0
        mapping[0] = 0
        inv_mapping[0] = 0
        return backtrack(1)
        
    def count_q3s(adj_mask):
        count = 0
        for subset in subsets:
            if is_isomorphic_to_q3(subset, adj_mask):
                count += 1
        return count

    def creates_q3(u, v, adj_mask):
        # Temporarily add edge
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
        # Check only subsets containing u and v?
        # No, the Q3 might not contain the new edge?
        # If adding edge (u,v) creates a Q3, that Q3 must contain the edge (u,v).
        # Because if the Q3 didn't contain (u,v), it would have existed before.
        # So we only need to check subsets that contain u and v.
        
        # Precomputed subsets containing u and v?
        # We can filter on the fly or precompute.
        # C(14,6) = 3003 subsets per edge. 120 edges * 3003 = 360k checks.
        # Each check is a small isomorphism.
        
        # Let's precompute subsets_by_pair
        # subsets_by_pair[(u,v)] = list of subsets containing u and v
        
        # But we don't have that yet. Let's just filter.
        count = 0
        for subset in subsets:
            if u in subset and v in subset:
                if is_isomorphic_to_q3(subset, adj_mask):
                    count += 1
        adj_mask[u] ^= (1 << v)
        adj_mask[v] ^= (1 << u)
        return count > 0

    # Precompute subsets containing each pair for speed?
    # There are 120 pairs.
    # subsets_by_pair = {}
    # for s in subsets:
    #   for i in range(8):
    #     for j in range(i+1, 8):
    #       u, v = s[i], s[j]
    #       if u > v: u, v = v, u
    #       subsets_by_pair.setdefault((u,v), []).append(s)
        
    # Let's do that.
    subsets_by_pair = {}
    for s in subsets:
        for i in range(8):
            for j in range(i+1, 8):
                u, v = s[i], s[j]
                if u > v:
                    u, v = v, u
                if (u, v) not in subsets_by_pair:
                    subsets_by_pair[(u, v)] = []
                subsets_by_pair[(u, v)].append(s)

    def creates_q3_fast(u, v, adj_mask):
        if u > v:
            u, v = v, u
        relevant_subsets = subsets_by_pair.get((u, v), [])
        for subset in relevant_subsets:
            if is_isomorphic_to_q3(subset, adj_mask):
                return True
        return False

    # Local Search Strategy
    # 1. Start with a random dense graph (e.g. 60-70% density)
    # 2. While Q3s exist, remove an edge from a random Q3
    # 3. Try to add random missing edges
    # 4. Repeat
    
    best_count = 0
    best_adj = [0] * n
    best_edges = []
    
    random.seed(42)
    
    # Initial random graph
    for _ in range(50): # 50 restarts
        if time.time() > deadline - 20:
            break
            
        adj_mask = [0] * n
        edges = set()
        
        # Start with ~50% density
        for u, v in all_pairs:
            if random.random() < 0.5:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                edges.add((u, v))
        
        # Phase 1: Remove edges to make Q3-free
        q3_count = count_q3s(adj_mask)
        while q3_count > 0:
            if time.time() > deadline - 10:
                break
            # Find a Q3 and remove a random edge from it
            # We need to find a specific Q3.
            # We can iterate through subsets and find one that is Q3.
            found = False
            for subset in subsets:
                if is_isomorphic_to_q3(subset, adj_mask):
                    # Found a Q3
                    # Get the edges of this Q3 in the graph
                    q3_edges_in_graph = []
                    for i in range(8):
                        for j in range(i+1, 8):
                            u, v = subset[i], subset[j]
                            if (adj_mask[u] >> v) & 1:
                                q3_edges_in_graph.append((u, v))
                    # Remove a random edge
                    if q3_edges_in_graph:
                        u, v = random.choice(q3_edges_in_graph)
                        adj_mask[u] ^= (1 << v)
                        adj_mask[v] ^= (1 << u)
                        edges.discard((u, v))
                    found = True
                    break
            if not found:
                break
            # Recalculate count? Or just decrement?
            # Removing one edge might break multiple Q3s.
            # Let's just recalculate if time permits, or assume it decreases.
            # For speed, let's just re-run the loop condition check.
            # Actually, count_q3s is expensive. Let's just try to remove edges until no Q3 is found in a scan.
            # But we need to know when to stop.
            # Let's just scan for a Q3. If found, remove edge. Repeat.
            # This is effectively the while loop above.
            # To avoid infinite loops, we can limit iterations.
            pass
        
        # Phase 2: Add edges
        missing = [e for e in all_pairs if e not in edges]
        random.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline - 10:
                break
            if not creates_q3_fast(u, v, adj_mask):
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                edges.add((u, v))
                
        # Phase 3: Local search / Hill climbing
        # Try swapping edges? Or just re-adding?
        # Let's try a few rounds of adding/removing
        
        current_count = len(edges)
        if current_count > best_count:
            best_count = current_count
            best_adj = adj_mask[:]
            best_edges = set(edges)
            
        # Try to improve further
        for _ in range(10):
            if time.time() > deadline - 10:
                break
            missing = [e for e in all_pairs if e not in edges]
            random.shuffle(missing)
            added_any = False
            for u, v in missing:
                if not creates_q3_fast(u, v, adj_mask):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    edges.add((u, v))
                    added_any = True
            if not added_any:
                break

    # Convert best_adj to edge list
    result_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj[u] >> v) & 1:
                result_edges.append([u, v])
                
    return result_edges
