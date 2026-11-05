import time
import itertools

# Q3 vertices: 0..7
# Edges of Q3:
# 0-1, 1-2, 2-3, 3-0 (square)
# 4-5, 5-6, 6-7, 7-4 (square)
# 0-4, 1-5, 2-6, 3-7 (connections)
Q3_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]
Q3_ADJ = {i: set() for i in range(8)}
for u, v in Q3_EDGES:
    Q3_ADJ[u].add(v)
    Q3_ADJ[v].add(u)

def contains_q3_in_8(vertices, graph_edges):
    """
    Check if the induced subgraph on 'vertices' (list of 8 vertex ids) 
    contains Q3 as a subgraph.
    graph_edges: set of edges (u,v) in the main graph.
    """
    # Map vertex ids to 0..7 for local indexing
    v_map = {v: i for i, v in enumerate(vertices)}
    inv_map = {i: v for i, v in enumerate(vertices)}
    
    # Build local adjacency
    local_adj = {i: set() for i in range(8)}
    for u, v in graph_edges:
        if u in v_map and v in v_map:
            lu = v_map[u]
            lv = v_map[v]
            local_adj[lu].add(lv)
            local_adj[lv].add(lu)
            
    # Check if Q3 is a subgraph of local_adj
    # Try all permutations of Q3 vertices to local vertices?
    # 8! is 40320. 12870 * 40320 is too much if done naively.
    # But we only call this for subsets containing a specific edge.
    # And we can prune.
    
    # Better: Backtracking isomorphism check.
    # Q3 is 3-regular.
    # If local_adj has any vertex with degree < 3, it can't contain Q3?
    # No, Q3 is a subgraph, not induced. So degrees in local_adj must be >= degrees in Q3.
    # So all 8 vertices must have degree >= 3 in local_adj.
    for i in range(8):
        if len(local_adj[i]) < 3:
            return False
            
    # Now check isomorphism.
    # We can use a simple recursive check.
    # Map Q3 vertices to local vertices.
    # Order Q3 vertices by degree (all 3).
    # Just try to map 0,1,2,3,4,5,6,7.
    
    # Optimization: Use the structure.
    # Pick a vertex in Q3, say 0. Map to a vertex in local.
    # Try all 8 possibilities.
    
    def is_subgraph(mapping, q3_idx):
        if q3_idx == 8:
            # Check if all Q3 edges exist in local_adj
            for u, v in Q3_EDGES:
                lu = mapping[u]
                lv = mapping[v]
                if lv not in local_adj[lu]:
                    return False
            return True
            
        # Heuristic: pick next Q3 vertex.
        # We can just use 0,1,2...
        # But we need to ensure consistency.
        
        # Actually, since all degrees are 3, we can just try all permutations?
        # 8! is 40320.
        # Number of calls to this function:
        # In the main search, we add one edge. We check ~3000 subsets.
        # 3000 * 40320 = 120 million operations. Too slow.
        
        # We need a faster check.
        # Count edges. Q3 has 12 edges.
        # If local_adj has < 12 edges, return False.
        edge_count = sum(len(s) for s in local_adj.values()) // 2
        if edge_count < 12:
            return False
            
        # Use a bipartite check? Q3 is bipartite.
        # If local_adj is not bipartite, it can still contain Q3 (if it has odd cycles elsewhere).
        # But Q3 itself is bipartite.
        
        # Let's use the fact that Q3 is the only 3-regular bipartite graph on 8 vertices?
        # No, $K_{4,4}$ is 4-regular.
        # But if we restrict to subgraphs with 8 vertices and 12 edges...
        
        # Let's just do the permutation check but with pruning.
        # Map Q3 vertices 0..7 to local vertices.
        # We can fix the mapping of 0 to any of the 8 local vertices.
        # Then 1 must be a neighbor of 0 in Q3, so map to a neighbor of mapped(0) in local.
        
        # This is a standard backtracking isomorphism.
        
        used = [False] * 8
        mapping = [None] * 8
        
        def backtrack(q3_v):
            if q3_v == 8:
                # Verify all edges
                for u, v in Q3_EDGES:
                    if mapping[v] not in local_adj[mapping[u]]:
                        return False
                return True
                
            # Try all unused local vertices for q3_v
            for l_v in range(8):
                if not used[l_v]:
                    # Check consistency with already mapped neighbors
                    ok = True
                    for nb in Q3_ADJ[q3_v]:
                        if mapping[nb] is not None:
                            if mapping[nb] not in local_adj[l_v]:
                                ok = False
                                break
                    if ok:
                        used[l_v] = True
                        mapping[q3_v] = l_v
                        if backtrack(q3_v + 1):
                            return True
                        used[l_v] = False
                        mapping[q3_v] = None
            return False
            
        return backtrack(0)

    return False

def construct():
    start_time = time.time()
    
    # Initial graph: K_{3,13}
    # Vertices 0,1,2 in A, 3..15 in B
    edges = set()
    for a in range(3):
        for b in range(3, 16):
            edges.add((a, b) if a < b else (b, a))
            
    # Candidates: all edges not in edges
    all_edges = set()
    for u in range(16):
        for v in range(u+1, 16):
            all_edges.add((u, v))
            
    candidates = list(all_edges - edges)
    
    # Heuristic: sort candidates? 
    # Edges within B (3..15) are 78. Edges within A (0..2) are 3.
    # Adding edges in B is more likely to be safe?
    # Let's try edges in B first, then A.
    def edge_type(e):
        u, v = e
        if u < 3 and v < 3:
            return 1 # A-A
        elif u >= 3 and v >= 3:
            return 0 # B-B
        else:
            return 2 # A-B (should not be in candidates)
            
    candidates.sort(key=edge_type)
    
    best_edges = set(edges)
    best_count = len(edges)
    
    def is_q3_free_with_new_edge(current_edges, new_edge):
        u, v = new_edge
        # Check all 8-vertex subsets containing u and v
        # Other 6 vertices from remaining 14
        other_vertices = [x for x in range(16) if x != u and x != v]
        for combo in itertools.combinations(other_vertices, 6):
            vertices = list(combo) + [u, v]
            # Check if induced subgraph on these 8 vertices contains Q3
            # We can pass the full current_edges + new_edge
            # But contains_q3_in_8 filters by vertices
            if contains_q3_in_8(vertices, current_edges | {new_edge}):
                return False
        return True

    def search(current_edges, candidates):
        nonlocal best_edges, best_count
        
        if time.time() - start_time > 110:
            return
            
        if not candidates:
            if len(current_edges) > best_count:
                best_edges = set(current_edges)
                best_count = len(current_edges)
            return
            
        # Pick a candidate
        e = candidates[0]
        rest = candidates[1:]
        
        # Try adding e
        if is_q3_free_with_new_edge(current_edges, e):
            new_edges = set(current_edges)
            new_edges.add(e)
            search(new_edges, rest)
            
        # Try skipping e
        search(current_edges, rest)
        
    search(edges, candidates)
    
    # Return best found
    return [list(e) for e in best_edges]

# To ensure we don't time out on the first run, we might need to limit the search depth or use a faster check.
# The current check is O(3000 * 8!) which is too slow.
# We need to optimize contains_q3_in_8.

# Let's replace contains_q3_in_8 with a faster check.
# We know Q3 is bipartite.
# We can check if the 8-vertex graph has a bipartite subgraph isomorphic to Q3.
# Actually, just checking if the 8-vertex graph *is* Q3 or contains it.
# Given the small size, we can precompute all 8-vertex graphs that contain Q3? No.

# Alternative: Use a library? No, standard library only.
# Let's optimize the backtracking in contains_q3_in_8.
# The degree check < 3 is good.
# The edge count < 12 is good.
# The backtracking is the bottleneck.
# We can prune more aggressively.
# In Q3, any two non-adjacent vertices have 2 common neighbors.
# In local_adj, if we map u, v to lu, lv, and they are not adjacent in Q3, they must have at least 2 common neighbors in local_adj.

# Let's rewrite contains_q3_in_8 with better pruning.

def contains_q3_in_8_optimized(vertices, graph_edges):
    v_map = {v: i for i, v in enumerate(vertices)}
    local_adj = {i: set() for i in range(8)}
    for u, v in graph_edges:
        if u in v_map and v in v_map:
            lu = v_map[u]
            lv = v_map[v]
            local_adj[lu].add(lv)
            local_adj[lv].add(lu)
            
    # Degree check
    for i in range(8):
        if len(local_adj[i]) < 3:
            return False
            
    edge_count = sum(len(s) for s in local_adj.values()) // 2
    if edge_count < 12:
        return False
        
    # Backtracking with pruning
    used = [False] * 8
    mapping = [None] * 8
    
    # Order Q3 vertices. 0 is connected to 1,3,4.
    # We can fix 0 to 0? No, we must try all.
    # But we can fix 0 to the first available vertex to reduce symmetry?
    # No, the graph might not be symmetric.
    
    def backtrack(q3_v):
        if q3_v == 8:
            for u, v in Q3_EDGES:
                if mapping[v] not in local_adj[mapping[u]]:
                    return False
            return True
            
        for l_v in range(8):
            if not used[l_v]:
                # Check neighbors already mapped
                ok = True
                for nb in Q3_ADJ[q3_v]:
                    if mapping[nb] is not None:
                        if mapping[nb] not in local_adj[l_v]:
                            ok = False
                            break
                if ok:
                    # Additional pruning: 
                    # If q3_v is not 0, check non-neighbors?
                    # Q3 is 3-regular. Non-neighbors are 4 vertices.
                    # In Q3, any non-neighbor pair has 2 common neighbors.
                    # This is complex to check dynamically.
                    
                    used[l_v] = True
                    mapping[q3_v] = l_v
                    if backtrack(q3_v + 1):
                        return True
                    used[l_v] = False
                    mapping[q3_v] = None
        return False
        
    return backtrack(0)

# Update the main function to use the optimized check.

def construct():
    start_time = time.time()
    
    edges = set()
    for a in range(3):
        for b in range(3, 16):
            edges.add((a, b) if a < b else (b, a))
            
    all_edges = set()
    for u in range(16):
        for v in range(u+1, 16):
            all_edges.add((u, v))
            
    candidates = list(all_edges - edges)
    
    def edge_type(e):
        u, v = e
        if u < 3 and v < 3:
            return 1
        elif u >= 3 and v >= 3:
            return 0
        else:
            return 2
            
    candidates.sort(key=edge_type)
    
    best_edges = set(edges)
    best_count = len(edges)
    
    def is_q3_free_with_new_edge(current_edges, new_edge):
        u, v = new_edge
        other_vertices = [x for x in range(16) if x != u and x != v]
        for combo in itertools.combinations(other_vertices, 6):
            vertices = list(combo) + [u, v]
            if contains_q3_in_8_optimized(vertices, current_edges | {new_edge}):
                return False
        return True

    def search(current_edges, candidates):
        nonlocal best_edges, best_count
        
        if time.time() - start_time > 110:
            return
            
        if not candidates:
            if len(current_edges) > best_count:
                best_edges = set(current_edges)
                best_count = len(current_edges)
            return
            
        e = candidates[0]
        rest = candidates[1:]
        
        if is_q3_free_with_new_edge(current_edges, e):
            new_edges = set(current_edges)
            new_edges.add(e)
            search(new_edges, rest)
            
        search(current_edges, rest)
        
    search(edges, candidates)
    
    return [list(e) for e in best_edges]
