import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 16
    # Precompute all pairs
    all_pairs = list(combinations(range(n), 2))
    pair_to_idx = {p: i for i, p in enumerate(all_pairs)}
    
    # Precompute Q3 subgraphs
    # A Q3 is an 8-vertex induced subgraph isomorphic to the 3-cube.
    # We can represent a potential Q3 by its 8 vertices.
    # There are C(16, 8) = 12870 ways to choose 8 vertices.
    # For each set of 8 vertices, we check if the induced subgraph is Q3.
    # But checking isomorphism for every step is slow.
    
    # Better approach:
    # A Q3 is bipartite with parts of size 4.
    # The vertices of Q3 can be identified by their adjacency structure.
    # However, enumerating all subgraphs isomorphic to Q3 in a general graph is hard.
    
    # Let's use a different bitset trick.
    # We want to check if a graph G contains a Q3.
    # A Q3 has 8 vertices and 12 edges.
    # We can precompute all 12870 subsets of 8 vertices.
    # For a given graph (represented as a set of edges or bitmasks),
    # we check if any of these subsets induces a Q3.
    
    # To speed up checking:
    # Precompute for each subset S of 8 vertices, the set of edges that *would* form a Q3
    # is not fixed, because the labeling of the cube vertices varies.
    # There are 48 automorphisms of the cube.
    # So for a fixed set of 8 vertices, there are 48 distinct edge sets that form a Q3 on those vertices.
    # Total "Q3 patterns" = 12870 * 48 = 617,760.
    # This is too many to check individually per step in a naive way.
    
    # Alternative:
    # Use a recursive backtracking search on edges.
    # State: current set of edges.
    # We want to maximize edges such that no Q3 is present.
    # We can order edges and decide include/exclude.
    # Pruning: if current edges + all remaining edges still don't allow a Q3? No, that's for finding a Q3.
    # We want to AVOID Q3s.
    # So, we check if adding an edge creates a Q3.
    # If adding edge e creates a Q3, we MUST exclude e.
    # This is a constraint satisfaction problem.
    
    # Let's try a greedy approach with backtracking limited to a small depth,
    # or simply a fast heuristic that is better than K3,13.
    
    # The inspirations mention scores of 1.38 (approx 54 edges) and 1.28.
    # K3,13 has 39 edges.
    # A random graph G(16, p) with p=0.6 has expected edges 120*0.6 = 72.
    # Does G(16, 0.6) contain a Q3?
    # Number of Q3s in K16 is 12870 * 48 = 617760.
    # Probability a specific Q3 is present in G(16, p) is p^12.
    # Expected number of Q3s = 617760 * p^12.
    # For p=0.6, p^12 ~ 0.002. Expected Q3s ~ 1236. Very likely to have Q3.
    # For p=0.5, p^12 ~ 0.00024. Expected Q3s ~ 148.
    # For p=0.4, p^12 ~ 0.000016. Expected Q3s ~ 9.8.
    # For p=0.3, p^12 ~ 0.00000053. Expected Q3s ~ 0.33.
    # So a graph with ~36-40 edges (p~0.3-0.33) might be Q3-free.
    # But we want to MAXIMIZE edges.
    # The Turan number ex(16, Q3) is likely higher than 39.
    
    # Let's implement a fast local search.
    # Start with a random dense graph.
    # While it contains a Q3, remove an edge from a Q3.
    # Then try to add edges back.
    
    def contains_q3(adj_matrix):
        # Check all 8-vertex subsets
        # This is expensive if done naively.
        # We can use bitsets for adjacency.
        # adj[i] is a bitmask of neighbors of i.
        
        # For a set of 8 vertices V, we need to check if the induced subgraph is Q3.
        # Q3 is bipartite with parts A, B of size 4.
        # Every vertex in A is connected to exactly 3 vertices in B.
        # Every vertex in B is connected to exactly 3 vertices in A.
        # No edges within A or within B.
        
        # So, for a given set of 8 vertices:
        # 1. Check if it's bipartite. If not, return False.
        # 2. Check if the bipartition has sizes 4 and 4. If not, return False.
        # 3. Check if every vertex has degree 3 within the induced subgraph. If not, return False.
        # If all pass, it's a Q3.
        
        # To find a Q3, we iterate through all 12870 subsets.
        # We can optimize by precomputing the subsets? No, 12870 is small enough.
        
        for combo in combinations(range(n), 8):
            # Check bipartiteness and degrees
            # Let's try to color the subgraph.
            # Since it's small, we can just check all 2^8 colorings? No, that's 256.
            # Or just BFS.
            
            # Let's extract the induced subgraph edges.
            # Actually, we can check the conditions directly.
            
            # Condition: Bipartite with parts of size 4.
            # This implies the subgraph is a 3-regular bipartite graph with 8 vertices.
            # There is only one such graph: Q3.
            
            # So we just need to check:
            # 1. Is the induced subgraph bipartite?
            # 2. If yes, are the partition sizes 4 and 4?
            # 3. Is every vertex degree 3?
            
            # Let's do a quick check.
            # We can check if the number of edges is 12. If not, it's not Q3.
            edge_count = 0
            for u in combo:
                for v in combo:
                    if u < v and (adj_matrix[u] >> v) & 1:
                        edge_count += 1
            if edge_count != 12:
                continue
                
            # Check bipartiteness
            # Try to 2-color the graph.
            # Since it's sparse, BFS is fast.
            colors = [-1] * 8
            combo_to_idx = {v: i for i, v in enumerate(combo)}
            
            found_bipartite = False
            # Try both starting colors for the first node
            for start_color in [0, 1]:
                colors[0] = start_color
                queue = [0]
                valid = True
                while queue:
                    u_idx = queue.pop(0)
                    u = combo[u_idx]
                    for v_idx in range(8):
                        if v_idx == u_idx: continue
                        v = combo[v_idx]
                        if (adj_matrix[u] >> v) & 1:
                            if colors[v_idx] == -1:
                                colors[v_idx] = 1 - colors[u_idx]
                                queue.append(v_idx)
                            elif colors[v_idx] == colors[u_idx]:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    # Check partition sizes
                    c0 = sum(1 for c in colors if c == 0)
                    c1 = sum(1 for c in colors if c == 1)
                    if c0 == 4 and c1 == 4:
                        # Check degrees
                        deg_ok = True
                        for u_idx in range(8):
                            u = combo[u_idx]
                            deg = 0
                            for v_idx in range(8):
                                if u_idx != v_idx:
                                    v = combo[v_idx]
                                    if (adj_matrix[u] >> v) & 1:
                                        deg += 1
                            if deg != 3:
                                deg_ok = False
                                break
                        if deg_ok:
                            return True, list(combo)
            # If we get here, no valid bipartition found for this combo
            pass
            
        return False, []

    # Initialize with a random graph
    # We want a dense graph. Let's start with p=0.5
    random.seed(42)
    adj = [0] * n
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append((i, j))
    
    best_edges = list(edges)
    best_count = len(edges)
    
    # Remove Q3s
    while True:
        has_q3, q3_vertices = contains_q3(adj)
        if not has_q3:
            break
        # Remove an edge from the Q3
        # Pick a random edge from the Q3
        u, v = random.choice(combinations(q3_vertices, 2))
        # Check if edge exists
        if (adj[u] >> v) & 1:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            edges.remove((u, v))
        else:
            # Should not happen if contains_q3 is correct, but safeguard
            # Force removal of any edge in the induced subgraph?
            # If contains_q3 says it's a Q3, the edges must be there.
            # Let's just break to avoid infinite loop
            break
            
    if len(edges) > best_count:
        best_edges = list(edges)
        best_count = len(edges)
        
    # Try to add edges back
    # Iterate through all missing edges
    missing = [p for p in all_pairs if p not in edges]
    random.shuffle(missing)
    
    for u, v in missing:
        if time.time() > deadline:
            break
        # Try adding edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edges.append((u, v))
        
        has_q3, _ = contains_q3(adj)
        if has_q3:
            # Remove it
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            edges.remove((u, v))
        else:
            if len(edges) > best_count:
                best_edges = list(edges)
                best_count = len(edges)
                
    # Return best found
    return [list(e) for e in best_edges]
# EVOLVE-BLOCK-END
