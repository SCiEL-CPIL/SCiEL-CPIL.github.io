"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin (evaluator kills at 150s)
    
    n = 40
    best_edges = None
    best_count = 0
    
    # Helper: check if adding edge (u,v) creates a C6
    # A C6 exists if there's a path of length 5 from u to v that doesn't use the edge (u,v)
    # Equivalently, in G - {(u,v)}, is there a path of length 5 from u to v?
    # Or: in G, if we add (u,v), a C6 is created iff there exists a path of length 5 from u to v in G (not using edge (u,v))
    
    # For efficiency, we'll use a different approach:
    # When considering adding edge (u,v), check if there's a path of length 5 from u to v in the current graph.
    # A path of length 5 from u to v means: u - a - b - c - d - v where all are distinct.
    
    # Actually, let's use a more direct approach with adjacency matrix and boolean matrix powers.
    # But for 40 vertices, we can do BFS/DFS.
    
    # Let me think about this differently. The standard approach for C6-free graphs:
    # A graph has no C6 iff for every pair of vertices u,v at distance 5, they are not adjacent.
    # Or equivalently, when adding edge (u,v), we need to ensure no path of length 5 exists between u and v.
    
    # Let's use a greedy approach with backtracking for local improvement.
    
    # Start with the known good solution: 8 copies of K5 = 8*10 = 80 edges
    # Then try to add edges greedily.
    
    # Initialize with 8 disjoint K5s
    adj = np.zeros((n, n), dtype=bool)
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                adj[u, v] = True
                adj[v, u] = True
    
    def count_edges(adj):
        return int(np.sum(adj)) // 2
    
    def has_c6_through_edge(adj, u, v):
        """Check if adding edge (u,v) would create a C6.
        This is true iff there exists a path of length 5 from u to v in the current graph
        (i.e., in the graph without edge (u,v))."""
        # We need to find if there's a path u - x1 - x2 - x3 - x4 - v where all vertices are distinct
        # and none of the edges used is (u,v) itself.
        
        # BFS from u up to depth 5, tracking paths
        # Since n=40, this is feasible.
        
        # Use a set to track (vertex, depth) pairs we've visited, but we need to track the full path
        # for distinctness. Actually, for C6 detection, we need a simple path of length 5.
        
        # Alternative: use the fact that a C6 through (u,v) exists iff
        # there are vertices a,b,c,d such that u-a, a-b, b-c, c-d, d-v are all edges,
        # and all of u,v,a,b,c,d are distinct.
        
        # Let's do a BFS/DFS from u to depth 5, looking for v.
        # To ensure simplicity, we track visited nodes in the current path.
        
        # For efficiency with n=40, let's use a recursive DFS with memoization-like pruning.
        
        def dfs(node, depth, visited, target):
            if depth == 5:
                return node == target
            for nb in np.where(adj[node, :])[0]:
                if nb not in visited:
                    visited.add(nb)
                    if dfs(nb, depth + 1, visited, target):
                        visited.discard(nb)
                        return True
                    visited.discard(nb)
            return False
        
        visited = {u, v}  # u is start, v is target (can't be in middle)
        # Wait, v should not be in the path before the end. Let me reconsider.
        # Path: u - x1 - x2 - x3 - x4 - v
        # So x1,x2,x3,x4 must be distinct from u, v, and each other.
        
        visited = {u}
        # x1 can be any neighbor of u except v
        for x1 in np.where(adj[u, :])[0]:
            if x1 == v:
                continue
            visited.add(x1)
            # x2 can be any neighbor of x1 except u, v, x1
            for x2 in np.where(adj[x1, :])[0]:
                if x2 in visited or x2 == v:
                    continue
                visited.add(x2)
                # x3
                for x3 in np.where(adj[x2, :])[0]:
                    if x3 in visited or x3 == v:
                        continue
                    visited.add(x3)
                    # x4
                    for x4 in np.where(adj[x3, :])[0]:
                        if x4 in visited or x4 == v:
                            continue
                        # Check if x4 is adjacent to v
                        if adj[x4, v]:
                            return True
                    visited.discard(x3)
            visited.discard(x1)
        return False
    
    def get_edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    edges.append([i, j])
        return edges
    
    # Greedy edge addition
    current_adj = adj.copy()
    
    # Try to add edges greedily
    all_possible = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.seed(42)
    
    # Multiple random orderings for greedy
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        
        test_adj = current_adj.copy()
        added = 0
        
        candidates = all_possible[:]
        random.shuffle(candidates)
        
        for u, v in candidates:
            if time.time() > deadline - 5:
                break
            if test_adj[u, v]:
                continue
            # Check if adding this edge creates a C6
            if not has_c6_through_edge(test_adj, u, v):
                test_adj[u, v] = True
                test_adj[v, u] = True
                added += 1
        
        edge_count = count_edges(test_adj)
        if edge_count > best_count:
            best_count = edge_count
            best_edges = get_edges_from_adj(test_adj)
            current_adj = test_adj.copy()
        
        if best_count >= 105:
            break
    
    # Try local search: remove an edge and try to add more
    if time.time() < deadline - 10 and best_edges is not None:
        for attempt in range(30):
            if time.time() > deadline - 5:
                break
            
            test_adj = np.array(best_edges and np.zeros((n, n), dtype=bool))
            for e in best_edges:
                test_adj[e[0], e[1]] = True
                test_adj[e[1], e[0]] = True
            
            # Remove a random edge
            if not best_edges:
                continue
            idx = random.randint(0, len(best_edges) - 1)
            u, v = best_edges[idx]
            test_adj[u, v] = False
            test_adj[v, u] = False
            
            # Try to add edges greedily
            candidates = all_possible[:]
            random.shuffle(candidates)
            
            for cu, cv in candidates:
                if time.time() > deadline - 5:
                    break
                if test_adj[cu, cv]:
                    continue
                if not has_c6_through_edge(test_adj, cu, cv):
                    test_adj[cu, cv] = True
                    test_adj[cv, cu] = True
            
            edge_count = count_edges(test_adj)
            if edge_count > best_count:
                best_count = edge_count
                best_edges = get_edges_from_adj(test_adj)
                
                if best_count >= 105:
                    break
    
    if best_edges is None:
        # Fallback to initial solution
        best_edges = get_edges_from_adj(adj)
    
    return best_edges
# EVOLVE-BLOCK-END
