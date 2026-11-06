import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    def time_check():
        return time.time() > deadline - 5.0

    # We aim to find a valid coloring for n=17.
    # If we find one, we return it.
    # The known solution exists. We use a CSP approach to find it efficiently.
    
    n = 17
    edges = []
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            edge_idx[(i, j)] = idx
            idx += 1
            
    num_edges = len(edges)
    
    # Precompute for each edge (u, v), the list of other edges that share a vertex with it.
    # This helps in forward checking.
    # Actually, for C6 detection, we need to check if adding an edge (u,v) of color c
    # completes a C6. A C6 containing (u,v) is a path of length 5 from u to v in color c.
    # In a partial coloring, we can check if there is a path of length 5 in the *currently assigned* 
    # edges of color c. If the path uses only assigned edges, and the endpoints are u and v, 
    # and (u,v) is now assigned c, then we have a C6.
    
    # However, forward checking is hard here because the "domain" of an edge is just {0,1,2,3}.
    # The constraint is global.
    
    # Better approach: Backtracking search with a strong heuristic.
    # Order edges by some heuristic (e.g., degree in the graph of uncolored edges? or just fixed order).
    # Since n=17 is small, a well-guided backtracking search should find the solution quickly.
    # The solution space is 4^136, which is huge, but the constraint "no C6" is very tight.
    
    # Let's use a randomized restart backtracking search.
    # Variable ordering: We will pick the next edge to color based on a heuristic.
    # Heuristic: Pick the edge (u,v) such that for the most colors c, 
    # assigning (u,v) color c would create a C6. 
    # This is the "Minimum Remaining Values" (MRV) heuristic.
    
    # To compute MRV efficiently, we need to know for each uncolored edge (u,v) and each color c,
    # if there is a path of length 5 from u to v in color c using only *already colored* edges.
    # If such a path exists, color c is forbidden for (u,v).
    
    # State:
    # colors: list of length num_edges, -1 if uncolored, else 0-3.
    # adj: list of 4 sets, adj[c] contains edges (u,v) colored c.
    # path_counts: We can maintain counts of paths of length k between all pairs for each color.
    # But updating path counts incrementally for all pairs is expensive.
    
    # Alternative: Just check for C6 completion on the fly when assigning an edge.
    # When assigning edge (u,v) color c:
    # 1. Check if there is a path of length 5 from u to v in color c using current adj[c].
    #    If yes, this assignment is invalid.
    # 2. If valid, assign it.
    
    # To speed up the path check, we can use BFS/DFS from u in adj[c] up to depth 5, 
    # and see if we reach v.
    # Since the graph is sparse (average degree ~ 17*3/4 = 12.75 initially, but grows),
    # and depth is small (5), this is fast.
    
    # We also need to ensure we don't get stuck. We will use randomized restarts.
    
    def has_path_length_5(u, v, adj_c):
        """Check if there is a path of length 5 from u to v in graph adj_c."""
        # BFS from u
        # dist[v] = min steps
        # We need exactly 5? No, "path of length 5" in the context of C6 completion
        # means a simple path of 5 edges connecting u and v.
        # If there is a shorter path, it doesn't form a C6 with (u,v).
        # But wait, if there is a path of length 3, and we add (u,v), we get a C4, not C6.
        # The constraint is specifically C6.
        # So we need a simple path of length 5.
        
        # BFS to find shortest path? No, we need *any* simple path of length 5.
        # Actually, if the shortest path is < 5, there might still be a longer simple path of length 5.
        # However, in a dense graph, paths are abundant.
        # For n=17, the diameter is small.
        
        # Let's just do a DFS with depth limit 5.
        # visited set to ensure simple path.
        
        def dfs(current, target, depth, visited):
            if depth == 5:
                return current == target
            for next_node in adj_c[current]:
                if next_node not in visited:
                    visited.add(next_node)
                    if dfs(next_node, target, depth + 1, visited):
                        return True
                    visited.remove(next_node)
            return False
            
        return dfs(u, v, 0, {u})

    def is_valid_assignment(u, v, c, adj):
        """Check if assigning edge (u,v) color c creates a C6."""
        # A C6 is formed if there is a path of length 5 from u to v in color c.
        # The path must not use (u,v) itself (which is not in adj[c] yet).
        # So we just check if u and v are connected by a path of length 5 in adj[c].
        return has_path_length_5(u, v, adj[c])

    def backtrack(colors, adj, edge_order, idx):
        """
        Recursive backtracking search.
        colors: list of assigned colors (-1 for unassigned)
        adj: list of sets of edges for each color
        edge_order: list of edge indices to process
        idx: current index in edge_order
        """
        if time_check():
            return None
            
        if idx == len(edge_order):
            return colors[:]
            
        e_idx = edge_order[idx]
        u, v = edges[e_idx]
        
        # Try colors in random order? Or fixed?
        # Let's try fixed order 0,1,2,3 but shuffle the edge order in restarts.
        
        for c in range(4):
            if colors[e_idx] != -1:
                continue # Should not happen if we pick unassigned
            
            # Check validity
            if not is_valid_assignment(u, v, c, adj):
                # Assign
                colors[e_idx] = c
                adj[c].add((u, v))
                
                # Recurse
                result = backtrack(colors, adj, edge_order, idx + 1)
                if result is not None:
                    return result
                
                # Unassign
                colors[e_idx] = -1
                adj[c].remove((u, v))
                
        return None

    # We will use randomized restarts with different edge orderings.
    # Edge ordering heuristic: 
    # 1. Random shuffle.
    # 2. Or, prioritize edges that are "central" in the graph.
    
    # Let's try a few restarts.
    for restart in range(100):
        if time_check():
            break
            
        # Random edge order
        edge_order = list(range(num_edges))
        random.shuffle(edge_order)
        
        # Initialize
        colors = [-1] * num_edges
        adj = [set() for _ in range(4)]
        
        # Run backtracking
        result = backtrack(colors, adj, edge_order, 0)
        
        if result is not None:
            # Verify the result
            # Re-check for C6s to be safe
            adj_final = [set() for _ in range(4)]
            for i in range(num_edges):
                u, v = edges[i]
                c = result[i]
                adj_final[c].add((u, v))
                
            valid = True
            for c in range(4):
                for u, v in adj_final[c]:
                    if has_path_length_5(u, v, adj_final[c]):
                        valid = False
                        break
                if not valid:
                    break
                    
            if valid:
                return {'n': n, 'colors': result}
                
    # If backtracking fails, fall back to the cyclic search from the previous version?
    # The prompt asks for a DIFFERENT strategy. 
    # If the CSP fails, we should try to return the best known valid one if we can find it quickly.
    # The cyclic search is very fast (4^8 iterations).
    # Let's include it as a fallback if the CSP doesn't find a solution in time.
    
    # Cyclic Z_17 Search
    pairs = [(1, 16), (2, 15), (3, 14), (4, 13), (5, 12), (6, 11), (7, 10), (8, 9)]
    
    def has_c6_in_diffs(D):
        D_set = set(D)
        for v1 in D_set:
            for v2 in [(v1 + d) % 17 for d in D_set]:
                if v2 == 0 or v2 == v1: continue
                for v3 in [(v2 + d) % 17 for d in D_set]:
                    if v3 in (0, v1, v2): continue
                    for v4 in [(v3 + d) % 17 for d in D_set]:
                        if v4 in (0, v1, v2, v3): continue
                        for v5 in [(v4 + d) % 17 for d in D_set]:
                            if v5 in (0, v1, v2, v3, v4): continue
                            if (-v5) % 17 in D_set:
                                return True
        return False

    for assignment in range(4**8):
        if time_check(): break
        
        colors_for_pairs = [0] * 8
        temp = assignment
        for i in range(8):
            colors_for_pairs[i] = temp % 4
            temp //= 4
        
        diff_sets = [set() for _ in range(4)]
        for i, c in enumerate(colors_for_pairs):
            p = pairs[i]
            diff_sets[c].add(p[0])
            diff_sets[c].add(p[1])
        
        valid = True
        for c in range(4):
            if has_c6_in_diffs(diff_sets[c]):
                valid = False
                break
        
        if valid:
            color_map = {}
            for c in range(4):
                for d in diff_sets[c]:
                    color_map[d] = c
            
            n = 17
            colors_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    diff = (j - i) % 17
                    colors_list.append(color_map[diff])
            
            # Quick verify
            adj_final = [set() for _ in range(4)]
            for i in range(136):
                u, v = edges[i]
                c = colors_list[i]
                adj_final[c].add((u, v))
                
            valid_final = True
            for c in range(4):
                for u, v in adj_final[c]:
                    if has_path_length_5(u, v, adj_final[c]):
                        valid_final = False
                        break
                if not valid_final:
                    break
                    
            if valid_final:
                return {'n': n, 'colors': colors_list}

    # If all else fails, return a small valid coloring
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
