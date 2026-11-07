"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = P['n']
    base_edges_set = set()
    for u, v in P['edges']:
        base_edges_set.add((u, v))
        base_edges_set.add((v, u))
        
    # Base adjacency
    base_adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        base_adj[u].add(v)
        base_adj[v].add(u)
    
    # Generate valid candidate edges (not base, not |u-v|=2)
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_edges_set:
                continue
            if abs(u - v) == 2:
                continue
            candidates.append((u, v))
    
    c_len = len(candidates)
    
    # Precompute conflict matrix: conflict[i][j] = True if edges i and j can't coexist
    # (they share a common neighbor in the full graph)
    conflict = [set() for _ in range(c_len)]
    for i in range(c_len):
        ui, vi = candidates[i]
        # Neighbors of u in full graph: base_adj[u] + {v} + other added neighbors of u
        # For conflict check: two edges (u1,v1) and (u2,v2) conflict if they share a vertex
        # or have a common neighbor.
        # Actually, the triangle-free condition: for edge (u,v), no w with w in N(u) and w in N(v).
        # N(u) includes base neighbors and added neighbors.
        # Two added edges (u1,v1) and (u2,v2) create a triangle if:
        #   - They share a vertex (e.g., u1=u2, then edge (u1,v1) and (u1,v2) plus (v1,v2) if exists)
        #   - Actually: triangle (a,b,c) exists if all three edges present.
        # For two added edges (u1,v1) and (u2,v2), they form a triangle with a third edge if:
        #   - u1=u2: then need (v1,v2) edge. v1≠v2. (v1,v2) could be base or added.
        #   - u1=v2: then need (v1,u2) edge.
        #   - etc.
        # Simpler: after building the full graph, check all triangles.
        # For pruning: if two edges share a vertex, and the two other endpoints are adjacent (base or each other), conflict.
        for j in range(i + 1, c_len):
            u2, v2 = candidates[j]
            # Check if edges (u1,v1) and (u2,v2) form a triangle with some third edge
            # Triangle vertices must be 3 distinct vertices from {u1,v1,u2,v2}
            # Case 1: share one vertex, say u1==u2. Then triangle (u1,v1,v2) if (v1,v2) is edge.
            #   (v1,v2) is edge if in base or if (v1,v2) is also an added edge (but we're checking pairwise).
            #   For pairwise conflict: (v1,v2) in base? or (v1,v2)==(u2,v2)? No, v2 is endpoint.
            #   Actually if u1==u2, the two edges are (u1,v1) and (u1,v2). Triangle needs (v1,v2).
            #   (v1,v2) in base? Check. Also (v1,v2) could be another added edge, but that's a 3-way conflict.
            #   For pairwise: check if (v1,v2) in base_edges_set.
            # Case 2: u1==v2. Edges (u1,v1) and (v2,u2)=(u1,u2). Triangle needs (v1,u2).
            # Case 3: v1==u2. Similar.
            # Case 4: v1==v2. Edges (u1,v1) and (u2,v1). Triangle needs (u1,u2).
            # Case 5: u1==u2 and v1==v2: same edge, not possible (distinct candidates).
            # Case 6: u1==v2 and v1==u2: edges (u1,v1) and (v1,u1) - same edge reversed. Not possible.
            
            shared = False
            third_edge_in_base = False
            
            if u1 == u2:
                # Need (v1, v2)
                if (v1, v2) in base_edges_set or (v2, v1) in base_edges_set:
                    third_edge_in_base = True
            elif u1 == v2:
                # Edges (u1,v1) and (u1,u2). Need (v1,u2)
                if (v1, u2) in base_edges_set or (u2, v1) in base_edges_set:
                    third_edge_in_base = True
            elif v1 == u2:
                # Edges (u1,v1) and (v1,v2). Need (u1,v2)
                if (u1, v2) in base_edges_set or (v2, u1) in base_edges_set:
                    third_edge_in_base = True
            elif v1 == v2:
                # Edges (u1,v1) and (u2,v1). Need (u1,u2)
                if (u1, u2) in base_edges_set or (u2, u1) in base_edges_set:
                    third_edge_in_base = True
            elif u1 == v1 or u2 == v2:
                # Can't happen since u<v in each edge
                pass
            
            if third_edge_in_base:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Additional conflict: an added edge (u,v) conflicts with base edge (u,w) if w is also neighbor of v
    # But base edges are always present. So for edge (u,v), if there's base neighbor w of u that is also base neighbor of v, conflict.
    # We already excluded |u-v|=2. For |u-v|>2, base neighbors don't overlap.
    # But: if u has base neighbor w, and we add (u,v), then w and v must not be adjacent.
    # (w,v) adjacent means |w-v|=1 (base) or (w,v) is another added edge.
    # For pairwise with base: (w,v) in base means |w-v|=1.
    # w is neighbor of u: w=u-1 or w=u+1.
    # So if v==u-2 or v==u+2, conflict (excluded by |u-v|=2).
    # If v==u-1 or v==u+1, it's a base edge (excluded).
    # So no additional pairwise conflicts with base beyond |u-v|=2.
    
    # BFS diameter check
    def check_diameter(adj, limit=4):
        # adj is list of sets
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > limit:
                            return False
                        q.append(v)
            if any(d == -1 for d in dist):
                return False  # Not connected
        return True
    
    # Build adjacency from chosen edges
    def build_adj(chosen_indices):
        adj = [set(s) for s in base_adj]
        for idx in chosen_indices:
            u, v = candidates[idx]
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    # Try k edges using DFS with backtracking
    # Order candidates by "usefulness" - prefer long edges
    # Actually, let's sort by distance descending to cover more ground
    cand_order = sorted(range(c_len), key=lambda i: abs(candidates[i][0] - candidates[i][1]), reverse=True)
    
    def dfs(chosen, remaining, depth, k):
        if time.time() > deadline:
            return None
        if depth == k:
            adj = build_adj(chosen)
            if check_diameter(adj, 4):
                return list(chosen)
            return None
        
        # Try each candidate from remaining
        for pos in range(len(remaining)):
            idx = remaining[pos]
            # Check conflict with all chosen
            if any(idx in conflict[c] for c in chosen):
                continue
            chosen.append(idx)
            result = dfs(chosen, remaining[pos+1:], depth + 1, k)
            if result is not None:
                return result
            chosen.pop()
        return None
    
    # Also try with a pruning: if even with all remaining edges we can't achieve diameter 4, prune.
    # But that's expensive. Let's just do iterative deepening.
    
    best_solution = None
    
    for k in range(1, 20):
        if time.time() > deadline:
            break
        result = dfs([], cand_order, 0, k)
        if result is not None:
            best_solution = result
            break
    
    if best_solution is None:
        # Fallback: try a greedy + local search or return best effort
        # This should not happen for a path of 24
        return []
    
    # Convert to [u, v] lists
    final_added = [[int(candidates[i][0]), int(candidates[i][1])] for i in best_solution]
    
    # Final validation
    added_set = set()
    for u, v in final_added:
        if (u, v) in base_edges_set or (v, u) in base_edges_set:
            return []
        if (u, v) in added_set or (v, u) in added_set:
            return []
        added_set.add((u, v))
        added_set.add((v, u))
    
    final_adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        final_adj[u].add(v)
        final_adj[v].add(u)
    for u, v in final_added:
        final_adj[u].add(v)
        final_adj[v].add(u)
    
    # Check triangle-free
    for u in range(n):
        for v in final_adj[u]:
            if u < v:
                if final_adj[u].intersection(final_adj[v]):
                    return []
    
    # Check connected and diameter
    dist = [-1] * n
    dist[0] = 0
    q = deque([0])
    while q:
        u = q.popleft()
        for v in final_adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    
    if any(d == -1 for d in dist):
        return []
    
    if max(dist) > 4:
        return []
    
    return final_added

# EVOLVE-BLOCK-END
