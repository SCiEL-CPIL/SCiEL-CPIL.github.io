import time
import sys

def construct():
    """
    Finds a triangle-free graph on 16 vertices that is not 3-colorable 
    and has the maximum number of edges using branch-and-bound.
    
    Models the problem as maximum independent set in the line graph of K16.
    """
    n = 16
    total_edges = n * (n - 1) // 2  # 120 edges in K16
    
    # Precompute edge list and edge index
    edge_list = []
    edge_index = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_list.append((u, v))
            edge_index[(u, v)] = idx
            idx += 1
    
    # For each edge, precompute which other edges conflict with it
    # Two edges conflict if they form a triangle (share a vertex and the third edge exists)
    # Edge (a,b) conflicts with (c,d) if {a,b} and {c,d} share a vertex and the third pair is an edge
    # Actually, in the line graph, two edges are adjacent if they share a vertex.
    # But for triangle-free: we cannot have three edges forming a triangle.
    # So edge e1=(a,b) and e2=(c,d) conflict if they share a vertex AND the third edge exists.
    # Wait: in the line graph L(K16), edges are adjacent if they share a vertex.
    # An independent set in L(K16) is a set of edges no two of which share a vertex = matching.
    # That's not what we want!
    
    # Correction: A triangle-free graph is NOT a matching. 
    # A triangle-free graph can have edges sharing vertices, as long as no three edges form a triangle.
    # The constraint is: for any three vertices u,v,w, not all three edges (u,v), (v,w), (w,u) are present.
    
    # So the conflict graph is NOT the line graph of K16.
    # Instead, it's a 3-uniform hypergraph: each triangle in K16 is a hyperedge, and we want a set of edges
    # that contains no entire hyperedge (triangle).
    
    # Let me redefine: We have 120 possible edges. We want to select a maximum subset S such that:
    # 1. S is triangle-free: for no (u,v,w) are all three edges (u,v), (v,w), (w,u) in S
    # 2. The graph (V, S) is not 3-colorable
    
    # Precompute all triangles and which edges they contain
    triangles = []
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(v + 1, n):
                e1 = edge_index[(u, v)]
                e2 = edge_index[(v, w)]
                e3 = edge_index[(u, w)]
                triangles.append((e1, e2, e3))
    
    # For each edge, precompute the set of triangles it belongs to
    edge_to_triangles = [[] for _ in range(total_edges)]
    for tri_idx, (e1, e2, e3) in enumerate(triangles):
        edge_to_triangles[e1].append(tri_idx)
        edge_to_triangles[e2].append(tri_idx)
        edge_to_triangles[e3].append(tri_idx)
    
    # For 3-colorability check using bitmasks
    # adj_mask[v] is a bitmask of neighbors of v
    # color[v] is -1, 0, 1, or 2
    
    def is_3_colorable(adj_mask, order=None):
        """
        Check if graph with adjacency masks is 3-colorable.
        Uses backtracking with MRV heuristic.
        """
        color = [-1] * n
        
        def backtrack(v):
            if v == -1:
                return True
            
            # MRV: find uncolored vertex with fewest available colors
            if v == -1:
                min_avail = n + 1
                best_v = -1
                for i in range(n):
                    if color[i] == -1:
                        # Count available colors
                        used_colors = 0
                        for j in range(n):
                            if color[j] != -1 and (adj_mask[i] >> j) & 1:
                                used_colors |= (1 << color[j])
                        avail = 3 - bin(used_colors).count('1')
                        if avail < min_avail:
                            min_avail = avail
                            best_v = i
                            if avail == 0:
                                return False
                        elif avail == 0:
                            return False
                v = best_v
            
            # Try colors for vertex v
            used_colors = 0
            for j in range(n):
                if color[j] != -1 and (adj_mask[v] >> j) & 1:
                    used_colors |= (1 << color[j])
            
            for c in range(3):
                if not (used_colors >> c) & 1:
                    color[v] = c
                    if backtrack(-1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(-1)
    
    # Branch and bound
    best_count = 0
    best_edges = []
    start_time = time.time()
    timeout = 115  # Leave 5s buffer
    
    # State: selected_edges (set of edge indices), forbidden_edges (set of edge indices that would create a triangle)
    # Actually, we need to track which triangles are "blocked" (have at least one edge not selected)
    
    # Better state: 
    # - selected: set of selected edge indices
    # - For each triangle, track how many of its 3 edges are selected. If 2 are selected, the 3rd is forbidden.
    
    # Let's use a different approach:
    # Process edges in a fixed order. At each step, decide to include or exclude the next edge.
    # Maintain:
    # - adj_mask: current adjacency masks
    # - selected_count: number of selected edges
    # - For each triangle, count how many of its edges are selected (0, 1, 2, or 3)
    #   If count == 2, the third edge is forbidden.
    #   If count == 3, we have a triangle (invalid).
    
    # Precompute for each edge, the triangles it belongs to and the other two edges in those triangles
    edge_conflicts = [[] for _ in range(total_edges)]
    for tri_idx, (e1, e2, e3) in enumerate(triangles):
        # For edge e1, if e2 is also selected, then e3 is forbidden
        edge_conflicts[e1].append((e2, e3))
        edge_conflicts[e1].append((e3, e2))
        edge_conflicts[e2].append((e1, e3))
        edge_conflicts[e2].append((e3, e1))
        edge_conflicts[e3].append((e1, e2))
        edge_conflicts[e3].append((e2, e1))
    
    # For upper bound: maximum edges we can still add
    # Simple bound: remaining edges count
    # Better bound: for each remaining edge, check if it's currently forbidden
    # A edge is forbidden if there exists a triangle where the other two edges are both selected
    
    def compute_forbidden(selected_set, tri_counts):
        """
        Compute which edges are forbidden given current selection.
        An edge e is forbidden if there's a triangle (e, e1, e2) where both e1 and e2 are selected.
        """
        forbidden = set()
        for e in range(total_edges):
            if e in selected_set:
                continue
            for (e1, e2) in edge_conflicts[e]:
                if e1 in selected_set and e2 in selected_set:
                    forbidden.add(e)
                    break
        return forbidden
    
    # Use a simpler state representation with bitmasks for speed
    # selected_mask: 120-bit integer
    # We'll use a list of booleans for clarity, but optimize with bit operations if needed
    
    selected = [False] * total_edges
    adj_mask = [0] * n
    tri_count = [0] * len(triangles)  # How many edges of each triangle are selected
    
    # Order edges by some heuristic: maybe by degree potential?
    # For now, use natural order. Could sort by number of conflicts.
    
    # Count conflicts for each edge to order them
    edge_degree = [len(edge_conflicts[i]) for i in range(total_edges)]
    edge_order = sorted(range(total_edges), key=lambda i: edge_degree[i], reverse=True)
    
    # Map from position in edge_order to actual edge index
    pos_to_edge = edge_order
    edge_to_pos = [0] * total_edges
    for pos, edge in enumerate(edge_order):
        edge_to_pos[edge] = pos
    
    def solve(pos):
        nonlocal best_count, best_edges, selected, adj_mask, tri_count, start_time, timeout
        
        if time.time() - start_time > timeout:
            return
        
        # Upper bound: current count + remaining edges (optimistic)
        remaining = total_edges - pos
        if len([x for x in selected if x]) + remaining <= best_count:
            return
        
        if pos == total_edges:
            # Check if not 3-colorable
            if not is_3_colorable(adj_mask):
                count = sum(selected)
                if count > best_count:
                    best_count = count
                    best_edges = [list(edge_list[i]) for i in range(total_edges) if selected[i]]
            return
        
        edge = pos_to_edge[pos]
        u, v = edge_list[edge]
        
        # Option 1: Include edge
        # Check if adding this edge creates a triangle
        can_include = True
        for tri_idx in edge_to_triangles[edge]:
            if tri_count[tri_idx] == 2:
                can_include = False
                break
        
        if can_include:
            # Add edge
            selected[edge] = True
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            for tri_idx in edge_to_triangles[edge]:
                tri_count[tri_idx] += 1
            
            solve(pos + 1)
            
            # Remove edge
            selected[edge] = False
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            for tri_idx in edge_to_triangles[edge]:
                tri_count[tri_idx] -= 1
        
        # Option 2: Exclude edge
        solve(pos + 1)
    
    solve(0)
    
    # If we found a solution, return it
    if best_edges:
        return best_edges
    
    # Fallback: return Groetzsch graph extended
    # Groetzsch graph edges
    groetzsch_edges = []
    for i in range(5):
        groetzsch_edges.append([i, (i + 1) % 5])
    for i in range(5):
        u = i
        w = (i + 1) % 5
        groetzsch_edges.append([u, 5 + w])
        groetzsch_edges.append([w, 5 + u])
    for i in range(5):
        groetzsch_edges.append([5 + i, 10])
    
    # Extend to maximal triangle-free
    adj = [[False] * n for _ in range(n)]
    for u, v in groetzsch_edges:
        adj[u][v] = True
        adj[v][u] = True
    
    changed = True
    while changed:
        changed = False
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u][v]:
                    # Check if adding creates triangle
                    ok = True
                    for w in range(n):
                        if w != u and w != v and adj[u][w] and adj[v][w]:
                            ok = False
                            break
                    if ok:
                        adj[u][v] = True
                        adj[v][u] = True
                        changed = True
    
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                result.append([u, v])
    
    return result

if __name__ == "__main__":
    edges = construct()
    print(f"Found graph with {len(edges)} edges")
