"""Rewritten program: deterministic construction of a 4-coloring of K50 with no monochromatic triangle.

Strategy:
- Start from the known Greenwood-Gleason coloring of K16 (Clebsch graph × 3 colors).
- Build K32 by doubling with color 3 on cross edges (valid: bipartite color class).
- Extend to K50 by adding 18 new vertices. For each new vertex, we need to assign
  colors to edges connecting it to the existing 32 vertices such that no monochromatic
  triangle is formed.

Key insight for extension:
- For a new vertex v, and for each color c, the set of existing vertices u where
  edge (u,v) gets color c must form an independent set in the subgraph of color c
  among the existing vertices. Equivalently, for each color c, the neighbors of v
  in color c must be a clique-free set in the c-colored graph.

- The Greenwood-Gleason K16 has the property that each color class is the Clebsch
  graph (a strongly regular graph with parameters (16,5,0,2)). The Clebsch graph
  has no triangles, so any independent set in the Clebsch graph can be the color-c
  neighborhood of a new vertex.

- We use a deterministic greedy approach: for each new vertex, try all 4^32 possible
  color assignments? No, that's too many. Instead, we use the structure.

Alternative structural approach:
- The known 50-vertex construction by Chung uses a specific combinatorial design.
- Let me use a different well-known construction: the one based on the 50-vertex
  graph from the literature.

Actually, let me use a simpler approach: I'll hardcode a known valid 50-vertex coloring.
But that's not ideal. Let me instead construct it algorithmically.

Better approach: Use the fact that we can take two copies of K25 if we have a valid
K25 coloring, but we don't. 

Let me use the following constructive method:
1. Start with K16 (Greenwood-Gleason).
2. Add 16 more vertices to make K32 (doubling with color 3).
3. Add 2 more vertices to make K34, then continue adding vertices one by one
   using a backtracking search with strong pruning.

For adding a single vertex to an existing k-vertex coloring:
- For each color c ∈ {0,1,2,3}, define S_c = {u in existing vertices : edge(u, new) = c}
- Constraint: S_c must be an independent set in the c-colored subgraph of existing vertices.
  (Because if u,w ∈ S_c and edge(u,w) has color c, then (u,w,new) is a monochromatic triangle.)

So for each color c, S_c must be a subset of the existing vertices that is independent
in the c-colored graph.

For the K32 construction:
- Within each K16 block, color c forms the Clebsch graph (no triangles, so any independent set works).
- Cross edges are all color 3. So for color 3, the "graph" on the 32 vertices has all cross edges
  plus the Clebsch edges within each block. We need S_3 to be independent in this graph.

Let me implement a backtracking search to add vertices one at a time, with the constraint
that for each color, the neighbor set must be independent in that color's subgraph.

To make this efficient, I'll precompute for each color c and each existing vertex v, the set
of vertices that are NOT adjacent to v in color c. Then S_c must be a subset of the non-neighbors
of every vertex in S_c (i.e., an independent set).

I'll use a simple backtracking with forward checking. Given the structure, this should be fast.
"""

import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    def time_left():
        return deadline - time.time()
    
    # Build the base K16 Greenwood-Gleason coloring
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    
    # Build K32 by doubling
    n_base = 32
    colors = [0] * (n_base * (n_base - 1) // 2)
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if i // 16 == j // 16:
                colors[idx] = log[(i % 16) ^ (j % 16)] % 3
            else:
                colors[idx] = 3
            idx += 1
    
    # Build adjacency: adj[i][j] = color of edge (i,j)
    def build_adj(n, col_list):
        adj = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = col_list[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1
        return adj
    
    adj = build_adj(n_base, colors)
    
    # For each color c, build the "forbidden" sets:
    # For a new vertex, if we assign color c to edge (u, new), then for any w with adj[u][w] == c,
    # we CANNOT assign color c to edge (w, new).
    # So for each color c and vertex u, the set of w where we CANNOT also use color c is:
    #   forbidden[c][u] = {w != u : adj[u][w] == c}
    # Plus, the new vertex itself is not in the existing set.
    
    # Precompute for each color c, for each existing vertex u, the bitmask of vertices w
    # such that adj[u][w] == c. This tells us which vertices are "conflicts" if we use color c at u.
    
    # Use bitmasks for efficiency
    # conflict_mask[c][u] = bitmask of vertices w where adj[u][w] == c
    
    conflict_mask = [[0] * n_base for _ in range(4)]
    for c in range(4):
        for u in range(n_base):
            mask = 0
            for w in range(n_base):
                if w != u and adj[u][w] == c:
                    mask |= (1 << w)
            conflict_mask[c][u] = mask
    
    # Now add vertices one by one using backtracking
    # State: for each new vertex, we need to assign a color to each edge to existing vertices.
    # We'll represent the coloring of a new vertex as a tuple of 32 colors (one per existing vertex).
    
    # For efficiency, we'll use a recursive backtracking that assigns colors to edges
    # of the new vertex one existing vertex at a time, with constraint propagation.
    
    # Actually, let's think differently. For a new vertex, the assignment is a function
    # f: {0..n_base-1} -> {0,1,2,3} such that for each color c, the set {u : f(u) = c}
    # is an independent set in the c-colored graph.
    
    # Independent set in c-colored graph means: for any u, w in S_c, adj[u][w] != c.
    # Equivalently: for any u in S_c, w in S_c, w is NOT in conflict_mask[c][u].
    
    # So the constraint is: for each c, for each u with f(u) = c, we need
    #   f(w) != c for all w in conflict_mask[c][u]
    
    # This is a constraint satisfaction problem. Let's solve it with backtracking.
    
    def add_vertex(n_current, adj_current, conflict_mask_current):
        """
        Try to add one vertex. Returns the color assignment (list of n_current colors)
        or None if impossible.
        """
        n = n_current
        # We'll assign colors to edges (new_vertex, u) for u in 0..n-1
        # Use backtracking with ordering by most-constrained-first
        
        # For each vertex u, the possible colors are those c where
        # the constraint can be satisfied. Initially all 4 colors are possible.
        
        # We'll use a simple DFS with forward checking.
        
        assignment = [None] * n
        
        # Order vertices by degree in the conflict structure (most constrained first)
        # A vertex u is more constrained if it has many conflicts across colors.
        order = list(range(n))
        # Sort by total conflicts (sum over colors of popcount(conflict_mask[c][u]))
        def total_conflicts(u):
            total = 0
            for c in range(4):
                total += bin(conflict_mask_current[c][u]).count('1')
            return total
        order.sort(key=total_conflicts, reverse=True)
        
        # For each position in the order, we assign a color.
        # We maintain for each color c, the set of already-assigned vertices with color c.
        # When assigning color c to vertex u, we check that no already-assigned vertex w
        # with color c has u in conflict_mask[c][w] (or equivalently w in conflict_mask[c][u]).
        
        # Also, we need to ensure that for future assignments, the constraint holds.
        
        # Let's use a recursive approach with pruning.
        
        def backtrack(pos, color_sets):
            """
            pos: index in the order
            color_sets: list of 4 sets (as bitmasks) of vertices assigned to each color
            """
            if pos == n:
                # All assigned, build the assignment
                return assignment
            
            u = order[pos]
            
            for c in range(4):
                # Check if assigning color c to u is consistent with existing assignments
                # u must not be in conflict_mask[c][w] for any w already assigned color c
                # i.e., conflict_mask[c][u] & color_sets[c] == 0
                if conflict_mask_current[c][u] & color_sets[c]:
                    continue
                
                # Also check the reverse: for any w already assigned color c,
                # w must not be in conflict_mask[c][u]
                # But conflict_mask[c][u] & color_sets[c] already checks this since
                # conflict_mask is symmetric (adj is symmetric)
                
                # Assign
                assignment[u] = c
                color_sets[c] |= (1 << u)
                
                # Forward checking: for each unassigned vertex v, check if it still has
                # at least one valid color
                valid = True
                for pos2 in range(pos + 1, n):
                    v = order[pos2]
                    has_valid = False
                    for c2 in range(4):
                        if not (conflict_mask_current[c2][v] & color_sets[c2]):
                            has_valid = True
                            break
                    if not has_valid:
                        valid = False
                        break
                    if time.time() > deadline:
                        valid = False
                        break
                
                if valid:
                    result = backtrack(pos + 1, color_sets)
                    if result is not None:
                        return result
                
                # Undo
                assignment[u] = None
                color_sets[c] &= ~(1 << u)
            
            return None
        
        color_sets = [0, 0, 0, 0]
        result = backtrack(0, color_sets)
        return result
    
    # Add 18 vertices
    current_n = n_base
    for step in range(18):
        if time.time() > deadline:
            break
        
        # Rebuild conflict masks for current size
        conflict_mask_cur = [[0] * current_n for _ in range(4)]
        for c in range(4):
            for u in range(current_n):
                mask = 0
                for w in range(current_n):
                    if w != u and adj[u][w] == c:
                        mask |= (1 << w)
                conflict_mask_cur[c][u] = mask
        
        # Try to add a vertex
        assignment = add_vertex(current_n, adj, conflict_mask_cur)
        
        if assignment is None:
            # Backtracking failed; break and return what we have
            break
        
        # Add the new vertex to adj and colors
        new_idx = current_n
        # Extend adj
        adj.append([0] * (current_n + 1))
        for i in range(current_n):
            adj[i].append(assignment[i])
            adj[new_idx][i] = assignment[i]
        adj[new_idx][new_idx] = -1
        
        # Extend colors list
        for i in range(current_n):
            colors.append(assignment[i])
        
        current_n += 1
    
    # Build final colors list
    final_colors = []
    idx = 0
    for i in range(current_n):
        for j in range(i + 1, current_n):
            final_colors.append(adj[i][j])
    
    return {'n': current_n, 'colors': final_colors}
