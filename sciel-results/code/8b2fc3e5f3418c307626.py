"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _has_bicolor_cycle(coloring, n, adj, color_a, color_b):
    """Check if the subgraph induced by vertices with color_a or color_b has a cycle using union-find."""
    vertices = [v for v in range(n) if coloring[v] == color_a or coloring[v] == color_b]
    if not vertices:
        return False
    vset = set(vertices)
    parent = {v: v for v in vertices}
    rank = {v: 0 for v in vertices}
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return True  # cycle detected
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return False
    
    for u in vertices:
        for v in adj[u]:
            if v in vset and u < v:
                if union(u, v):
                    return True
    return False

def _is_valid_assignment(coloring, n, adj, k):
    """Check all constraints: adjacent differ, and no bicolored cycle for any color pair."""
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    colors_used = list(set(coloring))
    for i in range(len(colors_used)):
        for j in range(i + 1, len(colors_used)):
            if _has_bicolor_cycle(coloring, n, adj, colors_used[i], colors_used[j]):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = P['n']
    adj = _build_adjacency(P['edges'], n)
    
    # Precompute a good vertex ordering using degeneracy-like heuristic
    # Process vertices with fewer uncolored neighbors first (forward degree)
    # For a 4-regular graph, use a BFS/DFS ordering that keeps subgraphs small
    # Actually, let's try: order by "bottleneck" - vertices that connect to many
    # already-processed vertices should come later
    
    # Use a simple heuristic: order vertices so that each vertex has as few
    # earlier neighbors as possible (minimize forward degree)
    # This is like a greedy coloring order
    
    # Try multiple orderings
    orderings = [
        list(range(n)),
        list(range(n-1, -1, -1)),
    ]
    
    # Generate a BFS-based ordering from vertex 0
    from collections import deque
    visited = [False] * n
    bfs_order = [0]
    visited[0] = True
    q = deque([0])
    while q:
        u = q.popleft()
        for w in sorted(adj[u]):
            if not visited[w]:
                visited[w] = True
                bfs_order.append(w)
                q.append(w)
    orderings.append(bfs_order)
    
    # Generate ordering by "peeling" - remove degree-2 vertices first
    # For degree-4 graph this won't help much, but let's try a different approach:
    # Order by distance from a center (median vertex)
    # Compute eccentricities
    from collections import defaultdict as dd2
    
    # Try: order vertices by their index in a space-filling curve
    # The graph seems to have a toroidal/grid structure
    # Let me analyze: vertices 0-6 form a path, 7-13 another, etc.
    # Looking at edges: 0-1-2-3-4-5-6-7-8... this is a Hamiltonian path
    # Plus "chord" edges creating the 4-regular structure
    
    # Let's try ordering by the Hamiltonian path index
    # Find the Hamiltonian path
    hp_order = list(range(n))  # 0,1,2,...,49 appears to be the HP
    
    orderings.append(hp_order)
    
    # Try a "diagonal" ordering if the graph is grid-like
    # From the edge list, it looks like a 5x10 or similar grid with wraparound
    # Let me try: group by (v // 10, v % 10) patterns
    # Actually from edges: 0 connects to 1,7,43,49
    # 1 connects to 0,2,8,44
    # This looks like a grid where row i has vertices i*7..i*7+6? No...
    # 0: neighbors 1,7,43,49
    # 1: neighbors 0,2,8,44  
    # 7: neighbors 0,6,8,14
    # So 0-1 is one direction, 0-7 is another, 0-43 and 0-49 are wraparounds
    # This is a 5x10 torus? 50 = 5*10
    # Let's assume grid: vertex (r,c) where r=v//10, c=v%10
    # 0=(0,0): neighbors (0,1)=1, (1,0)=10? No, 7 not 10
    # Maybe 7x? No, 50/7 isn't integer
    # 0: 1,7,43,49. If 10-col grid: (0,0)->(0,1)=1, (0,7)? no
    # If 7-col: 0=(0,0), 1=(0,1), 7=(1,0), 43=(6,1), 49=(7,0)
    # 50 = 7*7+1, not clean
    # Let's just use the path order and a few variants
    
    best_result = None
    
    for ordering in orderings:
        if time.time() > deadline - 10.0:
            break
        
        for k in range(2, 12):
            if time.time() > deadline - 3.0:
                break
            
            coloring = [-1] * n
            pos_of_vertex = {v: i for i, v in enumerate(ordering)}
            prior_neighbors = [[] for _ in range(n)]
            for pos, v in enumerate(ordering):
                for nb in adj[v]:
                    if pos_of_vertex[nb] < pos:
                        prior_neighbors[pos].append(nb)
            
            # Track which colors are used for fast cycle checking
            # Use a more efficient incremental approach:
            # When we assign color c to vertex v, we only need to check
            # pairs (c, d) for each d that is a neighbor color or already present
            # And only the subgraph induced by those two colors
            
            # Precompute for each color pair, the set of vertices with that color
            # and use a running edge count to detect cycles
            # A subgraph has a cycle iff edges >= vertices (for connected components)
            # More precisely: a graph is a forest iff for each component, edges = vertices - 1
            # So total edges <= total vertices - number_of_components
            # Equivalently: if we add an edge within the same component, we get a cycle
            
            # Use union-find per color pair, incrementally
            # Key insight: when we assign vertex v color c, for each other color d,
            # we add edges from v to its neighbors of color d. If any such edge
            # creates a cycle in the (c,d)-bicolored subgraph, it's invalid.
            
            # Maintain: for each color pair (a,b), a union-find structure
            # This is expensive to maintain, so let's use a simpler check:
            # For each pair (c,d), count edges and vertices in the induced subgraph
            # If edges >= vertices, there's a cycle (for a non-empty graph)
            # Actually that's not quite right for disconnected graphs
            # But: a graph is a forest iff |E| <= |V| - |C| where C is components
            # |E| = |V| - |C| means tree, |E| >= |V| means cycle
            # For a graph with |V| vertices and |E| edges: cycle exists iff |E| >= |V|
            # Wait no: a single edge has |V|=2, |E|=1, no cycle. Two edges forming path: |V|=3,|E|=2.
            # Cycle: |V|=3, |E|=3. So |E| >= |V| implies cycle? 
            # A graph with |V| vertices and |E| edges has a cycle iff |E| >= |V|
            # (since a forest on |V| vertices has at most |V|-1 edges)
            # Yes! So we just need to track edge counts and vertex counts per color pair.
            
            # This is MUCH faster than full union-find at each step!
            # vertex_count[(a,b)] = number of vertices colored a or b
            # edge_count[(a,b)] = number of edges where both endpoints are colored a or b
            # If edge_count[(a,b)] >= vertex_count[(a,b)] for any pair, there's a cycle
            
            # Wait, this isn't quite right either. Consider two separate trees:
            # Tree1: 3 vertices, 2 edges. Tree2: 3 vertices, 2 edges.
            # Total: 6 vertices, 4 edges. 4 < 6, no cycle. Correct.
            # Now add an edge between them: 6 vertices, 5 edges. 5 < 6, still no cycle. Correct.
            # Add another edge within one tree: 6 vertices, 6 edges. 6 >= 6, cycle. Correct!
            # Actually wait: 6 vertices, 6 edges. A forest on 6 vertices has at most 5 edges.
            # So 6 edges means there IS a cycle. Yes, the condition |E| >= |V| correctly
            # detects cycles (since a forest has |E| = |V| - |C| <= |V| - 1 < |V|).
            # So |E| >= |V| is equivalent to "not a forest" = "has a cycle". 
            # Wait, |E| = |V| - |C| + 1 for a unicyclic graph. So |E| >= |V| - |C| + 1.
            # Hmm, the condition |E| >= |V| is sufficient but not necessary.
            # Counterexample: two triangles sharing a vertex. 5 vertices, 6 edges. 6 >= 5, cycle. OK.
            # What about: a path of 4 vertices (3 edges) plus an isolated vertex. 5 vertices, 3 edges. No cycle. 3 < 5. OK.
            # A cycle of 4 + isolated vertex: 5 vertices, 4 edges. 4 < 5. But there IS a cycle!
            # So |E| < |V| does NOT guarantee no cycle. The condition is wrong.
            
            # I need to use union-find after all, or track components.
            # Let me use a hybrid: track vertex_count and edge_count, and only
            # do full union-find check when edge_count >= vertex_count - 1 (near threshold)
            # Actually, let me just optimize the union-find check to be faster.
            
            # Better approach: maintain per-pair union-find incrementally
            # When vertex v gets color c, for each neighbor u with color d != c:
            #   add edge (u,v) to pair (min(c,d), max(c,d))
            #   check if this edge creates a cycle in that pair's subgraph
            # This is O(degree) per assignment, very fast!
            
            # But we also need to handle: what if v's color c and some non-neighbor
            # vertex w has color d, and there's already a path between v and w
            # in the (c,d)-subgraph? No - we only add edges from v to its neighbors.
            # The bicolored subgraph only contains edges between vertices of colors c and d.
            # So when we assign v color c, the only new edges in pair (c,d) are
            # edges from v to neighbors of v that have color d.
            
            # So the incremental check is:
            # 1. Assign v color c
            # 2. For each neighbor u of v with color d (d != c, d != -1):
            #    pair = (min(c,d), max(c,d))
            #    Check if adding edge (v,u) to pair's graph creates a cycle
            #    (i.e., are v and u already connected in pair's union-find?)
            # 3. If any cycle detected, undo assignment
            
            # We need to maintain union-find per pair. Since k <= 11, there are
            # at most k*(k-1)/2 = 55 pairs. Each has its own UF.
            # When we undo, we need to remove edges. UF doesn't support deletion easily.
            # Alternative: just re-check from scratch for affected pairs.
            
            # Actually, let's use a simpler approach: for each assignment, only check
            # the pairs that involve the new color c. For each such pair (c,d),
            # check if the (c,d)-subgraph (restricted to assigned vertices) has a cycle.
            # But do it efficiently by only looking at the relevant subgraph.
            
            # Even simpler: since the graph is small (50 vertices), and we're checking
            # at most k-1 pairs per assignment, let's just do a BFS/DFS cycle check
            # on the induced subgraph for each pair. With 50 vertices max, this is fast.
            
            def check_pair_cycle(ca, cb, assigned_set):
                """Check if subgraph induced by vertices with color ca or cb (among assigned) has a cycle."""
                # Collect vertices
                verts = []
                for u in assigned_set:
                    cu = coloring[u]
                    if cu == ca or cu == cb:
                        verts.append(u)
                if len(verts) < 3:
                    return False
                vset = set(verts)
                # Union-find
                uf = {x: x for x in verts}
                def find(x):
                    while uf[x] != x:
                        uf[x] = uf[uf[x]]
                        x = uf[x]
                    return x
                def union(x, y):
                    rx, ry = find(x), find(y)
                    if rx == ry:
                        return True
                    uf[ry] = rx
                    return False
                for u in verts:
                    for w in adj[u]:
                        if w in vset and u < w:
                            if union(u, w):
                                return True
                return False
            
            def backtrack(pos):
                if time.time() > deadline - 1.0:
                    return False
                if pos == n:
                    return True
                
                v = ordering[pos]
                
                # Get colors used by already-assigned neighbors
                neighbor_colors = set()
                for nb in prior_neighbors[pos]:
                    c = coloring[nb]
                    if c != -1:
                        neighbor_colors.add(c)
                
                # Try colors not used by neighbors
                for c in range(k):
                    if c in neighbor_colors:
                        continue
                    coloring[v] = c
                    
                    # Check bicolored cycles only for pairs involving c
                    # that have at least 3 total vertices
                    valid = True
                    for d in range(k):
                        if d == c:
                            continue
                        # Check if pair (c,d) has enough vertices
                        cnt = 0
                        for i in range(pos + 1):
                            cu = coloring[ordering[i]]
                            if cu == c or cu == d:
                                cnt += 1
                        if cnt < 3:
                            continue
                        # Check for cycle
                        verts = [ordering[i] for i in range(pos + 1) 
                                 if coloring[ordering[i]] == c or coloring[ordering[i]] == d]
                        vset = set(verts)
                        uf = {x: x for x in verts}
                        def find(x):
                            while uf[x] != x:
                                uf[x] = uf[uf[x]]
                                x = uf[x]
                            return x
                        def union(x, y):
                            rx, ry = find(x), find(y)
                            if rx == ry:
                                return True
                            uf[ry] = rx
                            return False
                        has_cycle = False
                        for u in verts:
                            for w in adj[u]:
                                if w in vset and u < w:
                                    if union(u, w):
                                        has_cycle = True
                                        break
                            if has_cycle:
                                break
                        if has_cycle:
                            valid = False
                            break
                    
                    if valid:
                        if backtrack(pos + 1):
                            return True
                    
                    coloring[v] = -1
                
                return False
            
            if backtrack(0):
                if _is_valid_assignment(coloring, n, adj, k):
                    return [int(x) for x in coloring]
    
    # Fallback: return identity coloring (always valid but uses many colors)
    return list(range(n))

# EVOLVE-BLOCK-END
