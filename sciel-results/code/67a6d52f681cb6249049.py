"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 5-coloring of K_n with no monochromatic C4.
    Uses greedy construction with backtracking and cost-minimization.
    """
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    # Try to build the largest possible coloring
    # Strategy: greedy with lookahead, trying to extend a base coloring
    
    # Start with a known good base and try to extend
    # The K11 construction: vertices 0..10, edge (i,j) colored by circular distance
    # Let's try to build from scratch with a smart greedy
    
    def try_build(n_target, seed=42):
        """Try to build a valid 5-coloring of K_n_target."""
        rng = random.Random(seed)
        
        # colors[i][j] = color of edge (i,j), stored in 2D array
        # We'll build vertex by vertex
        # When adding vertex v, we need to assign colors to edges (v, u) for u < v
        # such that no C4 is formed in any color class
        
        # State: for each color c, we track the adjacency structure
        # A C4 in color c means: exist a,b,c,d with edges (a,b),(b,c),(c,d),(d,a) all color c
        # Equivalently: two vertices have at least 2 common neighbors in color c
        
        # We'll track for each color c, for each pair of vertices (u,v), 
        # how many common neighbors they share in color c.
        # When adding edge (v, u) with color c, we need to check that for all w < v, w != u:
        #   if (v,w) is color c, then u and w must not already have a common neighbor in color c
        #   (because adding (v,u) color c would make v a common neighbor of u and w in color c)
        
        # More carefully: when we add edge (v, u) with color c:
        # For any pair (x, y) where x, y are both neighbors of v in color c (excluding u),
        # adding (v, u) doesn't directly create a C4 involving v.
        # The C4 would be: u - v - x - y - u, but we need (u,y) and (x,y) and (v,x) and (v,u) all color c.
        # Actually a C4 is four vertices a,b,c,d with edges ab, bc, cd, da all same color.
        # So if we add edge (v,u) color c, a new C4 could be formed if there exist x,y such that:
        #   (v,x) color c, (x,y) color c, (y,u) color c  -> C4: v-x-y-u-v
        #   or (v,x) color c, (x,y) color c, (y,u) color c is one case
        #   More generally: for any x, if (v,x) is color c, then for any y, if (x,y) is color c and (y,u) is color c, we have C4 v-x-y-u.
        #   Also: for any x, if (u,x) is color c, then for any y, if (x,y) is color c and (y,v) is color c, we have C4 u-x-y-v.
        
        # Let me think more carefully. When adding edge (v, u) with color c:
        # A C4 is a cycle of length 4. The new edge is (v, u). 
        # A C4 containing edge (v,u) has form: v - u - a - b - v
        # So we need (u,a) color c, (a,b) color c, (b,v) color c for some a, b.
        # Equivalently: a is a neighbor of u in color c, b is a neighbor of v in color c, and (a,b) is color c.
        # So we need: there exist a in N_c(u), b in N_c(v) such that (a,b) is color c.
        
        # Wait, that's not quite right either. The C4 is v-u-a-b-v. 
        # Edges: (v,u), (u,a), (a,b), (b,v) all color c.
        # So a must be in N_c(u) \ {v}, b must be in N_c(v) \ {u}, and (a,b) must be color c.
        
        # But also, a and b could be the same? No, in a simple graph C4 has 4 distinct vertices.
        # So a != b, a != v, b != u, a != u (since a is neighbor of u), b != v (since b is neighbor of v).
        
        # So the check is: for the proposed edge (v,u) color c,
        #   let Nu = {x : (u,x) is color c} (already assigned edges from u)
        #   let Nv = {x : (v,x) is color c} (already assigned edges from v, excluding u since we're adding it)
        #   Check: does there exist a in Nu, b in Nv with a != b and (a,b) color c?
        #   If yes, adding (v,u) color c creates a C4.
        
        # Actually wait: Nu should be neighbors of u in color c among vertices < v (since we're building in order).
        # And Nv should be neighbors of v in color c among vertices < v.
        # And (a,b) color c means the edge between a and b is color c.
        
        # This is O(n^2) per edge assignment, which is fine for n up to ~29.
        
        n = n_target
        # colors[v][u] for v > u
        colors = [[-1] * n for _ in range(n)]
        
        # For each vertex v (in order), assign colors to edges (v, u) for u = 0, ..., v-1
        # We process vertices in order 0, 1, 2, ...
        # When processing vertex v, edges (v, u) for u < v need to be assigned.
        
        # For efficiency, track for each color c, the adjacency list
        adj = [[] for _ in range(5)]  # adj[c][v] = list of neighbors of v in color c
        
        # Also track for quick C4 check: for each color c, for each pair (a,b), count of common neighbors
        # Actually, let's just do the direct check.
        
        def can_assign(v, u, c, colors, adj):
            """Check if we can assign color c to edge (v, u) where v > u."""
            # Check for C4: v-u-a-b-v
            # a in adj[c][u], b in adj[c][v], a != b, and colors[min(a,b)][max(a,b)] == c
            Nu = adj[c][u]
            Nv = adj[c][v]
            for a in Nu:
                for b in Nv:
                    if a == b:
                        continue
                    lo, hi = (a, b) if a < b else (b, a)
                    if colors[hi][lo] == c:
                        return False
            return True
        
        def assign(v, u, c, colors, adj):
            """Assign color c to edge (v, u) where v > u."""
            colors[v][u] = c
            adj[c][v].append(u)
            adj[c][u].append(v)
        
        def unassign(v, u, c, colors, adj):
            """Unassign color c from edge (v, u) where v > u."""
            colors[v][u] = -1
            adj[c][v].remove(u)
            adj[c][u].remove(v)
        
        # Process vertices in order
        # For vertex 0: no edges to assign
        # For vertex v > 0: assign edges (v, u) for u = 0, ..., v-1
        
        # We'll use backtracking if needed, but first try greedy
        
        # Greedy with cost minimization:
        # When assigning color to edge (v, u), prefer the color that creates the "least" future C4 risk.
        # Risk metric: number of pairs (a, b) with a in N_c(u), b in N_c(v), a != b, 
        #              such that (a,b) is NOT yet color c (could become color c later).
        # Actually, simpler: just count |N_c(u)| * |N_c(v)| as a proxy for risk.
        
        # Let's try a simple greedy first, then backtrack if stuck.
        
        # For each vertex v, assign colors to edges (v, 0), (v, 1), ..., (v, v-1)
        # Order of assignment: we can choose the order. Let's do u = 0, 1, ..., v-1.
        
        # Actually, let's try multiple orderings and strategies.
        
        # Simple greedy:
        for v in range(1, n):
            # Assign colors to edges (v, u) for u in 0..v-1
            # Try each edge in order u = 0, 1, ..., v-1
            for u in range(v):
                # Try colors 0..4, pick the one with minimum risk
                best_color = -1
                best_risk = float('inf')
                for c in range(5):
                    if not can_assign(v, u, c, colors, adj):
                        continue
                    # Risk: |adj[c][u]| * |adj[c][v]|
                    risk = len(adj[c][u]) * len(adj[c][v])
                    if risk < best_risk:
                        best_risk = risk
                        best_color = c
                if best_color == -1:
                    return None  # Failed
                assign(v, u, best_color, colors, adj)
        
        # Extract colors in lexicographic order
        result_colors = []
        for i in range(n):
            for j in range(i + 1, n):
                result_colors.append(colors[j][i])
        
        # Validate
        if not validate(n, result_colors):
            return None
        
        return result_colors
    
    def validate(n, colors_list):
        """Validate that no color class contains a C4."""
        # Build adjacency for each color
        adj = [set() for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors_list[idx]
                idx += 1
                adj[c].add((i, j))
        
        # For each color, check for C4
        # A C4 exists if there are two vertices with >= 2 common neighbors in that color
        for c in range(5):
            # Build adjacency list
            adj_list = [[] for _ in range(n)]
            for (i, j) in adj[c]:
                adj_list[i].append(j)
                adj_list[j].append(i)
            
            # For each pair (u, v), count common neighbors
            for u in range(n):
                for v in range(u + 1, n):
                    common = set(adj_list[u]) & set(adj_list[v])
                    if len(common) >= 2:
                        return False
        return True
    
    # Try to build colorings of increasing size
    best_n = 11  # Start with known good
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # First, try to extend beyond 11
    # Try n = 12, 13, ... up to some limit
    
    # Try multiple random seeds for each n
    max_n_to_try = 22  # Be conservative with time
    
    for n_try in range(12, max_n_to_try + 1):
        if time_left() < 10:
            break
        found = False
        for seed in range(20):
            if time_left() < 5:
                break
            result = try_build(n_try, seed)
            if result is not None:
                best_n = n_try
                best_colors = result
                found = True
                break
        if found:
            # Try to extend further
            continue
        else:
            # Couldn't build n_try, stop
            break
    
    # Also try the original K11 as fallback
    if best_n < 11:
        best_n = 11
        best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
