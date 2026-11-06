"""Rewritten program: greedy constructive algorithm with backtracking for C4-free 5-colouring."""
import sys
import random
from collections import defaultdict

def construct():
    # We want to maximize n such that K_n can be 5-coloured with no C4 in any colour class.
    # The Lazebnik-Woldar construction gives n=26.
    # We'll use a greedy algorithm with backtracking to try to build a valid colouring.
    # 
    # Key insight: For each colour c, we maintain for each vertex v the set of neighbours
    # of v in colour c. When adding a new vertex u and assigning colour c to edge (u, v),
    # we must ensure that u and v don't already share two common neighbours in colour c.
    # That is, |N_c(u) ∩ N_c(v)| must be at most 1 before adding the edge.
    #
    # Since we're building vertex-by-vertex, when we add vertex u (the newest), N_c(u) is
    # being built up. For existing vertices v, w, we need to track their common neighbours.
    #
    # Actually, a more efficient approach: maintain for each colour c, a dict mapping
    # each pair of existing vertices (v, w) to the count of common neighbours in colour c.
    # But that's O(n^2) per colour.
    #
    # Better: For each colour c and each vertex v, maintain N_c[v] as a set.
    # When considering adding edge (u, v) with colour c, check if |N_c[u] ∩ N_c[v]| <= 1.
    # Since u is new, N_c[u] only contains vertices already connected to u in colour c.
    #
    # Let's use a simpler approach: try to build up to n=26 (known feasible) and see if
    # we can push to 27 or 28 with backtracking.
    #
    # Given the 120s time limit, let's first try a deterministic greedy for n=26,
    # then attempt n=27 with limited backtracking.
    
    # First, let's implement a function that tries to colour K_n with 5 colours, C4-free.
    
    def try_colouring(n, max_backtracks=1000, seed=42):
        """Try to find a 5-colouring of K_n with no C4 in any colour class."""
        rng = random.Random(seed)
        
        # neighbours[color][vertex] = set of vertices connected to vertex in that colour
        neighbours = [defaultdict(set) for _ in range(5)]
        
        # For efficiency, we also track for each colour c, for each pair (v,w),
        # the number of common neighbours in colour c. But let's just use set intersection
        # since n is at most 28.
        
        def can_add_edge(u, v, c):
            """Check if adding edge (u,v) with colour c would create a C4."""
            # A C4 in colour c would be two vertices with two common neighbours in colour c.
            # If we add edge (u,v) in colour c, then for any existing vertex w,
            # if w is connected to both u and v in colour c, that's fine as long as
            # there's only one such w. But actually, the C4 condition is:
            # no two vertices have two common neighbours in the same colour.
            # 
            # When we add edge (u,v) in colour c, we need to check that u and v
            # don't already have two common neighbours in colour c.
            # 
            # Common neighbours of u in colour c: neighbours[c][u]
            # Common neighbours of v in colour c: neighbours[c][v]
            # The intersection is the set of vertices w such that (w,u) and (w,v) are both colour c.
            # If |intersection| >= 2, then adding (u,v) in colour c creates a C4.
            
            common = neighbours[c][u] & neighbours[c][v]
            return len(common) <= 1
        
        def add_edge(u, v, c):
            neighbours[c][u].add(v)
            neighbours[c][v].add(u)
        
        def remove_edge(u, v, c):
            neighbours[c][u].discard(v)
            neighbours[c][v].discard(u)
        
        # Build vertex by vertex. Vertex i connects to vertices 0..i-1.
        # For each new vertex i, assign colours to edges (i, 0), (i, 1), ..., (i, i-1).
        
        # colors[i][j] = colour of edge (i,j) for i<j
        colors = [[-1] * n for _ in range(n)]
        
        def backtrack(vertex, backtracks_remaining):
            if vertex >= n:
                return True
            
            # Try all possible colour assignments for edges (vertex, 0), (vertex, 1), ..., (vertex, vertex-1)
            # There are 5^vertex possibilities, which is too many. Use greedy with limited backtracking.
            
            # Greedy: for each edge (vertex, j) in order, try colours 0..4, pick the first valid one.
            # If stuck, backtrack.
            
            # To keep it simple and fast, let's do a simple backtracking with a cap.
            
            # For vertex, we need to assign colours to edges to 0..vertex-1
            # Let's process edges in order 0, 1, ..., vertex-1
            
            # Use recursion with memoization isn't feasible; just simple backtracking
            # but with a hard cap on total backtracks.
            
            # Actually, for n up to 28, 5^27 is way too large. We need a smarter approach.
            # 
            # Let's use a greedy heuristic: for each edge, prefer colours that have
            # fewer conflicts. If greedy fails, we can try a few random perturbations.
            
            # For now, let's just do greedy without backtracking first.
            
            for j in range(vertex):
                valid_colours = []
                for c in range(5):
                    if can_add_edge(vertex, j, c):
                        valid_colours.append(c)
                
                if not valid_colours:
                    return False, 0  # Stuck
                
                # Pick the colour that leaves the most options for future vertices
                # Heuristic: pick the colour with the smallest |N_c[vertex] ∩ N_c[j]|
                # after adding, to minimize future conflicts.
                
                best_c = valid_colours[0]
                best_score = float('inf')
                
                for c in valid_colours:
                    # Score: how many common neighbours will vertex and j share in colour c
                    # after adding this edge? That's |N_c[vertex] ∩ N_c[j]| (before adding,
                    # since adding doesn't change other edges).
                    # Actually, the score should reflect how constrained future edges will be.
                    # A simpler heuristic: pick randomly among valid colours, or pick the one
                    # with the smallest current intersection size.
                    inter_size = len(neighbours[c][vertex] & neighbours[c][j])
                    if inter_size < best_score:
                        best_score = inter_size
                        best_c = c
                
                # Sometimes randomize to escape local optima
                if rng.random() < 0.1:
                    best_c = rng.choice(valid_colours)
                
                add_edge(vertex, j, best_c)
                colors[vertex][j] = best_c
            
            # Recurse
            success, bt = backtrack(vertex + 1, backtracks_remaining)
            if success:
                return True, bt
            
            # Undo
            for j in range(vertex):
                c = colors[vertex][j]
                if c >= 0:
                    remove_edge(vertex, j, c)
                    colors[vertex][j] = -1
            
            return False, bt + 1
        
        # Try multiple seeds if greedy fails
        for seed in range(50):
            # Reset
            neighbours = [defaultdict(set) for _ in range(5)]
            colors = [[-1] * n for _ in range(n)]
            rng = random.Random(seed)
            
            # Redefine with new rng
            def can_add_edge(u, v, c):
                common = neighbours[c][u] & neighbours[c][v]
                return len(common) <= 1
            
            def add_edge(u, v, c):
                neighbours[c][u].add(v)
                neighbours[c][v].add(u)
            
            def remove_edge(u, v, c):
                neighbours[c][u].discard(v)
                neighbours[c][v].discard(u)
            
            def backtrack(vertex, backtracks_remaining):
                if vertex >= n:
                    return True, 0
                
                for j in range(vertex):
                    valid_colours = []
                    for c in range(5):
                        if can_add_edge(vertex, j, c):
                            valid_colours.append(c)
                    
                    if not valid_colours:
                        return False, 0
                    
                    best_c = valid_colours[0]
                    best_score = float('inf')
                    
                    for c in valid_colours:
                        inter_size = len(neighbours[c][vertex] & neighbours[c][j])
                        if inter_size < best_score:
                            best_score = inter_size
                            best_c = c
                    
                    if rng.random() < 0.15:
                        best_c = rng.choice(valid_colours)
                    
                    add_edge(vertex, j, best_c)
                    colors[vertex][j] = best_c
                
                success, bt = backtrack(vertex + 1, backtracks_remaining)
                if success:
                    return True, bt
                
                for j in range(vertex):
                    c = colors[vertex][j]
                    if c >= 0:
                        remove_edge(vertex, j, c)
                        colors[vertex][j] = -1
                
                return False, bt + 1
            
            success, _ = backtrack(0, 1000)
            if success:
                # Extract colours in lexicographic order
                result = []
                for i in range(n):
                    for j in range(i + 1, n):
                        result.append(colors[i][j])
                return result
        
        return None
    
    # Try n=28 first (most ambitious), then 27, then 26
    for n in [28, 27, 26]:
        result = try_colouring(n)
        if result is not None:
            return {'n': n, 'colors': result}
    
    # Fallback: use the known n=11 construction
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
