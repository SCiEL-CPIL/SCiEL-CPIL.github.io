"""Construct 5-colourings of K_n without monochromatic C4, maximising n."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # We know n=26 is achievable via Lazebnik-Woldar construction.
    # We try to find a valid coloring for n=26, 27, 28.
    # If we find one for 28, score is 28/11. If 27, 27/11. If 26, 26/11.
    # The seed program gives n=11 (score 1.0).
    
    # Strategy:
    # 1. Try to construct a valid coloring for n=28 using randomized backtracking.
    # 2. If that fails within time, try n=27.
    # 3. If that fails, try n=26.
    # 4. If that fails, return the seed n=11.
    
    # To speed up search, we use a greedy heuristic with randomization and backtracking.
    # Key constraint: No monochromatic C4.
    # This is equivalent to: For any color c, the graph G_c is C4-free.
    # A graph is C4-free iff for any two vertices u, v, |N(u) ∩ N(v)| <= 1.
    
    # We will use a backtracking solver that assigns colors to edges one by one.
    # To make it efficient, we order edges such that constraints are checked early.
    # A common heuristic is to process vertices in order, and for each vertex i,
    # assign colors to edges (i, j) for j > i.
    
    # However, a full backtracking on K_28 (378 edges) is huge.
    # We rely on the fact that valid colorings exist and are somewhat structured.
    # The Lazebnik-Woldar construction is based on finite geometry (projective planes).
    # Replicating that algebraic construction directly is safer and faster if we can code it.
    
    # Let's try to implement the Lazebnik-Woldar construction for n=26.
    # The construction uses a projective plane of order q=5? No, q=5 gives 31 points.
    # Actually, the known construction for R_5(C4) >= 26 uses a specific design.
    
    # Alternative: Use a randomized local search / hill climbing on the space of colorings?
    # No, C4-free is a hard constraint.
    
    # Let's try a simple randomized backtracking for n=26 first, as it's the known lower bound.
    # If we can't find it fast, we fall back to n=11.
    # But wait, the seed is n=11. The target is n=26.
    # A naive backtracker might not find n=26 in 115s.
    
    # Better Strategy:
    # 1. Hardcode a known valid coloring for n=26 if possible? 
    #    The Lazebnik-Woldar paper gives an explicit construction.
    #    It involves vertices as points in PG(2, q) or similar?
    #    Actually, for 5 colors, it's often related to symmetric balanced incomplete block designs.
    
    # Let's try a different approach: 
    # Use a greedy algorithm with random restarts to build a large C4-free 5-coloring.
    # Start with n=1, add vertices one by one.
    # When adding vertex v, assign colors to edges (v, u) for all existing u.
    # Check if adding (v, u) with color c creates a C4 in color c.
    # A C4 is created if there exists w such that (v, w) and (u, w) are already color c? No.
    # A C4 in color c involves vertices a, b, c, d such that edges (a,b), (b,c), (c,d), (d,a) are color c.
    # When adding edge (v, u) with color c, we must ensure that there is no vertex w 
    # such that (v, w) is color c AND (u, w) is color c.
    # This is because if (v,w) and (u,w) are color c, then v-w-u is a path of length 2.
    # If we also have a path of length 2 from v to u via another vertex x (which would mean (v,x) and (u,x) are color c), 
    # wait, the condition for C4 is: 
    # If we assign color c to edge (v, u), we must check if there is any w 
    # such that color(v, w) == c and color(u, w) == c.
    # If such a w exists, then v-w-u is a 2-path. 
    # Is that a C4? No, a C4 requires a cycle of length 4.
    # The vertices involved would be v, w, u, and...?
    # Actually, if v and u have a common neighbor w in color c, that forms a triangle v-w-u-v? No, (v,u) is the new edge.
    # So v-w-u-v is a triangle. Triangles are allowed.
    # A C4 is formed if v and u have TWO common neighbors w1, w2 in color c.
    # Then v-w1-u-w2-v is a C4.
    
    # So the check when adding edge (v, u) with color c is:
    # Count the number of vertices w such that color(v, w) == c and color(u, w) == c.
    # This count must be 0. (Because if it were 1, it's a triangle. If >=2, it's a C4).
    # Wait, if count is 1, say w1, we have edges (v,w1), (u,w1), (v,u). This is a triangle.
    # Does it create a C4? No.
    # So the condition is: The number of common neighbors in color c for v and u must be 0?
    # Let's re-read the C4 definition.
    # A 4-cycle is a set of 4 distinct vertices {a,b,c,d} such that (a,b), (b,c), (c,d), (d,a) are in the color class.
    # If we add edge (v,u) with color c, it can only complete a C4 if there is already a path of length 3 from v to u in color c?
    # No, adding an edge (v,u) creates a cycle if there is a path from v to u.
    # A cycle of length 4 is created if there is a path of length 3 from v to u in color c.
    # Path of length 3: v - w1 - w2 - u.
    # This requires edges (v,w1), (w1,w2), (w2,u) to be color c.
    # This is harder to check incrementally.
    
    # However, there is a simpler equivalent condition for C4-free graphs:
    # For any two vertices v, u, they can have at most 1 common neighbor in any color class.
    # Proof: If they have 2 common neighbors w1, w2, then v-w1-u-w2-v is a C4.
    # Conversely, if a C4 exists, its opposite vertices have 2 common neighbors.
    # So the condition is: For any pair (v, u), and any color c, |N_c(v) ∩ N_c(u)| <= 1.
    
    # When we assign color c to edge (v, u), we are effectively saying v and u are adjacent in G_c.
    # This does NOT directly change the common neighbor counts for other pairs, 
    # BUT it changes the neighbor sets N_c(v) and N_c(u).
    # Specifically, u is added to N_c(v) and v is added to N_c(u).
    # This increases the common neighbor count for any pair (x, y) where x is connected to v in color c 
    # and y is connected to u in color c? No.
    # Let's check the constraint for a pair (x, y) after adding (v, u) with color c.
    # The new common neighbors in color c for (x, y) are the old ones PLUS:
    # - If x=v and y=u, they now share no new common neighbors directly from this edge? 
    #   Wait, the condition is on the graph G_c.
    #   Adding edge (v, u) to G_c means v and u become neighbors.
    #   For any pair (x, y), the set N_c(x) ∩ N_c(y) might change.
    #   It changes only if x or y is v or u.
    #   Case 1: x=v, y=u. 
    #       Before adding (v,u), N_c(v) and N_c(u) were some sets. 
    #       After adding, u ∈ N_c(v) and v ∈ N_c(u).
    #       Do v and u have common neighbors? 
    #       A common neighbor w must be in N_c(v) ∩ N_c(u).
    #       The new edge doesn't add w to the intersection unless w was already there.
    #       Actually, the constraint |N_c(v) ∩ N_c(u)| <= 1 must hold.
    #       Before adding (v,u), we must have ensured that |N_c(v) ∩ N_c(u)| <= 1.
    #       Adding the edge (v,u) doesn't change the set of common neighbors, it just makes them adjacent.
    #       So if |N_c(v) ∩ N_c(u)| was 1, it's still 1. If it was 0, it's still 0.
    #       So the constraint for the pair (v,u) itself is just that they don't already have 2 common neighbors.
    #       
    #   Case 2: x=v, y=some other vertex z.
    #       N_c(v) gains u. N_c(z) is unchanged.
    #       New intersection N_c(v) ∩ N_c(z) = (Old N_c(v) ∩ N_c(z)) ∪ ({u} if u ∈ N_c(z) else {}).
    #       So the size can increase by 1 if u is already a neighbor of z in color c.
    #       We must ensure that if u ∈ N_c(z), then |Old N_c(v) ∩ N_c(z)| was 0 (so new size is 1).
    #       If |Old N_c(v) ∩ N_c(z)| was 1, and u ∈ N_c(z), then new size is 2 -> C4 formed!
    #       So we must check: For all z != v, u:
    #       If color(z, u) == c, then we must have |N_c(v) ∩ N_c(z)| == 0 before adding (v,u).
    #       But N_c(v) includes all current neighbors of v in color c.
    #       So the condition is: For all z such that color(z, u) == c, 
    #       v must not share any common neighbor with z in color c.
    #       i.e., N_c(v) ∩ N_c(z) must be empty.
    
    # This check is O(n) per edge assignment, which is feasible.
    
    def is_valid_assignment(n, colors, edge_idx, color):
        """
        Check if assigning 'color' to edge 'edge_idx' (which is (v, u)) 
        maintains the C4-free property.
        colors is a list of length n*(n-1)/2, initialized with -1.
        """
        # Map edge_idx to (v, u)
        # This is slow if done every time. Precompute mapping?
        # For n=26, edges=325. 
        # We'll just compute v, u from edge_idx.
        
        # Inverse of lexicographic order
        # edge_idx = k
        # Find v, u such that sum_{i=0}^{v-1} (n-1-i) + (u-v-1) = k
        # This is O(n) to invert.
        
        # Better: Precompute a list of (v, u) for each edge index.
        # But we are inside a function. Let's pass the mapping or compute it.
        # For speed, let's assume we iterate in lexicographic order and maintain state.
        
        # Actually, for a backtracker, it's better to have the mapping.
        # Let's create the mapping once outside the recursive call if possible.
        
        # For now, let's just return True and implement a simpler check later.
        # This placeholder is just to structure the code.
        return True

    # Given the complexity and time limit, a pure Python backtracker for n=26 might be too slow.
    # The Lazebnik-Woldar construction is deterministic and fast.
    # Let's try to find a Python implementation of the LW construction online or derive it.
    # The construction for R_5(C4) >= 26 is based on a 5-coloring of K_26.
    # It is related to the projective plane of order 5? No, that's 31 points.
    # It is related to a symmetric (26, 10, 4) design?
    
    # Actually, there is a known result: R_5(C4) >= 26.
    # A simple randomized greedy approach might find a smaller n (e.g., 15-20) quickly.
    # Let's try to maximize n using a greedy randomized construction.
    
    def greedy_construct(n, time_limit):
        """
        Attempt to construct a valid 5-coloring for K_n using greedy random assignment.
        Returns colors list or None if fails.
        """
        colors = [-1] * (n * (n - 1) // 2)
        
        # Precompute edge indices
        edges = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
                idx += 1
        
        # To make it efficient, we can use a heuristic ordering.
        # But random order with backtracking is complex.
        # Let's try a simple forward checking greedy:
        # Process edges in random order.
        # For each edge (v, u), try colors 0-4 randomly.
        # Check if valid.
        
        # The validity check:
        # For the pair (v, u) with color c:
        # 1. Check if v and u already have 2 common neighbors in color c.
        #    (This would mean a C4 was already formed? No, if they have 2 common neighbors, 
        #     adding the edge doesn't create a C4, the C4 exists in the subgraph? 
        #     Wait. If v and u have 2 common neighbors w1, w2 in color c, 
        #     then v-w1-u-w2-v is a C4. The edge (v,u) is not part of this C4.
        #     So the C4 already exists in the graph without (v,u).
        #     So we must ensure that before adding (v,u), v and u have AT MOST 1 common neighbor in color c.
        #     If they have 1, adding (v,u) creates a triangle, which is fine.
        #     If they have 0, adding (v,u) is fine for the pair (v,u).
        #
        # 2. Check for new C4s formed involving (v,u).
        #    A new C4 must use the edge (v,u).
        #    So it looks like v - w1 - w2 - u - v.
        #    This requires edges (v,w1), (w1,w2), (w2,u) to be color c.
        #    This is equivalent to: There exists a path of length 3 from v to u in color c.
        #    Alternatively, using the common neighbor property:
        #    Adding (v,u) to color c is valid iff:
        #    For every vertex z:
        #    If z is a neighbor of v in color c AND z is a neighbor of u in color c, 
        #    then we have a common neighbor. 
        #    Wait, the condition |N_c(v) ∩ N_c(u)| <= 1 is for the FINAL graph.
        #    If we add (v,u), N_c(v) and N_c(u) change.
        #    Let N_v = N_c(v) before adding, N_u = N_c(u) before adding.
        #    After adding, N_c'(v) = N_v ∪ {u}, N_c'(u) = N_u ∪ {v}.
        #    We need |N_c'(v) ∩ N_c'(u)| <= 1.
        #    N_c'(v) ∩ N_c'(u) = (N_v ∪ {u}) ∩ (N_u ∪ {v})
        #    = (N_v ∩ N_u) ∪ (N_v ∩ {v}) ∪ ({u} ∩ N_u) ∪ ({u} ∩ {v})
        #    Note: v is not in N_v (no self loops). u is not in N_u.
        #    So N_v ∩ {v} = empty. {u} ∩ N_u = empty (unless u in N_u, which is false).
        #    {u} ∩ {v} = empty (v != u).
        #    So N_c'(v) ∩ N_c'(u) = N_v ∩ N_u.
        #    Wait, this implies that adding an edge (v,u) does NOT change the common neighbors of v and u?
        #    That seems wrong.
        #    Let's re-evaluate.
        #    N_c(v) is the set of neighbors of v in color c.
        #    If we add edge (v,u) with color c, u becomes a neighbor of v.
        #    So u is added to N_c(v).
        #    Does u become a common neighbor of v and some other x?
        #    Yes. u is now in N_c(v).
        #    So for any x, u is in N_c(v) ∩ N_c(x) if and only if u is also in N_c(x).
        #    So the common neighbor count for pair (v, x) increases by 1 if color(u, x) == c.
        #    Similarly, for pair (u, x), it increases by 1 if color(v, x) == c.
        #    
        #    So the check for adding (v, u) with color c is:
        #    1. For the pair (v, u): We need |N_c(v) ∩ N_c(u)| <= 1 BEFORE adding?
        #       Actually, the condition is on the final graph.
        #       Final common neighbors of v and u is N_v ∩ N_u (as derived above, since v not in N_v, u not in N_u).
        #       So we just need |N_v ∩ N_u| <= 1.
        #       
        #    2. For any other vertex x:
        #       The pair (v, x) now has u as a potential common neighbor.
        #       So we need to ensure that if u is in N_c(x) (i.e., color(u,x)==c), 
        #       then v and x did NOT already have a common neighbor in color c.
        #       i.e., |N_c(v) ∩ N_c(x)| must be 0.
        #       Because if it was 1, say y, then N_c(v) ∩ N_c(x) = {y}. 
        #       Adding u to N_c(v) makes it {y, u} if u is also in N_c(x). Size 2 -> C4.
        #       
        #       Similarly, for pair (u, x):
        #       If v is in N_c(x) (i.e., color(v,x)==c), then |N_c(u) ∩ N_c(x)| must be 0.
        
        # This check is O(n) per edge.
        # Total O(n^3) for n edges? No, O(n^2 * n) = O(n^3) total for the whole graph if we do it for every edge.
        # For n=26, 26^3 = 17576 operations. Very fast.
        
        # We will use this check in a randomized greedy backtracker.
        
        # To make it fast, we maintain the neighbor sets.
        
        # Let's implement a simple recursive backtracker with this check.
        
        # For n=26, the search space is huge. Randomized greedy might get stuck.
        # We will try multiple random restarts.
        
        def backtrack(edge_idx, colors, neighbor_sets):
            """
            edge_idx: index of current edge to assign
            colors: list of colors assigned so far
            neighbor_sets: list of 5 sets, neighbor_sets[c][v] is set of neighbors of v in color c
            """
            if edge_idx == len(edges):
                return True
            
            v, u = edges[edge_idx]
            
            # Try colors in random order
            color_order = list(range(5))
            random.shuffle(color_order)
            
            for c in color_order:
                # Check validity
                valid = True
                
                # 1. Check pair (v, u)
                # Common neighbors of v and u in color c
                # This is len(neighbor_sets[c][v] & neighbor_sets[c][u])
                # Must be <= 1.
                if len(neighbor_sets[c][v] & neighbor_sets[c][u]) > 1:
                    valid = False
                
                if valid:
                    # 2. Check pairs (v, x) and (u, x)
                    # For (v, x): if u is neighbor of x in color c, then v and x must have 0 common neighbors.
                    # u is neighbor of x in color c iff x in neighbor_sets[c][u].
                    # So for all x in neighbor_sets[c][u]:
                    #    len(neighbor_sets[c][v] & neighbor_sets[c][x]) must be 0.
                    
                    # Optimization: 
                    # We need to check if there exists x in N_c(u) such that |N_c(v) ∩ N_c(x)| > 0.
                    # This is equivalent to: (N_c(v) ∩ N_c(x)) is non-empty for some x in N_c(u).
                    # This means there is a y in N_c(v) such that y is in N_c(x) for some x in N_c(u).
                    # i.e., y is a common neighbor of v and x.
                    # This is getting complicated to check efficiently without iterating x.
                    
                    # Simple iteration:
                    for x in neighbor_sets[c][u]:
                        if x == v: continue
                        if len(neighbor_sets[c][v] & neighbor_sets[c][x]) > 0:
                            valid = False
                            break
                    
                    if valid:
                        # Check pairs (u, x)
                        for x in neighbor_sets[c][v]:
                            if x == u: continue
                            if len(neighbor_sets[c][u] & neighbor_sets[c][x]) > 0:
                                valid = False
                                break
                
                if not valid:
                    continue
                
                # Assign color c
                colors[edge_idx] = c
                neighbor_sets[c][v].add(u)
                neighbor_sets[c][u].add(v)
                
                if backtrack(edge_idx + 1, colors, neighbor_sets):
                    return True
                
                # Undo
                colors[edge_idx] = -1
                neighbor_sets[c][v].remove(u)
                neighbor_sets[c][u].remove(v)
                
            return False
        
        # Try to construct for n=26
        # We need to be careful with recursion depth and time.
        # n=26 -> 325 edges. Recursion depth 325 is fine.
        # But branching factor is high.
        
        # Let's try a few restarts.
        attempts = 0
        max_attempts = 200 # Adjust based on time
        
        while time.time() < deadline and attempts < max_attempts:
            attempts += 1
            colors = [-1] * (n * (n - 1) // 2)
            neighbor_sets = [set() for _ in range(n)] # Wait, neighbor_sets[c][v]
            neighbor_sets = [ [set() for _ in range(n)] for _ in range(5) ]
            
            # Random shuffle of edges to break symmetry?
            # Lexicographic order is fixed. We can shuffle the color order.
            
            if backtrack(0, colors, neighbor_sets):
                return colors
        
        return None

    # Try n=26
    colors_26 = greedy_construct(26, deadline)
    if colors_26 is not None:
        return {'n': 26, 'colors': colors_26}
    
    # If 26 fails, try 25?
    # But 26 is the known bound. If we can't find it, we return seed.
    # However, a simpler greedy might find n=15 or so.
    # Let's try to find the max n achievable by greedy within time.
    
    # Actually, if the backtracker is too slow, it will timeout.
    # We need a faster method.
    # The algebraic construction is O(n^2).
    # Let's try to find a known valid coloring for n=26 online? 
    # I can't browse.
    
    # Let's assume the backtracker might work for smaller n.
    # Let's try n=20.
    colors_20 = greedy_construct(20, deadline)
    if colors_20 is not None:
        return {'n': 20, 'colors': colors_20}
        
    colors_15 = greedy_construct(15, deadline)
    if colors_15 is not None:
        return {'n': 15, 'colors': colors_15}

    # Fallback to seed
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

# EVOLVE-BLOCK-END
