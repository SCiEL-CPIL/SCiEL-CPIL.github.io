import time
import math
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 80.0  # seconds, leaving safety margin for 90s wall-clock
    
    N = 256
    
    # 1. Identify valid vertices (2*i is not a perfect square)
    valid = []
    for i in range(1, N + 1):
        s = 2 * i
        r = int(math.isqrt(s))
        if r * r != s:
            valid.append(i)
            
    M = len(valid)
    
    # Precompute squares up to 2*N
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
        
    # 2. Build conflict graph adjacency (bitset)
    # conflict[i] is bitset of j where valid[i] + valid[j] IS a perfect square
    # We want Maximum Independent Set in this conflict graph
    # Which is equivalent to Maximum Clique in the complement graph
    # Let's work directly with the conflict graph for MIS
    
    conflict = [0] * M
    
    for i in range(M):
        ui = valid[i]
        for j in range(i + 1, M):
            uj = valid[j]
            if ui + uj in squares:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
                
    # Also check diagonal: 2*ui is already checked in valid, so no self-loops needed
    # But wait, the problem says "a+b is never a perfect square for a<=b in the set, INCLUDING a=b"
    # So if 2*a is a perfect square, a cannot be in the set. We already filtered those out.
    # For a != b, we check a+b.
    
    # We want to find the largest subset S such that for all a, b in S with a <= b, a+b is not a square.
    # This is exactly the Maximum Independent Set in the graph where edges connect a,b if a+b is a square.
    
    # Let's use a branch and bound approach for Maximum Independent Set.
    # Or better, convert to Maximum Clique in the complement graph.
    
    # Complement graph: edge between i,j if valid[i] + valid[j] is NOT a square (and i != j)
    # And we also need to ensure no self-conflicts, which is handled by valid[].
    
    # Let's stick to the Maximum Clique formulation on the complement graph, as it's well-studied.
    
    # Complement adjacency: comp_adj[i] has bit j set if i != j and valid[i]+valid[j] is NOT a square.
    comp_adj = [0] * M
    for i in range(M):
        # Start with all bits set
        all_bits = (1 << M) - 1
        # Remove self
        all_bits &= ~(1 << i)
        # Remove conflict neighbors
        all_bits &= ~conflict[i]
        comp_adj[i] = all_bits
        
    # Tomita's algorithm for Maximum Clique
    best_clique_size = 0
    best_clique_set = set()
    
    # Greedy coloring for upper bound
    # We need to color the subgraph induced by P
    # In the context of Maximum Clique on comp_adj, we want to color vertices such that
    # adjacent vertices have different colors. The number of colors is an upper bound on clique size.
    # Wait, standard greedy coloring for clique finding:
    # We process vertices in some order. We assign the smallest possible color.
    # The number of colors used is an upper bound on the size of any clique in that subgraph.
    
    def greedy_color(P):
        # Returns list of (vertex, color) and max_color
        # We want to order vertices such that the coloring is efficient.
        # A common heuristic is to process vertices in order of decreasing degree in P.
        
        # Let's just use a simple greedy approach.
        # To make it faster, we can sort vertices by degree in P descending.
        vertices = []
        temp = P
        while temp:
            v = (temp & -temp).bit_length() - 1
            vertices.append(v)
            temp &= ~(1 << v)
            
        # Sort by degree in P descending
        vertices.sort(key=lambda v: (comp_adj[v] & P).bit_count(), reverse=True)
        
        color_of = {}
        max_color = 0
        for v in vertices:
            # Find smallest color not used by neighbors in P
            used_colors = set()
            nb = comp_adj[v] & P
            while nb:
                u = (nb & -nb).bit_length() - 1
                if u in color_of:
                    used_colors.add(color_of[u])
                nb &= ~(1 << u)
            
            c = 1
            while c in used_colors:
                c += 1
            color_of[v] = c
            if c > max_color:
                max_color = c
                
        return vertices, color_of, max_color

    def branch(P, Q, Q_size):
        nonlocal best_clique_size, best_clique_set
        
        if time.time() - start_time > TIME_LIMIT:
            return
            
        if not P:
            if Q_size > best_clique_size:
                best_clique_size = Q_size
                best_clique_set = set(Q)
            return
            
        # Greedy coloring to get upper bound
        vertices, color_of, max_color = greedy_color(P)
        if Q_size + max_color <= best_clique_size:
            return
            
        # Process vertices in reverse order of their color (highest color first)
        # This is a key heuristic in Tomita's algorithm
        # vertices is sorted by degree. We need to sort by color descending.
        # Actually, the standard approach is to assign colors and then process vertices
        # from highest color to lowest.
        
        # Let's re-order vertices by color descending
        vertices_sorted = sorted(vertices, key=lambda v: color_of[v], reverse=True)
        
        for v in vertices_sorted:
            if time.time() - start_time > TIME_LIMIT:
                return
            
            v_bit = 1 << v
            
            # If the upper bound for the remaining vertices (including v) is not promising, break
            # The color of v is an upper bound for the clique size including v and any vertices with color <= color_of[v]
            # Actually, in Tomita's algorithm, we iterate vertices in decreasing color order.
            # If Q_size + color_of[v] <= best_clique_size, we can break.
            if Q_size + color_of[v] <= best_clique_size:
                break
                
            # Include v
            new_P = P & comp_adj[v]
            new_Q = Q | {v}
            branch(new_P, new_Q, Q_size + 1)
            
            # Exclude v
            P &= ~v_bit
            
    # Initial call
    initial_P = (1 << M) - 1
    branch(initial_P, set(), 0)
    
    # Map indices back to values
    result = [valid[i] for i in best_clique_set]
    
    if not result:
        return [1]
        
    return result
