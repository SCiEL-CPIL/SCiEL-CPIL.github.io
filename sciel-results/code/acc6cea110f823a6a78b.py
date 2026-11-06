"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 115.0  # safety margin before 120s limit
    
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
    
    # Base: n=32 construction (two Clebsch 16-vertex blocks, cross edges color 3)
    base_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                base_colors.append(log[i ^ j] % 3)
            else:
                base_colors.append(3)
    
    best_n = 32
    best_colors = base_colors[:]
    
    # Build adjacency bitsets for each color class for fast triangle checking
    # adj[c][i] is a bitmask of vertices connected to i by color c
    def build_adj(colors, n):
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        return adj
    
    def check_no_mono_triangle(adj, n):
        """Check if any color class has a triangle using bitsets."""
        for c in range(4):
            for i in range(n):
                # For each neighbor j of i in color c, check if any common neighbor k
                neighbors = adj[c][i]
                j_mask = neighbors
                while j_mask:
                    j_bit = j_mask & (-j_mask)
                    j = j_bit.bit_length() - 1
                    j_mask ^= j_bit
                    if j > i:
                        # Check if i and j share a common neighbor in color c
                        common = adj[c][i] & adj[c][j]
                        if common:
                            return True
        return False
    
    def try_add_vertex(adj, colors, n, deadline):
        """Try to add vertex n. Returns new_edge_colors list or None."""
        # For new vertex v=n, we assign colors to edges (v, i) for i=0..n-1
        # Constraint: for each pair (i,j) with i<j<n, if colors[i,j]=c,
        # then NOT(new_color[i]==c AND new_color[j]==c)
        # Equivalently: for each color c, the set S_c = {i : new_color[i]=c}
        # must be an independent set in the c-colored subgraph
        
        # Build the constraint: for each color c, which pairs (i,j) have color c
        # If i and j are both in S_c, that's a violation
        
        # Use backtracking: assign colors to edges (v,0), (v,1), ..., (v,n-1)
        # with constraint propagation
        
        # Precompute: for each color c, the adjacency within existing vertices
        # adj_c[i] = bitmask of j such that edge (i,j) has color c
        
        # For the new vertex, new_color[i] = c means i is in S_c
        # S_c must be independent in color-c subgraph
        
        # Greedy + backtracking approach
        new_colors = [-1] * n
        
        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == n:
                return True
            
            # Try colors in order 0,1,2,3
            for c in range(4):
                # Check if adding pos to S_c violates independence
                # S_c currently = {i < pos : new_colors[i] == c}
                # Need: for all i in S_c, edge (i, pos) is NOT color c
                # i.e., pos not in adj[c][i] for any i in S_c
                
                valid = True
                for i in range(pos):
                    if new_colors[i] == c:
                        if (adj[c][i] >> pos) & 1:
                            valid = False
                            break
                
                if valid:
                    new_colors[pos] = c
                    if backtrack(pos + 1):
                        return True
                    new_colors[pos] = -1
            
            return False
        
        # Try multiple random orderings for diversity
        for attempt in range(50):
            if time.time() > deadline:
                break
            # Shuffle the order of vertices to assign
            order = list(range(n))
            random.shuffle(order)
            
            # Re-map: we assign in shuffled order but need consistent indexing
            # Actually, let's just try the natural order with different random seeds
            # or use a smarter heuristic
            
            # Try: assign in order of most constrained first
            # For each vertex i, count how many other vertices j have edge (i,j)
            # in each color - more constrained = fewer options
            
            # Simple: just try natural order
            if backtrack(0):
                return new_colors
        
        return None
    
    current_n = 32
    current_colors = base_colors[:]
    current_adj = build_adj(current_colors, current_n)
    
    # Verify base is valid
    assert not check_no_mono_triangle(current_adj, current_n), "Base coloring invalid!"
    
    # Try to extend
    while current_n < 62:
        if time.time() > deadline:
            break
        
        new_vertex_colors = try_add_vertex(current_adj, current_colors, current_n, deadline)
        if new_vertex_colors is None:
            break
        
        # Add the new vertex
        for i in range(current_n):
            current_colors.append(new_vertex_colors[i])
        
        # Update adjacency
        for i in range(current_n):
            c = new_vertex_colors[i]
            current_adj[c][i] |= (1 << current_n)
            current_adj[c][current_n] |= (1 << i)
        
        # Add self-row for new vertex
        current_adj[0].append(0)
        current_adj[1].append(0)
        current_adj[2].append(0)
        current_adj[3].append(0)
        for c in range(4):
            current_adj[c][current_n] = 0  # already set above
        
        current_n += 1
        
        # Verify no monochromatic triangle
        if check_no_mono_triangle(current_adj, current_n):
            # This shouldn't happen, but roll back
            current_n -= 1
            break
        
        if current_n > best_n:
            best_n = current_n
            best_colors = current_colors[:]
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
