import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # 1. Construct base K32 coloring
    # Greenwood-Gleason colouring of K16 by cubic-residue classes of GF(16)
    # Doubled: two copies with colour 3 on all edges between them
    
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
    
    base_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                # Inside a copy of K16
                base_colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                # Between copies
                base_colors.append(3)
                
    n = 32
    colors = base_colors
    
    # Helper to check if adding a vertex with given edge colors is valid
    # edge_colors is a list of length n, where edge_colors[u] is the color of (new_vertex, u)
    # u is 0..n-1
    def is_valid_insertion(n_current, edge_colors):
        # For each color c, let S_c be the set of vertices u such that edge_colors[u] == c
        # Check if there is any edge (u, w) in the existing graph with color c where u, w are both in S_c
        # We need to map the list of colors to pairs.
        # The existing colors are stored in a flat list.
        # Let's create a lookup or just iterate.
        
        # For efficiency, pre-compute for each color c, the list of pairs (u, w) with color c.
        # But n changes. Let's just check on the fly.
        
        # Optimization: For each c, collect vertices in S_c.
        # Then for each pair in S_c, check if their color is c.
        
        sets = [[], [], [], []]
        for u in range(n_current):
            sets[edge_colors[u]].append(u)
            
        # Iterate through all existing edges
        idx = 0
        for i in range(n_current):
            for j in range(i + 1, n_current):
                c_existing = colors[idx]
                # Check if both i and j are in sets[c_existing]
                # This is O(1) if we use a set, but O(n) if list.
                # Since n is small (<50), list is fine.
                if i in sets[c_existing] and j in sets[c_existing]:
                    return False
                idx += 1
        return True

    # Backtracking search for vertex insertion
    # We want to assign colors to edges (new_v, 0), (new_v, 1), ..., (new_v, n-1)
    # such that is_valid_insertion returns True.
    
    def insert_vertex(n_current):
        # Order vertices by some heuristic? 
        # Vertices with high degree in color c might be more constrained.
        # For now, simple order 0..n-1.
        
        # We can use a recursive backtracker
        # To avoid deep recursion and stack overflow, we limit depth or use iterative.
        # But n is small.
        
        # Heuristic: Try colors in a specific order.
        # Maybe randomize? No, deterministic is safer for reproducibility, but we can try a few orders.
        
        # Let's try to find a valid assignment.
        # We can prune if a color c is assigned to vertex u, 
        # then for all v such that color(u,v)==c, the new vertex cannot have color c with v.
        
        # This is a CSP.
        # Let's use a simple DFS with pruning.
        
        assignment = [None] * n_current
        # Pre-compute constraints?
        # For each u, and each color c, if we assign c to (new_v, u),
        # then for all v where color(u,v)==c, we cannot assign c to (new_v, v).
        
        # Let's just use the is_valid_insertion check at the end, but that's slow.
        # Better to check incrementally.
        
        # Let's use a different approach:
        # For each color c, the set of vertices S_c assigned to color c must be an independent set in the graph G_c (where G_c has edges of color c).
        
        # We can iterate over subsets? No, 2^50 is too big.
        
        # Let's use backtracking on vertices.
        # Order vertices by the number of neighbors they have in each color?
        
        # Simple backtracking:
        def backtrack(idx):
            if time.time() > deadline - 5:
                return False # Timeout safety
            if idx == n_current:
                return True
            
            for c in range(4):
                # Check if assigning color c to vertex idx is consistent with previous assignments
                # Previous assignments are for vertices 0..idx-1
                # We need to ensure that for any k < idx:
                # If assignment[k] == c, then color(idx, k) != c.
                # Wait, the condition is:
                # If assignment[k] == c AND assignment[idx] == c, then color(k, idx) != c.
                
                valid = True
                for k in range(idx):
                    if assignment[k] == c:
                        # Check color of edge (k, idx)
                        # Need to get color of (k, idx) from the flat list
                        # Index calculation:
                        # Sum of lengths of rows 0..k-1 + (idx - k - 1) ? No.
                        # Number of edges before row k is k*(k-1)/2.
                        # In row k, the edge to idx is at position idx - k - 1.
                        # Total index = k*(k-1)//2 + (idx - k - 1)
                        edge_idx = k * (k - 1) // 2 + (idx - k - 1)
                        if colors[edge_idx] == c:
                            valid = False
                            break
                
                if valid:
                    assignment[idx] = c
                    if backtrack(idx + 1):
                        return True
                    assignment[idx] = None
            return False
        
        if backtrack(0):
            return assignment
        return None

    # Try to insert vertices
    for target_n in range(33, 51):
        if time.time() > deadline - 10:
            break
            
        # We need to maintain the list of colors.
        # When we add a vertex, we append n_new colors to the end of the list.
        # The new edges are (new_v, 0), (new_v, 1), ..., (new_v, n_old-1).
        # These are appended to the end of the `colors` list.
        
        assignment = insert_vertex(n)
        
        if assignment is not None:
            # Append the new edge colors to the global list
            colors.extend(assignment)
            n += 1
        else:
            # Failed to insert
            break
            
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
