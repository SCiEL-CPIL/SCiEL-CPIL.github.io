import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # 1. Generate the base n=32 solution
    # Greenwood-Gleason coloring of K16
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
        
    base_colors_16 = {}
    for i in range(16):
        for j in range(i + 1, 16):
            base_colors_16[(i, j)] = log[i ^ j] % 3

    # Double to 32
    n_base = 32
    # neighbors[c][v] is a bitmask of vertices connected to v by color c
    neighbors = [[0] * n_base for _ in range(4)]
    
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if i // 16 == j // 16:
                c = base_colors_16[(i % 16, j % 16)]
            else:
                c = 3
            neighbors[c][i] |= 1 << j
            neighbors[c][j] |= 1 << i

    current_n = n_base
    best_n = n_base
    best_neighbors = [row[:] for row in neighbors]

    # 2. Backtracking to extend
    # We try to add vertices one by one.
    # We use a recursive function that tries to color edges for the new vertex.
    
    def can_extend(v_new, neighbors, deadline):
        """
        Tries to add vertex v_new to the graph defined by neighbors (size v_new).
        Returns True if successful, updating neighbors.
        """
        # neighbors is a list of 4 lists of ints, size v_new
        # We need to assign colors to (v_new, u) for u in 0..v_new-1
        # We do this in order u=0, 1, ..., v_new-1
        
        # To make it faster, we can use symmetry or heuristics.
        # Let's just backtrack.
        
        # Pre-compute the "forbidden" sets?
        # When we pick color c for (v_new, u), we add u to neighbors[c][v_new].
        # The constraint is neighbors[c][v_new] & neighbors[c][u] == 0.
        
        # We'll use a recursive helper
        def backtrack(u, current_neighbors):
            if time.time() > deadline:
                return False
            
            if u == v_new:
                return True
            
            # Try colors 0,1,2,3
            # Heuristic: try colors that have fewer conflicts first?
            # Or just random order?
            colors_to_try = [0, 1, 2, 3]
            random.shuffle(colors_to_try)
            
            for c in colors_to_try:
                # Check if adding u to color c for v_new is valid
                # current_neighbors[c][v_new] is the bitmask of already assigned neighbors of v_new with color c
                # neighbors[c][u] is the bitmask of neighbors of u with color c (among 0..v_new-1)
                if current_neighbors[c][v_new] & neighbors[c][u]:
                    continue
                
                # Assign
                current_neighbors[c][v_new] |= 1 << u
                # Note: we don't need to update neighbors[c][u] because u < v_new and we are only adding v_new
                # But wait, the neighbor list for u should include v_new?
                # For future vertices, yes. But for the current constraint check, we only care about existing edges.
                # However, if we are building the structure for the NEXT vertex, we need neighbors[c][u] to include v_new.
                # So we should update neighbors[c][u] as well.
                
                # Let's modify the approach:
                # We maintain a global `neighbors` structure.
                # When we add v_new, we update neighbors[c][v_new] and neighbors[c][u].
                
                # Save state
                old_v_new = current_neighbors[c][v_new]
                old_u = neighbors[c][u]
                
                current_neighbors[c][v_new] |= 1 << u
                neighbors[c][u] |= 1 << v_new
                
                if backtrack(u + 1, current_neighbors):
                    return True
                
                # Undo
                current_neighbors[c][v_new] = old_v_new
                neighbors[c][u] = old_u
                
            return False

        # Initialize neighbors for the new vertex
        new_neighbors = [row[:] for row in neighbors]
        for c in range(4):
            new_neighbors[c].append(0) # v_new has no neighbors yet
        
        # We need to pass the updated neighbors structure to the recursive function
        # The recursive function modifies `neighbors` and `current_neighbors`
        # This is tricky with Python closures. Let's use a class or pass by reference carefully.
        
        # Let's rewrite the extension logic more cleanly.
        
        return False # Placeholder, will implement properly below

    # Proper implementation of extension
    def try_add_vertex(v_new, neighbors, deadline):
        # neighbors is 4 x v_new matrix of bitmasks
        # We create a new matrix of size 4 x (v_new + 1)
        new_neighbors = [[0] * (v_new + 1) for _ in range(4)]
        for c in range(4):
            for u in range(v_new):
                new_neighbors[c][u] = neighbors[c][u]
        
        # We need to assign colors to (v_new, u) for u in 0..v_new-1
        # We will use a recursive function
        def backtrack(u):
            if time.time() > deadline:
                return False
            if u == v_new:
                return True
            
            # Try colors
            # Heuristic: check which colors are "open" for u
            # A color c is open if new_neighbors[c][v_new] & new_neighbors[c][u] == 0
            
            # Order: try colors with least flexibility?
            # Just iterate 0-3
            for c in range(4):
                if new_neighbors[c][v_new] & new_neighbors[c][u]:
                    continue
                
                # Assign
                new_neighbors[c][v_new] |= 1 << u
                new_neighbors[c][u] |= 1 << v_new
                
                if backtrack(u + 1):
                    return True
                
                # Undo
                new_neighbors[c][v_new] &= ~(1 << u)
                new_neighbors[c][u] &= ~(1 << v_new)
                
            return False

        if backtrack(0):
            return new_neighbors
        return None

    # Main loop: try to add vertices
    current_neighbors = [row[:] for row in neighbors]
    
    while time.time() < deadline:
        next_n = current_n + 1
        new_neighbors = try_add_vertex(next_n - 1, current_neighbors, deadline)
        if new_neighbors:
            current_n = next_n
            current_neighbors = new_neighbors
            best_n = current_n
            best_neighbors = [row[:] for row in current_neighbors]
            # Optional: restart with randomization if we get stuck?
            # For now, greedy extension.
        else:
            # Stuck. Try to randomize the last few edges?
            # Or just break?
            # To improve chances, we can try a few random restarts of the last vertex assignment.
            found = False
            for _ in range(100):
                if time.time() > deadline:
                    break
                # Shuffle the order of colors in try_add_vertex?
                # We can modify try_add_vertex to accept a random seed or shuffle colors.
                # Let's just break for now to keep it simple.
                # A better strategy: if we fail, we can try to "relax" the constraints? No, they are hard.
                # We can try to add vertex with a different random order of colors.
                # Let's implement a randomized version of try_add_vertex.
                pass
            break

    # Convert best_neighbors to the required format
    colors = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            # Find color
            for c in range(4):
                if best_neighbors[c][i] & (1 << j):
                    colors.append(c)
                    break
    
    return {'n': best_n, 'colors': colors}
