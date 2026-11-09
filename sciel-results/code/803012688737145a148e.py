"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 110.0  # 110 seconds to leave margin

    # 1. Generate the base Greenwood-Gleason colouring of K16 (Clebsch graph x 3)
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

    def get_seed_colors(n):
        """Generates colors for a graph of size n based on the K16 seed."""
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                # We use the Clebsch construction for vertices < 16.
                # For vertices >= 16, we need a strategy.
                # The seed program doubles it: two copies of K16 with color 3 between them.
                # This gives K32.
                # For n > 32, we need to extend.
                # A simple extension strategy:
                # If n <= 32, use the standard doubling.
                # If n > 32, we can try to add vertices by backtracking or random assignment + repair.
                
                if i // 16 == j // 16:
                    # Same "block" of 16
                    # Map to 0..15
                    ii = i % 16
                    jj = j % 16
                    colors.append(log[ii ^ jj] % 3)
                else:
                    # Different "blocks"
                    # If we have more than 2 blocks, this simple logic breaks.
                    # The seed only supports n=32 (2 blocks).
                    # Let's stick to the n=32 seed as the starting point for search.
                    # If we want to try n>32, we must handle this.
                    # For now, let's just return the K32 seed if n==32.
                    # If n != 32, this function is not directly applicable without modification.
                    # Let's assume we start with n=32.
                    colors.append(3)
        return colors

    # Start with n=32
    current_n = 32
    current_colors = get_seed_colors(current_n)
    
    # Convert to 2D array for fast access
    graph = [[0] * current_n for _ in range(current_n)]
    idx = 0
    for i in range(current_n):
        for j in range(i + 1, current_n):
            c = current_colors[idx]
            graph[i][j] = c
            graph[j][i] = c
            idx += 1

    best_n = current_n
    best_colors = current_colors[:]

    def validate(graph, n):
        """Check if the graph has any monochromatic triangles."""
        for i in range(n):
            for j in range(i + 1, n):
                c_ij = graph[i][j]
                for k in range(j + 1, n):
                    if graph[i][k] == c_ij and graph[j][k] == c_ij:
                        return False
        return True

    # Optimization: Precompute neighbors for each color?
    # Or just use the 2D array. n=32..50 is small enough.
    
    def try_add_vertex(graph, n):
        """
        Attempts to add a new vertex (index n) to the existing graph of size n.
        Returns (success, new_graph, new_colors_list) or (False, None, None).
        """
        # New vertex is n. Edges are (i, n) for i in 0..n-1.
        # We need to assign colors to these edges such that no triangle (i, j, n) is monochromatic.
        # Triangle (i, j, n) is monochromatic if graph[i][j] == color(i,n) == color(j,n).
        
        # Backtracking search
        # Order of vertices: 0, 1, ..., n-1
        # We can prune if for some i, all 4 colors are forbidden.
        
        # A color c is forbidden for edge (i, n) if there exists j < i such that graph[i][j] == c and color(j, n) == c.
        # Wait, triangle is (j, i, n). Colors: graph[j][i], color(j,n), color(i,n).
        # If graph[j][i] == c, and color(j,n) == c, then color(i,n) cannot be c.
        
        # Let's maintain a list of forbidden colors for each new edge.
        # Actually, we can just backtrack.
        
        def backtrack(i, assignments):
            if i == n:
                return True
            
            # Determine forbidden colors for edge (i, n)
            forbidden = [False] * 4
            for j in range(i):
                c_ji = graph[j][i]
                c_jn = assignments[j]
                if c_ji == c_jn:
                    forbidden[c_ji] = True
            
            # Try colors
            for c in range(4):
                if not forbidden[c]:
                    assignments[i] = c
                    if backtrack(i + 1, assignments):
                        return True
            return False

        assignments = [-1] * n
        if backtrack(0, assignments):
            # Success
            new_graph = [row[:] for row in graph]
            new_graph.append([0] * (n + 1))
            for i in range(n):
                new_graph[i].append(assignments[i])
                new_graph[n][i] = assignments[i]
                new_graph[n][n] = 0 # Diagonal doesn't matter, but keep consistent? No, just 0.
            
            # Build new colors list
            new_colors = []
            for i in range(n + 1):
                for j in range(i + 1, n + 1):
                    new_colors.append(new_graph[i][j])
            
            return True, new_graph, new_colors
        return False, None, None

    def local_search(graph, n, max_steps=1000, time_limit=None):
        """
        Attempts to improve the coloring or prepare it for vertex addition by flipping edges.
        Returns (success, new_graph, new_colors).
        Success means we found a valid configuration (which it already is) or maybe a better one?
        Actually, we just want to escape local minima in the "can we add a vertex" search.
        So, we flip random edges, check validity, and keep if valid.
        """
        current_graph = [row[:] for row in graph]
        current_n = n
        steps = 0
        
        while time_limit is None or time.time() < time_limit:
            if steps >= max_steps: break
            steps += 1
            
            # Pick a random edge
            i = random.randint(0, current_n - 1)
            j = random.randint(0, current_n - 1)
            if i == j: continue
            if i > j: i, j = j, i
            
            old_c = current_graph[i][j]
            new_c = random.randint(0, 3)
            if new_c == old_c: continue
            
            current_graph[i][j] = new_c
            current_graph[j][i] = new_c
            
            # Check validity
            # Only check triangles involving (i, j)
            valid = True
            for k in range(current_n):
                if k == i or k == j: continue
                # Triangle (i, j, k)
                c_ik = current_graph[i][k]
                c_jk = current_graph[j][k]
                if c_ik == c_jk == new_c:
                    valid = False
                    break
            
            if not valid:
                # Revert
                current_graph[i][j] = old_c
                current_graph[j][i] = old_c
            else:
                # Accept
                pass
        
        # Build colors
        colors = []
        for i in range(current_n):
            for j in range(i + 1, current_n):
                colors.append(current_graph[i][j])
        
        return current_graph, colors

    # Main loop
    # Strategy:
    # 1. Try to add vertex.
    # 2. If fails, do a local search to "shake up" the graph, then try again.
    # 3. Keep the best n found.
    
    # We start with n=32.
    # We can also try to start with a random valid K32? No, the seed is good.
    
    # Let's try to go up to 50 or 62.
    
    while time.time() < deadline:
        # Try to add a vertex
        success, new_graph, new_colors = try_add_vertex(graph, current_n)
        
        if success:
            current_n += 1
            graph = new_graph
            current_colors = new_colors
            if current_n > best_n:
                best_n = current_n
                best_colors = current_colors[:]
            
            # If we reached a high number, maybe stop?
            # The record is 50. 51 is new.
            # Let's keep trying until time runs out or we fail repeatedly.
            # But adding a vertex is hard.
            # If we succeed, we are in a new state.
            # We should probably do some local search to make it more "robust" for the next addition?
            # Or just try to add again immediately.
            
            # If we are at n=50, we are doing great.
            
        else:
            # Failed to add vertex.
            # Perform local search to find a different valid coloring of K_current_n
            # that might allow adding a vertex.
            
            # Time check
            if time.time() > deadline - 5:
                break
                
            # Do local search for a short time
            search_time_limit = time.time() + 0.5 # 0.5s per attempt?
            # Or fixed steps?
            # Let's do a few hundred flips.
            
            new_graph, new_colors = local_search(graph, current_n, max_steps=500, time_limit=search_time_limit)
            
            # Update state
            graph = new_graph
            current_colors = new_colors
            
            # Optionally, try to add again immediately after search?
            # The loop will handle it.
            
            # If we fail many times, maybe we are stuck at this n?
            # We could try to reduce n? No, we want to maximize.
            # We just keep searching for a valid K_n that extends to K_{n+1}.
            
            # If current_n is small (e.g. 32), we might be able to jump?
            # No, we must add one by one.
            
            # Heuristic: If we are stuck at n=32, maybe the Clebsch structure is too rigid?
            # Local search should break the symmetry.
            
            # If we are stuck at n=50, we are at the limit of known constructions.
            # Breaking to 51 is the goal.

    # Final validation of best result
    # best_colors corresponds to best_n
    # We should ensure best_colors is valid.
    # It was generated by valid steps, so it should be.
    
    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
