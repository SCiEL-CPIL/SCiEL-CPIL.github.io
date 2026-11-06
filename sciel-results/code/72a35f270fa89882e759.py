import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    max_n_target = 26
    current_n = 1
    # colors[i][j] for i < j
    # We store colors in a list of lists or a flat array?
    # Flat array is faster for indexing if we map (i,j) to index.
    # idx = i*n + j - (i+1)*(i+2)//2 ? No, standard triangular index.
    # Let's use a 2D list of lists for simplicity, or a dict.
    # For n=26, 26*25/2 = 325 edges. Small.
    
    # State:
    # n: current number of vertices
    # colors: list of lists, colors[i][j] is color of edge (i,j)
    # neighbors: neighbors[i][k] is bitmask of neighbors of i in color k
    # conflict: conflict[i][j][k] is True if i and j have >= 2 common neighbors in color k
    
    # We will use a recursive backtracking search to extend n by 1.
    
    def get_edge_idx(i, j, n):
        # Not needed if we use 2D array
        pass

    # Initialize
    n = 1
    colors = [[0] * n for _ in range(n)]
    neighbors = [[0] * 5 for _ in range(n)]
    # conflict[i][j] is a 5-bit mask? Or list of bools?
    # conflict[i][j][k] = 1 if conflict
    conflict = [[[0]*5 for _ in range(n)] for __ in range(n)]
    
    best_n = n
    best_colors = colors
    
    # To allow backtracking, we need to save states or use a stack.
    # Simpler: Use a randomized local search on the whole graph for a fixed n?
    # No, extension is better because it preserves validity of previous steps.
    
    # Let's implement a function try_extend(n) that returns True if it can extend to n+1.
    # It modifies colors, neighbors, conflict.
    # If it fails, we need to restore state.
    # Copying state for n=26 is cheap.
    
    def copy_state(n, colors, neighbors, conflict):
        c = [row[:] for row in colors]
        nb = [row[:] for row in neighbors]
        cf = [[[val for val in row] for row in mat] for mat in conflict]
        return c, nb, cf

    def restore_state(n, colors, neighbors, conflict, c, nb, cf):
        for i in range(n):
            for j in range(n):
                colors[i][j] = c[i][j]
            for k in range(5):
                neighbors[i][k] = nb[i][k]
            for j in range(n):
                for k in range(5):
                    conflict[i][j][k] = cf[i][j][k]

    # Extension logic
    # We want to add vertex v = n.
    # We need to assign colors to edges (v, u) for u in 0..n-1.
    # Let's use backtracking to assign these colors.
    
    def can_extend(n, colors, neighbors, conflict):
        v = n
        # Variables: c_u for u in 0..n-1
        # We assign one by one.
        # Order: random or by constraint tightness.
        # Let's use random order for diversification.
        order = list(range(n))
        random.shuffle(order)
        
        # Forward checking:
        # When we assign color k to (v, u), we update potential conflicts for other unassigned vertices?
        # Actually, the constraint is between pairs (u, w).
        # If we assign color k to (v, u), then for any w, if we later assign color k to (v, w),
        # we must ensure that u and w do NOT have a conflict in color k.
        # conflict[u][w][k] is precomputed from the graph of 0..n-1.
        
        # So the constraint is purely on the assignment vector C[0..n-1].
        # C[u] = color of (v, u).
        # Constraint: For all u < w, if conflict[u][w][k] is True, then NOT (C[u]==k and C[w]==k).
        
        # This is a graph coloring problem on the graph G_conflict_k for each k?
        # No, it's a single assignment with constraints across colors.
        
        # Backtracking search:
        assigned = [-1] * n
        # To speed up, we can use a simple recursive search with pruning.
        
        def backtrack(idx):
            if idx == n:
                return True
            
            u = order[idx]
            # Try colors 0..4
            # Heuristic: Try colors that are less constrained?
            # For simplicity, try 0..4 in random order?
            colors_to_try = [0, 1, 2, 3, 4]
            random.shuffle(colors_to_try)
            
            for c in colors_to_try:
                # Check if assigning c to u is valid with already assigned vertices
                valid = True
                for prev_idx in range(idx):
                    w = order[prev_idx]
                    if conflict[u][w][c]:
                        if assigned[w] == c:
                            valid = False
                            break
                if valid:
                    assigned[u] = c
                    if backtrack(idx + 1):
                        return True
                    assigned[u] = -1
            return False
        
        return backtrack(0), assigned

    # Main loop
    for target_n in range(2, max_n_target + 1):
        if time.time() > deadline:
            break
        
        success = False
        # Try multiple random restarts for the extension step
        # But the state of n-1 is fixed.
        # If we fail to extend n-1 to n, we might need to change the coloring of n-1.
        # This implies we should have a stack of states or restart the whole search for smaller n.
        
        # Strategy:
        # Maintain a stack of states for n=1..current_n.
        # If extension fails, pop the last state, try a different extension for it, or pop further.
        # This is DFS over the space of valid colorings.
        
        # To keep it simple and fast, let's use a "restart" strategy.
        # If we fail to extend from a specific state of n-1, we generate a new random valid coloring for n-1?
        # That's expensive.
        
        # Better: Use a local search on the extension step.
        # If backtracking fails, it means no solution for this specific n-1 coloring.
        # We should backtrack to n-2 and try a different coloring for n-2.
        
        # Let's implement a simple DFS with backtracking over the extension steps.
        # State stack:
        # Each entry: (n, colors, neighbors, conflict)
        
        # Since we are inside a loop, let's manage the state manually.
        
        # We already have a valid coloring for n-1.
        # Let's try to extend.
        # If it fails, we need to "undo" the last extension and try a different one.
        # This requires storing the previous state.
        
        # Let's restructure:
        # We keep a list `history` of states.
        # history[n] is a tuple (colors, neighbors, conflict) for K_n.
        
        # Initialize history[1]
        
        # This might be memory heavy but n is small.
        
        # Let's try a simpler approach:
        # Run a local search on the full graph K_n for n=26 directly?
        # Start with random coloring, minimize C4 count.
        # If 0, we win.
        
        # Given the difficulty of backtracking implementation in this constrained prompt,
        # and the success of the "incremental neighbor mask" program (score 2.0),
        # I will implement a fast local search for n=26.
        
        pass

    # Let's switch to Local Search for n=26.
    # It's more robust to implement correctly in a short time.
    
    n = 26
    # Initialize random coloring
    colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
    
    # Helper to get/set color
    def get_idx(i, j):
        if i > j: i, j = j, i
        return i * n + j - (i + 1) * (i + 2) // 2 # This formula is for 1-based?
        # Standard 0-based triangular:
        # 0: (0,1)
        # 1: (0,2)
        # 2: (1,2)
        # 3: (0,3)
        # idx(i,j) = i*n - i*(i+1)//2 + j - i - 1 ?
        # Let's just use a 2D list for safety and speed (n=26 is small).
        
    # Re-initialize with 2D list
    colors_2d = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            colors_2d[i][j] = random.randint(0, 4)
            colors_2d[j][i] = colors_2d[i][j]
            
    # Precompute neighbors
    neighbors = [[0] * 5 for _ in range(n)]
    def update_neighbors(i, j, old_c, new_c):
        neighbors[i][old_c] &= ~(1 << j)
        neighbors[i][new_c] |= (1 << j)
        neighbors[j][old_c] &= ~(1 << i)
        neighbors[j][new_c] |= (1 << i)

    # Initialize neighbors
    for i in range(n):
        for j in range(i + 1, n):
            c = colors_2d[i][j]
            neighbors[i][c] |= (1 << j)
            neighbors[j][c] |= (1 << i)

    # Calculate violations
    # A violation is a C4.
    # Count C4s: For each pair (u, v), for each color k, if they have 2 common neighbors, that's a violation.
    # Actually, each C4 is counted 4 times (once per vertex pair?)
    # Let's just count the number of pairs (u,v) that have >= 2 common neighbors in any color.
    # This is a proxy for C4 count.
    
    def count_violations():
        viol = 0
        for u in range(n):
            for v in range(u + 1, n):
                for k in range(5):
                    common = neighbors[u][k] & neighbors[v][k]
                    if common & (common - 1): # has at least 2 bits
                        viol += 1
        return viol

    best_viol = count_violations()
    best_colors = [row[:] for row in colors_2d]
    
    # Local Search
    # Simulated Annealing or Hill Climbing with restarts.
    # Given 120s, we can do many steps.
    
    step = 0
    while time.time() < deadline:
        # Pick a random edge (i, j) and change its color
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j: continue
        if i > j: i, j = j, i
        
        old_c = colors_2d[i][j]
        new_c = random.randint(0, 4)
        if old_c == new_c: continue
        
        # Update neighbors
        update_neighbors(i, j, old_c, new_c)
        
        # Calculate delta violation
        # Only violations involving i or j change.
        # Pairs (u, v) where u or v is i or j.
        
        # Re-calculate violation count for pairs involving i or j.
        # This is O(n) per step.
        
        # To be faster, we can just re-calculate total if it's fast enough.
        # n=26, pairs=325. 325 * 5 ops = 1625 ops. Very fast.
        # Let's just re-calculate total.
        
        new_viol = count_violations()
        
        if new_viol <= best_viol:
            # Accept
            best_viol = new_viol
            best_colors = [row[:] for row in colors_2d]
            
            if new_viol == 0:
                break
        else:
            # Reject (or accept with prob exp(-delta/T) for SA)
            # Let's use simple hill climbing first, with random restarts if stuck.
            # Revert
            update_neighbors(i, j, new_c, old_c)
            
        step += 1
        if step % 10000 == 0 and best_viol > 0:
             # Random restart if stuck?
             # Just keep going, randomization in edge picking helps.
             pass

    # If we found 0 violations, we have a valid n=26.
    # If not, we return the best we found?
    # The task requires a VALID coloring.
    # So if best_viol > 0, we must return a valid one.
    # We should keep a history of valid colorings for smaller n?
    # Or just return n=11 seed if we fail?
    
    # To be safe, let's try to find a valid coloring for smaller n if n=26 fails.
    # But the score is n/11.
    # If we fail 26, try 25, 24...
    
    # Actually, the local search might get stuck.
    # Let's add a fallback: If we don't find 0 violations for n=26,
    # we try n=25, etc.
    
    # But re-running local search for each n is expensive.
    
    # Alternative: The backtracking extension method is more likely to find a valid solution
    # if one exists, because it maintains validity.
    
    # Let's go back to the extension method. It's more robust for "find ANY valid solution".
    
    pass

    # Final Decision: Implement the Extension Method.
    # It's O(n * 5^n) worst case, but pruning makes it fast.
    # And it guarantees validity.
    
    # Re-implement extension method cleanly.
    
    def solve_max_n():
        n = 1
        colors = [[0] * n for _ in range(n)]
        neighbors = [[0] * 5 for _ in range(n)]
        conflict = [[[0] * 5 for _ in range(n)] for __ in range(n)]
        
        # Stack for backtracking
        # We'll use a simple recursive function with global state and undo.
        
        max_reached = 1
        
        def extend(n, colors, neighbors, conflict):
            nonlocal max_reached
            v = n
            if time.time() > deadline:
                return False
            
            # Save state for backtracking?
            # We'll use a recursive approach where we pass copies or use undo.
            # Copies are safer.
            
            # Try to assign colors to (v, u) for u in 0..n-1
            # Use backtracking
            
            order = list(range(n))
            random.shuffle(order)
            
            assigned = [-1] * n
            
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if idx == n:
                    # Success! Update global state
                    # Add vertex v
                    new_n = n + 1
                    # Extend colors
                    new_colors = [row[:] for row in colors]
                    for i in range(new_n):
                        if i < len(new_colors):
                            pass
                        # We need to add a row/col
                    # This copy logic is messy.
                    
                    # Let's just return the assigned colors.
                    return True
                
                u = order[idx]
                colors_to_try = [0, 1, 2, 3, 4]
                random.shuffle(colors_to_try)
                
                for c in colors_to_try:
                    valid = True
                    for prev_idx in range(idx):
                        w = order[prev_idx]
                        if conflict[u][w][c]:
                            if assigned[w] == c:
                                valid = False
                                break
                    if valid:
                        assigned[u] = c
                        if backtrack(idx + 1):
                            return True
                        assigned[u] = -1
                return False
            
            if backtrack(0):
                # Apply the extension
                # Add vertex v to structures
                # ...
                return True, assigned
            else:
                return False, None

        # This is getting complicated to manage state updates.
        # Given the time, I will output the local search version but with a fallback to the seed n=11 if it fails.
        # The local search for n=26 is likely to succeed given 120s.
        
        pass
        
    # I will provide the local search code, which is simpler to verify.
    # If it finds 0 violations, return n=26.
    # Else, return n=11 (seed) to guarantee a valid score.
    
    # Wait, the seed n=11 is valid.
    # If I return n=11, score is 1.0.
    # If I return n=26, score is 2.36.
    # If I return invalid, score is 0.
    
    # So I must ensure validity.
    # The local search `best_colors` is only valid if `best_viol == 0`.
    
    if best_viol == 0:
        # Flatten colors
        flat = []
        for i in range(n):
            for j in range(i + 1, n):
                flat.append(best_colors[i][j])
        return {'n': n, 'colors': flat}
    else:
        # Fallback to seed
        n_seed = 11
        return {'n': n_seed, 'colors': [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]}

# EVOLVE-BLOCK-END
