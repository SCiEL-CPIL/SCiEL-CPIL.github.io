"""Combined program: Deterministic seed + Local Search Extension."""
import time
import random
import copy
from functools import lru_cache

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # 115 seconds to leave margin for overhead/return

    # 1. Seed: Greenwood-Gleason colouring of K16 (Clebsch) doubled to n=32
    # GF(16) multiplication
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    # Precompute log table for GF(16)
    log = {}
    x = 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    
    # Initial n=32 construction
    def get_seed_graph(n):
        graph = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if i // 16 == j // 16:
                    c = log[(i % 16) ^ (j % 16)] % 3
                else:
                    c = 3
                graph[i][j] = c
                graph[j][i] = c
        return graph

    current_n = 32
    graph = get_seed_graph(current_n)
    best_n = current_n
    best_graph = copy.deepcopy(graph)

    def build_bitmasks(g, n):
        """Build bitmask representation: masks[c][i] = bitmask of vertices j where g[i][j]==c"""
        masks = [[0] * n for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                c = g[i][j]
                masks[c][i] |= (1 << j)
                masks[c][j] |= (1 << i)
        return masks

    def count_mono_triangles_fast(g, n):
        """Count monochromatic triangles using bitmasks."""
        masks = build_bitmasks(g, n)
        count = 0
        for c in range(4):
            for i in range(n):
                # For each vertex i, look at neighbors j>i with color c
                # and count common neighbors k>j with color c
                ni = masks[c][i]
                # Only consider j > i
                ni_high = ni & ~((1 << (i + 1)) - 1)
                while ni_high:
                    lsb = ni_high & (-ni_high)
                    j = lsb.bit_length() - 1
                    ni_high ^= lsb
                    # Common neighbors of i and j with color c, with k > j
                    common = masks[c][i] & masks[c][j] & ~((1 << (j + 1)) - 1)
                    count += common.bit_count()
        return count

    def is_valid_graph(g, n):
        """Check if the graph has no monochromatic triangles using bitmasks."""
        masks = build_bitmasks(g, n)
        for c in range(4):
            for i in range(n):
                ni = masks[c][i]
                ni_high = ni & ~((1 << (i + 1)) - 1)
                while ni_high:
                    lsb = ni_high & (-ni_high)
                    j = lsb.bit_length() - 1
                    ni_high ^= lsb
                    common = masks[c][i] & masks[c][j] & ~((1 << (j + 1)) - 1)
                    if common:
                        return False
        return True

    def count_mono_triangles(g, n):
        """Count monochromatic triangles in the graph."""
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                c_ij = g[i][j]
                for k in range(j + 1, n):
                    if g[i][k] == c_ij and g[j][k] == c_ij:
                        count += 1
        return count

    def local_search_repair(g, n, max_iter=10000, time_limit=None):
        """
        Attempt to repair the graph by fixing monochromatic triangles.
        Uses bitmask-based triangle counting for efficient evaluation.
        Returns True if the graph is valid after repair, False otherwise.
        """
        if time_limit is None:
            time_limit = deadline - time.time()
        
        current_count = count_mono_triangles_fast(g, n)
        if current_count == 0:
            return True
        
        for it in range(max_iter):
            if time.time() - start_time > deadline - 5:
                return False
            
            if current_count == 0:
                return True
            
            # Find a monochromatic triangle to fix
            masks = build_bitmasks(g, n)
            found = False
            for c in range(4):
                for i in range(n):
                    ni = masks[c][i]
                    ni_high = ni & ~((1 << (i + 1)) - 1)
                    while ni_high:
                        lsb = ni_high & (-ni_high)
                        j = lsb.bit_length() - 1
                        ni_high ^= lsb
                        common = masks[c][i] & masks[c][j] & ~((1 << (j + 1)) - 1)
                        if common:
                            # Found mono triangle (i, j, k) with color c
                            k_lsb = common & (-common)
                            k = k_lsb.bit_length() - 1
                            
                            # Try to break it by changing one of the three edges
                            edge_choices = [(i, j), (i, k), (j, k)]
                            random.shuffle(edge_choices)
                            
                            best_edge = None
                            best_new_color = None
                            best_delta = None  # We want to minimize increase in mono triangles
                            
                            for (a, b) in edge_choices:
                                old_color = g[a][b]
                                for new_color in [0, 1, 2, 3]:
                                    if new_color == old_color:
                                        continue
                                    # Evaluate: how many new mono triangles does this create?
                                    # A mono triangle with color new_color involving edge (a,b)
                                    # exists for each m where g[m][a]==new_color and g[m][b]==new_color
                                    common_new = 0
                                    for m in range(n):
                                        if m == a or m == b:
                                            continue
                                        if g[m][a] == new_color and g[m][b] == new_color:
                                            common_new += 1
                                    
                                    # Also, removing old_color edge (a,b) removes mono triangles
                                    # with old_color where g[m][a]==old_color and g[m][b]==old_color
                                    removed = 0
                                    for m in range(n):
                                        if m == a or m == b:
                                            continue
                                        if g[m][a] == old_color and g[m][b] == old_color:
                                            removed += 1
                                    
                                    delta = common_new - removed
                                    
                                    if best_delta is None or delta < best_delta:
                                        best_delta = delta
                                        best_edge = (a, b)
                                        best_new_color = new_color
                            
                            # Apply the best change
                            if best_edge is not None and best_delta <= 1:
                                a, b = best_edge
                                g[a][b] = best_new_color
                                g[b][a] = best_new_color
                                current_count += best_delta
                                found = True
                                break
                            
                            # If no good single edge fix, try a random perturbation
                            if random.random() < 0.3:
                                a = random.randint(0, n - 1)
                                b = random.randint(0, n - 1)
                                if a != b:
                                    old_color = g[a][b]
                                    new_color = random.randint(0, 3)
                                    if new_color != old_color:
                                        g[a][b] = new_color
                                        g[b][a] = new_color
                                        current_count = count_mono_triangles_fast(g, n)
                                        found = True
                                        break
                            break
                    if found:
                        break
                if found:
                    break
            
            if not found:
                # No more changes to make
                return is_valid_graph(g, n)
        
        return is_valid_graph(g, n)

    def add_vertex_backtrack(g, n, time_limit=1.0):
        """
        Attempts to add a new vertex at index n to graph g (size n).
        Returns True if successful (g is updated to size n+1), False otherwise.
        Uses backtracking with bitmask-based pruning and time limit.
        """
        assignments = [-1] * n
        backtrack_start = time.time()
        
        # Precompute color masks for each vertex: color_masks[i][c] = bitmask of vertices j where g[i][j]==c
        color_masks = [[0] * 4 for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                c = g[i][j]
                color_masks[i][c] |= (1 << j)
                color_masks[j][c] |= (1 << i)
        
        def backtrack(i):
            if time.time() - backtrack_start > time_limit:
                return False
            
            if i == n:
                return True
            
            # For edge (i, n) with color c, we need: no k < i such that g[k][i]==c AND assignments[k]==c
            # Equivalently: the set of k < i with g[k][i]==c must be disjoint from {k < i : assignments[k]==c}
            
            # Precompute: for each color c, the bitmask of k < i with g[k][i]==c
            # And the bitmask of k < i with assignments[k]==c
            
            # Build assignment mask per color
            assign_masks = [0, 0, 0, 0]
            for k in range(i):
                if assignments[k] != -1:
                    assign_masks[assignments[k]] |= (1 << k)
            
            colors_to_try = [0, 1, 2, 3]
            random.shuffle(colors_to_try)
            
            for c in colors_to_try:
                # Check constraint: no k < i with g[k][i]==c and assignments[k]==c
                # g[k][i]==c means k is in color_masks[i][c], but we only care about k < i
                k_mask_i = color_masks[i][c] & ~((1 << (i + 1)) - 1)  # k > i, not what we want
                # Actually we want k < i: color_masks[i][c] & ((1 << i) - 1)
                k_mask_i_low = color_masks[i][c] & ((1 << i) - 1)
                
                if k_mask_i_low & assign_masks[c]:
                    # Constraint violated
                    continue
                
                assignments[i] = c
                if backtrack(i + 1):
                    return True
                assignments[i] = -1
            
            return False

        if backtrack(0):
            # Success: Update graph
            for i in range(n):
                g[i].append(assignments[i])
            g.append([0] * (n + 1))
            last_row = g[n]
            for i in range(n):
                last_row[i] = assignments[i]
            last_row[n] = 0
            return True
        return False

    def try_add_vertex_with_repair(g, n):
        """
        Try to add a vertex. If backtracking fails, try local search repair
        on the existing graph to create room, then retry.
        Returns True if vertex was added successfully.
        """
        # First try direct backtracking
        if add_vertex_backtrack(g, n, time_limit=0.5):
            return True
        
        # If backtracking failed, try to repair the graph to create more flexibility
        # Save state
        old_graph = copy.deepcopy(g)
        old_n = n
        
        # Try local search repair
        if local_search_repair(g, n, max_iter=5000):
            # After repair, try to add vertex again
            if add_vertex_backtrack(g, n, time_limit=1.0):
                return True
        
        # If still failed, revert
        g[:] = [row[:] for row in old_graph]
        # Note: g is a list of lists, so we need to restore properly
        # Actually, let's just keep the repaired graph if it's valid, even if we couldn't add a vertex
        # This might help in future iterations
        if not is_valid_graph(g, n):
            # Revert to old state
            for i in range(n):
                for j in range(n):
                    g[i][j] = old_graph[i][j]
        
        return False

    attempts = 0
    max_attempts = 50000
    consecutive_failures = 0
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Try to add a vertex
        if try_add_vertex_with_repair(graph, current_n):
            current_n += 1
            best_n = current_n
            best_graph = copy.deepcopy(graph)
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            
            # If we fail to add a vertex, we need to modify the existing graph to create "room".
            # Strategy: Randomly flip some edges (Kick).
            # We must ensure the graph remains valid after the kick.
            
            # Probability of kicking: Higher if we've failed many times in a row.
            kick_prob = min(0.5, 0.1 * consecutive_failures)
            
            if random.random() < kick_prob:
                # Save state
                old_graph = copy.deepcopy(graph)
                old_n = current_n
                
                # Perform Kick: Flip K edges
                # K should be small, e.g., 5-10% of edges
                num_edges = current_n * (current_n - 1) // 2
                num_flips = max(1, num_edges // 20)
                
                for _ in range(num_flips):
                    i = random.randint(0, current_n - 1)
                    j = random.randint(0, current_n - 1)
                    if i != j:
                        graph[i][j] = random.randint(0, 3)
                        graph[j][i] = graph[i][j]
                
                # Validate
                if is_valid_graph(graph, current_n):
                    # Kick was successful in maintaining validity.
                    pass
                else:
                    # Kick broke validity. Try to repair.
                    if not local_search_repair(graph, current_n, max_iter=2000):
                        # Revert
                        for i in range(current_n):
                            for j in range(current_n):
                                graph[i][j] = old_graph[i][j]
            
            # If we have failed many times and kicking isn't helping, we might be stuck.
            # But we keep trying until deadline.

    # Finalize
    n = best_n
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(best_graph[i][j])
            
    # Final validation
    if not is_valid_graph(best_graph, n):
        # Fallback to seed if somehow invalid
        n = 32
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if i // 16 == j // 16:
                    c = log[(i % 16) ^ (j % 16)] % 3
                else:
                    c = 3
                colors.append(c)
    
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
