import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # GF(64) implementation
    # Primitive polynomial: x^6 + x + 1 (0b1000011)
    # Actually, let's use a standard one. x^6 + x + 1 is primitive.
    PRIM_POLY = 0b1000011  # 67
    
    def gf64_mul(a, b):
        r = 0
        for i in range(6):
            if b & (1 << i):
                r ^= a << i
        # Reduce by x^6 + x + 1
        # If bit 6 is set, subtract (x^6 + x + 1)
        for i in range(11, 5, -1):
            if r & (1 << i):
                r ^= PRIM_POLY << (i - 6)
        return r & 0x3F  # Mask to 6 bits
    
    # Find primitive element and log table
    log_table = {}
    log_table[1] = 0
    x = 1
    for i in range(1, 63):
        x = gf64_mul(x, 2)
        if x in log_table:
            # 2 is not primitive? Let's try 3 or check.
            # If 2 generates the group, this loop finds all.
            # If not, we need a different generator.
            pass
        log_table[x] = i
    
    # Check if 2 is primitive
    if len(log_table) != 63:
        # Try to find a primitive element
        for g in range(1, 64):
            log_table = {1: 0}
            x = 1
            for i in range(1, 63):
                x = gf64_mul(x, g)
                log_table[x] = i
            if len(log_table) == 63:
                break
    
    # Precompute edge colors based on discrete log of ratio
    # Vertex i corresponds to element 2^i? No, let's map vertex index to element.
    # Let vertex v correspond to element g^v, where g is the primitive element found.
    # Actually, simpler: Let vertices be 1..63.
    # Element for vertex v is just v? No, we need the group structure.
    # Let's use the log table to get the "coordinate" of each element.
    # Element e has log log_table[e].
    # For edge (u, v), difference is log_table[u] - log_table[v] mod 63.
    
    # We need to assign colors to differences 1..62.
    # Let diff_color[d] be the color for difference d.
    # Note: diff(u,v) = (log(u) - log(v)) % 63.
    # diff(v,u) = (log(v) - log(u)) % 63 = (63 - diff(u,v)) % 63.
    # For the coloring to be well-defined on undirected edges, we need diff_color[d] == diff_color[63-d].
    # So we only need to assign colors to d in 1..31.
    # Let's verify: 63 is odd, so d != 63-d.
    
    # Initialize random diff_color for 1..31
    # Colors 0,1,2,3
    def random_diff_color():
        return [random.randint(0, 3) for _ in range(31)]
    
    def count_conflicts(dc):
        # dc[d] for d in 1..31
        # Full color array for 1..62
        full_dc = [0] * 63
        for d in range(1, 32):
            full_dc[d] = dc[d-1]
            full_dc[63-d] = dc[d-1]
        
        conflicts = 0
        # Triples (d1, d2, d3) such that d1+d2+d3 == 0 mod 63
        # d1, d2, d3 are in 1..62 (non-zero)
        # Iterate d1, d2, compute d3 = (-d1-d2) % 63
        # If d3 != 0, check if full_dc[d1] == full_dc[d2] == full_dc[d3]
        # To avoid double counting, we can just count all and divide, or just count.
        # Since we want to minimize, exact count doesn't matter, just relative.
        for d1 in range(1, 63):
            c1 = full_dc[d1]
            for d2 in range(1, 63):
                d3 = (-d1 - d2) % 63
                if d3 == 0:
                    continue
                c2 = full_dc[d2]
                if c1 != c2:
                    continue
                c3 = full_dc[d3]
                if c1 == c3:
                    conflicts += 1
        return conflicts
    
    def solve_diff_colors():
        # Local search on diff_color (length 31)
        dc = random_diff_color()
        best_dc = dc[:]
        best_cost = count_conflicts(dc)
        
        if best_cost == 0:
            return dc
        
        current_cost = best_cost
        temp = 1.0
        cooling = 0.999
        
        while time.time() < deadline - 5.0:
            # Pick random index
            idx = random.randint(0, 30)
            old_val = dc[idx]
            new_val = random.randint(0, 3)
            if new_val == old_val:
                continue
            dc[idx] = new_val
            new_cost = count_conflicts(dc)
            
            if new_cost < current_cost:
                current_cost = new_cost
                best_dc = dc[:]
                best_cost = new_cost
                if best_cost == 0:
                    break
            else:
                # Simulated Annealing acceptance
                # Simple: accept with small prob
                if random.random() < 0.1:
                    current_cost = new_cost
                else:
                    dc[idx] = old_val
                    current_cost = best_cost # Reset to best? No, usually keep current.
                    # Let's stick to Hill Climbing for simplicity and speed
                    dc[idx] = old_val
            
            # Cooling
            temp *= cooling
            if temp < 0.01:
                temp = 0.01
        
        return best_dc
    
    # Try to find a perfect diff_color
    # Run multiple restarts
    best_overall_dc = None
    best_overall_cost = float('inf')
    
    # Limit restarts to ensure time
    restarts = 0
    while time.time() < deadline - 10.0 and restarts < 20:
        dc = solve_diff_colors()
        cost = count_conflicts(dc)
        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall_dc = dc
        if cost == 0:
            break
        restarts += 1
    
    if best_overall_dc is None:
        best_overall_dc = random_diff_color()
    
    # Build the full coloring for n=63
    # Vertices 0..62 correspond to elements 2^0 .. 2^62?
    # Let's map vertex i to element g^i.
    # We need the element values to compute logs.
    # We have log_table: element -> log
    # We need inv_log: log -> element
    inv_log = [0] * 63
    for e, l in log_table.items():
        inv_log[l] = e
    
    # Vertex i is element inv_log[i]
    # Edge (i, j):
    # log_u = i, log_v = j
    # diff = (i - j) % 63
    # color = best_overall_dc[(diff - 1) // 2 if diff <= 31 else (63-diff-1)//2]
    # Wait, diff is in 1..62.
    # If diff <= 31, index is diff-1.
    # If diff > 31, index is 63-diff-1.
    
    def get_color_for_diff(d):
        if d <= 31:
            return best_overall_dc[d-1]
        else:
            return best_overall_dc[63-d-1]
    
    n = 63
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            d = (i - j) % 63
            if d == 0:
                d = 63 # Should not happen for i!=j
            colors.append(get_color_for_diff(d))
    
    # Check validity of n=63
    def is_valid(colors, n):
        # Build adjacency matrix for each color
        # adj[c][i][j] = True if edge (i,j) is color c
        # Or just check triangles directly
        # For each color c, check if any triangle exists.
        # A triangle (i,j,k) is mono in c if colors for (i,j), (j,k), (i,k) are all c.
        # To speed up, we can use bitsets or just triple loops if n is small.
        # n=63, triples ~ 40,000. Fast.
        
        # Map (i,j) to color index
        # colors list is lexicographic.
        # Index for (i,j) with i<j:
        # i=0: 0..n-2
        # i=1: n-1..2n-3
        # idx = i*n - i*(i+1)/2 + (j-i-1)
        
        def get_edge_color(i, j):
            if i > j:
                i, j = j, i
            return colors[i*n - i*(i+1)//2 + (j-i-1)]
            
        for c in range(4):
            for i in range(n):
                for j in range(i+1, n):
                    if get_edge_color(i, j) != c:
                        continue
                    for k in range(j+1, n):
                        if get_edge_color(i, k) == c and get_edge_color(j, k) == c:
                            return False
        return True
    
    if is_valid(colors, n):
        return {'n': n, 'colors': colors}
    
    # If n=63 is not valid, prune vertices
    # We have a coloring with some conflicts.
    # We want to remove vertices to eliminate all monochromatic triangles.
    
    # Calculate conflict count for each vertex
    # A vertex v is in a mono triangle if there exist u,w such that (u,v), (v,w), (u,w) are same color.
    
    def get_conflict_counts(colors, n):
        counts = [0] * n
        def get_edge_color(i, j):
            if i > j:
                i, j = j, i
            return colors[i*n - i*(i+1)//2 + (j-i-1)]
            
        # Precompute color matrix for faster access
        cm = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                c = get_edge_color(i, j)
                cm[i][j] = c
                cm[j][i] = c
                
        for i in range(n):
            for j in range(i+1, n):
                c = cm[i][j]
                for k in range(j+1, n):
                    if cm[i][k] == c and cm[j][k] == c:
                        counts[i] += 1
                        counts[j] += 1
                        counts[k] += 1
        return counts
    
    current_n = n
    current_colors = colors[:]
    
    # Pruning loop
    while current_n > 32:
        if is_valid(current_colors, current_n):
            break
        
        counts = get_conflict_counts(current_colors, current_n)
        
        # Find vertex with max conflicts
        max_conf = -1
        max_idx = -1
        for i in range(current_n):
            if counts[i] > max_conf:
                max_conf = counts[i]
                max_idx = i
        
        if max_conf == 0:
            # Should be valid, but is_valid said no?
            # Maybe is_valid is too strict or slow?
            break
            
        # Remove max_idx
        # Rebuild colors list without max_idx
        new_colors = []
        for i in range(current_n):
            if i == max_idx:
                continue
            for j in range(i+1, current_n):
                if j == max_idx:
                    continue
                # Get original index
                # We need to map new indices to old.
                # This is getting complex.
                pass
        
        # Simpler: Keep a list of active vertices
        # Let's restart the pruning with a list of active vertices
        
        break # Fallback to simpler pruning if this gets too complex
    
    # Alternative Pruning Strategy:
    # Keep a list of active vertices.
    # Start with all 63.
    # While conflicts exist:
    #   Find vertex with most conflicts.
    #   Remove it from active list.
    #   Rebuild colors from active list.
    
    active = list(range(63))
    base_colors = colors # The full 63x63 colors
    
    def get_base_color(i, j):
        # i, j are original indices 0..62
        if i > j:
            i, j = j, i
        return base_colors[i*63 - i*(i+1)//2 + (j-i-1)]
    
    def build_and_check(active):
        n = len(active)
        cols = []
        for idx_i in range(n):
            u = active[idx_i]
            for idx_j in range(idx_i+1, n):
                v = active[idx_j]
                cols.append(get_base_color(u, v))
        
        # Check validity
        def get_color(i, j):
            if i > j:
                i, j = j, i
            return cols[i*n - i*(i+1)//2 + (j-i-1)]
            
        for c in range(4):
            for i in range(n):
                for j in range(i+1, n):
                    if get_color(i, j) != c:
                        continue
                    for k in range(j+1, n):
                        if get_color(i, k) == c and get_color(j, k) == c:
                            return False, cols, n
        return True, cols, n
    
    # Prune
    while len(active) > 32 and time.time() < deadline - 5.0:
        valid, cols, n = build_and_check(active)
        if valid:
            return {'n': n, 'colors': cols}
        
        # Find vertex to remove
        # Count conflicts in the current active subgraph
        counts = [0] * len(active)
        def get_color(i, j):
            if i > j:
                i, j = j, i
            return cols[i*n - i*(i+1)//2 + (j-i-1)]
            
        for i in range(n):
            for j in range(i+1, n):
                c = get_color(i, j)
                for k in range(j+1, n):
                    if get_color(i, k) == c and get_color(j, k) == c:
                        counts[i] += 1
                        counts[j] += 1
                        counts[k] += 1
        
        max_conf = -1
        max_idx = -1
        for i in range(n):
            if counts[i] > max_conf:
                max_conf = counts[i]
                max_idx = i
        
        if max_conf == 0:
            break
            
        # Remove active[max_idx]
        active.pop(max_idx)
        
    # Fallback to seed solution
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
    colors_seed = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors_seed.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors_seed}

# EVOLVE-BLOCK-END
