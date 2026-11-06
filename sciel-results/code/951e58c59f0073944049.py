import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # We know n=26 is achievable via algebraic construction (Lazebnik-Woldar).
    # The seed program returns n=11.
    # The inspirations found n=26 (score 2.0) and n=26 (score 2.09).
    # To maximize the score (n/11), we should aim for the highest n possible.
    # n=26 is the known bound. n=27 or 28 would be new but unlikely to be found by heuristic search in 120s.
    # However, the inspirations used local search to *find* a valid coloring for n=26.
    # The algebraic construction is deterministic and instant.
    # Let's implement the algebraic construction for n=26 directly.
    # If that fails (it shouldn't), we fall back to local search.
    
    # Lazebnik-Woldar construction for 5-coloring K_26 without monochromatic C4.
    # Vertices are elements of Z_5 x Z_5.
    # n = 25? Wait, the task says "algebraic construction ... gives 26 vertices".
    # Let's check the standard construction.
    # R_5(C4) > 25 is known. R_5(C4) > 26 is known?
    # The prompt says: "The algebraic construction of Lazebnik and Woldar gives 26 vertices; 27 or 28 would be new."
    # So n=26 is the target.
    
    # Construction details for n=26:
    # One common construction uses vertices (x, y) in Z_5 x Z_5 plus one point at infinity?
    # Or is it simply 26 vertices?
    # Let's look at the inspirations. They used local search to find a valid coloring for n=26.
    # This implies a valid coloring exists and can be found.
    # Since I don't have the explicit algebraic formula memorized for n=26 specifically (often it's 25 for affine plane of order 5, which gives R_5(C4) > 25), 
    # and the prompt explicitly mentions 26, I will rely on the local search approach from the inspirations, 
    # but optimized to be faster and more robust.
    
    # Actually, let's try to construct n=26 using the local search method from inspiration 2, 
    # but with a tighter deadline and better initialization if possible.
    # The inspirations scored 2.0 and 2.09, which corresponds to n=22 and n=23? 
    # Score = n / 11.
    # 2.0 * 11 = 22.
    # 2.09 * 11 = 22.99 -> 23.
    # So the inspirations only found n=22 and n=23.
    # The prompt says "The algebraic construction ... gives 26 vertices". This implies n=26 is possible.
    # If I can implement the algebraic construction, I get score 26/11 = 2.36.
    # If I can't, I should try to beat 23.
    
    # Let's try to find the algebraic construction for n=26.
    # A known result is R_5(C4) > 25 (from affine plane of order 5).
    # Is there a construction for 26?
    # Yes, Lazebnik and Woldar (1995) proved R_5(C4) > 26.
    # The construction involves a specific graph.
    # Without the explicit formula, I will use a high-performance local search.
    # To improve upon the inspirations (which got 22-23), I need a better search.
    # The inspirations used simulated annealing.
    # A key improvement is using a greedy initialization or a more efficient violation calculation.
    
    # Let's implement a fast local search for n=26.
    # If it fails, try n=25, 24, etc.
    
    target_ns = [26, 25, 24, 23, 22]
    
    for n in target_ns:
        if time.time() > deadline - 10:
            break
        result = local_search_fast(n, time_limit=20)
        if result is not None:
            return result
            
    # Fallback to seed
    return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}

def local_search_fast(n, time_limit=20):
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute edge index for O(1) lookup
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
        
    # Initialize colors randomly
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    # Build adjacency masks for each color
    # neighbors[c][v] is a bitmask of neighbors of v in color c
    neighbors = [[0] * n for _ in range(5)]
    
    def update_neighbors(idx, old_color, new_color):
        i, j = edges[idx]
        # Remove from old_color
        neighbors[old_color][i] &= ~(1 << j)
        neighbors[old_color][j] &= ~(1 << i)
        # Add to new_color
        neighbors[new_color][i] |= (1 << j)
        neighbors[new_color][j] |= (1 << i)
        
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)
        
    # Calculate violations
    # A violation is a pair of vertices (u, v) and a color c such that they have >1 common neighbor in color c.
    # We can count the number of C4s.
    def calc_violations():
        violations = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = neighbors[c][u] & neighbors[c][v]
                    k = common.bit_count()
                    if k >= 2:
                        violations += k * (k - 1) // 2
        return violations
        
    current_violations = calc_violations()
    if current_violations == 0:
        return {'n': n, 'colors': colors}
        
    best_violations = current_violations
    best_colors = colors[:]
    
    start_time = time.time()
    temperature = 1.0
    cooling_rate = 0.999
    
    # Local search loop
    while time.time() < start_time + time_limit:
        # Pick a random edge
        e_idx = random.randint(0, num_edges - 1)
        old_color = colors[e_idx]
        
        # Try changing to a different color
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
            
        # Apply change
        colors[e_idx] = new_color
        update_neighbors(e_idx, old_color, new_color)
        
        # Calculate new violations
        # Optimization: Only recalculate violations for the affected color classes?
        # Changing edge (i, j) from c1 to c2 affects:
        # 1. Violations in c1 involving i or j
        # 2. Violations in c2 involving i or j
        # But calculating the full violation count is O(5 * n^2 * bit_count).
        # For n=26, n^2 = 676. 5*676 = 3380 checks. bit_count is fast.
        # It's likely fast enough to just recalculate everything.
        
        new_violations = calc_violations()
        delta = new_violations - current_violations
        
        if delta < 0 or random.random() < pow(2.71828, -delta / temperature):
            current_violations = new_violations
            if new_violations < best_violations:
                best_violations = new_violations
                best_colors = colors[:]
                if best_violations == 0:
                    return {'n': n, 'colors': colors}
        else:
            # Revert
            colors[e_idx] = old_color
            update_neighbors(e_idx, new_color, old_color)
            
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
            
    return None
