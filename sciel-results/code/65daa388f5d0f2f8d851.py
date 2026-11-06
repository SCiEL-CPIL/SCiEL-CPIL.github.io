import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_n = 11
    best_colors = None
    
    # Fallback solution: n=11 (known valid)
    # K11 split into five Hamiltonian cycles by circular distance 1..5
    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    best_n = n_fallback
    best_colors = colors_fallback

    # Helper to check if a coloring is valid (no C4 in any color class)
    def is_valid_coloring(colors, n):
        num_edges = n * (n - 1) // 2
        if len(colors) != num_edges:
            return False
        
        # Build adjacency masks for each color
        adj_mask = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                if c < 0 or c > 4:
                    return False
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
                idx += 1
        
        # Check for C4: any pair (u,v) with >1 common neighbor in same color
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj_mask[c][u] & adj_mask[c][v]
                    if common.bit_count() > 1:
                        return False
        return True

    # Helper to count violations (number of pairs with >1 common neighbor)
    def count_violations(colors, n):
        num_edges = n * (n - 1) // 2
        adj_mask = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
                idx += 1
        
        violations = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj_mask[c][u] & adj_mask[c][v]
                    cnt = common.bit_count()
                    if cnt > 1:
                        violations += cnt - 1
        return violations

    # Algebraic construction for n=26 using AG(2,5)
    # Vertices 0..24 correspond to (x,y) in Z5 x Z5, vertex 25 is the "infinity" vertex
    # Color edge (u,v) based on slope of line through them
    def construct_26():
        n = 26
        colors = [0] * (n * (n - 1) // 2)
        
        # Map vertex index to (x, y) or infinity
        # Vertex 25 is infinity
        def vertex_to_coord(v):
            if v == 25:
                return None  # infinity
            return (v % 5, v // 5)
        
        def slope_color(u, v):
            """
            Return color 0-4 for edge (u,v) based on slope.
            Colors 0-3 for finite slopes, color 4 for vertical (infinity) slope.
            Actually, we have 5 colors. In AG(2,5), there are 6 parallel classes (5 finite slopes + 1 vertical).
            Wait, in Z5 x Z5, the slopes are 0, 1, 2, 3, 4 (finite) and infinity (vertical). That's 6 classes.
            But we only have 5 colors.
            
            The Lazebnik-Woldar construction uses a different mapping or a different structure.
            Let's re-evaluate. The standard affine plane AG(2,5) has 25 points and 6 parallel classes of lines.
            Each parallel class is a 1-factorization (perfect matching) of K25? No, a parallel class is a set of 5 disjoint lines covering all 25 points.
            So each parallel class gives a 5-regular graph? No, each line has 5 points, so it's K5.
            Wait, a line in AG(2,5) has 5 points. The edges of that line form a K5.
            If we color all edges on lines of a given parallel class with one color, then each color class is a union of 5 K5s.
            Does a K5 contain a C4? Yes, K5 contains many C4s.
            So this simple slope coloring does NOT work directly because K5 has C4s.
            
            The Lazebnik-Woldar construction is more subtle. It uses a group-based construction or a different combinatorial design.
            Since I don't have the exact formula memorized, and deriving it is complex,
            I will rely on a strong local search initialized with a good heuristic.
            """
            # For now, return a random-ish slope-based color to serve as a heuristic init
            cu = vertex_to_coord(u)
            cv = vertex_to_coord(v)
            
            if cu is None or cv is None:
                # One is infinity
                # Color based on the finite vertex's x-coordinate?
                if cu is None:
                    return cv[0]
                else:
                    return cu[0]
            
            dx = (cv[0] - cu[0]) % 5
            dy = (cv[1] - cu[1]) % 5
            
            if dx == 0:
                return 4  # Vertical slope
            else:
                # slope = dy * dx^{-1} mod 5
                # Inverse of 1..4 mod 5: 1->1, 2->3, 3->2, 4->4
                inv_dx = [0, 1, 3, 2, 4][dx]
                slope = (dy * inv_dx) % 5
                return slope  # 0..4
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                colors[idx] = slope_color(i, j)
                idx += 1
        
        return colors

    # Try to construct n=26 using local search with algebraic init
    if time.time() < deadline - 20.0:
        n_target = 26
        num_edges = n_target * (n_target - 1) // 2
        init_colors = construct_26()
        
        # Check if init is already valid (unlikely)
        if is_valid_coloring(init_colors, n_target):
            best_n = n_target
            best_colors = init_colors
        else:
            # Local search to fix violations
            # We need a fast local search that can handle n=26 (325 edges)
            # Violations are likely many initially.
            
            def local_search_26(colors, time_limit):
                current_violations = count_violations(colors, n_target)
                if current_violations == 0:
                    return colors, 0
                
                best_colors = colors[:]
                best_violations = current_violations
                temp = 2.0
                cooling = 0.999
                
                # Precompute edge list for speed
                edge_list = [(i, j) for i in range(n_target) for j in range(i + 1, n_target)]
                
                # We can optimize count_violations by only checking affected pairs,
                # but for simplicity and robustness, we use full count initially.
                # If too slow, we'll switch to incremental.
                
                for it in range(20000):  # Max iterations
                    if time.time() > time_limit:
                        break
                    if current_violations == 0:
                        break
                    
                    e_idx = random.randint(0, num_edges - 1)
                    old_color = colors[e_idx]
                    new_color = random.randint(0, 4)
                    if new_color == old_color:
                        continue
                    
                    colors[e_idx] = new_color
                    new_violations = count_violations(colors, n_target)
                    delta = new_violations - current_violations
                    
                    if delta < 0 or random.random() < math.exp(-delta / temp):
                        current_violations = new_violations
                        if current_violations < best_violations:
                            best_violations = current_violations
                            best_colors = colors[:]
                    else:
                        colors[e_idx] = old_color
                    
                    temp *= cooling
                
                return best_colors, best_violations
            
            colors, viol = local_search_26(init_colors, deadline - 5.0)
            if viol == 0:
                best_n = n_target
                best_colors = colors
            else:
                # If we couldn't get 0 violations, try to salvage a smaller n
                # Or just return the best we have (which might be invalid if we didn't check)
                # We must return a VALID coloring.
                # If local search didn't reach 0, we cannot return n=26.
                # We fall back to the previous best_n (11) or try to find a valid smaller n.
                pass

    # If we didn't find n=26, try to find a valid smaller n using random search
    # to improve upon n=11.
    if best_n == 11 and time.time() < deadline - 5.0:
        for n_try in [12, 13, 14, 15, 16, 17, 18, 19, 20]:
            if time.time() > deadline - 5.0:
                break
            
            num_edges = n_try * (n_try - 1) // 2
            found = False
            
            for restart in range(10):
                if time.time() > deadline - 5.0:
                    break
                
                colors = [random.randint(0, 4) for _ in range(num_edges)]
                
                # Quick local search
                current_violations = count_violations(colors, n_try)
                temp = 1.0
                cooling = 0.999
                edge_list = [(i, j) for i in range(n_try) for j in range(i + 1, n_try)]
                
                for it in range(2000):
                    if time.time() > deadline - 5.0:
                        break
                    if current_violations == 0:
                        break
                    
                    e_idx = random.randint(0, num_edges - 1)
                    old_color = colors[e_idx]
                    new_color = random.randint(0, 4)
                    if new_color == old_color:
                        continue
                    
                    colors[e_idx] = new_color
                    new_violations = count_violations(colors, n_try)
                    delta = new_violations - current_violations
                    
                    if delta < 0 or random.random() < math.exp(-delta / temp):
                        current_violations = new_violations
                    else:
                        colors[e_idx] = old_color
                    
                    temp *= cooling
                
                if current_violations == 0:
                    best_n = n_try
                    best_colors = colors
                    found = True
                    break
            
            if found:
                # Continue to try larger n
                continue
            else:
                # If we failed this n, it's likely we'll fail larger ones too
                # But maybe just bad luck? Let's break to be safe and return best_n
                break

    # Final validation check before returning
    if not is_valid_coloring(best_colors, best_n):
        # Fallback to known valid n=11
        best_n = 11
        best_colors = colors_fallback

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
