import time
import random
import math
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Fallback: n=11 (Known valid: K11 decomposed into 5 Hamiltonian cycles)
    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    best_n = n_fallback
    best_colors = colors_fallback

    # Helper: Index of edge (i, j) in lexicographic list
    def get_idx(i, j, n):
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

    # Helper: Check if a coloring is valid
    def check_validity(colors, n):
        adj_mask = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
                idx += 1
        
        for c in range(5):
            for u in range(n):
                mask_u = adj_mask[c][u]
                if mask_u == 0:
                    continue
                for v in range(u + 1, n):
                    mask_v = adj_mask[c][v]
                    if mask_v == 0:
                        continue
                    common = mask_u & mask_v
                    if common:
                        if common.bit_count() > 1:
                            return False
        return True

    # Helper: Count violations (C4s)
    def count_violations(colors, n):
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
                mask_u = adj_mask[c][u]
                if mask_u == 0:
                    continue
                for v in range(u + 1, n):
                    mask_v = adj_mask[c][v]
                    if mask_v == 0:
                        continue
                    common = mask_u & mask_v
                    if common:
                        cnt = common.bit_count()
                        if cnt > 1:
                            violations += cnt - 1
        return violations

    # Helper: Build adjacency masks from colors
    def build_masks(colors, n):
        adj_mask = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
                idx += 1
        return adj_mask

    # Helper: Calculate delta in violations when changing edge (u, v) from color c_old to c_new
    # adj_mask: current masks
    # colors: current colors
    def calc_delta(adj_mask, colors, u, v, c_old, c_new, n):
        if c_old == c_new:
            return 0
        
        delta = 0
        
        # Removing edge (u, v) from color c_old
        # This affects common neighbor counts for pairs (x, y) where x, y are neighbors of u or v in c_old?
        # Actually, a C4 is a set of 4 vertices {a, b, c, d} such that edges (a,b), (b,c), (c,d), (d,a) exist.
        # Or equivalently, two vertices share 2 common neighbors.
        # Let's use the "common neighbors" definition.
        # Violation count = sum over all pairs (x, y) of max(0, common_neighbors(x, y) - 1).
        
        # Changing edge (u, v) affects:
        # 1. The pair (u, v) itself.
        # 2. Pairs (u, w) for w in N_{c_old}(v) \ {u}
        # 3. Pairs (v, w) for w in N_{c_old}(u) \ {v}
        # 4. Pairs (x, y) where x in N_{c_old}(u) \ {v}, y in N_{c_old}(v) \ {u}
        #    Wait, this is getting complicated. Let's just recompute the relevant parts.
        
        # A faster way:
        # For a specific color c, let N_c(x) be the set of neighbors of x in color c.
        # The number of C4s involving color c is sum_{x < y} (|N_c(x) ∩ N_c(y)| - 1)_+.
        
        # When we change edge (u, v) from c_old to c_new:
        # In c_old:
        #   u loses neighbor v.
        #   v loses neighbor u.
        #   For any w != u, v:
        #     If w in N_{c_old}(u) and w in N_{c_old}(v), then (u, v) was a C4 edge? No.
        #     The common neighbors of u and v in c_old change? No, u and v are not neighbors of each other in the common neighbor count for OTHER pairs.
        #     Actually, the common neighbors of (u, v) in c_old is |N_{c_old}(u) ∩ N_{c_old}(v)|.
        #     Before: u was in N_{c_old}(v) and v was in N_{c_old}(u).
        #     So N_{c_old}(u) contained v. N_{c_old}(v) contained u.
        #     After removal:
        #     N'_{c_old}(u) = N_{c_old}(u) \ {v}
        #     N'_{c_old}(v) = N_{c_old}(v) \ {u}
        
        # Let's just calculate the change in the sum S_c = sum_{x < y} (|N_c(x) ∩ N_c(y)| - 1)_+
        
        # This is still complex to do incrementally without significant bookkeeping.
        # Given the time limit, a full recomputation is ~ n^2 * 5 operations. For n=26, 26^2 * 5 = 3380 ops.
        # 10,000 steps * 3380 = 33 million ops. In Python, this might take 10-30 seconds.
        # So full recomputation is feasible for n=26 if we limit steps.
        
        # Let's try a different approach for the delta:
        # We only need to check if the change creates or removes a C4.
        # A C4 is removed if the edge (u, v) was part of a C4.
        # A C4 is created if the edge (u, v) in c_new creates a C4.
        
        # This is hard to count exactly without scanning.
        # Let's stick to full recomputation but optimize it.
        
        # Actually, let's just use the full recomputation in the search loop, but limit the number of steps.
        # Or, better yet, use a "greedy repair" on the structured solution.
        
        return 0 # Placeholder, we will use full recomputation in the search

    # Local Search with Full Recomputation (Optimized)
    def local_search(colors, n, max_time):
        adj_mask = build_masks(colors, n)
        current_violations = count_violations(colors, n)
        
        if current_violations == 0:
            return colors, True
        
        best_colors = colors[:]
        best_violations = current_violations
        
        # Precompute edge list
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        
        num_edges = len(edges)
        
        # Simulated Annealing
        temp = 10.0
        cooling = 0.995
        min_temp = 0.01
        
        while temp > min_temp and time.time() < max_time:
            # Pick random edge
            e_idx = random.randint(0, num_edges - 1)
            u, v = edges[e_idx]
            
            old_color = colors[e_idx]
            new_color = random.randint(0, 4)
            if new_color == old_color:
                continue
            
            colors[e_idx] = new_color
            new_violations = count_violations(colors, n)
            
            delta = new_violations - current_violations
            
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_violations = new_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
            else:
                colors[e_idx] = old_color
            
            temp *= cooling
            
        if best_violations == 0:
            return best_colors, True
        return best_colors, False

    # Strategy:
    # 1. Try to construct n=26 using the Lazebnik-Woldar algebraic construction.
    # 2. If it works, return it.
    # 3. If not, try to repair it or find a smaller n via search.
    
    # Lazebnik-Woldar Construction for n=26
    # Vertices: (x, y) where x, y in Z_5, plus 2 points at infinity? No, 26 = 5*5 + 1? No.
    # 26 = 5^2 + 1? No.
    # The construction is based on the affine plane AG(2, 5).
    # Points: (x, y) in Z_5 x Z_5. Total 25 points.
    # Plus 1 point?
    # Actually, the known R_5(C4) > 26 construction uses 26 vertices.
    # Let's try to generate a valid coloring for n=26.
    
    # A simple algebraic construction for C4-free colorings of K_{q^2+q+1} or similar?
    # For q=5, q^2=25.
    # Let's try to build n=26.
    
    # If we can't easily construct 26, we can try to search for it.
    # But searching for 26 from scratch is hard.
    # Let's try to search for n=12, 13, ... up to 26 using the structured init if possible.
    
    # For now, let's try to find a valid coloring for n=26 using a heuristic.
    # If we fail, we fall back to smaller n.
    
    # Heuristic for n=26:
    # Use the structure of the affine plane.
    # Color (x, y), (x', y') based on the slope of the line connecting them.
    # Slopes in Z_5: 0, 1, 2, 3, 4, infinity. 6 slopes. We have 5 colors.
    # So we must merge two slopes?
    # Or use a different geometry.
    
    # Let's try a random search for n=26 with a time limit.
    # If it fails, try n=25, etc.
    
    # To save time, let's first try to see if n=26 is possible with a simple pattern.
    # Pattern: Color(i, j) = (i + j) % 5?
    # Check for C4.
    # If i, j, k, l form a C4, then colors must be consistent.
    
    # Let's just run the local search for n=26 with a short time limit.
    # If it finds a solution, great. If not, try smaller n.
    
    # However, random init for n=26 will have millions of violations.
    # SA might not escape the local minima.
    
    # Better: Try to find a valid coloring for n=12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26.
    # Start from 12 and go up.
    
    # But we know 26 is possible.
    # Let's try to implement the actual Lazebnik-Woldar construction.
    # Reference: Lazebnik, F., & Woldar, A. (1995). On a problem of Erdős.
    # The construction uses the finite field GF(5).
    # Vertices are pairs (a, b) in GF(5)^2.
    # Color of edge ((a, b), (c, d)) is determined by the slope (d-b)/(c-a) if c!=a, else infinity.
    # There are 6 slopes. We need 5 colors.
    # Maybe we can map 6 slopes to 5 colors?
    # If we map two slopes to the same color, we might create a C4.
    # Specifically, if we map slope 0 and slope infinity to the same color, we get a grid of 5x5.
    # In a 5x5 grid, do we have a C4 in the color class?
    # Yes, any 2x2 subgrid is a C4.
    # So we cannot merge slopes that form a grid.
    
    # Actually, the construction might be more subtle.
    # Let's try to search for n=26 with a specific initialization.
    
    # Initialization: Random.
    # But we can bias it.
    
    # Let's try n=26.
    n_target = 26
    colors = [random.randint(0, 4) for _ in range(n_target * (n_target - 1) // 2)]
    
    # Run local search for n=26
    t_limit = time.time() + 10.0
    result_colors, is_valid = local_search(colors, n_target, t_limit)
    
    if is_valid:
        best_n = n_target
        best_colors = result_colors
    else:
        # Try smaller n
        for n in range(25, 11, -1):
            if time.time() > deadline - 20.0:
                break
            
            colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
            t_limit = time.time() + 5.0
            result_colors, is_valid = local_search(colors, n, t_limit)
            
            if is_valid:
                best_n = n
                best_colors = result_colors
                break

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
