import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Lazebnik-Woldar construction for n=26
    # Uses Z_5 x Z_5 x Z_5 with 5 colors
    # The construction: vertices are (a,b,c) in Z_5^3, but we need 26 vertices
    # Actually, the LW construction gives a C4-free 5-coloring of K_26
    
    # Let me implement the known algebraic construction
    # The standard LW construction uses the group Z_5 x Z_5
    # Vertices: 25 elements of Z_5 x Z_5 plus one extra vertex
    # Colors: 5 colors based on differences
    
    # Actually, let me use a simpler approach:
    # The LW construction for R_5(C4) >= 26 uses:
    # - 25 vertices from Z_5 x Z_5
    # - Color of edge (u,v) determined by v-u in Z_5 x Z_5
    # - Map Z_5 x Z_5 \ {0} to 5 colors such that each color class is C4-free
    
    # The key insight: partition Z_5 x Z_5 \ {0} into 5 sets of 4 elements each
    # where each set is a "cap" (no 3 collinear points in the affine plane)
    
    # For Z_5 x Z_5, we can use:
    # Color 0: {(1,0), (2,0), (3,0), (4,0)} - x-axis non-zero
    # But this might contain C4s... Let me think differently.
    
    # Alternative: use a known explicit construction
    # The LW construction uses finite geometry
    # Let me try a different approach: use the fact that q+1 = 6 > 5
    # So we can't directly use the standard q+1 color construction
    
    # Let me just use simulated annealing with a good initialization
    # and hope to find n=26 or better
    
    def get_idx(i, j, n):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    def count_violations_fast(colors, n):
        """Count C4 violations using adjacency lists"""
        neighbors = [[[] for _ in range(n)] for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_idx(i, j, n)
                c = colors[idx]
                neighbors[c][i].append(j)
                neighbors[c][j].append(i)
        
        bad_total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    nu = neighbors[c][u]
                    nv = neighbors[c][v]
                    # Count common neighbors
                    cnt = 0
                    for x in nu:
                        if x in nv:
                            cnt += 1
                    if cnt >= 2:
                        bad_total += cnt * (cnt - 1) // 2
        return bad_total
    
    def solve_sa(n, max_time, initial_colors=None):
        m = n * (n - 1) // 2
        if initial_colors is not None:
            colors = list(initial_colors)
        else:
            colors = [random.randint(0, 4) for _ in range(m)]
        
        neighbors = [[[] for _ in range(n)] for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_idx(i, j, n)
                c = colors[idx]
                neighbors[c][i].append(j)
                neighbors[c][j].append(i)
        
        # Precompute common neighbor counts
        common = [[[0]*n for _ in range(n)] for _ in range(5)]
        bad_total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    nu = neighbors[c][u]
                    nv = neighbors[c][v]
                    cnt = 0
                    for x in nu:
                        if x in nv:
                            cnt += 1
                    common[c][u][v] = cnt
                    common[c][v][u] = cnt
                    if cnt >= 2:
                        bad_total += cnt * (cnt - 1) // 2
                        
        if bad_total == 0:
            return colors, True
        
        T = 1.0
        cooling = 0.9995
        best_colors = list(colors)
        best_bad = bad_total
        current_bad = bad_total
        
        time_limit = start_time + max_time
        while time.time() < time_limit:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            idx = get_idx(i, j, n)
            old_c = colors[idx]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
            
            delta = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                if j in neighbors[old_c][k]:
                    old_cnt = common[old_c][i][k]
                    new_cnt = old_cnt - 1
                    delta -= (old_cnt * (old_cnt - 1) // 2) - (new_cnt * (new_cnt - 1) // 2)
                    common[old_c][i][k] = new_cnt
                    common[old_c][k][i] = new_cnt
                if i in neighbors[old_c][k]:
                    old_cnt = common[old_c][j][k]
                    new_cnt = old_cnt - 1
                    delta -= (old_cnt * (old_cnt - 1) // 2) - (new_cnt * (new_cnt - 1) // 2)
                    common[old_c][j][k] = new_cnt
                    common[old_c][k][j] = new_cnt
                if j in neighbors[new_c][k]:
                    old_cnt = common[new_c][i][k]
                    new_cnt = old_cnt + 1
                    delta += (new_cnt * (new_cnt - 1) // 2) - (old_cnt * (old_cnt - 1) // 2)
                    common[new_c][i][k] = new_cnt
                    common[new_c][k][i] = new_cnt
                if i in neighbors[new_c][k]:
                    old_cnt = common[new_c][j][k]
                    new_cnt = old_cnt + 1
                    delta += (new_cnt * (new_cnt - 1) // 2) - (old_cnt * (old_cnt - 1) // 2)
                    common[new_c][j][k] = new_cnt
                    common[new_c][k][j] = new_cnt
            
            new_bad = current_bad + delta
            
            if new_bad < current_bad or (delta >= 0 and random.random() < math.exp(-delta / T)):
                current_bad = new_bad
                colors[idx] = new_c
                try:
                    neighbors[old_c][i].remove(j)
                except ValueError:
                    pass
                try:
                    neighbors[old_c][j].remove(i)
                except ValueError:
                    pass
                neighbors[new_c][i].append(j)
                neighbors[new_c][j].append(i)
                if current_bad < best_bad:
                    best_bad = current_bad
                    best_colors = list(colors)
                    if best_bad == 0:
                        return best_colors, True
            else:
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if j in neighbors[old_c][k]:
                        common[old_c][i][k] += 1
                        common[old_c][k][i] += 1
                    if i in neighbors[old_c][k]:
                        common[old_c][j][k] += 1
                        common[old_c][k][j] += 1
                    if j in neighbors[new_c][k]:
                        common[new_c][i][k] -= 1
                        common[new_c][k][i] -= 1
                    if i in neighbors[new_c][k]:
                        common[new_c][j][k] -= 1
                        common[new_c][k][j] -= 1
            T *= cooling
        
        return best_colors, (best_bad == 0)
    
    best_n = 1
    best_colors = [0]
    
    # Try n from 26 down to 12
    for n in range(26, 11, -1):
        if time.time() > deadline - 15:
            break
        found = False
        restarts = 8
        for r in range(restarts):
            if time.time() > deadline - 10:
                break
            random.seed(n * 7919 + r * 13)
            colors, is_valid = solve_sa(n, max_time=15)
            if is_valid:
                best_n = n
                best_colors = colors
                found = True
                break
        if found:
            break
    
    # If we didn't find anything good, use a small valid coloring
    if best_n < 12:
        # Use the cyclic construction for n=11
        n = 11
        best_colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        best_n = n
    
    return {'n': best_n, 'colors': best_colors}
