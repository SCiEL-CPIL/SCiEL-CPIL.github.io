import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = 140  # 10s margin
    
    # The known good solution with 17 colors
    def get_17_sol():
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    best_colors = 17
    best_sol = get_17_sol()
    
    # Try multiplicative structure approach first - assign vertex values and use (v_i * v_j) mod m
    # This might find a valid coloring with fewer colors
    for m in [15, 14, 16]:
        if time.time() > deadline - 20:
            break
        
        # Try different vertex assignments
        for attempt in range(50):
            if time.time() > deadline - 15:
                break
            
            # Assign random distinct values to vertices in Z_m
            # We need 16 vertices, so m must be >= 16 or we allow some values to repeat
            # But for multiplicative structure, we want distinct non-zero values ideally
            if m < 16:
                # Can't have 16 distinct values in Z_m if m < 16
                # Use random assignment allowing repeats
                vertex_vals = [random.randint(1, m - 1) for _ in range(16)]
            else:
                # Pick 16 distinct non-zero values from Z_m
                candidates = list(range(1, m))
                if len(candidates) >= 16:
                    vertex_vals = random.sample(candidates, 16)
                else:
                    vertex_vals = [random.randint(1, m - 1) for _ in range(16)]
            
            # Generate coloring using multiplicative structure
            colors = []
            for i in range(16):
                for j in range(i + 1, 16):
                    colors.append((vertex_vals[i] * vertex_vals[j]) % m)
            
            # Check if valid
            if is_valid_fast(colors):
                # Count distinct colors used
                distinct = len(set(colors))
                if distinct < best_colors:
                    best_colors = distinct
                    best_sol = colors
                    # If we found a solution with <= 15 colors, that's great
                    if best_colors <= 15:
                        return best_sol
        
        # If multiplicative didn't work well, try local search
        sol = find_solution(m, time.time(), deadline)
        if sol is not None:
            if is_valid_fast(sol):
                distinct = len(set(sol))
                if distinct < best_colors:
                    best_colors = distinct
                    best_sol = sol
                    if best_colors <= 15:
                        return best_sol
    
    # Final fallback: try additive with smaller moduli
    for m in [15, 14]:
        if time.time() > deadline - 5:
            break
        sol = [(i + j) % m for i in range(16) for j in range(i + 1, 16)]
        if is_valid_fast(sol):
            distinct = len(set(sol))
            if distinct < best_colors:
                best_colors = distinct
                best_sol = sol
    
    return best_sol

def is_valid(colors):
    return is_valid_fast(colors)

def is_valid_fast(colors):
    """Fast validity check using precomputed K4 edge indices"""
    # Precompute K4 edge indices once (module-level would be better, but this works)
    # Edge index formula: for edge (i,j) with i<j,
    # idx = sum_{k=0}^{i-1} (15-k) + (j-i-1) = 15*i - i*(i-1)//2 + j - i - 1
    
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return 15 * i - i * (i - 1) // 2 + (j - i - 1)
    
    # Check all K4s
    for a in range(13):
        for b in range(a + 1, 14):
            for c in range(b + 1, 15):
                for d in range(c + 1, 16):
                    # Get the 6 edges of this K4
                    e1 = edge_idx(a, b)
                    e2 = edge_idx(a, c)
                    e3 = edge_idx(a, d)
                    e4 = edge_idx(b, c)
                    e5 = edge_idx(b, d)
                    e6 = edge_idx(c, d)
                    
                    c1 = colors[e1]
                    c2 = colors[e2]
                    c3 = colors[e3]
                    c4 = colors[e4]
                    c5 = colors[e5]
                    c6 = colors[e6]
                    
                    # Count distinct colors
                    # Use a small set for speed
                    if c1 == c2:
                        distinct = 1
                        if c3 != c1: distinct += 1
                        if c4 != c1 and c4 != c3: distinct += 1
                        if c5 != c1 and c5 != c3 and c5 != c4: distinct += 1
                        if c6 != c1 and c6 != c3 and c6 != c4 and c6 != c5: distinct += 1
                    else:
                        distinct = 2
                        if c3 != c1 and c3 != c2: distinct += 1
                        if c4 != c1 and c4 != c2 and c4 != c3: distinct += 1
                        if c5 != c1 and c5 != c2 and c5 != c3 and c5 != c4: distinct += 1
                        if c6 != c1 and c6 != c2 and c6 != c3 and c6 != c4 and c6 != c5: distinct += 1
                    
                    if distinct < 5:
                        return False
    return True

def find_solution(C, start_time, deadline):
    """Local search to find valid coloring with at most C colors"""
    
    # Precompute K4 edge indices
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return 15 * i - i * (i - 1) // 2 + (j - i - 1)
    
    k4_edges = []
    for a in range(13):
        for b in range(a + 1, 14):
            for c in range(b + 1, 15):
                for d in range(c + 1, 16):
                    e = [
                        edge_idx(a, b), edge_idx(a, c), edge_idx(a, d),
                        edge_idx(b, c), edge_idx(b, d), edge_idx(c, d)
                    ]
                    k4_edges.append(e)
    
    max_iters = 3000
    
    for restart in range(5):
        if time.time() > deadline - 10:
            return None
        
        cur_colors = [random.randint(0, C - 1) for _ in range(120)]
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Count violations and find one
            viol_found = False
            for _ in range(20):
                idx = random.randint(0, len(k4_edges) - 1)
                e = k4_edges[idx]
                
                c1 = cur_colors[e[0]]
                c2 = cur_colors[e[1]]
                c3 = cur_colors[e[2]]
                c4 = cur_colors[e[3]]
                c5 = cur_colors[e[4]]
                c6 = cur_colors[e[5]]
                
                # Quick distinct count
                distinct = 1
                if c2 != c1: distinct += 1
                if c3 != c1 and c3 != c2: distinct += 1
                if c4 != c1 and c4 != c2 and c4 != c3: distinct += 1
                if c5 != c1 and c5 != c2 and c5 != c3 and c5 != c4: distinct += 1
                if c6 != c1 and c6 != c2 and c6 != c3 and c6 != c4 and c6 != c5: distinct += 1
                
                if distinct < 5:
                    # Found violation - fix it
                    edge_idx_in_k4 = random.randint(0, 5)
                    global_edge_idx = e[edge_idx_in_k4]
                    
                    old_color = cur_colors[global_edge_idx]
                    new_color = random.randint(0, C - 1)
                    if new_color == old_color:
                        new_color = (new_color + 1) % C
                    
                    cur_colors[global_edge_idx] = new_color
                    viol_found = True
                    break
            
            if not viol_found:
                # No violation found in 20 random samples - likely valid
                if is_valid_fast(cur_colors):
                    return cur_colors[:]
        
        # Restart with new random initialization
    
    return None
