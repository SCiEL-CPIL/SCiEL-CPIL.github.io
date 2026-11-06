import time
import itertools
import random

def _check_validity_fast(colors, n=16):
    """Check if edge coloring satisfies: every K4 spans >= 5 distinct colors."""
    # Build adjacency color matrix for fast lookup
    col = [[0]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            col[i][j] = col[j][i] = colors[idx]
            idx += 1
    
    # Check all K4s
    for a,b,c,d in itertools.combinations(range(n), 4):
        cset = {col[a][b], col[a][c], col[a][d], col[b][c], col[b][d], col[c][d]}
        if len(cset) < 5:
            return False
    return True

def _try_linear_form(m, a_coeff, b_coeff, n=16):
    """Try coloring (i,j) -> (a_coeff*i + b_coeff*j) % m"""
    colors = [(a_coeff*i + b_coeff*j) % m for i in range(n) for j in range(i+1, n)]
    if _check_validity_fast(colors):
        return colors
    return None

def _try_quadratic_form(m, a_coeff, b_coeff, c_coeff, n=16):
    """Try coloring (i,j) -> (a_coeff*i + b_coeff*j + c_coeff*i*j) % m"""
    colors = [(a_coeff*i + b_coeff*j + c_coeff*i*j) % m 
              for i in range(n) for j in range(i+1, n)]
    if _check_validity_fast(colors):
        return colors
    return None

def _local_search(C, deadline, n=16, seed=42):
    """Randomized local search for a valid coloring with C colors."""
    rng = random.Random(seed)
    
    # Precompute K4 edge indices
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append((i, j))
    
    edge_index = {}
    for idx, (i, j) in enumerate(edges):
        edge_index[(i, j)] = idx
        edge_index[(j, i)] = idx
    
    k4s = list(itertools.combinations(range(n), 4))
    k4_edges = []
    for quad in k4s:
        a, b, c, d = quad
        e = [edge_index[(a,b)], edge_index[(a,c)], edge_index[(a,d)],
             edge_index[(b,c)], edge_index[(b,d)], edge_index[(c,d)]]
        k4_edges.append(e)
    
    def count_violations(colors):
        viol = 0
        for e in k4_edges:
            cset = {colors[e[0]], colors[e[1]], colors[e[2]],
                    colors[e[3]], colors[e[4]], colors[e[5]]}
            if len(cset) < 5:
                viol += 1
        return viol
    
    best_colors = None
    best_viol = float('inf')
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Initialize: start from (i+j) % C perturbed
        cur_colors = [rng.randint(0, C-1) for _ in range(120)]
        cur_viol = count_violations(cur_colors)
        
        if cur_viol == 0:
            return cur_colors
        
        if cur_viol < best_viol:
            best_viol = cur_viol
            best_colors = cur_colors[:]
        
        # Local search: hill climbing with random restarts
        for it in range(2000):
            if time.time() > deadline - 5:
                break
            
            # Pick a random violating K4
            found_viol = False
            for _ in range(20):
                idx = rng.randint(0, len(k4_edges)-1)
                e = k4_edges[idx]
                cset = {cur_colors[e[0]], cur_colors[e[1]], cur_colors[e[2]],
                        cur_colors[e[3]], cur_colors[e[4]], cur_colors[e[5]]}
                if len(cset) < 5:
                    # Try changing a random edge in this K4
                    edge_pos = rng.randint(0, 5)
                    global_idx = e[edge_pos]
                    old_c = cur_colors[global_idx]
                    new_c = rng.randint(0, C-1)
                    if new_c == old_c:
                        new_c = (new_c + 1) % C
                    cur_colors[global_idx] = new_c
                    
                    new_viol = count_violations(cur_colors)
                    if new_viol <= cur_viol:
                        cur_viol = new_viol
                        if cur_viol == 0:
                            return cur_colors
                        if cur_viol < best_viol:
                            best_viol = cur_viol
                            best_colors = cur_colors[:]
                    else:
                        # Revert
                        cur_colors[global_idx] = old_c
                    
                    found_viol = True
                    break
            
            if not found_viol:
                break
    
    return best_colors

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    n = 16
    
    # First, try the known 17-color solution as baseline
    baseline = [(i + j) % 17 for i in range(n) for j in range(i+1, n)]
    if _check_validity_fast(baseline):
        best_colors = baseline
        best_num = 17
    else:
        best_colors = None
        best_num = 18
    
    # Search for linear forms (a*i + b*j) % m with m < 17
    for m in range(13, 17):
        if time.time() > deadline - 20:
            break
        for a_coeff in range(1, m):
            for b_coeff in range(1, m):
                if time.time() > deadline - 20:
                    break
                colors = _try_linear_form(m, a_coeff, b_coeff, n)
                if colors is not None:
                    num_used = len(set(colors))
                    if num_used < best_num:
                        best_num = num_used
                        best_colors = colors
                        if num_used <= 13:
                            return best_colors
    
    # Search for quadratic forms
    if time.time() < deadline - 30 and best_num > 13:
        for m in range(13, 17):
            if time.time() > deadline - 30:
                break
            for a_coeff in range(1, m):
                for b_coeff in range(1, m):
                    for c_coeff in range(1, m):
                        if time.time() > deadline - 30:
                            break
                        colors = _try_quadratic_form(m, a_coeff, b_coeff, c_coeff, n)
                        if colors is not None:
                            num_used = len(set(colors))
                            if num_used < best_num:
                                best_num = num_used
                                best_colors = colors
                                if num_used <= 13:
                                    return best_colors
    
    # Try local search with fewer colors
    if time.time() < deadline - 10 and best_num > 13:
        for C in range(13, best_num):
            if time.time() > deadline - 10:
                break
            result = _local_search(C, deadline, n)
            if result is not None:
                if _check_validity_fast(result):
                    best_colors = result
                    best_num = C
                    break
    
    # Fallback
    if best_colors is None:
        best_colors = [(i + j) % 17 for i in range(n) for j in range(i+1, n)]
    
    return best_colors
