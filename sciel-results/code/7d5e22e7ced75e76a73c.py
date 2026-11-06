import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Target ns in descending order
    target_ns = [26, 25, 24, 23, 22, 21, 20]
    
    # 1. Try Local Search with delta evaluation (fast) - multiple restarts
    for n in target_ns:
        if time.time() > deadline - 30:
            break
        # Try multiple restarts with increasing time budget
        restarts = 3
        for r in range(restarts):
            remaining = deadline - time.time()
            if remaining < 5:
                break
            t_budget = min(10, remaining / (restarts - r))
            result = local_search_delta(n, time_limit=t_budget)
            if result is not None:
                return result

    # 2. Try Cyclic Search
    for n in target_ns:
        if time.time() > deadline - 5:
            break
        result = search_cyclic(n, time_limit=min(5, max(1, deadline - time.time() - 5)))
        if result is not None:
            return result

    # Fallback: construct a valid coloring for small n
    return {'n': 11, 'colors': [(i * 3 + j) % 5 for i in range(11) for j in range(i + 1, 11)]}

def search_cyclic(n, time_limit=5):
    k = n // 2
    # params: colors for differences 1..k
    # params[d-1] is the color for difference d (and -d)
    params = [random.randint(0, 4) for _ in range(k)]
    
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Precompute edge-to-diff mapping for speed
    edge_diff = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = j - i
            if diff <= k:
                d = diff
            else:
                d = n - diff
            edge_diff.append(d - 1)
    
    def gen_colors(params):
        return [params[ed] for ed in edge_diff]

    def calc_violations(colors):
        neighbors = [[0] * n for _ in range(5)]
        for idx, (i, j) in enumerate(edges):
            c = colors[idx]
            neighbors[c][i] |= (1 << j)
            neighbors[c][j] |= (1 << i)
            
        violations = 0
        for c in range(5):
            for u in range(n):
                nu = neighbors[c][u]
                for v in range(u + 1, n):
                    common = nu & neighbors[c][v]
                    cnt = common.bit_count()
                    if cnt > 1:
                        violations += cnt * (cnt - 1) // 2
        return violations

    current_colors = gen_colors(params)
    current_violations = calc_violations(current_colors)
    best_violations = current_violations
    best_params = params[:]
    
    if current_violations == 0:
        return {'n': n, 'colors': current_colors}
        
    start_time = time.time()
    temperature = 2.0
    cooling_rate = 0.999
    
    while time.time() < start_time + time_limit:
        idx = random.randint(0, k - 1)
        old_val = params[idx]
        new_val = random.randint(0, 4)
        if new_val == old_val:
            continue
            
        params[idx] = new_val
        new_colors = gen_colors(params)
        new_violations = calc_violations(new_colors)
        
        delta = new_violations - current_violations
        
        if delta < 0 or random.random() < pow(2.71828, -delta / max(temperature, 0.01)):
            current_violations = new_violations
            current_colors = new_colors
            if new_violations < best_violations:
                best_violations = new_violations
                best_params = params[:]
                if best_violations == 0:
                    return {'n': n, 'colors': current_colors}
        else:
            params[idx] = old_val
            
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
            
    return None

def local_search_delta(n, time_limit=15):
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Build edge index lookup
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
    
    # Initial random coloring
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    # Build neighbor bitmasks per color
    neighbors = [[0] * n for _ in range(5)]
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)
    
    # Count C4s: for each color, for each pair (u,v), count common neighbors
    # violations = sum over c, u<v of C(common_c(u,v), 2)
    def count_c4s():
        total = 0
        for c in range(5):
            for u in range(n):
                nu = neighbors[c][u]
                for v in range(u + 1, n):
                    common = nu & neighbors[c][v]
                    cnt = common.bit_count()
                    if cnt > 1:
                        total += cnt * (cnt - 1) // 2
        return total
    
    current_violations = count_c4s()
    best_violations = current_violations
    best_colors = colors[:]
    
    if current_violations == 0:
        return {'n': n, 'colors': colors}
    
    start_time = time.time()
    temperature = 3.0
    cooling_rate = 0.9998
    min_temp = 0.05
    
    while time.time() < start_time + time_limit:
        e_idx = random.randint(0, num_edges - 1)
        u, v = edges[e_idx]
        old_color = colors[e_idx]
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
        
        # Compute delta in violations when changing edge (u,v) from old_color to new_color
        delta = 0
        
        nu_old = neighbors[old_color][u]
        nv_old = neighbors[old_color][v]
        nu_new = neighbors[new_color][u]
        nv_new = neighbors[new_color][v]
        
        for w in range(n):
            if w == u or w == v:
                continue
            nw_old = neighbors[old_color][w]
            nw_new = neighbors[new_color][w]
            
            # Old color: removing (u,v)
            # Pair (u, w): if v in N_oc(w), common count decreases by 1
            if (1 << v) & nw_old:
                old_k = (nu_old & nw_old).bit_count()
                delta -= (old_k - 1)
            # Pair (v, w): if u in N_oc(w), common count decreases by 1
            if (1 << u) & nw_old:
                old_k = (nv_old & nw_old).bit_count()
                delta -= (old_k - 1)
            
            # New color: adding (u,v)
            # Pair (u, w): if v in N_nc(w), common count increases by 1
            if (1 << v) & nw_new:
                old_k = (nu_new & nw_new).bit_count()
                delta += old_k
            # Pair (v, w): if u in N_nc(w), common count increases by 1
            if (1 << u) & nw_new:
                old_k = (nv_new & nw_new).bit_count()
                delta += old_k
        
        # Apply the change
        colors[e_idx] = new_color
        neighbors[old_color][u] &= ~(1 << v)
        neighbors[old_color][v] &= ~(1 << u)
        neighbors[new_color][u] |= (1 << v)
        neighbors[new_color][v] |= (1 << u)
        
        current_violations += delta
        
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            if best_violations == 0:
                return {'n': n, 'colors': colors}
        
        # Simulated annealing acceptance
        if delta >= 0:
            if random.random() < pow(2.71828, -delta / max(temperature, min_temp)):
                pass  # accept
            else:
                # Revert
                colors[e_idx] = old_color
                neighbors[new_color][u] &= ~(1 << v)
                neighbors[new_color][v] &= ~(1 << u)
                neighbors[old_color][u] |= (1 << v)
                neighbors[old_color][v] |= (1 << u)
                current_violations -= delta
        
        temperature *= cooling_rate
        if temperature < min_temp:
            temperature = min_temp
    
    return None
