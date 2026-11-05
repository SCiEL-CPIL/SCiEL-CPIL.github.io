import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Prioritize n=26
    # If we find it, great.
    
    # Try n=26 with a significant time chunk
    time_for_26 = 80.0
    result = solve_n(26, time_for_26)
    if result:
        return result
        
    # Try n=25 with remaining time
    remaining = deadline - time.time()
    if remaining > 5.0:
        result = solve_n(25, remaining - 2.0)
        if result:
            return result
            
    # Fallback
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def solve_n(n, time_budget):
    if n <= 1:
        return {'n': n, 'colors': []}
        
    num_edges = n * (n - 1) // 2
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    edge_idx = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1
            
    neighbors = [[0] * n for _ in range(5)]
    
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[edge_idx[i][j]]
            neighbors[c][i] |= (1 << j)
            neighbors[c][j] |= (1 << i)
            
    c4_count = [0] * 5
    
    def calc_c4_count(c):
        count = 0
        for u in range(n):
            nu = neighbors[c][u]
            for v in range(u + 1, n):
                nv = neighbors[c][v]
                common = (nu & nv).bit_count()
                count += common * (common - 1) // 2
        return count
        
    for c in range(5):
        c4_count[c] = calc_c4_count(c)
        
    total_violations = sum(c4_count)
    best_total = total_violations
    best_colors = colors.copy()
    
    if total_violations == 0:
        return {'n': n, 'colors': colors}
        
    T = 1.0
    cooling = 0.9999
    min_T = 0.001
    
    binom2 = [0] * (n + 1)
    for k in range(n + 1):
        binom2[k] = k * (k - 1) // 2
        
    bits = [1 << i for i in range(n)]
    
    # Local variables for speed
    ne = neighbors
    ei = edge_idx
    co = colors
    bc = binom2
    bt = bits
    
    while time.time() < start_time + time_budget:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        c_old = co[ei[u][v]]
        c_new = random.randint(0, 4)
        if c_new == c_old:
            continue
            
        delta = 0
        
        nu_old = ne[c_old][u]
        nv_old = ne[c_old][v]
        
        for z in range(n):
            if z == u or z == v:
                continue
            nz = ne[c_old][z]
            
            c_uz_old = (nu_old & nz).bit_count()
            if nz & bt[v]:
                c_uz_new = c_uz_old - 1
            else:
                c_uz_new = c_uz_old
                
            c_vz_old = (nv_old & nz).bit_count()
            if nz & bt[u]:
                c_vz_new = c_vz_old - 1
            else:
                c_vz_new = c_vz_old
                
            delta += bc[c_uz_new] - bc[c_uz_old]
            delta += bc[c_vz_new] - bc[c_vz_old]
            
        delta_old = delta
        
        nu_new = ne[c_new][u]
        nv_new = ne[c_new][v]
        
        for z in range(n):
            if z == u or z == v:
                continue
            nz = ne[c_new][z]
            
            c_uz_old = (nu_new & nz).bit_count()
            if nz & bt[v]:
                c_uz_new = c_uz_old + 1
            else:
                c_uz_new = c_uz_old
                
            c_vz_old = (nv_new & nz).bit_count()
            if nz & bt[u]:
                c_vz_new = c_vz_old + 1
            else:
                c_vz_new = c_vz_old
                
            delta += bc[c_uz_new] - bc[c_uz_old]
            delta += bc[c_vz_new] - bc[c_vz_old]
            
        if delta < 0 or random.random() < math.exp(-delta / T):
            co[ei[u][v]] = c_new
            
            ne[c_old][u] &= ~bt[v]
            ne[c_old][v] &= ~bt[u]
            ne[c_new][u] |= bt[v]
            ne[c_new][v] |= bt[u]
            
            c4_count[c_old] += delta_old
            c4_count[c_new] += (delta - delta_old)
            total_violations += delta
            
            if total_violations < best_total:
                best_total = total_violations
                best_colors = co.copy()
                
                if total_violations == 0:
                    return {'n': n, 'colors': co}
                    
        T *= cooling
        if T < min_T:
            T = min_T
            
    if best_total == 0:
        return {'n': n, 'colors': best_colors}
    else:
        return None
