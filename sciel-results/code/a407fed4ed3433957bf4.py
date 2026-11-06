import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    for n in range(29, 10, -1):
        result = local_search(n, deadline)
        if result is not None:
            return result
            
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def local_search(n, deadline):
    max_restarts = 5
    max_iter = 50000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        col = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                c = random.randint(0, 4)
                col[i][j] = c
                col[j][i] = c
                
        N = [[0]*5 for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                c = col[i][j]
                N[i][c] |= (1 << j)
                N[j][c] |= (1 << i)
                
        current_viol = 0
        for i in range(n):
            for j in range(i+1, n):
                current_viol += calc_pair_viol(N[i], N[j])
                
        if current_viol == 0:
            return format_result(n, col)
            
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if u > v: u, v = v, u
            
            old_c = col[u][v]
            new_c = random.randint(0, 4)
            if new_c == old_c: continue
            
            delta = 0
            
            for w in range(n):
                if w == u or w == v: continue
                
                # Pair (u, w)
                if (N[w][old_c] >> v) & 1:
                    delta += delta_viol(N[u][old_c] & N[w][old_c], 1)
                if (N[w][new_c] >> v) & 1:
                    delta += delta_viol(N[u][new_c] & N[w][new_c], -1)
                
                # Pair (v, w)
                if (N[w][old_c] >> u) & 1:
                    delta += delta_viol(N[v][old_c] & N[w][old_c], 1)
                if (N[w][new_c] >> u) & 1:
                    delta += delta_viol(N[v][new_c] & N[w][new_c], -1)
            
            new_total = current_viol + delta
            if new_total < current_viol:
                N[u][old_c] &= ~(1 << v)
                N[v][old_c] &= ~(1 << u)
                N[u][new_c] |= (1 << v)
                N[v][new_c] |= (1 << u)
                
                col[u][v] = new_c
                col[v][u] = new_c
                
                current_viol = new_total
                
                if current_viol == 0:
                    return format_result(n, col)
                
    return None

def calc_pair_viol(N_u, N_v):
    v = 0
    for c in range(5):
        common = N_u[c] & N_v[c]
        count = common.bit_count()
        if count > 1:
            v += count - 1
    return v

def delta_viol(common, delta_k):
    k_old = common.bit_count()
    k_new = k_old + delta_k
    return max(0, k_new - 1) - max(0, k_old - 1)

def format_result(n, col):
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            colors.append(col[i][j])
    return {'n': n, 'colors': colors}
