import numpy as np
import time
import random

N = 16
M = N * N
ALL_MASK = (1 << M) - 1

points = [(x, y) for y in range(N) for x in range(N)]

# Precompute conflict masks
conflict_mask = [[0] * M for _ in range(M)]

def dist_sq(p, q):
    return (p[0] - q[0])**2 + (p[1] - q[1])**2

for i in range(M):
    for j in range(i + 1, M):
        p_i = points[i]
        p_j = points[j]
        d_ij = dist_sq(p_i, p_j)
        mask = 0
        for k in range(M):
            if k == i or k == j:
                continue
            p_k = points[k]
            d_ik = dist_sq(p_i, p_k)
            d_jk = dist_sq(p_j, p_k)
            if d_ik == d_jk or d_ik == d_ij or d_jk == d_ij:
                mask |= (1 << k)
        conflict_mask[i][j] = mask
        conflict_mask[j][i] = mask

def get_bits(mask):
    bits = []
    while mask:
        lsb = mask & -mask
        bits.append(lsb.bit_length() - 1)
        mask ^= lsb
    return bits

def greedy_construct(order):
    S = 0
    for idx in order:
        valid = True
        s_list = get_bits(S)
        for i in range(len(s_list)):
            u = s_list[i]
            for j in range(i + 1, len(s_list)):
                v = s_list[j]
                if (conflict_mask[u][v] >> idx) & 1:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            S |= (1 << idx)
    return S

def is_maximal(S):
    covered = 0
    s_list = get_bits(S)
    for i in range(len(s_list)):
        u = s_list[i]
        for j in range(i + 1, len(s_list)):
            v = s_list[j]
            covered |= conflict_mask[u][v]
    not_in_S = ALL_MASK & ~S
    return (not_in_S & ~covered) == 0

def minimize_maximal(S):
    changed = True
    while changed:
        changed = False
        indices = get_bits(S)
        random.shuffle(indices)
        for idx in indices:
            S_new = S & ~(1 << idx)
            if is_maximal(S_new):
                S = S_new
                changed = True
                break
    return S

def solve_bnb():
    best_size = M + 1
    best_S = 0
    start_time = time.time()
    time_limit = 280.0
    nodes_visited = 0
    max_nodes = 1000000

    for _ in range(10):
        order = list(range(M))
        random.shuffle(order)
        S_g = greedy_construct(order)
        S_g = minimize_maximal(S_g)
        size = bin(S_g).count('1')
        if size < best_size:
            best_size = size
            best_S = S_g

    def dfs(S, Candidates, Covered, current_size):
        nonlocal best_size, best_S, nodes_visited, start_time
        
        if time.time() - start_time > time_limit or nodes_visited > max_nodes:
            return
            
        nodes_visited += 1
        
        Uncovered = ALL_MASK & ~S & ~Covered
        
        if Uncovered == 0:
            if current_size < best_size:
                best_size = current_size
                best_S = S
            return

        if current_size >= best_size:
            return

        if Candidates == 0:
            return

        s_list = get_bits(S)
        cand_list = get_bits(Candidates)
        
        covers = {}
        for z in cand_list:
            cov = 0
            for v in s_list:
                cov |= conflict_mask[z][v]
            covers[z] = cov
            
        total_cover = 0
        for z in cand_list:
            total_cover |= covers[z]
            
        if (Uncovered & ~total_cover) != 0:
            return

        forced_candidate = -1
        force_found = False
        u_list = get_bits(Uncovered)
        
        for u in u_list:
            cnt = 0
            last_z = -1
            for z in cand_list:
                if (covers[z] >> u) & 1:
                    cnt += 1
                    last_z = z
                    if cnt > 1:
                        break
            if cnt == 1:
                forced_candidate = last_z
                force_found = True
                break
                
        if force_found:
            z = forced_candidate
            new_S = S | (1 << z)
            new_Covered = Covered
            forbidden = 0
            for v in s_list:
                new_Covered |= conflict_mask[z][v]
                forbidden |= conflict_mask[z][v]
            new_Candidates = Candidates & ~(1 << z) & ~forbidden
            dfs(new_S, new_Candidates, new_Covered, current_size + 1)
            return

        best_z = -1
        max_cov = -1
        for z in cand_list:
            cov_count = bin(covers[z] & Uncovered).count('1')
            if cov_count > max_cov:
                max_cov = cov_count
                best_z = z
                
        if best_z == -1:
            return
            
        z = best_z
        
        new_S = S | (1 << z)
        new_Covered = Covered
        forbidden = 0
        for v in s_list:
            new_Covered |= conflict_mask[z][v]
            forbidden |= conflict_mask[z][v]
        new_Candidates = Candidates & ~(1 << z) & ~forbidden
        
        dfs(new_S, new_Candidates, new_Covered, current_size + 1)
        
        if time.time() - start_time > time_limit:
            return

        new_Candidates = Candidates & ~(1 << z)
        dfs(S, new_Candidates, Covered, current_size)

    dfs(0, ALL_MASK, 0, 0)
    
    if best_S == 0:
        return None
        
    result = []
    s = best_S
    while s:
        lsb = s & -s
        idx = lsb.bit_length() - 1
        s ^= lsb
        result.append(points[idx])
    return result

def construct():
    bnb_result = solve_bnb()
    if bnb_result is not None:
        return bnb_result
        
    best_S = 0
    best_size = M + 1
    for _ in range(20):
        order = list(range(M))
        random.shuffle(order)
        S = greedy_construct(order)
        S = minimize_maximal(S)
        size = bin(S).count('1')
        if size < best_size:
            best_size = size
            best_S = S
            
    result = []
    s = best_S
    while s:
        lsb = s & -s
        idx = lsb.bit_length() - 1
        s ^= lsb
        result.append(points[idx])
    return result
