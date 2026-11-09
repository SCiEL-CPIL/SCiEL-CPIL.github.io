import time
import random

N = 16
POINTS = N * N
DEADLINE = 325

def construct():
    start_time = time.time()
    
    pts = [(x, y) for y in range(N) for x in range(N)]
    
    # Precompute squared distances
    dist2 = [[0] * POINTS for _ in range(POINTS)]
    for i in range(POINTS):
        xi, yi = pts[i]
        for j in range(i + 1, POINTS):
            xj, yj = pts[j]
            d = (xi - xj) ** 2 + (yi - yj) ** 2
            dist2[i][j] = d
            dist2[j][i] = d
    
    # Precompute isosceles pairs for each point
    iso_pairs = [[] for _ in range(POINTS)]
    for i in range(POINTS):
        for j in range(i + 1, POINTS):
            d_ij = dist2[i][j]
            for k in range(j + 1, POINTS):
                d_ik = dist2[i][k]
                d_jk = dist2[j][k]
                if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                    a, b = min(j, k), max(j, k)
                    iso_pairs[i].append((a, b))
                    a, b = min(i, k), max(i, k)
                    iso_pairs[j].append((a, b))
                    a, b = min(i, j), max(i, j)
                    iso_pairs[k].append((a, b))
    
    iso_pairs_set = [set(p) for p in iso_pairs]
    
    def is_no_isosceles(S_set):
        for i in S_set:
            for (a, b) in iso_pairs_set[i]:
                if a in S_set and b in S_set:
                    return False
        return True
    
    def is_maximal(S_set):
        for p in range(POINTS):
            if p in S_set:
                continue
            for (a, b) in iso_pairs_set[p]:
                if a in S_set and b in S_set:
                    break
            else:
                return False
        return True
    
    def greedy_random():
        order = list(range(POINTS))
        random.shuffle(order)
        chosen = []
        chosen_set = set()
        for p in order:
            ok = True
            for (a, b) in iso_pairs_set[p]:
                if a in chosen_set and b in chosen_set:
                    ok = False
                    break
            if ok:
                chosen.append(p)
                chosen_set.add(p)
        return chosen
    
    def reduce_by_removal(S):
        S = list(S)
        improved = True
        while improved:
            improved = False
            random.shuffle(S)
            for i in range(len(S)):
                candidate = S[i]
                new_S = S[:i] + S[i+1:]
                new_S_set = set(new_S)
                if is_maximal(new_S_set):
                    S = new_S
                    improved = True
                    break
        return S
    
    best = None
    best_len = float('inf')
    
    for restart in range(500):
        if time.time() - start_time > DEADLINE:
            break
        
        S = greedy_random()
        S = reduce_by_removal(S)
        
        if len(S) < best_len:
            best = S
            best_len = len(S)
    
    if best is None:
        best = greedy_random()
    
    return [pts[i] for i in best]
