def construct():
    import time
    import itertools
    import numpy as np
    import random
    
    start_time = time.time()
    DEADLINE = 115
    
    n = 16
    adj = np.zeros((n, n), dtype=bool)
    for a in range(3):
        for b in range(3, n):
            adj[a, b] = True
            adj[b, a] = True
    for i in range(3, 15):
        adj[i, i+1] = True
        adj[i+1, i] = True
    adj[15, 3] = True
    adj[3, 15] = True
    
    R = list(range(3, 16))
    adj_R = np.zeros((13, 13), dtype=bool)
    for i in range(13):
        for j in range(13):
            adj_R[i, j] = adj[R[i], R[j]]
    
    def check_q3_after_add(adj_R, i, j):
        others = [k for k in range(13) if k != i and k != j]
        # Case 1: Q3 with 2 L vertices and 6 R vertices (including i, j)
        for four in itertools.combinations(others, 4):
            six_set = [i, j] + list(four)
            for part1 in itertools.combinations(six_set, 3):
                part2 = tuple(v for v in six_set if v not in part1)
                ok = True
                for a in part1:
                    deg = sum(1 for b in part2 if adj_R[a, b])
                    if deg != 2:
                        ok = False
                        break
                if not ok:
                    continue
                for b in part2:
                    deg = sum(1 for a in part1 if adj_R[a, b])
                    if deg != 2:
                        ok = False
                        break
                if not ok:
                    continue
                return True
        # Case 2: Q3 with 8 R vertices (including i, j)
        for six in itertools.combinations(others, 6):
            eight_set = [i, j] + list(six)
            for part1 in itertools.combinations(eight_set, 4):
                part2 = tuple(v for v in eight_set if v not in part1)
                ok = True
                for a in part1:
                    deg = sum(1 for b in part2 if adj_R[a, b])
                    if deg != 3:
                        ok = False
                        break
                if not ok:
                    continue
                for b in part2:
                    deg = sum(1 for a in part1 if adj_R[a, b])
                    if deg != 3:
                        ok = False
                        break
                if not ok:
                    continue
                return True
        return False
    
    candidates = [(i, j) for i in range(13) for j in range(i+1, 13) if not adj_R[i, j]]
    random.seed(42)
    random.shuffle(candidates)
    
    for (i, j) in candidates:
        if time.time() > DEADLINE:
            break
        if adj_R[i, j]:
            continue
        adj_R[i, j] = True
        adj_R[j, i] = True
        if check_q3_after_add(adj_R, i, j):
            adj_R[i, j] = False
            adj_R[j, i] = False
    
    edges = []
    for a in range(3):
        for b in range(3, n):
            edges.append([a, b])
    for i in range(13):
        for j in range(i+1, 13):
            if adj_R[i, j]:
                edges.append([R[i], R[j]])
    return edges
