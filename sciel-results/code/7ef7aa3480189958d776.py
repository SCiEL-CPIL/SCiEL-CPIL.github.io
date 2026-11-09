import random
import time

def construct():
    n = 18
    start_time = time.time()
    deadline = start_time + 100
    
    best_score = n
    best_graph = None
    
    adj = [0] * n
    
    def is_triangle_free(S):
        s = S
        while s:
            u = (s & -s).bit_length() - 1
            s = s & (s - 1)
            W = adj[u] & S
            w = W
            while w:
                v = (w & -w).bit_length() - 1
                w = w & (w - 1)
                if adj[v] & W:
                    return False
        return True

    def get_score():
        max_size = 0
        # Iterate subsets
        for S in range(1, 1 << n):
            pc = bin(S).count('1')
            if pc <= max_size:
                continue
            if is_triangle_free(S):
                max_size = pc
                if max_size == n:
                    break
        return max_size

    def remove_k4s():
        changed = True
        while changed and time.time() < deadline:
            changed = False
            found = False
            for i in range(n):
                for j in range(i+1, n):
                    if not (adj[i] & (1 << j)): continue
                    for k in range(j+1, n):
                        if not (adj[i] & (1 << k)) or not (adj[j] & (1 << k)): continue
                        for l in range(k+1, n):
                            if (adj[i] & (1 << l)) and (adj[j] & (1 << l)) and (adj[k] & (1 << l)):
                                edges = [(i,j), (i,k), (i,l), (j,k), (j,l), (k,l)]
                                e = random.choice(edges)
                                adj[e[0]] &= ~(1 << e[1])
                                adj[e[1]] &= ~(1 << e[0])
                                changed = True
                                found = True
                                break
                        if found: break
                    if found: break
                if found: break
            if found: break

    def try_improve():
        score = get_score()
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u] & (1 << v): continue
                    common = adj[u] & adj[v]
                    safe = True
                    c = common
                    while c:
                        w = (c & -c).bit_length() - 1
                        c = c & (c - 1)
                        if adj[w] & common:
                            safe = False
                            break
                    if safe:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        new_score = get_score()
                        if new_score < score:
                            score = new_score
                            improved = True
                        else:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
        return score

    for seed in range(100):
        if time.time() > deadline: break
        random.seed(seed)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        remove_k4s()
        score = try_improve()
        
        if score < best_score:
            best_score = score
            best_graph = adj[:]
            print(f"Seed {seed}: Score {score}")

    if best_graph is None:
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
        
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_graph[u] & (1 << v):
                edges.append([u, v])
    return edges
