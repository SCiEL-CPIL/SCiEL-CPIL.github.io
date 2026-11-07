import time
import random
import math

N = 15
DEADLINE = 85.0

def construct():
    start_time = time.time()
    
    adj = [[0] * N for _ in range(N)]
    deg = [0] * N
    neighbors = [set() for _ in range(N)]
    
    def add_edge(u, v):
        if u == v or adj[u][v]:
            return False
        adj[u][v] = 1
        adj[v][u] = 1
        deg[u] += 1
        deg[v] += 1
        neighbors[u].add(v)
        neighbors[v].add(u)
        return True

    def remove_edge(u, v):
        if u == v or not adj[u][v]:
            return False
        adj[u][v] = 0
        adj[v][u] = 0
        deg[u] -= 1
        deg[v] -= 1
        neighbors[u].remove(v)
        neighbors[v].remove(u)
        return True

    def try_add_edge(u, v):
        if u == v or adj[u][v]:
            return False
        
        # 1. Check s-u-v-t
        deg_map_u = {}
        for s in neighbors[u]:
            if s != v:
                d = deg[s]
                if d not in deg_map_u:
                    deg_map_u[d] = []
                deg_map_u[d].append(s)
        
        for t in neighbors[v]:
            if t == u:
                continue
            d_t = deg[t]
            if d_t in deg_map_u:
                for s in deg_map_u[d_t]:
                    if s != t:
                        return False
        
        new_deg_u = deg[u] + 1
        new_deg_v = deg[v] + 1
        
        # 2. Check u-w-x-y
        for w in neighbors[u]:
            if w == v:
                continue
            for x in neighbors[w]:
                if x == u:
                    continue
                for y in neighbors[x]:
                    if y == w:
                        continue
                    if u == x or u == y or w == y:
                        continue
                    if new_deg_u == deg[y]:
                        return False
        
        # 3. Check w-x-y-v
        for w in neighbors[v]:
            if w == u:
                continue
            for x in neighbors[w]:
                if x == v:
                    continue
                for y in neighbors[x]:
                    if y == w:
                        continue
                    if v == x or v == y or w == y:
                        continue
                    if deg[y] == new_deg_v:
                        return False
        
        # 4. Check u-w-x-v
        for w in neighbors[u]:
            if w == v:
                continue
            for x in neighbors[w]:
                if x == u:
                    continue
                if adj[x][v]:
                    if u != x and u != v and w != x and w != v and x != v and x != u:
                        if new_deg_u == new_deg_v:
                            return False

        add_edge(u, v)
        return True

    # Initialize
    random.seed(42)
    edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    random.shuffle(edges)
    for u, v in edges:
        try_add_edge(u, v)
    
    current_count = sum(deg) // 2
    best_adj = [row[:] for row in adj]
    best_count = current_count
    
    T = 10.0
    cooling_rate = 0.995
    min_T = 0.01
    
    while time.time() - start_time < DEADLINE:
        if T < min_T:
            break
            
        move_type = random.random()
        
        if move_type < 0.4:
            missing = [(u, v) for u in range(N) for v in range(u + 1, N) if not adj[u][v]]
            if missing:
                u, v = random.choice(missing)
                if try_add_edge(u, v):
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = [row[:] for row in adj]
                        
        elif move_type < 0.7:
            existing = [(u, v) for u in range(N) for v in range(u + 1, N) if adj[u][v]]
            if existing:
                u, v = random.choice(existing)
                remove_edge(u, v)
                current_count -= 1
                if random.random() < math.exp(-1.0 / T):
                    pass
                else:
                    add_edge(u, v)
                    current_count += 1
        else:
            existing = [(u, v) for u in range(N) for v in range(u + 1, N) if adj[u][v]]
            missing = [(u, v) for u in range(N) for v in range(u + 1, N) if not adj[u][v]]
            if existing and missing:
                u1, v1 = random.choice(existing)
                u2, v2 = random.choice(missing)
                remove_edge(u1, v1)
                current_count -= 1
                if try_add_edge(u2, v2):
                    current_count += 1
                else:
                    add_edge(u1, v1)
                    current_count += 1
                    
        T *= cooling_rate

    result = []
    for u in range(N):
        for v in range(u + 1, N):
            if best_adj[u][v]:
                result.append([int(u), int(v)])
    
    return result
