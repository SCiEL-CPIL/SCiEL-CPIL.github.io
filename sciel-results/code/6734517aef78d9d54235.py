import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = 120.0
    
    for n in range(35, 16, -1):
        if time.time() - start_time > deadline:
            break
        graph = _find_valid_graph(n, time.time(), deadline)
        if graph is not None:
            return graph
            
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}

def _find_valid_graph(n, start_time, deadline):
    max_attempts = 100
    
    for attempt in range(max_attempts):
        if time.time() - start_time > deadline:
            return None
            
        G = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    G[i][j] = 1
                    G[j][i] = 1
        
        for step in range(200):
            if time.time() - start_time > deadline:
                return None
            
            # Find one K4
            k4 = _find_one_k4(G, n)
            if k4:
                u, v, w, x = k4
                # Remove random edge from K4
                edges_in_k4 = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
                e = random.choice(edges_in_k4)
                G[e[0]][e[1]] = 0
                G[e[1]][e[0]] = 0
                continue
            
            # Find one Ind6
            ind6 = _find_one_ind6(G, n)
            if ind6:
                # Add random edge in Ind6
                u, v = random.sample(ind6, 2)
                G[u][v] = 1
                G[v][u] = 1
                continue
            
            # No violations
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if G[i][j]:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
    return None

def _find_one_k4(G, n):
    neighbors = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if G[i][j]:
                neighbors[i].add(j)
                neighbors[j].add(i)
                
    for i in range(n):
        for j in neighbors[i]:
            if j <= i: continue
            common_jw = neighbors[i] & neighbors[j]
            for w in common_jw:
                if w <= j: continue
                # Check if any common neighbor of i and w is also neighbor of j? No, K4 is i,j,w,x
                # x must be neighbor of i, j, w
                common_iw = neighbors[i] & neighbors[w]
                for x in common_iw:
                    if x in neighbors[j]:
                        return (i, j, w, x)
    return None

def _find_one_ind6(G, n):
    # Check all combinations of 6
    for combo in itertools.combinations(range(n), 6):
        is_ind = True
        for i in range(6):
            for j in range(i+1, 6):
                if G[combo[i]][combo[j]]:
                    is_ind = False
                    break
            if not is_ind:
                break
        if is_ind:
            return combo
    return None
