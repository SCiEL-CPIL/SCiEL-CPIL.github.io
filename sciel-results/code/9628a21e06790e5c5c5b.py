import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 150s limit
    
    n = 20
    
    # Build polarity graph of PG(2,4) as initial seed
    # GF(4) arithmetic
    gf4_add = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
    gf4_mul = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
    gf4_inv = [0,1,3,2]
    
    def gf4_dot(p, q):
        return gf4_add[gf4_add[gf4_mul[p[0]][q[0]]][gf4_mul[p[1]][q[1]]]][gf4_mul[p[2]][q[2]]]
    
    # Enumerate 21 points of PG(2,4)
    points = []
    seen = set()
    for a in range(4):
        for b in range(4):
            for c in range(4):
                if a == 0 and b == 0 and c == 0:
                    continue
                pt = [a, b, c]
                if pt[0] != 0:
                    inv = gf4_inv[pt[0]]
                    pt = [1, gf4_mul[pt[1]][inv], gf4_mul[pt[2]][inv]]
                elif pt[1] != 0:
                    inv = gf4_inv[pt[1]]
                    pt = [0, 1, gf4_mul[pt[2]][inv]]
                else:
                    pt = [0, 0, 1]
                key = tuple(pt)
                if key not in seen:
                    seen.add(key)
                    points.append(pt)
    
    # Build 21-vertex polarity graph, delete one vertex to get 20
    adj21 = [[0]*21 for _ in range(21)]
    for i in range(21):
        for j in range(i+1, 21):
            if gf4_dot(points[i], points[j]) == 0:
                adj21[i][j] = 1
                adj21[j][i] = 1
    
    # Delete vertex 0 to get 20-vertex graph
    A = np.zeros((n, n), dtype=np.int8)
    for i in range(1, 21):
        for j in range(1, 21):
            if adj21[i][j]:
                A[i-1][j-1] = 1
    
    # Compute common neighbor matrix K
    K = (A @ A).astype(np.int16)
    
    current_edges = int(np.sum(A)) // 2
    best_edges = current_edges
    best_A = A.copy()
    best_K = K.copy()
    
    def can_add(A, K, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        for x in range(n):
            if A[u][x] == 1 and x != v:
                if K[x][v] > 1:
                    return False
        for y in range(n):
            if A[v][y] == 1 and y != u:
                if K[u][y] > 1:
                    return False
        return True
    
    def add_edge(A, K, u, v):
        A[u][v] = 1
        A[v][u] = 1
        for x in range(n):
            if A[u][x] == 1 and x != v:
                K[x][v] += 1
                K[v][x] += 1
        for y in range(n):
            if A[v][y] == 1 and y != u:
                K[u][y] += 1
                K[y][u] += 1
    
    def remove_edge(A, K, u, v):
        A[u][v] = 0
        A[v][u] = 0
        for x in range(n):
            if A[u][x] == 1 and x != v:
                K[x][v] -= 1
                K[v][x] -= 1
        for y in range(n):
            if A[v][y] == 1 and y != u:
                K[u][y] -= 1
                K[y][u] -= 1
    
    def greedy_fill(A, K, max_edges=None):
        """Greedily add all possible edges. Returns number of edges added."""
        added = 0
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u+1, n):
                    if A[u][v] == 0 and can_add(A, K, u, v):
                        add_edge(A, K, u, v)
                        added += 1
                        changed = True
                        if max_edges and added >= max_edges:
                            return added
        return added
    
    # Initial greedy fill
    greedy_fill(A, K)
    current_edges = int(np.sum(A)) // 2
    if current_edges > best_edges:
        best_edges = current_edges
        best_A = A.copy()
        best_K = K.copy()
    
    rng = random.Random(42)
    
    # Simulated annealing
    t = 1.0
    t_min = 0.005
    
    while time.time() - start_time < deadline:
        move_type = rng.random()
        
        if move_type < 0.5:
            # Try to add a random non-edge
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if A[u][v]:
                continue
            if can_add(A, K, u, v):
                add_edge(A, K, u, v)
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_A = A.copy()
                    best_K = K.copy()
        elif move_type < 0.8:
            # Remove a random edge
            edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if A[i][j]]
            if not edges_list:
                continue
            u, v = rng.choice(edges_list)
            remove_edge(A, K, u, v)
            current_edges -= 1
        else:
            # Remove k random edges, then greedy refill
            k = rng.randint(2, 6)
            edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if A[i][j]]
            if len(edges_list) < k:
                k = len(edges_list)
            if k == 0:
                continue
            
            to_remove = rng.sample(edges_list, k)
            for u, v in to_remove:
                remove_edge(A, K, u, v)
                current_edges -= 1
            
            # Greedy refill
            greedy_fill(A, K)
            current_edges = int(np.sum(A)) // 2
            
            if current_edges > best_edges:
                best_edges = current_edges
                best_A = A.copy()
                best_K = K.copy()
            elif current_edges < best_edges - 15:
                # Revert to best
                A = best_A.copy()
                K = best_K.copy()
                current_edges = best_edges
        
        # Cool
        t *= 0.9995
        if t < t_min:
            t = t_min
    
    # Final greedy polish
    A = best_A.copy()
    K = best_K.copy()
    greedy_fill(A, K)
    final_edges = int(np.sum(A)) // 2
    
    # Extract edges
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if A[u][v]:
                edges.append([u, v])
    
    return edges
