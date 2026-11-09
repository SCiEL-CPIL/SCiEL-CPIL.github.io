import random
import time
import math

def construct():
    n = 16
    # Initialize K_8,8 + (0,1)
    A = [[0]*n for _ in range(n)]
    for u in range(8):
        for v in range(8, 16):
            A[u][v] = 1
            A[v][u] = 1
    A[0][1] = 1
    A[1][0] = 1
    
    # Get list of edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if A[i][j]:
                edges.append((i, j))
                
    def get_score():
        degs = [sum(A[i]) for i in range(n)]
        max_sum = 0
        for i in range(n):
            for j in range(i+1, n):
                if A[i][j]:
                    for k in range(j+1, n):
                        if A[i][k] and A[j][k]:
                            s = degs[i] + degs[j] + degs[k]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    current_score = get_score()
    best_score = current_score
    best_A = [row[:] for row in A]
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    T = 5.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        # Pick random edge to remove
        idx_rm = random.randint(0, len(edges) - 1)
        u, v = edges[idx_rm]
        
        # Pick random non-edge to add
        # Generate random pair
        while True:
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x == y:
                continue
            if x > y:
                x, y = y, x
            if A[x][y] == 0:
                break
        
        # Perform move
        A[u][v] = 0
        A[v][u] = 0
        edges[idx_rm] = (x, y)
        
        A[x][y] = 1
        A[y][x] = 1
        
        new_score = get_score()
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_A = [row[:] for row in A]
        else:
            # Accept with probability
            delta = new_score - current_score
            if T > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    # Revert
                    A[u][v] = 1
                    A[v][u] = 1
                    edges[idx_rm] = (u, v)
                    
                    A[x][y] = 0
                    A[y][x] = 0
            else:
                # Revert
                A[u][v] = 1
                A[v][u] = 1
                edges[idx_rm] = (u, v)
                
                A[x][y] = 0
                A[y][x] = 0
        
        T *= cooling

    # Reconstruct edges from best_A
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i][j]:
                final_edges.append([i, j])
                
    return {'n': n, 'edges': final_edges}
