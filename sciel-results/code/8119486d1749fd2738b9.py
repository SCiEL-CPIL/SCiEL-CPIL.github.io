"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen(U, V, W, i0, j0, k0, col):
        # M1
        U[i0 * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M2
        U[(i0 + 1) * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] -= 1
        col += 1
        
        # M3
        U[i0 * m + j0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] -= 1
        W[(k0 + 1) * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M4
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0][col] += 1
        V[j0 * p + k0][col] -= 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        col += 1
        
        # M5
        U[i0 * m + j0][col] += 1
        U[i0 * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0][col] += 1
        W[k0 * n + i0][col] -= 1
        col += 1
        
        # M6
        U[(i0 + 1) * m + j0][col] += 1
        U[i0 * m + j0][col] -= 1
        V[j0 * p + k0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M7
        U[i0 * m + j0 + 1][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] -= 1
        V[(j0 + 1) * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1
        
        return col

    # Generate all potential 2x2x2 blocks
    blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                inds = set()
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            inds.add((i0+di, j0+dj, k0+dk))
                blocks.append((i0, j0, k0, inds))
                
    num_blocks = len(blocks)
    
    # Build conflict graph
    conflict = [[False] * num_blocks for _ in range(num_blocks)]
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if blocks[i][3].intersection(blocks[j][3]):
                conflict[i][j] = True
                conflict[j][i] = True
            else:
                conflict[i][j] = False
                conflict[j][i] = False
                
    best_set = []
    best_size = 0
    
    # Greedy MIS
    def greedy_mis():
        available = set(range(num_blocks))
        chosen = []
        while available:
            min_deg = float('inf')
            min_node = -1
            for u in available:
                deg = 0
                for v in available:
                    if u != v and conflict[u][v]:
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
                    min_node = u
            if min_node == -1: break
            chosen.append(min_node)
            to_remove = {min_node}
            for v in available:
                if conflict[min_node][v]:
                    to_remove.add(v)
            available -= to_remove
        return chosen

    best_set = greedy_mis()
    best_size = len(best_set)
    
    # Random restarts
    for _ in range(100):
        if time.time() > deadline - 10: break
        priorities = list(range(num_blocks))
        random.shuffle(priorities)
        available = set(range(num_blocks))
        chosen = []
        for node in priorities:
            if node not in available: continue
            can_add = True
            for c in chosen:
                if conflict[node][c]:
                    can_add = False
                    break
            if can_add:
                chosen.append(node)
                to_remove = {node}
                for v in available:
                    if conflict[node][v]:
                        to_remove.add(v)
                available -= to_remove
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen
            
    rank = 105 - best_size
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    used_indices = set()
    
    for b_idx in best_set:
        i0, j0, k0, inds = blocks[b_idx]
        col = add_strassen(U, V, W, i0, j0, k0, col)
        used_indices.update(inds)
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i,j,k) not in used_indices:
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
