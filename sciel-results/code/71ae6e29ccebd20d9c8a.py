import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    # Try multiple block configurations and pick the best
    all_blocks_options = [
        [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)],
        [(0, 0, 0), (0, 2, 0), (1, 0, 0), (1, 2, 0)],
        [(0, 0, 0), (0, 2, 2), (1, 0, 2), (1, 2, 2)],
        [(0, 0, 0), (1, 0, 0), (0, 2, 0), (1, 2, 0)],
        [(0, 0, 0), (0, 2, 0), (2, 0, 0), (2, 2, 0)],
        [(0, 0, 0), (0, 2, 0), (0, 0, 2), (1, 0, 2)],
        [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)],
        [(0, 0, 0), (0, 2, 0), (1, 0, 0), (1, 2, 2)],
    ]
    
    best_rank = float('inf')
    best_blocks = None
    best_uncovered = None
    
    for blocks in all_blocks_options:
        covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
        valid = True
        for i0, j0, k0 in blocks:
            if i0 + 1 >= n or j0 + 1 >= m or k0 + 1 >= p:
                valid = False
                break
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered_3d[i0+di][j0+dj][k0+dk] = True
        if not valid:
            continue
        
        uncovered = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered_3d[i][j][k]:
                        uncovered.append((i, j, k))
        
        # Try to reduce uncovered via grouping
        # For each (i,j) with multiple k uncovered, or (j,k) with multiple i, 
        # we can't save with pure schoolbook, but check if we can add more Strassen blocks
        # to cover some of these.
        
        # Try adding 1-2 more Strassen blocks to cover remaining gaps
        # Find 2x2x2 subcubes within uncovered that could be covered by another Strassen
        # But they must not overlap with already-covered regions in a way that wastes rank
        
        # For now, just count
        rank = len(blocks) * 7 + len(uncovered)
        if rank < best_rank:
            best_rank = rank
            best_blocks = blocks
            best_uncovered = uncovered
    
    blocks = best_blocks
    uncovered = best_uncovered
    total_rank = best_rank
    
    # Build factor matrices
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
    
    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    for (i, j, k) in uncovered:
        if time.time() > deadline:
            break
        U[i*m+j][col] += 1
        V[j*p+k][col] += 1
        W[k*n+i][col] += 1
        col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
