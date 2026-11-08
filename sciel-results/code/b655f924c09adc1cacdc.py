import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    # 4 Strassen 2x2x2 blocks (maximal packing in 3x5x5)
    blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
    
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    # Uncovered terms
    uncovered = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    uncovered.append((i, j, k))
    
    # Try to group uncovered terms into small sub-problems
    # For a 1x2x2 subproblem (1 row of A, 2x2 block of B):
    # Schoolbook needs 4 multiplications.
    # But we can use: C0 = A0*(B0+B1), C1 = A0*B1  -> C0 - C1 = A0*B0, C1 = A0*B1
    # That's still 2 mults for 2 products. No gain.
    # For 2x1x1: similar, no gain.
    # For 1x1x2: C = A*(B0+B1) - A*B1 gives A*B0, and A*B1. Still 2 mults for 2.
    # So no simple gain for 1x2x2 or similar.
    
    # However, for a 2x2x1 or 2x1x2 we can potentially save.
    # 2x2x1: C00=A00*B0, C01=A01*B1, C10=A10*B0, C11=A11*B1
    # = A00*B0 + A10*B0 = (A00+A10)*B0 ... no, B is 2x1 so B0, B1 are scalars.
    # Actually for 2x2 * 2x1: result is 2x1.
    # C0 = A00*B0 + A01*B1, C1 = A10*B0 + A11*B1
    # M1 = (A00+A10)*B0 = A00*B0 + A10*B0
    # M2 = (A01+A11)*B1 = A01*B1 + A11*B1
    # M3 = (A00+A01)*B1 = A00*B1 + A01*B1  -- not directly useful
    # This doesn't help directly.
    
    # Let's try a different approach: use the fact that some uncovered terms
    # share A or B indices to combine.
    
    # Group by (i,j) pairs: for each (i,j), how many k are uncovered?
    from collections import defaultdict
    by_ij = defaultdict(list)
    for (i, j, k) in uncovered:
        by_ij[(i,j)].append(k)
    
    # Group by (j,k): for each (j,k), how many i are uncovered?
    by_jk = defaultdict(list)
    for (i, j, k) in uncovered:
        by_jk[(j,k)].append(i)
    
    # Group by (i,k): for each (i,k), how many j are uncovered?
    by_ik = defaultdict(list)
    for (i, j, k) in uncovered:
        by_ik[(i,k)].append(j)
    
    # Strategy: for groups where multiple terms share the same A entry (same i,j) 
    # or same B entry (same j,k), we can potentially use linearity.
    # But each term A[i,j]*B[j,k] is a separate product, so sharing doesn't help
    # unless we can express multiple products as fewer products via Strassen-like tricks.
    
    # For now, use pure schoolbook for uncovered terms.
    # The key insight: we already have the minimum rank for this block decomposition.
    
    total_rank = len(blocks) * 7 + len(uncovered)
    
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
