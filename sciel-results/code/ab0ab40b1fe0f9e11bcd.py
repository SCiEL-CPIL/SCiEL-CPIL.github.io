import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    def add_strassen_block(U, V, W, i0, j0, k0, col_offset):
        """Add a Strassen 2x2x2 block starting at (i0, j0, k0) with rank 7."""
        # M1 = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M2 = (A[i0+1,j0] + A[i0+1,j0+1]) * B[j0,k0]
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        # M3 = A[i0,j0] * (B[j0,k0+1] - B[j0+1,k0+1])
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M4 = A[i0+1,j0+1] * (B[j0+1,k0] - B[j0,k0])
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        # M5 = (A[i0,j0] + A[i0,j0+1]) * B[j0+1,k0+1]
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        # M6 = (A[i0+1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k0+1])
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M7 = (A[i0,j0+1] - A[i0+1,j0+1]) * (B[j0+1,k0] + B[j0+1,k0+1])
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is correct."""
        # Check that the contraction gives the correct tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r] for r in range(rank))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # All possible 2x2x2 block positions
    block_positions = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                block_positions.append((i0, j0, k0))
    
    def block_entries(i0, j0, k0):
        entries = set()
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    entries.add((i0+di, j0+dj, k0+dk))
        return entries
    
    # Try all subsets of blocks (up to a reasonable size)
    best_rank = n * m * p  # Schoolbook rank
    best_blocks = []
    
    # For efficiency, only try subsets of size up to 6
    from itertools import combinations
    
    max_blocks_to_try = 6
    for num_blocks in range(0, min(len(block_positions), max_blocks_to_try) + 1):
        if time.time() > deadline - 30:
            break
        for blocks in combinations(block_positions, num_blocks):
            # Check if blocks are non-overlapping
            covered = set()
            valid = True
            for (i0, j0, k0) in blocks:
                entries = block_entries(i0, j0, k0)
                if covered & entries:
                    valid = False
                    break
                covered |= entries
            
            if not valid:
                continue
            
            # Calculate rank
            strassen_rank = num_blocks * 7
            schoolbook_rank = n * m * p - num_blocks * 8
            total_rank = strassen_rank + schoolbook_rank
            
            if total_rank < best_rank:
                # Verify the decomposition
                U = [[0.0] * total_rank for _ in range(n * m)]
                V = [[0.0] * total_rank for _ in range(m * p)]
                W = [[0.0] * total_rank for _ in range(p * n)]
                
                col = 0
                for (i0, j0, k0) in blocks:
                    add_strassen_block(U, V, W, i0, j0, k0, col)
                    col += 7
                
                # Add schoolbook terms for uncovered entries
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if (i, j, k) not in covered:
                                U[i*m + j][col] += 1
                                V[j*p + k][col] += 1
                                W[k*n + i][col] += 1
                                col += 1
                
                if verify_decomposition(U, V, W, total_rank):
                    best_rank = total_rank
                    best_blocks = list(blocks)
    
    # If no valid blocks found, use schoolbook
    if not best_blocks:
        total_rank = n * m * p
        U = [[0.0] * total_rank for _ in range(n * m)]
        V = [[0.0] * total_rank for _ in range(m * p)]
        W = [[0.0] * total_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] += 1
                    V[j*p + k][col] += 1
                    W[k*n + i][col] += 1
                    col += 1
    else:
        total_rank = len(best_blocks) * 7 + (n * m * p - len(best_blocks) * 8)
        U = [[0.0] * total_rank for _ in range(n * m)]
        V = [[0.0] * total_rank for _ in range(m * p)]
        W = [[0.0] * total_rank for _ in range(p * n)]
        
        col = 0
        for (i0, j0, k0) in best_blocks:
            add_strassen_block(U, V, W, i0, j0, k0, col)
            col += 7
        
        covered = set()
        for (i0, j0, k0) in best_blocks:
            covered |= block_entries(i0, j0, k0)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        U[i*m + j][col] += 1
                        V[j*p + k][col] += 1
                        W[k*n + i][col] += 1
                        col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
