import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    # Rank 28 decomposition
    # Split into j=0 (rank 10) and j=1,2 (rank 18)
    # j=1,2 split into k=0,1 (Strassen rank 7) and k=2,3,4 (Rank 11: k=2 schoolbook 4 + k=3,4 Strassen 7)
    # Wait, the previous comment said:
    # j=1,2 split into k=0,1 (Strassen rank 7) and k=2,3,4 (Rank 11: k=2 schoolbook 4 + k=3,4 Strassen 7)
    # This implies k=2 is schoolbook?
    # Let's re-verify the split.
    # m=2 part (j=1,2) has p=5.
    # Split p=5 into 2, 2, 1.
    # k=0,1: Strassen (7)
    # k=2,3: Strassen (7)
    # k=4: Schoolbook (4)
    # Total 18.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # M1
        U[(0 + i_off) * m + (0 + j_off)][col] = 1
        U[(1 + i_off) * m + (1 + j_off)][col] = 1
        V[(0 + j_off) * p + (0 + k_off)][col] = 1
        V[(1 + j_off) * p + (1 + k_off)][col] = 1
        W[(0 + k_off) * n + (0 + i_off)][col] = 1
        W[(1 + k_off) * n + (1 + i_off)][col] = 1
        col += 1
        
        # M2
        U[(1 + i_off) * m + (0 + j_off)][col] = 1
        U[(1 + i_off) * m + (1 + j_off)][col] = 1
        V[(0 + j_off) * p + (0 + k_off)][col] = 1
        W[(0 + k_off) * n + (1 + i_off)][col] = 1
        W[(1 + k_off) * n + (1 + i_off)][col] = -1
        col += 1
        
        # M3
        U[(0 + i_off) * m + (0 + j_off)][col] = 1
        V[(0 + j_off) * p + (1 + k_off)][col] = 1
        V[(1 + j_off) * p + (1 + k_off)][col] = -1
        W[(1 + k_off) * n + (0 + i_off)][col] = 1
        W[(1 + k_off) * n + (1 + i_off)][col] = 1
        col += 1
        
        # M4
        U[(1 + i_off) * m + (1 + j_off)][col] = 1
        V[(1 + j_off) * p + (0 + k_off)][col] = 1
        V[(0 + j_off) * p + (0 + k_off)][col] = -1
        W[(0 + k_off) * n + (0 + i_off)][col] = 1
        W[(0 + k_off) * n + (1 + i_off)][col] = 1
        col += 1
        
        # M5
        U[(0 + i_off) * m + (0 + j_off)][col] = 1
        U[(0 + i_off) * m + (1 + j_off)][col] = 1
        V[(1 + j_off) * p + (1 + k_off)][col] = 1
        W[(0 + k_off) * n + (0 + i_off)][col] = -1
        W[(1 + k_off) * n + (0 + i_off)][col] = 1
        col += 1
        
        # M6
        U[(1 + i_off) * m + (0 + j_off)][col] = 1
        U[(0 + i_off) * m + (0 + j_off)][col] = -1
        V[(0 + j_off) * p + (0 + k_off)][col] = 1
        V[(0 + j_off) * p + (1 + k_off)][col] = 1
        W[(1 + k_off) * n + (1 + i_off)][col] = 1
        col += 1
        
        # M7
        U[(0 + i_off) * m + (1 + j_off)][col] = 1
        U[(1 + i_off) * m + (1 + j_off)][col] = -1
        V[(1 + j_off) * p + (0 + k_off)][col] = 1
        V[(1 + j_off) * p + (1 + k_off)][col] = 1
        W[(0 + k_off) * n + (0 + i_off)][col] = 1
        col += 1

    # Block 1: j=0 (Rank 10)
    # This is 2x1 by 1x5
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Block 2: j=1,2 (Rank 18)
    # Split k into 0,1 (Strassen), 2,3 (Strassen), 4 (Schoolbook)
    
    # k=0,1: Strassen
    add_strassen(0, 1, 0)
    
    # k=2,3: Strassen
    add_strassen(0, 1, 2)
    
    # k=4: Schoolbook (Rank 4)
    # 2x2 by 2x1
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
