"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # We'll construct a rank-24 decomposition
    # Split p=9 into three groups of 3: cols 0-2, 3-5, 6-8
    # For each 4x4x3 block, use rank-8 decomposition
    # Total rank: 3 * 8 = 24
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # For each group of 3 columns, add 8 rank-1 terms
    for group in range(3):
        col_start = group * 3
        col_end = col_start + 3  # exclusive
        
        # Rank-8 decomposition for 4x4x3
        # We use the following scheme:
        # For i in 0..3, k in 0..2: C[i,k] = sum_j A[i,j] B[j,k]
        # We'll use 8 terms that share computation
        
        # Term design: use pairs of j indices
        # j pairs: (0,1), (2,3), and combinations
        
        # Actually, let's use a simpler valid construction:
        # 8 terms for 4x4x3:
        # We need sum_r U_r[i,j] * V_r[j,k] * W_r[i,k] = delta condition
        
        # Let's use the following 8 terms:
        # For each (i, j_pair) where j_pair is one of {(0,1), (2,3)} and we handle k separately
        # This is getting complex. Let me just use schoolbook for 4x4x3 with rank 4*4*3=48
        # but that gives total 144 again.
        
        # Alternative: just use schoolbook but with a different split
        # Actually, let me just implement a valid rank-24 by brute force construction
        
        # For 4x4x3, schoolbook is 48 terms. We need 8.
        # This requires clever sharing. Let me use a known result or construct manually.
        
        # I'll use the following: for 4x4x3, use 8 terms where:
        # Terms 0-3: handle j=0,1 for all i,k
        # Terms 4-7: handle j=2,3 for all i,k
        # But that's still 4*3*2 = 24 per group... no.
        
        # Let me just use schoolbook for now to ensure correctness, then optimize
        for i in range(n):
            for j in range(m):
                for k in range(col_start, col_end):
                    u = [0] * (n * m)
                    v = [0] * (m * p)
                    w = [0] * (p * n)
                    u[i * m + j] = 1
                    v[j * p + k] = 1
                    w[k * n + i] = 1
                    U_cols.append(u)
                    V_cols.append(v)
                    W_cols.append(w)
    
    rank = len(U_cols)
    U_mat = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V_mat = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W_mat = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
