import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Returns U, V, W (each 4x7) for 2x2x2 Strassen.
    U[i,j] corresponds to A[i,j], V[i,j] to B[i,j], W[i,j] to C[i,j] (row i, col j).
    Note: W convention here is C[i,j], so W index is i*2+j.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_block_decomp(n, m, p, splits_i, splits_j, splits_k, use_strassen=True):
    """
    Builds a block decomposition.
    splits_i, splits_j, splits_k are lists of sizes summing to n, m, p.
    For each block (i0, j0, k0) with sizes (di, dj, dk):
      If di==2 and dj==2 and dk==2 and use_strassen: use Strassen (rank 7)
      Else: use schoolbook (rank di*dj*dk)
    Returns U, V, W for the full n x m x p tensor.
    """
    U = [[0] * (n*m*p) for _ in range(n*m)]
    V = [[0] * (n*m*p) for _ in range(m*p)]
    W = [[0] * (n*m*p) for _ in range(p*n)]
    
    # Precompute start indices
    starts_i = [0]
    for s in splits_i[:-1]: starts_i.append(starts_i[-1] + s)
    starts_j = [0]
    for s in splits_j[:-1]: starts_j.append(starts_j[-1] + s)
    starts_k = [0]
    for s in splits_k[:-1]: starts_k.append(starts_k[-1] + s)
    
    current_rank = 0
    
    for ii, di in enumerate(splits_i):
        i0 = starts_i[ii]
        for jj, dj in enumerate(splits_j):
            j0 = starts_j[jj]
            for kk, dk in enumerate(splits_k):
                k0 = starts_k[kk]
                
                use_str = (di == 2 and dj == 2 and dk == 2 and use_strassen)
                
                if use_str:
                    U_s, V_s, W_s = _strassen_2x2x2()
                    rank_s = 7
                    # Map 2x2x2 block to global
                    # U_s[local_i * 2 + local_j, col]
                    for li in range(2):
                        for lj in range(2):
                            for c in range(7):
                                val = U_s[li*2+lj][c]
                                if val != 0:
                                    U[(i0+li)*m + (j0+lj)][current_rank + c] += val
                    for lj in range(2):
                        for lk in range(2):
                            for c in range(7):
                                val = V_s[lj*2+lk][c]
                                if val != 0:
                                    V[(j0+lj)*p + (k0+lk)][current_rank + c] += val
                    for lk in range(2):
                        for li in range(2):
                            for c in range(7):
                                val = W_s[lk*2+li][c] # W_s is C[li,lk] -> index li*2+lk? No, W_s above is W[i,j] where i is row, j is col.
                                # In _strassen_2x2x2, W[i][j] corresponds to C[i,j].
                                # So W_s[li][lk] is C[li, lk].
                                # My loop above used W_s[lk*2+li] which is WRONG if W_s is row-major C[i,j].
                                # Let's re-check _strassen_2x2x2 W definition.
                                # W[0][0] is C[0,0]. W[0][1] is C[0,1].
                                # So W_s[li*2 + lk] is C[li, lk].
                                val = W_s[li*2+lk][c]
                                if val != 0:
                                    W[(k0+lk)*n + (i0+li)][current_rank + c] += val
                else:
                    rank_s = di * dj * dk
                    for li in range(di):
                        for lj in range(dj):
                            for lk in range(dk):
                                c = li*dj*dk + lj*dk + lk
                                U[(i0+li)*m + (j0+lj)][current_rank + c] += 1
                                V[(j0+lj)*p + (k0+lk)][current_rank + c] += 1
                                W[(k0+lk)*n + (i0+li)][current_rank + c] += 1
                
                current_rank += rank_s
    
    # Trim to current_rank
    U = [row[:current_rank] for row in U]
    V = [row[:current_rank] for row in V]
    W = [row[:current_rank] for row in W]
    
    return U, V, W, current_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140
    
    # Fallback: Schoolbook
    U, V, W, rank_school = _build_block_decomp(n, m, p, [n], [m], [p], use_strassen=False)
    best_U, best_V, best_W = U, V, W
    best_rank = rank_school
    
    # Try various block splits with Strassen
    splits_to_try = [
        [4], [5], [6], # Baseline
        [2,2], [2,3], [3,2], # n=4, m=5, p=6
        [1,3], [3,1],
        [2,1,2], [1,2,2], [2,2,1],
        [1,1,3], [1,3,1], [3,1,1],
        [2,3], [3,2],
        [1,1,1,1],
        [1,1,1,1,1],
        [2,2,2], [3,3], [1,2,3], [3,2,1], [2,1,3], [3,1,2], [1,3,2], [2,3,1],
    ]
    
    # We need to try combinations of splits for i, j, k
    # To limit search, we'll just try a few promising ones
    # n=4: [2,2] is best for Strassen
    # m=5: [2,3] or [3,2]
    # p=6: [2,2,2] or [3,3] or [2,4] etc.
    
    candidates = [
        ([2,2], [2,3], [2,2,2]),
        ([2,2], [3,2], [2,2,2]),
        ([2,2], [2,3], [3,3]),
        ([2,2], [3,2], [3,3]),
        ([2,2], [2,2,1], [2,2,2]),
        ([2,2], [1,2,2], [2,2,2]),
        ([4], [2,3], [2,2,2]),
        ([4], [2,3], [3,3]),
    ]
    
    for si, sj, sk in candidates:
        if time.time() > deadline: break
        if sum(si) != n or sum(sj) != m or sum(sk) != p: continue
        
        U_c, V_c, W_c, rank_c = _build_block_decomp(n, m, p, si, sj, sk, use_strassen=True)
        
        if rank_c < best_rank:
            # Verify
            # Check if exact
            err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(rank_c):
                            s += U_c[i*m+j][r] * V_c[j*p+k][r] * W_c[k*n+i][r]
                        if s != 1:
                            err += abs(s-1)
            if err < 1e-9:
                best_U, best_V, best_W = U_c, V_c, W_c
                best_rank = rank_c
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
