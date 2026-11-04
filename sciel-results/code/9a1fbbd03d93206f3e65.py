"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s limit
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Strategy: partition n=4=2+2, m=5=1+1+1+1+1, p=6=2+2+2
    # Each sub-block is 2x1x2, which has rank 1
    # Total sub-blocks: 2 * 5 * 3 = 30, each rank 1, total rank = 30
    
    n_bounds = [0, 2, 4]
    m_bounds = [0, 1, 2, 3, 4, 5]
    p_bounds = [0, 2, 4, 6]
    
    total_rank = 2 * 5 * 3  # 30
    U_full = [[0] * total_rank for _ in range(nm)]
    V_full = [[0] * total_rank for _ in range(mp)]
    W_full = [[0] * total_rank for _ in range(pn)]
    
    col_offset = 0
    for bi in range(2):
        for bj in range(5):
            for bk in range(3):
                ni = n_bounds[bi]
                nj = n_bounds[bi+1]
                mi = m_bounds[bj]
                mj = m_bounds[bj+1]
                pi = p_bounds[bk]
                pj = p_bounds[bk+1]
                
                n_size = nj - ni  # 2
                m_size = mj - mi  # 1
                p_size = pj - pi  # 2
                
                # 2x1x2 block: rank 1
                # A is 2x1, B is 1x2, C is 2x2
                # C[i,k] = A[i,0] * B[0,k]
                # U[(ni+ii)*m + mi][col] = 1 for ii in range(2)
                # V[mi*p + (pi+kk)][col] = 1 for kk in range(2)
                # W[(pi+kk)*n + (ni+ii)][col] = 1 for kk in range(2), ii in range(2)
                for ii in range(n_size):
                    U_full[(ni+ii)*m + mi][col_offset] = 1
                for kk in range(p_size):
                    V_full[mi*p + (pi+kk)][col_offset] = 1
                    for ii in range(n_size):
                        W_full[(pi+kk)*n + (ni+ii)][col_offset] = 1
                col_offset += 1
    
    # Verify the decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(total_rank):
                    s += U_full[i * m + j][r] * V_full[j * p + k][r] * W_full[k * n + i][r]
                if abs(s - 1.0) > 1e-10:
                    valid = False
                    break
        if not valid:
            break
    
    if valid:
        return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback: schoolbook
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(nm)]
    V_sb = [[0] * schoolbook_rank for _ in range(mp)]
    W_sb = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a][col] = 1
                V_sb[b][col] = 1
                W_sb[c][col] = 1
                col += 1
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
