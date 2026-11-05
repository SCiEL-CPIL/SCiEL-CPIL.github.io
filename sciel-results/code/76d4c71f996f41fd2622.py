"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Block structure:
    # n=4: [2, 2]
    # m=5: [2, 2, 1]
    # p=6: [2, 2, 2]
    
    i_offsets = [0, 2]
    i_sizes = [2, 2]
    
    j_offsets = [0, 2, 4]
    j_sizes = [2, 2, 1]
    
    k_offsets = [0, 2, 4]
    k_sizes = [2, 2, 2]
    
    # Pre-compute total rank
    total_rank = 0
    for i_idx in range(len(i_offsets)):
        n_b = i_sizes[i_idx]
        for j_idx in range(len(j_offsets)):
            m_b = j_sizes[j_idx]
            for k_idx in range(len(k_offsets)):
                p_b = k_sizes[k_idx]
                if n_b == 2 and m_b == 2 and p_b == 2:
                    total_rank += 7
                elif n_b == 2 and m_b == 1 and p_b == 2:
                    total_rank += 4
                else:
                    total_rank += n_b * m_b * p_b
    
    # Allocate matrices
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    col = 0
    
    for i_idx, i_0 in enumerate(i_offsets):
        n_b = i_sizes[i_idx]
        for j_idx, j_0 in enumerate(j_offsets):
            m_b = j_sizes[j_idx]
            for k_idx, k_0 in enumerate(k_offsets):
                p_b = k_sizes[k_idx]
                
                if n_b == 2 and m_b == 2 and p_b == 2:
                    # Strassen 2x2x2: 7 terms
                    # Term 1: M1 = (A00+A11)(B00+B11)
                    # U: A00, A11; V: B00, B11; W: C00, C11
                    U[(i_0+0)*m + (j_0+0), col] = 1
                    U[(i_0+1)*m + (j_0+1), col] = 1
                    V[(j_0+0)*p + (k_0+0), col] = 1
                    V[(j_0+1)*p + (k_0+1), col] = 1
                    W[(k_0+0)*n + (i_0+0), col] = 1
                    W[(k_0+1)*n + (i_0+1), col] = 1
                    col += 1
                    
                    # Term 2: M2 = (A10+A11)B00
                    # U: A10, A11; V: B00; W: C10, C11(-)
                    U[(i_0+1)*m + (j_0+0), col] = 1
                    U[(i_0+1)*m + (j_0+1), col] = 1
                    V[(j_0+0)*p + (k_0+0), col] = 1
                    W[(k_0+0)*n + (i_0+1), col] = 1
                    W[(k_0+1)*n + (i_0+1), col] = -1
                    col += 1
                    
                    # Term 3: M3 = A00(B01-B11)
                    # U: A00; V: B01, B11(-); W: C01, C11
                    U[(i_0+0)*m + (j_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+1), col] = 1
                    V[(j_0+1)*p + (k_0+1), col] = -1
                    W[(k_0+1)*n + (i_0+0), col] = 1
                    W[(k_0+1)*n + (i_0+1), col] = 1
                    col += 1
                    
                    # Term 4: M4 = A11(B10-B00)
                    # U: A11; V: B10, B00(-); W: C00, C10
                    U[(i_0+1)*m + (j_0+1), col] = 1
                    V[(j_0+1)*p + (k_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+0), col] = -1
                    W[(k_0+0)*n + (i_0+0), col] = 1
                    W[(k_0+0)*n + (i_0+1), col] = 1
                    col += 1
                    
                    # Term 5: M5 = (A00+A01)B11
                    # U: A00, A01; V: B11; W: C01, C00(-)
                    U[(i_0+0)*m + (j_0+0), col] = 1
                    U[(i_0+0)*m + (j_0+1), col] = 1
                    V[(j_0+1)*p + (k_0+1), col] = 1
                    W[(k_0+1)*n + (i_0+0), col] = 1
                    W[(k_0+0)*n + (i_0+0), col] = -1
                    col += 1
                    
                    # Term 6: M6 = (A10-A00)(B00+B01)
                    # U: A10, A00(-); V: B00, B01; W: C11
                    U[(i_0+1)*m + (j_0+0), col] = 1
                    U[(i_0+0)*m + (j_0+0), col] = -1
                    V[(j_0+0)*p + (k_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+1), col] = 1
                    W[(k_0+1)*n + (i_0+1), col] = 1
                    col += 1
                    
                    # Term 7: M7 = (A01-A11)(B10+B11)
                    # U: A01, A11(-); V: B10, B11; W: C00
                    U[(i_0+0)*m + (j_0+1), col] = 1
                    U[(i_0+1)*m + (j_0+1), col] = -1
                    V[(j_0+1)*p + (k_0+0), col] = 1
                    V[(j_0+1)*p + (k_0+1), col] = 1
                    W[(k_0+0)*n + (i_0+0), col] = 1
                    col += 1
                    
                elif n_b == 2 and m_b == 1 and p_b == 2:
                    # Schoolbook 2x1x2: 4 terms
                    for u in range(2):
                        for w in range(2):
                            U[(i_0+u)*m + (j_0+0), col] = 1
                            V[(j_0+0)*p + (k_0+w), col] = 1
                            W[(k_0+w)*n + (i_0+u), col] = 1
                            col += 1
                else:
                    # Generic schoolbook fallback
                    for u in range(n_b):
                        for v in range(m_b):
                            for w in range(p_b):
                                U[(i_0+u)*m + (j_0+v), col] = 1
                                V[(j_0+v)*p + (k_0+w), col] = 1
                                W[(k_0+w)*n + (i_0+u), col] = 1
                                col += 1
    
    # Verify the decomposition
    if col != total_rank:
        # Should not happen, but trim if needed
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
    
    # Quick verification of a few entries to ensure correctness
    # Check all tensor entries equal 1
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    rank = col
    
    # Verify a sample of entries (or all if fast enough)
    verified = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U_list[i*m+j][c] * V_list[j*p+k][c] * W_list[k*n+i][c]
                if s != 1:
                    verified = False
                    break
            if not verified:
                break
        if not verified:
            break
    
    if not verified:
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j][c] = 1
                    V_school[j*p+k][c] = 1
                    W_school[k*n+i][c] = 1
                    c += 1
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
