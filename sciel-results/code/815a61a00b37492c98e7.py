"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook_decomp(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition for multiplying n x m by m x p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i*m+j]
                v_row = V_arr[j*p+k]
                w_row = W_arr[k*n+i]
                val = np.dot(u_row, v_row * w_row)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2(U, V, W, n_off, m_off, p_off, n, m, p, col_start):
    """Apply Strassen's 7-multiplication algorithm for a 2x2x2 block.
    Returns the number of columns used."""
    # Strassen's algorithm for 2x2 matrix multiplication uses 7 multiplications
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e), M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # In our tensor decomposition, each "multiplication" M_r corresponds to one rank term
    # where U and V have entries that form the left and right factors, and W has the output coefficient
    
    # For the 2x2x2 block at offset (n_off, m_off, p_off):
    # A indices: (n_off, m_off), (n_off, m_off+1), (n_off+1, m_off), (n_off+1, m_off+1)
    # B indices: (m_off, p_off), (m_off, p_off+1), (m_off+1, p_off), (m_off+1, p_off+1)
    # C indices: (n_off, p_off), (n_off, p_off+1), (n_off+1, p_off), (n_off+1, p_off+1)
    
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1]-B[1,1])
    # M4 = A[1,1] * (B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    col = col_start
    
    # For each Strassen multiplication M_r, we need:
    # U entries for the A factors, V entries for the B factors, W entries for C coefficients
    
    # M1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # U: A[0,0] has coeff 1, A[1,1] has coeff 1
    U[n_off * m + m_off][col] = 1
    U[(n_off+1) * m + (m_off+1)][col] = 1
    # V: B[0,0] has coeff 1, B[1,1] has coeff 1
    V[m_off * p + p_off][col] = 1
    V[(m_off+1) * p + (p_off+1)][col] = 1
    # W: C[0,0] coeff 1, C[1,1] coeff 1
    W[p_off * n + n_off][col] = 1
    W[(p_off+1) * n + (n_off+1)][col] = 1
    col += 1
    
    # M2: (A[1,0]+A[1,1]) * B[0,0]
    U[(n_off+1) * m + m_off][col] = 1
    U[(n_off+1) * m + (m_off+1)][col] = 1
    V[m_off * p + p_off][col] = 1
    # W: C[1,0] coeff 1
    W[p_off * n + (n_off+1)][col] = 1
    col += 1
    
    # M3: A[0,0] * (B[0,1]-B[1,1])
    U[n_off * m + m_off][col] = 1
    V[m_off * p + (p_off+1)][col] = 1
    V[(m_off+1) * p + (p_off+1)][col] = -1
    # W: C[0,1] coeff 1, C[1,1] coeff 1
    W[(p_off+1) * n + n_off][col] = 1
    W[(p_off+1) * n + (n_off+1)][col] = 1
    col += 1
    
    # M4: A[1,1] * (B[1,0]-B[0,0])
    U[(n_off+1) * m + (m_off+1)][col] = 1
    V[(m_off+1) * p + p_off][col] = 1
    V[m_off * p + p_off][col] = -1
    # W: C[0,0] coeff 1, C[1,0] coeff 1
    W[p_off * n + n_off][col] = 1
    W[p_off * n + (n_off+1)][col] = 1
    col += 1
    
    # M5: (A[0,0]+A[0,1]) * B[1,1]
    U[n_off * m + m_off][col] = 1
    U[n_off * m + (m_off+1)][col] = 1
    V[(m_off+1) * p + (p_off+1)][col] = 1
    # W: C[0,0] coeff -1, C[0,1] coeff 1
    W[p_off * n + n_off][col] = -1
    W[(p_off+1) * n + n_off][col] = 1
    col += 1
    
    # M6: (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    U[(n_off+1) * m + m_off][col] = 1
    U[n_off * m + m_off][col] = -1
    V[m_off * p + p_off][col] = 1
    V[m_off * p + (p_off+1)][col] = 1
    # W: C[1,1] coeff 1
    W[(p_off+1) * n + (n_off+1)][col] = 1
    col += 1
    
    # M7: (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    U[n_off * m + (m_off+1)][col] = 1
    U[(n_off+1) * m + (m_off+1)][col] = -1
    V[(m_off+1) * p + p_off][col] = 1
    V[(m_off+1) * p + (p_off+1)][col] = 1
    # W: C[0,0] coeff 1
    W[p_off * n + n_off][col] = 1
    col += 1
    
    return col - col_start  # 7 columns used

def _construct_hybrid_decomp(n, m, p):
    """Construct a hybrid decomposition using Strassen for 2x2x2 blocks and schoolbook for the rest."""
    # For 3x5x6:
    # Split n=3 into 2+1, m=5 into 2+2+1, p=6 into 2+2+2
    # 
    # Blocks:
    # (2,2,2) at (0,0,0): Strassen, rank 7
    # (2,2,2) at (0,2,0): Strassen, rank 7
    # (2,2,2) at (0,2,2): Strassen, rank 7
    # (2,1,2) at (0,4,0): schoolbook, rank 2*1*2=4
    # (2,1,2) at (0,4,2): schoolbook, rank 2*1*2=4
    # (1,2,2) at (2,0,0): schoolbook, rank 1*2*2=4
    # (1,2,2) at (2,2,0): schoolbook, rank 1*2*2=4
    # (1,2,2) at (2,2,2): schoolbook, rank 1*2*2=4
    # (1,1,2) at (2,4,0): schoolbook, rank 1*1*2=2
    # (1,1,2) at (2,4,2): schoolbook, rank 1*1*2=2
    #
    # Total rank = 3*7 + 2*4 + 3*4 + 2*2 = 21 + 8 + 12 + 4 = 45
    
    n_offsets = [0, 2]  # 2-block, then 1-block
    m_offsets = [0, 2, 4]  # 2-block, 2-block, 1-block
    p_offsets = [0, 2, 4]  # 2-block, 2-block, 2-block
    
    n_sizes = [2, 1]
    m_sizes = [2, 2, 1]
    p_sizes = [2, 2, 2]
    
    total_rank = 0
    for ni in range(2):
        for mi in range(3):
            for pi in range(3):
                if n_sizes[ni] == 2 and m_sizes[mi] == 2 and p_sizes[pi] == 2:
                    total_rank += 7
                else:
                    total_rank += n_sizes[ni] * m_sizes[mi] * p_sizes[pi]
    
    rank = total_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for ni in range(2):
        for mi in range(3):
            for pi in range(3):
                n_off = n_offsets[ni]
                m_off = m_offsets[mi]
                p_off = p_offsets[pi]
                ns = n_sizes[ni]
                ms = m_sizes[mi]
                ps = p_sizes[pi]
                
                if ns == 2 and ms == 2 and ps == 2:
                    col += _strassen_2x2x2(U, V, W, n_off, m_off, p_off, n, m, p, col)
                else:
                    # Schoolbook for this block
                    for i_local in range(ns):
                        for j_local in range(ms):
                            for k_local in range(ps):
                                U[(n_off + i_local) * m + (m_off + j_local)][col] = 1
                                V[(m_off + j_local) * p + (p_off + k_local)][col] = 1
                                W[(p_off + k_local) * n + (n_off + i_local)][col] = 1
                                col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    deadline = t0 + 115.0  # safety margin
    
    # Start with schoolbook
    best = _schoolbook_decomp(n, m, p)
    best_rank = n * m * p
    
    # Try hybrid decomposition with Strassen blocks
    hybrid = _construct_hybrid_decomp(n, m, p)
    if hybrid is not None:
        hybrid_rank = len(hybrid["U"][0])
        if hybrid_rank < best_rank:
            # Verify the hybrid decomposition
            if _verify_decomposition(hybrid["U"], hybrid["V"], hybrid["W"], n, m, p):
                best = hybrid
                best_rank = hybrid_rank
            else:
                # Verification failed, stick with schoolbook
                pass
    
    # Verify the final decomposition
    if not _verify_decomposition(best["U"], best["V"], best["W"], n, m, p):
        raise RuntimeError("Final decomposition failed verification")
    
    return best
# EVOLVE-BLOCK-END
