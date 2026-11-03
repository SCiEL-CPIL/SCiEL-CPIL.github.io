import time
import numpy as np
import sys

DEADLINE = 140.0  # Safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify if the decomposition is exact.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Convention: W[k*n + i, r] corresponds to output index (k, i).
    """
    rank = U.shape[1]
    if rank == 0:
        return False
        
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    result = np.einsum('ijr,jkr,kr->ijk', U_3d, V_3d, W_3d)
    
    return np.allclose(result, 1.0, atol=tol)

def _construct_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return U, V, W

def _strassen_2x2x2():
    """
    Returns the 7 rank-1 tensors for 2x2x2 matrix multiplication.
    U, V, W are 2x2 matrices.
    """
    return [
        (np.array([[1,0],[0,1]], dtype=np.float64), 
         np.array([[0,1],[0,1]], dtype=np.float64), 
         np.array([[1,0],[0,1]], dtype=np.float64)),
        (np.array([[0,0],[1,1]], dtype=np.float64), 
         np.array([[1,0],[0,0]], dtype=np.float64), 
         np.array([[0,1],[-1,0]], dtype=np.float64)),
        (np.array([[1,0],[0,0]], dtype=np.float64), 
         np.array([[1,0],[-1,0]], dtype=np.float64), 
         np.array([[0,0],[1,1]], dtype=np.float64)),
        (np.array([[0,0],[0,1]], dtype=np.float64), 
         np.array([[0,-1],[0,1]], dtype=np.float64), 
         np.array([[1,0],[0,1]], dtype=np.float64)),
        (np.array([[1,1],[0,0]], dtype=np.float64), 
         np.array([[0,0],[0,1]], dtype=np.float64), 
         np.array([[-1,0],[0,1]], dtype=np.float64)),
        (np.array([[-1,0],[1,0]], dtype=np.float64), 
         np.array([[1,1],[0,0]], dtype=np.float64), 
         np.array([[0,0],[0,1]], dtype=np.float64)),
        (np.array([[0,1],[0,-1]], dtype=np.float64), 
         np.array([[0,0],[1,1]], dtype=np.float64), 
         np.array([[1,0],[0,0]], dtype=np.float64))
    ]

def _schoolbook_2x2x2():
    """
    Returns the 4 rank-1 tensors for 2x2x2 matrix multiplication (schoolbook).
    """
    return [
        (np.array([[1,0],[0,0]], dtype=np.float64), 
         np.array([[1,0],[0,0]], dtype=np.float64), 
         np.array([[1,0],[0,0]], dtype=np.float64)),
        (np.array([[0,1],[0,0]], dtype=np.float64), 
         np.array([[0,0],[1,0]], dtype=np.float64), 
         np.array([[1,0],[0,0]], dtype=np.float64)),
        (np.array([[0,0],[1,0]], dtype=np.float64), 
         np.array([[0,0],[0,1]], dtype=np.float64), 
         np.array([[0,1],[0,0]], dtype=np.float64)),
        (np.array([[0,0],[0,1]], dtype=np.float64), 
         np.array([[0,1],[0,0]], dtype=np.float64), 
         np.array([[0,1],[0,0]], dtype=np.float64))
    ]

def _add_block_to_decomp(U_list, V_list, W_list, n, m, p, 
                         U_b, V_b, W_b, 
                         i_off, j_off_A, j_off_B, k_off_B):
    """
    Adds a single rank-1 term to the decomposition lists.
    U_b: 2x2 (or smaller) matrix for the A-block
    V_b: 2xK matrix for the B-block
    W_b: Kx2 matrix for the C-block
    Offsers specify where in the full n x m, m x p, p x n grids this block sits.
    """
    U_flat = np.zeros(n*m)
    V_flat = np.zeros(m*p)
    W_flat = np.zeros(p*n)
    
    # U_b is (n_b, m_b) where n_b=2, m_b=2
    n_b, m_b = U_b.shape
    for ii in range(n_b):
        for jj in range(m_b):
            val = U_b[ii, jj]
            if val != 0:
                U_flat[(i_off+ii)*m + (j_off_A+jj)] = val
        
    # V_b is (m_b, k_b)
    m_b_v, k_b = V_b.shape
    for ii in range(m_b_v):
        for jj in range(k_b):
            val = V_b[ii, jj]
            if val != 0:
                V_flat[(j_off_B+ii)*p + (k_off_B+jj)] = val
        
    # W_b is (k_b, n_b)
    k_b_w, n_b_w = W_b.shape
    for jj in range(k_b_w):
        for ii in range(n_b_w):
            val = W_b[jj, ii]
            if val != 0:
                W_flat[(k_off_B+jj)*n + (i_off+ii)] = val
        
    U_list.append(U_flat)
    V_list.append(V_flat)
    W_list.append(W_flat)

def _construct_rank50_v2():
    """
    Constructs a rank-50 decomposition.
    Split 2x4 * 4x7 into:
    - 2x4 * 4x4 (rank 28 via 4 blocks of 2x2x2 Strassen)
    - 2x4 * 4x3 (rank 22 via 2 blocks of 2x2x2 Strassen + 2 blocks of 2x2x1 Schoolbook)
    
    Total: 28 + 22 = 50.
    """
    n, m, p = 2, 4, 7
    U_list = []
    V_list = []
    W_list = []
    
    strassen_terms = _strassen_2x2x2()
    school_221_terms = [
        (np.array([[1,0],[0,0]], dtype=np.float64), np.array([[1],[0]], dtype=np.float64), np.array([[1],[0]], dtype=np.float64)),
        (np.array([[0,1],[0,0]], dtype=np.float64), np.array([[0],[1]], dtype=np.float64), np.array([[1],[0]], dtype=np.float64)),
        (np.array([[0,0],[1,0]], dtype=np.float64), np.array([[1],[0]], dtype=np.float64), np.array([[0],[1]], dtype=np.float64)),
        (np.array([[0,0],[0,1]], dtype=np.float64), np.array([[0],[1]], dtype=np.float64), np.array([[0],[1]], dtype=np.float64))
    ]
    
    # Part 1: 2x4 * 4x4 -> Output 2x4
    # A is 2x4, B is 4x4, C is 2x4
    # Split A into two 2x2 blocks: A0 (cols 0-1), A1 (cols 2-3)
    # Split B into four 2x2 blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3), B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    # Split C into two 2x2 blocks: C0 (cols 0-1), C1 (cols 2-3)
    
    # Term: C0 += A0 * B00
    for U_b, V_b, W_b in strassen_terms:
        _add_block_to_decomp(U_list, V_list, W_list, n, m, p, U_b, V_b, W_b, 
                             i_off=0, j_off_A=0, j_off_B=0, k_off_B=0)
                             
    # Term: C0 += A0 * B10
    for U_b, V_b, W_b in strassen_terms:
        _add_block_to_decomp(U_list, V_list, W_list, n, m, p, U_b, V_b, W_b, 
                             i_off=0, j_off_A=0, j_off_B=2, k_off_B=0)
                             
    # Term: C0 += A1 * B01
    for U_b, V_b, W_b in strassen_terms:
        _add_block_to_decomp(U_list, V_list, W_list, n, m, p, U_b, V_b, W_b, 
                             i_off=0, j_off_A=2, j_off_B=0, k_off_B=2)
                             
    # Term: C0 += A1 * B11
    for U_b, V_b, W_b in strassen_terms:
        _add_block_to_decomp(U_list, V_list, W_list, n, m, p, U_b, V_b, W_b, 
                             i_off=0, j_off_A=2, j_off_B=2, k_off_B=2)
                             
    # Part 2: 2x4 * 4x3 -> Output 2x3
    # A is 2x4, B is 4x3, C is 2x3
    # Split A into two 2x2 blocks: A0 (cols 0-1), A1 (cols 2-3)
    # Split B into:
    #   B0: rows 0-1, cols 0-1 (2x2)
    #   B1: rows 0-1, col 2 (2x1)
    #   B2: rows 2-3, cols 0-1 (2x2)
    #   B3: rows 2-3, col 2 (2x1)
    # Split C into:
    #   C0: cols 0-1 (2x2)
    #   C1: col 2 (2x1)
    
    # Term: C0 += A0 * B0
    for U_b, V_b, W_b in strassen_terms:
        _add_block_to_decomp(U_list, V_list, W_list, n, m, p, U_b, V_b, W_b, 
                             i_off=0, j_off_A=0, j_off_B=0, k_off_B=4)
                             
    # Term: C1 += A0 * B1 (2x2 * 2x1 -> 2x1)
    for U_b, V_b, W_b in school_221_terms:
        _add_block_to_decomp(U_list, V_list, W_list, n, m, p, U_b, V_b, W_b, 
                             i_off=0, j_off_A=0, j_off_B=0, k_off_B=6)
                             
    # Term: C0 += A1 * B2
    for U_b, V_b, W_b in strassen_terms:
        _add_block_to_decomp(U_list, V_list, W_list, n, m, p, U_b, V_b, W_b, 
                             i_off=0, j_off_A=2, j_off_B=2, k_off_B=4)
                             
    # Term: C1 += A1 * B3 (2x2 * 2x1 -> 2x1)
    for U_b, V_b, W_b in school_221_terms:
        _add_block_to_decomp(U_list, V_list, W_list, n, m, p, U_b, V_b, W_b, 
                             i_off=0, j_off_A=2, j_off_B=2, k_off_B=6)
                             
    if len(U_list) == 50:
        U = np.column_stack(U_list)
        V = np.column_stack(V_list)
        W = np.column_stack(W_list)
        if _verify(U, V, W, n, m, p):
            return U, V, W
            
    return None

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 7
    
    # Try the robust Rank 50 construction first
    try:
        res = _construct_rank50_v2()
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        print("Rank 50 construction failed:", e)
        
    # Fallback to Schoolbook
    try:
        U, V, W = _construct_schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        print("Schoolbook construction failed:", e)
        return {"U": [], "V": [], "W": []}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]) if res['U'] else 0)
