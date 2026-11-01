"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE = 150.0
_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _deadline_reached(margin=5.0):
    return _elapsed() > _DEADLINE - margin

def _strassen_2x2():
    """Return the rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns U (4x7), V (4x7), W (4x7) where:
    U rows indexed by (i,j) with i in {0,1}, j in {0,1}, row = i*2+j
    V rows indexed by (j,k) with j in {0,1}, k in {0,1}, row = j*2+k
    W rows indexed by (k,i) with k in {0,1}, i in {0,1}, row = k*2+i  (trace convention: W[k*n+i, r])
    """
    # Strassen: C = A*B where A is 2x2, B is 2x2
    # Partition: A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C00 = M1 - M4 + M6 + M7
    # C01 = M5 + M4
    # C10 = M1 + M2
    # C11 = M3 - M1 + M5 + M7
    # where:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # U: factor for A, rows (i,j), columns 0..6 for M1..M7
    U = np.zeros((4, 7), dtype=float)
    # M1: A00+A11 -> (0,0) and (1,1)
    U[0, 0] = 1  # a00
    U[3, 0] = 1  # a11
    # M2: A10+A11 -> (1,0) and (1,1)
    U[2, 1] = 1  # a10
    U[3, 1] = 1  # a11
    # M3: A00 -> (0,0)
    U[0, 2] = 1
    # M4: A11 -> (1,1)
    U[3, 3] = 1
    # M5: A00+A01 -> (0,0) and (0,1)
    U[0, 4] = 1
    U[1, 4] = 1
    # M6: A10-A00 -> (1,0) and (0,0)
    U[2, 5] = 1
    U[0, 5] = -1
    # M7: A01-A11 -> (0,1) and (1,1)
    U[1, 6] = 1
    U[3, 6] = -1
    
    # V: factor for B, rows (j,k), columns 0..6
    V = np.zeros((4, 7), dtype=float)
    # M1: B00+B11 -> (0,0) and (1,1)
    V[0, 0] = 1
    V[3, 0] = 1
    # M2: B00 -> (0,0)
    V[0, 1] = 1
    # M3: B01-B11 -> (0,1) and (1,1)
    V[1, 2] = 1
    V[3, 2] = -1
    # M4: B10-B00 -> (1,0) and (0,0)
    V[2, 3] = 1
    V[0, 3] = -1
    # M5: B11 -> (1,1)
    V[3, 4] = 1
    # M6: B00+B01 -> (0,0) and (0,1)
    V[0, 5] = 1
    V[1, 5] = 1
    # M7: B10+B11 -> (1,0) and (1,1)
    V[2, 6] = 1
    V[3, 6] = 1
    
    # W: factor for C, rows (k,i) with k in {0,1}, i in {0,1}, row = k*2+i
    # C00 (k=0,i=0): M1 - M4 + M6 + M7
    # C01 (k=0,i=1): M5 + M4
    # C10 (k=1,i=0): M1 + M2
    # C11 (k=1,i=1): M3 - M1 + M5 + M7
    W = np.zeros((4, 7), dtype=float)
    # W row 0 = (k=0, i=0) = C00
    W[0, 0] = 1   # M1
    W[0, 3] = -1  # -M4
    W[0, 5] = 1   # M6
    W[0, 6] = 1   # M7
    # W row 1 = (k=0, i=1) = C01
    W[1, 4] = 1   # M5
    W[1, 3] = 1   # M4
    # W row 2 = (k=1, i=0) = C10
    W[2, 0] = 1   # M1
    W[2, 1] = 1   # M2
    # W row 3 = (k=1, i=1) = C11
    W[3, 2] = 1   # M3
    W[3, 0] = -1  # -M1
    W[3, 4] = 1   # M5
    W[3, 6] = 1   # M7
    
    return U, V, W

def _trivial_2x2x1():
    """Return the rank-4 trivial decomposition for 2x2 * 2x1 -> 2x1.
    U (4x4), V (2x4), W (2x4)
    """
    n, m, p = 2, 2, 1
    rank = n * m * p  # 4
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _block_decomposition(n, m, p):
    """Build a block decomposition using Strassen for 2x2 blocks where possible.
    
    For 2x2 * 2x9:
    - We can split the 9 columns into four 2x2 blocks and one 2x1 block.
    - Actually, let's think more carefully.
    
    The matrix product is (2x2) * (2x9) -> (2x9).
    
    We can view this as: for each output column k (0..8), we compute C[:,k] = A * B[:,k].
    
    A better approach: partition the 9 columns into blocks of 2, giving 4 blocks of 2 and 1 block of 1.
    For each block of 2 columns, we have a 2x2 * 2x2 subproblem (use Strassen, rank 7).
    For the remaining 2x1 subproblem, use trivial (rank 4).
    Total rank = 4*7 + 4 = 32.
    
    Let's construct this properly.
    """
    n, m, p = 2, 2, 9
    
    # Number of full 2x2 blocks
    num_full = p // 2  # 4
    remainder = p % 2   # 1
    
    rank = num_full * 7 + (remainder * 4 if remainder > 0 else 0)
    
    U = np.zeros((n*m, rank))  # (4, rank)
    V = np.zeros((m*p, rank))  # (18, rank)
    W = np.zeros((p*n, rank))  # (18, rank)
    
    U_s, V_s, W_s = _strassen_2x2()
    U_t, V_t, W_t = _trivial_2x2x1()
    
    col_offset = 0
    
    for block in range(num_full):
        # This block handles columns 2*block and 2*block+1 of B
        # U part: same for all blocks (A is the same)
        U[:, col_offset:col_offset+7] = U_s
        
        # V part: rows for this block's columns
        # B is 2x9, so B[j, 2*block + k] for j in {0,1}, k in {0,1}
        # V rows are indexed by (j, k_global) where k_global = 2*block + k
        # V_s has rows (j, k_local) with k_local in {0,1}
        for j in range(m):
            for k_local in range(2):
                k_global = 2 * block + k_local
                V[j * p + k_global, col_offset:col_offset+7] = V_s[j, :]
        
        # W part: output columns for this block
        # W rows are indexed by (k_global, i)
        for k_local in range(2):
            for i in range(n):
                k_global = 2 * block + k_local
                W[k_global * n + i, col_offset:col_offset+7] = W_s[k_local * n + i, :]
        
        col_offset += 7
    
    if remainder > 0:
        # Remaining single column at position 2*num_full
        k_start = 2 * num_full
        U[:, col_offset:col_offset+4] = U_t
        
        for j in range(m):
            V[j * p + k_start, col_offset:col_offset+4] = V_t[j, :]
        
        for i in range(n):
            W[k_start * n + i, col_offset:col_offset+4] = W_t[i, :]
        
        col_offset += 4
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    if U.shape[0] != n*m:
        return False
    if V.shape[0] != m*p:
        return False
    if W.shape[0] != p*n:
        return False
    
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 if i==j' and k==j'... 
    # Actually, the tensor T[i,j,k] = 1 for all i,j,k (since it's matrix multiplication)
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix multiplication, T[i,j,k] = 1 for all valid i,j,k.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _to_list(mat):
    """Convert numpy array to list of lists with integer/half-integer entries."""
    result = []
    for row in mat:
        r = []
        for val in row:
            # Round to nearest half-integer
            v = round(val * 2) / 2.0
            if abs(v - round(v)) < 1e-12:
                v = int(round(v))
            r.append(v)
        result.append(r)
    return result

def construct():
    _START_TIME = time.time()
    n, m, p = DIMENSIONS
    
    # Try block decomposition with Strassen
    try:
        U, V, W = _block_decomposition(n, m, p)
        if _verify(U, V, W, n, m, p):
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    except Exception:
        pass
    
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
# EVOLVE-BLOCK-END
