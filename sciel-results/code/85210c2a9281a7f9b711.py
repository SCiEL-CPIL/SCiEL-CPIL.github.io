import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _build_strassen_2x2x2():
    """
    Return (U, V, W) for 2x2x2 matrix multiplication (Strassen, rank 7).
    U is 4x7 (i*m+j, r), V is 4x7 (j*p+k, r), W is 4x7 (k*n+i, r).
    """
    # Strassen's 7 multiplications:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M7
    # C11 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    # A is 2x2, B is 2x2, C is 2x2
    
    # M1 = (A00+A11)(B00+B11)
    # U: A00=1, A11=1 -> rows 0 (i=0,j=0), 3 (i=1,j=1)
    U[0][0] = 1; U[3][0] = 1
    # V: B00=1, B11=1 -> rows 0 (j=0,k=0), 3 (j=1,k=1)
    V[0][0] = 1; V[3][0] = 1
    # W: C00=1, C11=1 -> rows 0 (k=0,i=0), 3 (k=1,i=1)
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C10: k=1,i=0 -> row 2
    
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C01: k=0,i=1 -> row 1
    
    # M4 = A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1  # C00: k=0,i=0 -> row 0
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1  # C01: k=0,i=1 -> row 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C11: k=1,i=1 -> row 3
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[2][6] = 1  # C10: k=1,i=0 -> row 2
    
    return U, V, W

def _build_decomposition():
    """
    Build the rank-23 decomposition for 2x3 * 3x6 matrix multiplication.
    
    Strategy:
    - Split the 3x6 matrix B into three 3x2 blocks: B[:,0:2], B[:,2:4], B[:,4:6]
    - For each block, use the Strassen 2x2x2 decomposition on the 2x2 core (j=0,1)
    - The j=2 row of B contributes to C via A[:,2] * B[2,:] which is a 2x6 outer product
      (rank 2 contribution)
    
    Total rank: 3*7 + 2 = 23
    """
    n, m, p = 2, 3, 6
    rank = 23
    
    U = [[0]*rank for _ in range(n*m)]  # 6 x 23
    V = [[0]*rank for _ in range(m*p)]  # 18 x 23
    W = [[0]*rank for _ in range(p*n)]  # 12 x 23
    
    # Get Strassen 2x2x2 decomposition
    Us, Vs, Ws = _build_strassen_2x2x2()
    
    # For each of the 3 blocks of B (columns 0-1, 2-3, 4-5):
    for block in range(3):
        base_r = block * 7
        k_offset = block * 2
        
        for r in range(7):
            col = base_r + r
            
            # U: same for all blocks (A part is always 2x2 with j=0,1)
            for i in range(2):
                for j in range(2):
                    U[i*m + j][col] = Us[i*2 + j][r]
            
            # V: for block b, B rows j=0,1 map to columns k_offset, k_offset+1
            for j in range(2):
                for k in range(2):
                    V[j*p + (k_offset + k)][col] = Vs[j*2 + k][r]
            
            # W: C rows i=0,1, columns k_offset, k_offset+1
            for k in range(2):
                for i in range(2):
                    W[(k_offset + k)*n + i][col] = Ws[k*2 + i][r]
    
    # Tail: j=2 contribution
    # C[i, k] += A[i, 2] * B[2, k]
    # This is a rank-2 term: for each i in {0,1}, U[i*3+2, r] = 1, V[2*6+k, r] = 1, W[k*2+i, r] = 1
    # But we can combine: use 2 terms, one for i=0 and one for i=1
    # Actually, this is an outer product of A[:,2] (2x1) and B[2,:] (1x6)
    # Rank of outer product of 2x1 and 1x6 is 2 (since min(2,6)=2... actually it's 1 if both are vectors)
    # Wait: A[:,2] is a column vector (2x1), B[2,:] is a row vector (1x6)
    # Their outer product is 2x6, rank 1.
    # But in tensor decomposition terms:
    # C[i,k] += A[i,2] * B[2,k]
    # U[i*3+2, r] * V[2*6+k, r] * W[k*2+i, r]
    # We need: sum_r U[i,2,r] * V[2,k,r] * W[k,i,r] = A[i,2] * B[2,k]
    # This can be done with rank 1:
    # U[0*3+2, r] = 1, U[1*3+2, r] = 1
    # V[2*6+k, r] = 1 for all k
    # W[k*2+i, r] = 1 for all k, i
    # Then sum = 1 * 1 * 1 = 1 for each (i,k), but we need A[i,2]*B[2,k]
    # 
    # Actually the tensor decomposition is:
    # T[i,j,k] = delta_{j,2} (the tensor is 1 when j=2, 0 otherwise for this part)
    # Wait no. The full tensor is T[i,j,k] = 1 if the indices match the multiplication pattern.
    # For matrix multiplication C = AB, T[i,j,k] = 1 (it's the identity tensor).
    # The decomposition is: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all (i,j,k)
    # 
    # For the j=2 part: we need sum_r U[i,2,r] V[2,k,r] W[k,i,r] to contribute to the j=2 slice.
    # But actually all (i,j,k) must sum to 1. The j=0,1 parts are handled by Strassen blocks.
    # For j=2: we need sum_r U[i,2,r] V[2,k,r] W[k,i,r] = 1 for all i,k.
    # 
    # With rank 1: U[i,2,r]=1, V[2,k,r]=1, W[k,i,r]=1 gives 1. 
    # But wait, that would make it 1 for all i,k, which is correct.
    # 
    # However, we used 2 terms in the plan. Let me use rank 1 for this.
    # Actually, let me re-examine. The total rank would be 21 + 1 = 22.
    
    # Let's use 1 term for the tail
    tail_r = 21
    
    for i in range(2):
        U[i*m + 2][tail_r] = 1
    for k in range(6):
        V[2*p + k][tail_r] = 1
    for k in range(6):
        for i in range(2):
            W[k*n + i][tail_r] = 1
    
    # We have 22 terms used, 1 unused. Let's keep rank=22.
    # Actually, let me verify: do we need rank 22 or can we do better?
    # Let's just use 22 and verify.
    
    return {"U": U, "V": V, "W": W}, 22

def _verify_decomposition(n, m, p, U, V, W, num_tests=50):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_reshaped = U_arr.reshape(n, m, -1)
    V_reshaped = V_arr.reshape(m, p, -1)
    W_reshaped = W_arr.reshape(p, n, -1)
    
    for _ in range(num_tests):
        A = np.random.randint(-5, 6, size=(n, m), dtype=np.float64)
        B = np.random.randint(-5, 6, size=(m, p), dtype=np.float64)
        
        C_direct = A @ B
        
        LA = np.einsum('ij,ijr->ir', A, U_reshaped)
        LB = np.einsum('jk,jkr->kr', B, V_reshaped)
        C_decomp = np.einsum('kir,ir,kr->ik', W_reshaped, LA, LB)
        
        if not np.allclose(C_direct, C_decomp, atol=1e-9):
            return False
    
    return True

def _build_schoolbook():
    """Build the schoolbook rank-36 decomposition as fallback."""
    n, m, p = 2, 3, 6
    rank = n * m * p
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Try the structured rank-22 decomposition
    solution, rank = _build_decomposition()
    
    # Verify it
    if _verify_decomposition(n, m, p, solution["U"], solution["V"], solution["W"], num_tests=100):
        # Trim to actual rank used
        # Find the last non-zero column
        used_cols = []
        for r in range(rank):
            col_u = [solution["U"][i][r] for i in range(n*m)]
            col_v = [solution["V"][i][r] for i in range(m*p)]
            col_w = [solution["W"][i][r] for i in range(p*n)]
            if any(x != 0 for x in col_u) or any(x != 0 for x in col_v) or any(x != 0 for x in col_w):
                used_cols.append(r)
        
        actual_rank = len(used_cols)
        U_trimmed = [[solution["U"][i][r] for r in used_cols] for i in range(n*m)]
        V_trimmed = [[solution["V"][i][r] for r in used_cols] for i in range(m*p)]
        W_trimmed = [[solution["W"][i][r] for r in used_cols] for i in range(p*n)]
        
        return {"U": U_trimmed, "V": V_trimmed, "W": W_trimmed}
    
    # Fallback to schoolbook
    return _build_schoolbook()
# EVOLVE-BLOCK-END
