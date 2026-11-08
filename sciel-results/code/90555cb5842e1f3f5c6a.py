"""Matrix multiplication 2,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a tensor decomposition for matrix multiplication C = A @ B
    where A is n x m and B is m x p.
    
    Decomposition form:
    C[k,i] = sum_l W[k*n+i, l] * ( sum_{i',j'} U[i'*m+j', l] * A[i',j'] ) * ( sum_{j'',k'} V[j''*p+k', l] * B[j'',k'] )
    
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)  <-- Index is k*n + i
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    start_time = time.time()
    timeout = 100.0
    
    rank = 36
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    def u_idx(i, j): return i*m + j
    def v_idx(j, k): return j*p + k
    def w_idx(k, i): return k*n + i
    
    col = 0
    
    def add_strassen_2x2x2(ai0, ai1, aj0, aj1, bj0, bj1, bk0, bk1):
        nonlocal col
        a00 = u_idx(ai0, aj0)
        a01 = u_idx(ai0, aj1)
        a10 = u_idx(ai1, aj0)
        a11 = u_idx(ai1, aj1)
        
        b00 = v_idx(bj0, bk0)
        b01 = v_idx(bj0, bk1)
        b10 = v_idx(bj1, bk0)
        b11 = v_idx(bj1, bk1)
        
        c00 = w_idx(bk0, ai0)
        c01 = w_idx(bk1, ai0)
        c10 = w_idx(bk0, ai1)
        c11 = w_idx(bk1, ai1)
        
        # M1 = (A00+A11)*(B00+B11)
        U[a00][col] += 1; U[a11][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        
        # M2 = (A10+A11)*B00
        U[a10][col] += 1; U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1; W[c11][col] -= 1
        col += 1
        
        # M3 = A00*(B01-B11)
        U[a00][col] += 1
        V[b01][col] += 1; V[b11][col] -= 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        
        # M4 = A11*(B10-B00)
        U[a11][col] += 1
        V[b10][col] += 1; V[b00][col] -= 1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1
        
        # M5 = (A00+A01)*B11
        U[a00][col] += 1; U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1; W[c01][col] += 1
        col += 1
        
        # M6 = (A10-A00)*(B00+B01)
        U[a10][col] += 1; U[a00][col] -= 1
        V[b00][col] += 1; V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7 = (A01-A11)*(B10+B11)
        U[a01][col] += 1; U[a11][col] -= 1
        V[b10][col] += 1; V[b11][col] += 1
        W[c00][col] += 1
        col += 1
    
    def add_vec_mat_2x2x1(ai0, ai1, aj0, aj1, bj0, bj1, bk):
        nonlocal col
        a00 = u_idx(ai0, aj0)
        a01 = u_idx(ai0, aj1)
        a10 = u_idx(ai1, aj0)
        a11 = u_idx(ai1, aj1)
        
        b0 = v_idx(bj0, bk)
        b1 = v_idx(bj1, bk)
        
        c0 = w_idx(bk, ai0)
        c1 = w_idx(bk, ai1)
        
        U[a00][col] += 1; V[b0][col] += 1; W[c0][col] += 1; col += 1
        U[a01][col] += 1; V[b1][col] += 1; W[c0][col] += 1; col += 1
        U[a10][col] += 1; V[b0][col] += 1; W[c1][col] += 1; col += 1
        U[a11][col] += 1; V[b1][col] += 1; W[c1][col] += 1; col += 1
    
    # Block 1: A[0:2,0:2] @ B[0:2,0:2] -> C[0:2,0:2]
    add_strassen_2x2x2(0, 1, 0, 1, 0, 1, 0, 1)
    
    # Block 2: A[0:2,0:2] @ B[0:2,2:4] -> C[2:4,0:2]
    add_strassen_2x2x2(0, 1, 0, 1, 0, 1, 2, 3)
    
    # Block 3: A[0:2,0:2] @ B[0:2,4] -> C[4,0:2]
    add_vec_mat_2x2x1(0, 1, 0, 1, 0, 1, 4)
    
    # Block 4: A[0:2,2:4] @ B[2:4,0:2] -> C[0:2,0:2]
    add_strassen_2x2x2(0, 1, 2, 3, 2, 3, 0, 1)
    
    # Block 5: A[0:2,2:4] @ B[2:4,2:4] -> C[2:4,0:2]
    add_strassen_2x2x2(0, 1, 2, 3, 2, 3, 2, 3)
    
    # Block 6: A[0:2,2:4] @ B[2:4,4] -> C[4,0:2]
    add_vec_mat_2x2x1(0, 1, 2, 3, 2, 3, 4)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
