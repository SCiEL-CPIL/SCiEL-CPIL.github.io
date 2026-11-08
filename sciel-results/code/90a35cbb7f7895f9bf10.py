import numpy as np
import time

DIMENSIONS = [2, 3, 9]
_start_time = time.time()
TIME_LIMIT = 145.0

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Build target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T_target[a, b, c] = 1.0
    
    # Compute decomposition
    T_decomp = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_decomp += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    
    return np.allclose(T_decomp, T_target, atol=1e-9)

def _rank50_decomposition():
    """
    Construct a rank 50 decomposition for 2x3 * 3x9.
    Split m=3 into 2+1.
    Part 1: 2x2 * 2x9 (rank 32)
    Part 2: 2x1 * 1x9 (rank 18)
    Total rank 50.
    """
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 2x2 * 2x9
    # Use Strassen for 2x2x2 blocks and schoolbook for 2x2x1
    # 9 columns = 4 + 4 + 1
    # 2x2x2 rank 7, 2x2x1 rank 4
    
    def _add_2x2x2_block(u_base, v_base, w_base, col_offset):
        # u_base: index of first row of A (2x2)
        # v_base: index of first row of B (2x2xk)
        # w_base: index of first row of C (2x2xk)
        # Indices:
        # A: u_base (0,0), u_base+1 (0,1), u_base+2 (1,0), u_base+3 (1,1) -- wait, A is 2x2, so 4 entries.
        # In our global U (size 6), A_L is rows 0-3?
        # A is 2x3. A_L is rows 0-3? No.
        # A indices: i*m + j.
        # i=0, j=0 -> 0
        # i=0, j=1 -> 1
        # i=1, j=0 -> 3
        # i=1, j=1 -> 4
        # So A_L uses U indices 0, 1, 3, 4.
        
        a00 = u_base
        a01 = u_base + 1
        a10 = u_base + 2 # This is wrong for global indexing.
        
        # Let's pass explicit indices to be safe.
        pass

    # Explicit indices for A_L (2x2 part of 2x3 A)
    # A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    u_a00 = 0
    u_a01 = 1
    u_a10 = 3
    u_a11 = 4
    
    # B_L (2x9 part of 3x9 B)
    # B[0,k], B[1,k]
    # B indices: j*p + k.
    # j=0, k=0..8 -> 0..8
    # j=1, k=0..8 -> 9..17
    
    # C (2x9)
    # C indices: k*n + i.
    # i=0,1. k=0..8.
    # C[0,k] -> k*2 + 0 = 2k
    # C[1,k] -> k*2 + 1 = 2k+1

    # Process 9 columns of B_L
    # Split into 4+4+1
    
    def _strassen_2x2x2(u00, u01, u10, u11, v0, v1, v2, v3, w0, w1, w2, w3, col):
        # v0, v1 are B[0, col0], B[0, col1]
        # v2, v3 are B[1, col0], B[1, col1]
        # w0, w1 are C[0, col0], C[0, col1]
        # w2, w3 are C[1, col0], C[1, col1]
        
        # M1 = (a00+a11)(v0+v3)
        U[u00][col] += 1
        U[u11][col] += 1
        V[v0][col] += 1
        V[v3][col] += 1
        W[w0][col] += 1
        W[w1][col] += 1
        W[w2][col] += 1
        W[w3][col] += 1
        
        # M2 = (a10+a11)v0
        U[u10][col+1] += 1
        U[u11][col+1] += 1
        V[v0][col+1] += 1
        W[w2][col+1] += 1
        W[w3][col+1] += 1
        
        # M3 = a00(v1-v3)
        U[u00][col+2] += 1
        V[v1][col+2] += 1
        V[v3][col+2] -= 1
        W[w0][col+2] += 1
        W[w3][col+2] += 1
        
        # M4 = a11(v2-v0)
        U[u11][col+3] += 1
        V[v2][col+3] += 1
        V[v0][col+3] -= 1
        W[w2][col+3] += 1
        W[w3][col+3] += 1
        
        # M5 = (a00+a01)v3
        U[u00][col+4] += 1
        U[u01][col+4] += 1
        V[v3][col+4] += 1
        W[w0][col+4] += 1
        W[w1][col+4] += 1
        
        # M6 = (a10-a00)(v0+v2)
        U[u10][col+5] += 1
        U[u00][col+5] -= 1
        V[v0][col+5] += 1
        V[v2][col+5] += 1
        W[w2][col+5] += 1
        W[w3][col+5] += 1
        
        # M7 = (a01-a11)(v1+v3)
        U[u01][col+6] += 1
        U[u11][col+6] -= 1
        V[v1][col+6] += 1
        V[v3][col+6] += 1
        W[w1][col+6] += 1
        W[w3][col+6] += 1

    def _schoolbook_2x2x1(u00, u01, u10, u11, v0, v1, w0, w1, w2, w3, col):
        # v0 = B[0, k], v1 = B[1, k]
        # w0 = C[0, k], w1 = C[1, k] -- wait, w indices are C[0,k] and C[1,k]
        # Actually for 2x2x1:
        # C00 = a00 v0 + a01 v1
        # C10 = a10 v0 + a11 v1
        
        U[u00][col] += 1
        V[v0][col] += 1
        W[w0][col] += 1
        
        U[u01][col+1] += 1
        V[v1][col+1] += 1
        W[w0][col+1] += 1
        
        U[u10][col+2] += 1
        V[v0][col+2] += 1
        W[w2][col+2] += 1
        
        U[u11][col+3] += 1
        V[v1][col+3] += 1
        W[w2][col+3] += 1

    # Process columns 0-3 (2x2x2)
    k0, k1 = 0, 1
    v0 = 0*p + k0
    v1 = 0*p + k1
    v2 = 1*p + k0
    v3 = 1*p + k1
    w0 = k0*n + 0
    w1 = k1*n + 0
    w2 = k0*n + 1
    w3 = k1*n + 1
    _strassen_2x2x2(u_a00, u_a01, u_a10, u_a11, v0, v1, v2, v3, w0, w1, w2, w3, col)
    col += 7
    
    # Process columns 2-3 (2x2x2)
    k0, k1 = 2, 3
    v0 = 0*p + k0
    v1 = 0*p + k1
    v2 = 1*p + k0
    v3 = 1*p + k1
    w0 = k0*n + 0
    w1 = k1*n + 0
    w2 = k0*n + 1
    w3 = k1*n + 1
    _strassen_2x2x2(u_a00, u_a01, u_a10, u_a11, v0, v1, v2, v3, w0, w1, w2, w3, col)
    col += 7
    
    # Process columns 4-7 (2x2x4 -> two 2x2x2)
    k0, k1 = 4, 5
    v0 = 0*p + k0
    v1 = 0*p + k1
    v2 = 1*p + k0
    v3 = 1*p + k1
    w0 = k0*n + 0
    w1 = k1*n + 0
    w2 = k0*n + 1
    w3 = k1*n + 1
    _strassen_2x2x2(u_a00, u_a01, u_a10, u_a11, v0, v1, v2, v3, w0, w1, w2, w3, col)
    col += 7
    
    k0, k1 = 6, 7
    v0 = 0*p + k0
    v1 = 0*p + k1
    v2 = 1*p + k0
    v3 = 1*p + k1
    w0 = k0*n + 0
    w1 = k1*n + 0
    w2 = k0*n + 1
    w3 = k1*n + 1
    _strassen_2x2x2(u_a00, u_a01, u_a10, u_a11, v0, v1, v2, v3, w0, w1, w2, w3, col)
    col += 7
    
    # Process column 8 (2x2x1)
    k0 = 8
    v0 = 0*p + k0
    v1 = 1*p + k0
    w0 = k0*n + 0
    w1 = k0*n + 1
    _schoolbook_2x2x1(u_a00, u_a01, u_a10, u_a11, v0, v1, w0, w1, w0, w1, col)
    col += 4
    
    # Part 2: 2x1 * 1x9
    # A_R is column 2 of A. Indices: A[0,2]=2, A[1,2]=5.
    # B_R is row 2 of B. Indices: B[2,k] = 2*p+k = 18+k.
    # C is 2x9.
    
    u_ar0 = 2
    u_ar1 = 5
    
    for k in range(9):
        v_br = 2*p + k
        w_c0 = k*n + 0
        w_c1 = k*n + 1
        
        U[u_ar0][col] += 1
        V[v_br][col] += 1
        W[w_c0][col] += 1
        
        U[u_ar1][col+1] += 1
        V[v_br][col+1] += 1
        W[w_c1][col+1] += 1
        col += 2
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    
    # Start with rank 50 Strassen decomposition
    result = _rank50_decomposition()
    
    # Verify
    if not _verify(result["U"], result["V"], result["W"], n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][col] = 1
                    V[b][col] = 1
                    W[c][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return result
