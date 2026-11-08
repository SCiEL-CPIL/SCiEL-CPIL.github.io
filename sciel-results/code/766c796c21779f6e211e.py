"""Matrix multiplication 2,4,6 — rank-minimal exact tensor decomposition."""

import numpy as np
import time

DIMENSIONS = [2, 4, 6]


# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _strassen_2x2x2():
    """Return U, V, W for Strassen's 2x2x2 matrix multiplication, rank 7.
    
    A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    Strassen's 7 products:
    p1 = (a+d)(e+h)
    p2 = (c+d)e
    p3 = a(f-h)
    p4 = d(g-e)
    p5 = (a+b)h
    p6 = (c-a)(e+f)
    p7 = (b-d)(g+h)
    
    C[0,0] = p1 + p4 - p5 + p7
    C[0,1] = p3 + p5
    C[1,0] = p2 + p4
    C[1,1] = p1 - p2 + p3 + p6
    """
    # U: shape (4, 7) - indices 0=a, 1=b, 2=c, 3=d
    # V: shape (4, 7) - indices 0=e, 1=f, 2=g, 3=h
    # W: shape (4, 7) - indices 0=C00, 1=C01, 2=C10, 3=C11
    
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # p1 = (a+d)(e+h)
    U[0, 0] = 1.0  # a
    U[3, 0] = 1.0  # d
    V[0, 0] = 1.0  # e
    V[3, 0] = 1.0  # h
    W[0, 0] = 1.0  # C00
    W[3, 0] = 1.0  # C11
    
    # p2 = (c+d)e
    U[2, 1] = 1.0  # c
    U[3, 1] = 1.0  # d
    V[0, 1] = 1.0  # e
    W[2, 1] = 1.0  # C10
    W[3, 1] = -1.0 # C11
    
    # p3 = a(f-h)
    U[0, 2] = 1.0  # a
    V[1, 2] = 1.0  # f
    V[3, 2] = -1.0 # h
    W[1, 2] = 1.0  # C01
    W[3, 2] = 1.0  # C11
    
    # p4 = d(g-e)
    U[3, 3] = 1.0  # d
    V[2, 3] = 1.0  # g
    V[0, 3] = -1.0 # e
    W[0, 3] = 1.0  # C00
    W[2, 3] = 1.0  # C10
    
    # p5 = (a+b)h
    U[0, 4] = 1.0  # a
    U[1, 4] = 1.0  # b
    V[3, 4] = 1.0  # h
    W[0, 4] = -1.0 # C00
    W[1, 4] = 1.0  # C01
    
    # p6 = (c-a)(e+f)
    U[2, 5] = 1.0  # c
    U[0, 5] = -1.0 # a
    V[0, 5] = 1.0  # e
    V[1, 5] = 1.0  # f
    W[3, 5] = 1.0  # C11
    
    # p7 = (b-d)(g+h)
    U[1, 6] = 1.0  # b
    U[3, 6] = -1.0 # d
    V[2, 6] = 1.0  # g
    V[3, 6] = 1.0  # h
    W[0, 6] = 1.0  # C00
    W[1, 6] = 1.0  # C01
    
    return U, V, W


def _construct_rank42():
    """Construct a rank-42 decomposition using block structure.
    
    A is 2x4: split into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3)
    B is 4x6: split into B1 (2x6, rows 0-1) and B2 (2x6, rows 2-3)
    C = A1*B1 + A2*B2
    
    Each A_i (2x2) * B_i (2x6) can be decomposed using Strassen-like approach.
    B_i (2x6) = [B_i,1 (2x2), B_i,2 (2x2), B_i,3 (2x2)]
    A_i * B_i = A_i * B_i,1 + A_i * B_i,2 + A_i * B_i,3
    
    Each A_i * B_i,j is a 2x2 * 2x2 multiplication with rank 7 (Strassen).
    Total: 2 (A blocks) * 3 (B sub-blocks) * 7 = 42
    """
    n, m, p = 2, 4, 6
    dim_a = n * m  # 8
    dim_b = m * p  # 24
    dim_c = p * n  # 12
    
    rank = 42
    U = np.zeros((dim_a, rank), dtype=np.float64)
    V = np.zeros((dim_b, rank), dtype=np.float64)
    W = np.zeros((dim_c, rank), dtype=np.float64)
    
    U_str, V_str, W_str = _strassen_2x2x2()
    
    col = 0
    for a_block in range(2):  # A1 (cols 0-1) or A2 (cols 2-3)
        for b_sub in range(3):  # B sub-blocks (cols 0-1, 2-3, 4-5)
            for s in range(7):  # Strassen's 7 products
                r = col
                
                # A indices: a_block determines which 2 cols of A (2x4)
                # A has rows 0-1, cols 0-3
                # a_block=0: cols 0,1; a_block=1: cols 2,3
                # U index: i*m + j where i=row, j=col
                # For A1: U[0, r] and U[1, r] (rows 0,1 of col 0,1)
                # For A2: U[2, r] and U[3, r] (rows 0,1 of col 2,3)
                # U_str indices: 0=a, 1=b, 2=c, 3=d for 2x2 matrix
                # a_block=0: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
                # a_block=1: a=A[0,2], b=A[0,3], c=A[1,2], d=A[1,3]
                for idx in range(4):
                    U_val = U_str[idx, s]
                    if U_val != 0:
                        i = idx // 2  # row
                        j_local = idx % 2  # local col in 2x2
                        j_global = a_block * 2 + j_local  # global col in 2x4
                        U[i * m + j_global, r] += U_val
                
                # B indices: b_sub determines which 2 cols of B_i (2x6)
                # B has rows 0-3, cols 0-5
                # B1 (rows 0-1): for b_sub=0: cols 0,1; b_sub=1: cols 2,3; b_sub=2: cols 4,5
                # B2 (rows 2-3): for b_sub=0: cols 0,1; b_sub=1: cols 2,3; b_sub=2: cols 4,5
                # V index: j*p + k where j=row, k=col
                # V_str indices: 0=e, 1=f, 2=g, 3=h for 2x2 matrix
                # e=B[i,0], f=B[i,1], g=B[i+1,0], h=B[i+1,1] (local)
                for idx in range(4):
                    V_val = V_str[idx, s]
                    if V_val != 0:
                        i_local = idx // 2  # local row in 2x2
                        k_local = idx % 2  # local col in 2x2
                        # Which B block? We need to determine from context
                        # Actually, we're iterating over a_block, and for each a_block
                        # we use the corresponding B block (B1 for a_block=0, B2 for a_block=1)
                        i_global = a_block * 2 + i_local  # global row in 4x6
                        k_global = b_sub * 2 + k_local  # global col in 2x6
                        V[i_global * p + k_global, r] += V_val
                
                # C indices: C is 2x6
                # W index: k*n + i where k=col, i=row
                # W_str indices: 0=C00, 1=C01, 2=C10, 3=C11
                # C00: row 0, col 0 of 2x2 output
                # C01: row 0, col 1
                # C10: row 1, col 0
                # C11: row 1, col 1
                # The 2x2 output corresponds to C[*, b_sub*2 : b_sub*2+2]
                for idx in range(4):
                    W_val = W_str[idx, s]
                    if W_val != 0:
                        i_out = idx // 2  # row in output
                        k_out_local = idx % 2  # local col in 2x2 output
                        k_out_global = b_sub * 2 + k_out_local  # global col in 2x6
                        W[k_out_global * n + i_out, r] += W_val
                
                col += 1
    
    return U, V, W


def _validate(U, V, W, T, tol=1e-6):
    """Check if U, V, W (each shape (dim, rank)) exactly reconstruct T."""
    T_hat = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.allclose(T_hat, T, atol=tol)


def construct():
    n, m, p = 2, 4, 6
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # First try the constructive rank-42 decomposition
    U, V, W = _construct_rank42()
    T = _build_tensor(n, m, p)
    
    if _validate(U, V, W, T, tol=1e-6):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If rank-42 doesn't work, fall back to schoolbook
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
