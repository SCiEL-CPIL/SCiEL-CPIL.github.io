import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    # Strassen 2x2 decomposition
    # Inputs: a00, a01, a10, a11
    # U rows correspond to a00, a01, a10, a11
    # V rows correspond to b00, b01, b10, b11
    # W rows correspond to c00, c01, c10, c11
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    U2 = np.array([
        [1, 0, 0, 0, 1, 0, 0], # a00: M1, M3, M5, M6
        [0, 0, 0, 0, 1, 0, 1], # a01: M5, M7
        [0, 1, 0, 0, 0, 1, 0], # a10: M2, M6
        [1, 1, 0, 1, 0, 0, -1] # a11: M1, M2, M4, M7
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # b00: M1, M2, M4, M6
        [0, 0, 1, 0, 0, 1, 0], # b01: M3, M6
        [0, 0, 0, 1, 0, 0, 1], # b10: M4, M7
        [1, 0, -1, 0, 1, 0, 1]  # b11: M1, M3, M5, M7
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, -1, 0, 1, 0], # c00: M1, M4, M6
        [0, 0, 1, 0, 1, 0, 0], # c01: M3, M5
        [0, 1, 0, 1, 0, 0, 0], # c10: M2, M4
        [1, 0, -1, 0, 0, 0, 1]  # c11: M1, M3, M7
    ], dtype=float)
    
    # Verify 2x2 decomposition
    # C = U.T @ diag(p) @ V.T ? No, C_ij = sum_k U_ik p_k V_jk?
    # The tensor is T = sum_k u_k v_k^T w_k^T?
    # Matrix mult: C = A B.
    # C_{ij} = sum_k A_{ik} B_{kj}.
    # Decomposition: C_{ij} = sum_r U_{i,r} V_{j,r} W_{k,r}? No.
    # The task format: U (nm x rank), V (mp x rank), W (pn x rank).
    # C_{i,k} = sum_r U_{i, r} V_{k, r} W_{j, r}? No.
    # Standard format:
    # C_{i, j, k} = sum_r U_{i, r} V_{j, r} W_{k, r}
    # Here C is the product matrix C[i][j].
    # Wait, the tensor is T_{i, j, k} where C[i][j] = sum_k A[i][k] B[k][j].
    # So T_{i, j, k} = A[i][k] B[k][j] is not a tensor in i,j,k for fixed A,B.
    # The multiplication tensor T is defined by T_{i, k, j} = 1 if we consider A[i][k] and B[k][j].
    # The output is C[i][j].
    # So the tensor is T[i, k, j] (A index, B index, C index).
    # The task says:
    # U: (n*m, rank) -> indices i, k? No, n is rows of A, m is cols of A.
    # A is n x m. B is m x p. C is n x p.
    # U index: i (0..n-1), k (0..m-1)? No, U is n*m.
    # Usually U corresponds to A, V to B, W to C.
    # U[i, r] * V[k, r] * W[j, r] ?
    # Let's check the seed program.
    # U[i*m+j][column] = 1. Index i*m+j. i is row of A, j is col of A.
    # V[j*p+k][column] = 1. Index j*p+k. j is col of A (row of B), k is col of B.
    # W[k*n+i][column] = 1. Index k*n+i. k is col of B (col of C), i is row of A (row of C).
    # So C[i][k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    # This matches C[i][k] = sum_j A[i][j] B[j][k].
    
    # So for 2x2:
    # n=2, m=2, p=2.
    # U index i*2+j.
    # V index j*2+k.
    # W index k*2+i.
    
    # My Strassen matrices U2, V2, W2 must follow this indexing.
    # U2 rows: a00 (i=0,j=0), a01 (i=0,j=1), a10 (i=1,j=0), a11 (i=1,j=1).
    # V2 rows: b00 (j=0,k=0), b01 (j=0,k=1), b10 (j=1,k=0), b11 (j=1,k=1).
    # W2 rows: c00 (k=0,i=0), c01 (k=1,i=0), c10 (k=0,i=1), c11 (k=1,i=1).
    
    # Let's re-verify W2 indices.
    # W[k*n+i]. n=2.
    # k=0, i=0 -> 0. c00.
    # k=1, i=0 -> 2. c10? No, c[i][k]. i=0, k=1 is c01.
    # Wait. W index is k*n+i.
    # If i=0, k=0 -> 0.
    # If i=0, k=1 -> 2.
    # If i=1, k=0 -> 1.
    # If i=1, k=1 -> 3.
    # So W row 0: c00.
    # W row 1: c10.
    # W row 2: c01.
    # W row 3: c11.
    
    # My previous W2 had rows c00, c01, c10, c11.
    # I need to reorder W2 rows to match k*n+i order: 0, 2, 1, 3.
    # c00 (0), c10 (1), c01 (2), c11 (3).
    
    W2_corrected = np.array([
        [1, 0, 0, -1, 0, 1, 0], # c00
        [0, 1, 0, 1, 0, 0, 0], # c10
        [0, 0, 1, 0, 1, 0, 0], # c01
        [1, 0, -1, 0, 0, 0, 1]  # c11
    ], dtype=float)
    
    # Check U2 and V2 ordering.
    # U index i*m+j.
    # i=0, j=0 -> 0.
    # i=0, j=1 -> 1.
    # i=1, j=0 -> 2.
    # i=1, j=1 -> 3.
    # My U2 rows: a00, a01, a10, a11. Matches.
    
    # V index j*p+k.
    # j=0, k=0 -> 0.
    # j=0, k=1 -> 1.
    # j=1, k=0 -> 2.
    # j=1, k=1 -> 3.
    # My V2 rows: b00, b01, b10, b11. Matches.
    
    # Now Kronecker product.
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2_corrected, W2_corrected)
    
    # Convert to list of lists for return
    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }

# EVOLVE-BLOCK-END
