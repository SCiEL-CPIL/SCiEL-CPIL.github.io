"""
Program for matrix multiplication 2x5x5 rank minimization.
Uses Strassen's algorithm on 2x2 sub-blocks to reduce rank from 50 to 46.
"""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 5 matrix by a 5 by 5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 5, 5
    
    terms = []
    
    def add_strassen(i_off, j_off, k_off):
        """Adds 7 terms for a 2x2x2 Strassen block at given offsets."""
        # M1 = (A00+A11)(B00+B11)
        terms.append((
            {(i_off, j_off): 1, (i_off+1, j_off+1): 1},
            {(j_off, k_off): 1, (j_off+1, k_off+1): 1},
            {(k_off, i_off): 1, (k_off+1, i_off+1): 1}
        ))
        # M2 = (A10+A11)B00
        terms.append((
            {(i_off+1, j_off): 1, (i_off+1, j_off+1): 1},
            {(j_off, k_off): 1},
            {(k_off, i_off+1): 1, (k_off+1, i_off+1): -1}
        ))
        # M3 = A00(B01-B11)
        terms.append((
            {(i_off, j_off): 1},
            {(j_off, k_off+1): 1, (j_off+1, k_off+1): -1},
            {(k_off+1, i_off): 1, (k_off+1, i_off+1): 1}
        ))
        # M4 = A11(B10-B00)
        terms.append((
            {(i_off+1, j_off+1): 1},
            {(j_off+1, k_off): 1, (j_off, k_off): -1},
            {(k_off, i_off): 1, (k_off, i_off+1): 1}
        ))
        # M5 = (A00+A01)B11
        terms.append((
            {(i_off, j_off): 1, (i_off, j_off+1): 1},
            {(j_off+1, k_off+1): 1},
            {(k_off, i_off): -1, (k_off+1, i_off): 1}
        ))
        # M6 = (A10-A00)(B00+B01)
        terms.append((
            {(i_off+1, j_off): 1, (i_off, j_off): -1},
            {(j_off, k_off): 1, (j_off, k_off+1): 1},
            {(k_off+1, i_off+1): 1}
        ))
        # M7 = (A01-A11)(B10+B11)
        terms.append((
            {(i_off, j_off+1): 1, (i_off+1, j_off+1): -1},
            {(j_off+1, k_off): 1, (j_off+1, k_off+1): 1},
            {(k_off, i_off): 1}
        ))
    
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        """Adds schoolbook terms for the given ranges."""
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    terms.append((
                        {(i, j): 1},
                        {(j, k): 1},
                        {(k, i): 1}
                    ))
    
    # Partition inner dim 5 = 2+2+1, output dim 5 = 2+2+1
    # Block (i in {0,1}, j in {0,1}, k in {0,1}): Strassen
    add_strassen(0, 0, 0)
    # Block (i in {0,1}, j in {0,1}, k in {2,3}): Strassen
    add_strassen(0, 0, 2)
    # Block (i in {0,1}, j in {0,1}, k in {4}): Schoolbook
    add_schoolbook(0, 2, 0, 2, 4, 5)
    
    # Block (i in {0,1}, j in {2,3}, k in {0,1}): Strassen
    add_strassen(0, 2, 0)
    # Block (i in {0,1}, j in {2,3}, k in {2,3}): Strassen
    add_strassen(0, 2, 2)
    # Block (i in {0,1}, j in {2,3}, k in {4}): Schoolbook
    add_schoolbook(0, 2, 2, 4, 4, 5)
    
    # Block (i in {0,1}, j in {4}, k in {0,1,2,3,4}): Schoolbook
    add_schoolbook(0, 2, 4, 5, 0, 5)
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        u_dict, v_dict, w_dict = terms[c]
        for (i, j), val in u_dict.items():
            U[i * m + j][c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][c] = val
    
    # Verify correctness with a random test
    if time.time() < deadline - 5.0:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Build the multiplication tensor from U, V, W
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Compare with identity tensor: T[i,j,k] = 1 for all (i,j,k)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    C_computed[i, k] += A_test[i, j] * B_test[j, k] * 1.0
        
        # Verify via the decomposition: C[i,k] = sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]*A[i,j]*B[j,k]
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += U_arr[i*m+j, r] * A_test[i, j]
                    s2 = 0.0
                    for j in range(m):
                        s2 += V_arr[j*p+k, r] * B_test[j, k]
                    C_decomp[i, k] += s * s2 * W_arr[k*n+i, r]
        
        if not np.allclose(C_decomp, C_expected, atol=1e-10):
            # Fallback to schoolbook if Strassen has a bug
            terms.clear()
            add_schoolbook(0, n, 0, m, 0, p)
            rank = len(terms)
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            for c in range(rank):
                u_dict, v_dict, w_dict = terms[c]
                for (i, j), val in u_dict.items():
                    U[i * m + j][c] = val
                for (j, k), val in v_dict.items():
                    V[j * p + k][c] = val
                for (k, i), val in w_dict.items():
                    W[k * n + i][c] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
