"""Program for 4x4x4 matrix multiplication tensor decomposition."""
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-49 decomposition for 4x4x4 matrix multiplication
    using the tensor square of Strassen's 7-term 2x2x2 algorithm.
    
    The 4x4x4 multiplication tensor is the tensor square of the 2x2x2
    multiplication tensor. Since Strassen gives a rank-7 decomposition for
    2x2x2, the tensor square gives a rank-49 decomposition for 4x4x4.
    
    Verification: For all A, B (4x4 matrices), the decomposition computes
    C = A*B exactly, where C[i,k] = sum_j A[i,j]*B[j,k].
    """
    n, m, p = 4, 4, 4
    
    # Strassen's 7 terms for 2x2x2 matrix multiplication
    def strassen_2x2():
        """Return list of 7 (U, V, W) tuples, each a 2x2 numpy array."""
        terms = []
        # M1 = (A00+A11)(B00+B11)
        U = np.array([[1,0],[0,1]], dtype=float)
        V = np.array([[1,0],[0,1]], dtype=float)
        W = np.array([[1,0],[0,1]], dtype=float)
        terms.append((U, V, W))
        
        # M2 = (A10+A11)B00
        U = np.array([[0,0],[1,1]], dtype=float)
        V = np.array([[1,0],[0,0]], dtype=float)
        W = np.array([[0,0],[1,0]], dtype=float)
        terms.append((U, V, W))
        
        # M3 = A00(B01-B11)
        U = np.array([[1,0],[0,0]], dtype=float)
        V = np.array([[0,1],[0,-1]], dtype=float)
        W = np.array([[0,1],[0,0]], dtype=float)
        terms.append((U, V, W))
        
        # M4 = A11(B10-B00)
        U = np.array([[0,0],[0,1]], dtype=float)
        V = np.array([[-1,0],[1,0]], dtype=float)
        W = np.array([[1,0],[0,1]], dtype=float)
        terms.append((U, V, W))
        
        # M5 = (A00+A10)B11
        U = np.array([[1,0],[1,0]], dtype=float)
        V = np.array([[0,0],[0,1]], dtype=float)
        W = np.array([[-1,0],[0,-1]], dtype=float)
        terms.append((U, V, W))
        
        # M6 = (A01-A10)(B00+B10)
        U = np.array([[0,1],[-1,0]], dtype=float)
        V = np.array([[1,0],[1,0]], dtype=float)
        W = np.array([[0,0],[0,1]], dtype=float)
        terms.append((U, V, W))
        
        # M7 = (A01-A11)(B11+B01)
        U = np.array([[0,1],[0,-1]], dtype=float)
        V = np.array([[0,1],[1,1]], dtype=float)
        W = np.array([[1,0],[0,0]], dtype=float)
        terms.append((U, V, W))
        
        return terms
    
    s2x2 = strassen_2x2()
    
    # Construct the rank-49 decomposition for 4x4x4
    # For 4x4 matrices, index i = 2*i2 + i1 where i2 in {0,1} is block row, i1 in {0,1} is within-block row
    # Similarly for j, k.
    # 
    # The tensor square construction:
    # For each pair (r1, r2) in {0..6}^2:
    #   U_4x4[i, j, col] = U_s[r1][i2, j2] * U_s[r2][i1, j1]  where i=2*i2+i1, j=2*j2+j1
    #   V_4x4[j, k, col] = V_s[r1][j2, k2] * V_s[r2][j1, k1]  where j=2*j2+j1, k=2*k2+k1
    #   W_4x4[k, i, col] = W_s[r1][k2, i2] * W_s[r2][k1, i1]  where k=2*k2+k1, i=2*i2+i1
    #
    # Flattened: U[i*4+j, col], V[j*4+k, col], W[k*4+i, col]
    
    rank = 49
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    
    col = 0
    for r1 in range(7):
        for r2 in range(7):
            U1, V1, W1 = s2x2[r1]
            U2, V2, W2 = s2x2[r2]
            
            for i2 in range(2):
                for j2 in range(2):
                    for k2 in range(2):
                        for i1 in range(2):
                            for j1 in range(2):
                                for k1 in range(2):
                                    i = 2*i2 + i1
                                    j = 2*j2 + j1
                                    k = 2*k2 + k1
                                    
                                    U[i*4 + j, col] = U1[i2, j2] * U2[i1, j1]
                                    V[j*4 + k, col] = V1[j2, k2] * V2[j1, k1]
                                    W[k*4 + i, col] = W1[k2, i2] * W2[k1, i1]
            
            col += 1
    
    # Verify the decomposition
    np.random.seed(42)
    for trial in range(5):
        A_test = np.random.randn(4, 4)
        B_test = np.random.randn(4, 4)
        C_true = A_test @ B_test
        
        C_decomp = np.zeros((4, 4))
        for r in range(rank):
            u_flat = U[:, r]
            v_flat = V[:, r]
            w_flat = W[:, r]
            # u_val = sum_{i,j} U[i*4+j, r] * A[i,j]
            u_val = np.dot(u_flat, A_test.flatten())
            # v_val = sum_{j,k} V[j*4+k, r] * B[j,k]
            v_val = np.dot(v_flat, B_test.flatten())
            # C_decomp += u_val * v_val * W[k*4+i, r] for all i,k
            C_decomp += u_val * v_val * w_flat.reshape(4, 4)
        
        max_err = np.max(np.abs(C_decomp - C_true))
        if max_err > 1e-10:
            raise AssertionError(f"Verification failed: max_err={max_err}")
    
    # Convert to lists of integers/half-integers
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                if abs(val - round(val)) < 1e-10:
                    row_list.append(int(round(val)))
                else:
                    doubled = val * 2
                    if abs(doubled - round(doubled)) < 1e-10:
                        row_list.append(int(round(doubled)) / 2.0)
                    else:
                        row_list.append(val)
            result.append(row_list)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
