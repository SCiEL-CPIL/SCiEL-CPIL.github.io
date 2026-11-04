"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Schoolbook rank: n*m*p = 4*4*7 = 112
    # We'll try to find a better decomposition using Strassen-like methods
    
    def strassen_2x2_decomp():
        """Return U, V, W for 2x2x2 matrix multiplication with rank 7."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # P1 = (a+d)(e+h)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1
        
        # P2 = (c+d)e
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1
        
        # P3 = a(f-h)
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[1,2] = 1; W[3,2] = 1
        
        # P4 = d(g-e)
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[2,3] = 1
        
        # P5 = (a+b)h
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,4] = -1; W[1,4] = 1
        
        # P6 = (c-a)(e+f)
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1
        
        # P7 = (b-d)(g+h)
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = 1; W[3,6] = 1
        
        return U, V, W
    
    def build_4x4x4_strassen():
        """Build 4x4x4 decomposition using Strassen (rank 49)."""
        U2, V2, W2 = strassen_2x2_decomp()
        
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        # Block indices in flattened 4x4
        A_blocks = {
            'A11': [0, 1, 4, 5],
            'A12': [2, 3, 6, 7],
            'A21': [8, 9, 12, 13],
            'A22': [10, 11, 14, 15]
        }
        B_blocks = {
            'B11': [0, 1, 4, 5],
            'B12': [2, 3, 6, 7],
            'B21': [8, 9, 12, 13],
            'B22': [10, 11, 14, 15]
        }
        C_blocks = {
            'C11': [0, 1, 4, 5],
            'C12': [2, 3, 6, 7],
            'C21': [8, 9, 12, 13],
            'C22': [10, 11, 14, 15]
        }
        
        strassen_products = [
            ({'A11': 1, 'A22': 1}, {'B11': 1, 'B22': 1}, {'C11': 1, 'C22': 1}),
            ({'A21': 1, 'A22': 1}, {'B11': 1}, {'C21': 1}),
            ({'A11': 1}, {'B12': 1, 'B22': -1}, {'C12': 1, 'C22': 1}),
            ({'A22': 1}, {'B21': 1, 'B11': -1}, {'C11': 1, 'C21': 1}),
            ({'A11': 1, 'A12': 1}, {'B22': 1}, {'C11': -1, 'C12': 1}),
            ({'A21': 1, 'A11': -1}, {'B11': 1, 'B12': 1}, {'C22': 1}),
            ({'A12': 1, 'A22': -1}, {'B21': 1, 'B22': 1}, {'C11': 1, 'C22': 1}),
        ]
        
        for s, (a_coeffs, b_coeffs, c_contribs) in enumerate(strassen_products):
            for c in range(7):
                col = s * 7 + c
                for block_name, coeff in a_coeffs.items():
                    for idx in A_blocks[block_name]:
                        pos = A_blocks[block_name].index(idx)
                        U[idx, col] += coeff * U2[pos, c]
                for block_name, coeff in b_coeffs.items():
                    for idx in B_blocks[block_name]:
                        pos = B_blocks[block_name].index(idx)
                        V[idx, col] += coeff * V2[pos, c]
                for block_name, coeff in c_contribs.items():
                    for idx in C_blocks[block_name]:
                        pos = C_blocks[block_name].index(idx)
                        W[idx, col] += coeff * W2[pos, c]
        
        return U, V, W
    
    def build_4x4x3_schoolbook():
        """Build 4x4x3 schoolbook decomposition (rank 48)."""
        U = np.zeros((16, 48))
        V = np.zeros((12, 48))  # j in 0..3, k in 0..2
        W = np.zeros((12, 48))  # k in 0..2, i in 0..3
        
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U[i*4+j, col] = 1
                    V[j*3+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        
        return U, V, W
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is correct."""
        rank = U.shape[1]
        # Check shapes
        if U.shape != (n*m, rank):
            return False
        if V.shape != (m*p, rank):
            return False
        if W.shape != (p*n, rank):
            return False
        
        # Check the tensor equation
        # For each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1
        # For all other combinations, it should be 0
        # Actually, the correct check is:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # In tensor form: T[i,j,k; i',j',k'] = delta[i,i']*delta[j,j']*delta[k,k']
        # The decomposition gives: T = sum_c U_c * V_c * W_c (outer products)
        # So: T[i,j,k; i',j',k'] = sum_c U[i*m+j, c] * V[j'*p+k', c] * W[k*n+i', c]
        # Wait, let me re-read the convention.
        # W[k*n+i, r] - so W is indexed by (k, i) where k is the column of C and i is the row of C
        # The tensor for matrix multiplication A*B=C is:
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This should equal delta[i,i']*delta[j,j']*delta[k,k'] for the identity case
        # Actually no. Let me think again.
        
        # The matrix multiplication tensor T has T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
        # So T[i,j,k] = delta[i,i']*delta[j,j']*delta[k,k']... no.
        
        # Actually, the tensor is defined such that:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # In terms of the decomposition:
        # C[i,k] = sum_c (sum_j A[i,j] U[j... wait no.
        
        # Let me use the standard formulation:
        # C = A*B where A is n x m, B is m x p, C is n x p
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # 
        # The tensor decomposition: C[i,k] = sum_c (sum_j A[i,j] * U[i*m+j, c]) * (sum_k' B[j,k'] * V[j*p+k', c]) ... no
        
        # Actually the standard formulation is:
        # C[i,k] = sum_c U_A[i, c] * U_B[k, c] * ... no
        
        # Let me look at the interface: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # The decomposition is:
        # C[i,k] = sum_c (sum_j A[i,j] * U[i*m+j, c]) * (sum_j' B[j',k] * V[j'*p+k, c]) * ... 
        # That doesn't work either.
        
        # The correct formulation for matrix multiplication tensor:
        # T[i,j,k] = 1 for the "diagonal" entries in the appropriate sense
        # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This means: for each (i,j,k), the sum over c of the product should give the tensor value.
        # For matrix multiplication, the tensor is:
        # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
        # But that's not quite right either.
        
        # Let me just verify numerically with random matrices.
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        
        # Compute C from the decomposition
        C_decomp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    # U[i*m+j, c] for each j
                    # V[j*p+k, c] for each j
                    # W[k*n+i, c]
                    for j in range(m):
                        val += A[i,j] * B[j,k] * U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                C_decomp[i,k] = val
        
        if not np.allclose(C_decomp, C_true, atol=1e-6):
            return False
        return True
    
    # First, try the Strassen-based approach: 49 (for 4x4x4) + 48 (for 4x4x3) = 97
    U4, V4, W4 = build_4x4x4_strassen()
    U3, V3, W3 = build_4x4x3_schoolbook()
    
    rank = 97
    U = np.zeros((16, rank))
    V = np.zeros((28, rank))
    W = np.zeros((28, rank))
    
    # Part 1: 4x4x4 (first 4 columns of B, first 4 columns of C)
    U[:, :49] = U4
    for j in range(4):
        for k in range(4):
            V[j*7+k, :49] = V4[j*4+k, :49]
    W[:16, :49] = W4
    
    # Part 2: 4x4x3 (last 3 columns of B, last 3 columns of C)
    col_offset = 49
    for i in range(4):
        for j in range(4):
            for k in range(3):
                col = col_offset + i*12 + j*3 + k
                U[i*4+j, col] = U3[i*4+j, i*12 + j*3 + k]
                V[j*7 + (k+4), col] = V3[j*3+k, i*12 + j*3 + k]
                W[(k+4)*4 + i, col] = W3[k*4+i, i*12 + j*3 + k]
    
    # Verify
    if verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank 112
    rank = 112
    U = np.zeros((16, rank))
    V = np.zeros((28, rank))
    W = np.zeros((28, rank))
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(7):
                U[i*4+j, col] = 1
                V[j*7+k, col] = 1
                W[k*4+i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
