import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 8]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    """
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = 2, 4, 8
    
    # Strategy: Use Kronecker product structure
    # M_{2,4,8} can be decomposed using the structure of smaller multiplications.
    # 
    # Approach: Use 6 independent rank-8 decompositions for 2x2 * 2x4 subproblems
    # Actually, let me try a direct structural decomposition.
    #
    # The matrix product C = A*B where A is 2x4, B is 4x8, C is 2x8.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    #
    # We can write A as [A1 | A2] where A1, A2 are 2x2
    # B as [B1; B2; B3; B4] where Bi are 2x2
    # Then C[i,k] = sum_{j} A[i,j]*B[j,k]
    # = (A1 row i) dot (B1 col k, B2 col k, B3 col k, B4 col k) ... not quite
    #
    # Better: C = A1 * B_top + A2 * B_bottom where B_top is 2x8 (rows 0,1 of B)
    # and B_bottom is 2x8 (rows 2,3 of B)
    # So C = A1 * B1 + A2 * B2 where B1 = B[0:2,:], B2 = B[2:4,:]
    # This is two 2x2 * 2x8 multiplications
    #
    # Each 2x2 * 2x8: rank = 2 * rank(2x1*1x8)? No.
    # 2x2 * 2x8: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This has rank at most 2*8 = 16 schoolbook, but can we do better?
    # Actually for fixed A, this is just linear combinations.
    # The tensor M_{2,2,8} has rank... 
    #
    # Let me just try the straightforward approach and verify.
    
    # Try rank 48 construction:
    # Split into 6 blocks. Each block handles a 2x2 portion of the computation.
    # Block r (r=0..5): 
    #   U covers some A entries, V covers some B entries, W covers some C entries
    
    # Alternative: use the fact that M_{2,4,8} is a sub-tensor of M_{2,4,4} x M_{1,1,2}
    # 
    # Let me try a direct construction with rank 48.
    # The idea: use 3 groups of 16 rank terms each, or 6 groups of 8.
    
    # Simplest correct approach: schoolbook rank 64, but let's try to do better.
    
    # Use Strassen-style: 
    # View as two 2x2 * 2x8 problems (split B rows into top 2 and bottom 2)
    # For 2x2 * 2x8: 
    #   C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #   This is rank 16 schoolbook.
    #   Can we use Strassen? Strassen is for 2x2*2x2.
    #   For 2x2*2x8, we can apply Strassen to each 2x2 column block of B:
    #   B has 4 column blocks of 2x2. So 4 * Strassen(2x2*2x2) = 4*7 = 28 per 2x2*2x8.
    #   Total: 2 * 28 = 56. Same as before.
    
    # Let me try a different split. 
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # Use the identity: write the computation as a sum of rank-1 tensors.
    # A rank-1 tensor is u * v * w where u is in R^{n*m}, v in R^{m*p}, w in R^{p*n}
    # and T[i,j,k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
    #
    # For our problem, T[i,j,k] = 1 for all i,j,k.
    #
    # Let me try to find a decomposition with rank R by solving a system.
    # This is hard in general, but for small dimensions we can try specific structures.
    
    # Let me try rank 48 with a structured approach.
    # 
    # Idea: Use the Kronecker product M_{2,4,8} = M_{2,2,4} ⊗ M_{1,2,2}
    # Rank(M_{2,2,4}) * Rank(M_{1,2,2})
    # M_{1,2,2}: rank = 4 (schoolbook: C[i,k] = B[i,k] for 1x1 * 2x2, but wait...)
    # Actually M_{1,2,2}: A is 1x2, B is 2x2, C is 1x2. C[k] = A[0]*B[0,k] + A[1]*B[1,k]. Rank 4.
    # M_{2,2,4}: A is 2x2, B is 2x4, C is 2x4. 
    #   C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #   Rank: at most 2*4=8 schoolbook. Can we do better?
    #   Using Strassen-like: split B into two 2x2 blocks, apply Strassen to each: 2*7=14. Worse.
    #   Schoolbook: 8.
    # So total: 8 * 4 = 32? That seems too good. Let me check if the tensor product structure is valid.
    
    # Actually, the tensor product M_{n1,m1,p1} ⊗ M_{n2,m2,p2} = M_{n1*n2, m1*m2, p1*p2}
    # So M_{2,4,8} = M_{2,2,2} ⊗ M_{1,2,4} or M_{2,4,4} ⊗ M_{1,1,2} or M_{2,2,4} ⊗ M_{1,2,2}
    
    # M_{2,2,2} ⊗ M_{1,2,4}: rank <= 7 * 8 = 56
    # M_{2,4,4} ⊗ M_{1,1,2}: rank(M_{2,4,4}) * rank(M_{1,1,2})
    #   M_{1,1,2}: A is 1x1, B is 1x2, C is 1x2. C[k] = A[0]*B[0,k]. Rank 2.
    #   M_{2,4,4}: A is 2x4, B is 4x4, C is 2x4.
    #     Schoolbook rank: 2*4*4 = 32.
    #     Can we do better? Using Strassen: split into 2x2 blocks.
    #     A = [A11 A12], B = [B11 B12; B21 B22] where each is 2x2.
    #     C = [C11 C12] where C11 = A11*B11 + A12*B21, C12 = A11*B12 + A12*B22.
    #     Each Cij is 2x2, computed using Strassen (rank 7 each).
    #     Total: 2 * 7 = 14 for M_{2,4,4}? No, that's not right either.
    #     Actually, C11 and C12 share the same A blocks but different B blocks.
    #     Let me count: C11 = A11*B11 + A12*B21. This is a 2x2 matrix product sum.
    #     Using Strassen for A11*B11 (7 mults) and A12*B21 (7 mults) = 14 mults for C11.
    #     Similarly C12 = A11*B12 + A12*B22: another 14 mults.
    #     Total: 28 for M_{2,4,4}.
    #     So M_{2,4,4} ⊗ M_{1,1,2}: rank <= 28 * 2 = 56.
    
    # M_{2,2,4} ⊗ M_{1,2,2}: 
    #   M_{2,2,4}: rank 8 (schoolbook)
    #   M_{1,2,2}: rank 4
    #   Total: 32. Let me verify this is valid.
    
    # Let me construct M_{2,2,4} ⊗ M_{1,2,2}.
    # M_{2,2,4}: A is 2x2, B is 2x4, C is 2x4.
    #   T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,1,2,3}
    #   Schoolbook: for each j,k: u[i*m+j]=1, v[j*p+k]=1, w[k*n+i]=1
    #   That's 2*4 = 8 terms.
    
    # M_{1,2,2}: A is 1x2, B is 2x2, C is 1x2.
    #   T[i,j,k] = 1 for i in {0}, j in {0,1}, k in {0,1}
    #   Schoolbook: 1*2*2 = 4 terms.
    
    # Tensor product: M_{2*1, 2*2, 4*2} = M_{2,4,8}. Yes!
    # Rank <= 8 * 4 = 32.
    
    # Let me construct this.
    # The tensor product of decompositions: if T1 has rank r1 with (U1,V1,W1)
    # and T2 has rank r2 with (U2,V2,W2), then T1⊗T2 has rank r1*r2 with
    # U = U1 ⊗ U2, V = V1 ⊗ V2, W = W1 ⊗ W2 (Kronecker product).
    
    # M_{2,2,4} schoolbook decomposition:
    n1, m1, p1 = 2, 2, 4
    r1 = n1 * m1 * p1  # 8
    U1 = np.zeros((n1 * m1, r1), dtype=int)
    V1 = np.zeros((m1 * p1, r1), dtype=int)
    W1 = np.zeros((p1 * n1, r1), dtype=int)
    col = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U1[i * m1 + j, col] = 1
                V1[j * p1 + k, col] = 1
                W1[k * n1 + i, col] = 1
                col += 1
    
    # M_{1,2,2} schoolbook decomposition:
    n2, m2, p2 = 1, 2, 2
    r2 = n2 * m2 * p2  # 4
    U2 = np.zeros((n2 * m2, r2), dtype=int)
    V2 = np.zeros((m2 * p2, r2), dtype=int)
    W2 = np.zeros((p2 * n2, r2), dtype=int)
    col = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U2[i * m2 + j, col] = 1
                V2[j * p2 + k, col] = 1
                W2[k * n2 + i, col] = 1
                col += 1
    
    # Kronecker product:
    # U = U1 ⊗ U2 has shape (n1*m1 * n2*m2, r1 * r2) = (4*2, 8*4) = (8, 32)
    # But we need U of shape (n*m, rank) = (2*4, rank) = (8, rank). Good.
    # V = V1 ⊗ V2 has shape (m1*p1 * m2*p2, r1*r2) = (8*4, 32) = (32, 32)
    # We need V of shape (m*p, rank) = (4*8, rank) = (32, rank). Good.
    # W = W1 ⊗ W2 has shape (p1*n1 * p2*n2, r1*r2) = (8*2, 32) = (16, 32)
    # We need W of shape (p*n, rank) = (8*2, rank) = (16, rank). Good.
    
    U = np.kron(U1, U2)
    V = np.kron(V1, V2)
    W = np.kron(W1, W2)
    
    rank = U.shape[1]
    
    # Verify
    U_np = U.astype(float)
    V_np = V.astype(float)
    W_np = W.astype(float)
    
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    T_computed = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    T_expected = np.ones((n, m, p))
    
    if not np.allclose(T_computed, T_expected, atol=1e-10):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
