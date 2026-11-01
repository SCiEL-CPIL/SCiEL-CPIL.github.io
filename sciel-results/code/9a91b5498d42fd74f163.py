import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Tensor product of the rank-7 Strassen decomposition for 2x2 matrices.
    # This yields a rank-49 decomposition for 4x4 matrices.
    # The index convention is:
    #   U row index: i*m + j  (m=4)
    #   V row index: j*p + k  (p=4)
    #   W row index: k*n + i  (n=4)
    
    # Standard Strassen rank-7 factors for 2x2 matrices
    # U_2 has shape (4, 7): rows indexed by (i, j) for i, j in {0, 1}
    # V_2 has shape (4, 7): rows indexed by (j, k) for j, k in {0, 1}
    # W_2 has shape (4, 7): rows indexed by (k, i) for k, i in {0, 1}
    
    U_2 = np.array([
        [1, 0, 0, 1, 0, 1, 0],  # (0,0)
        [0, 0, 0, 0, 0, -1, 1], # (0,1)
        [0, 1, 1, 0, 0, 0, -1], # (1,0)
        [1, 1, 1, 0, 1, 0, 0],  # (1,1)
    ], dtype=int)
    
    V_2 = np.array([
        [1, 1, 0, 0, 0, 0, 0],  # (0,0)
        [0, 0, 0, 0, -1, 1, 0], # (0,1)
        [0, 0, 0, 0, 1, 0, 1],  # (1,0)
        [1, 0, 1, -1, 0, 1, 1], # (1,1)
    ], dtype=int)
    
    W_2 = np.array([
        [1, 0, 0, 1, 0, 0, 0],  # (0,0)
        [0, 0, 1, 0, 0, 0, 1],  # (0,1)
        [0, -1, 0, 0, -1, 0, 0],# (1,0)
        [1, -1, 0, 0, 1, 0, 0], # (1,1)
    ], dtype=int)
    
    # Verify Strassen correctness for 2x2
    # For matrices A, B (2x2), C = A*B
    # The Strassen algorithm computes 7 products M1..M7
    # Let's verify the factorization: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = C[i,k]
    # Actually, the convention is W[k*n+i, c], so for 2x2, W index is k*2+i.
    # Let's double check the indices.
    # The problem statement says: "output factor follows the official trace convention W[k*n+i,r]"
    # So W row index is k*n + i.
    # For 2x2, n=2.
    # Row 0: k=0, i=0 -> C[0,0]
    # Row 1: k=0, i=1 -> C[1,0]? No, W[k*n+i].
    # Wait, the matrix multiplication result is C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor is T[i,j,k] = delta_{i,i} delta_{j,j} delta_{k,k}? No.
    # The multiplication tensor T[i,j,k] is 1 if ... no.
    # T is defined such that <T, U V W> = sum_{i,j,k} T[i,j,k] U[i,j] V[j,k] W[k,i] = sum C[i,k].
    # Actually, the standard tensor for matrix multiplication is T[i,j,k] = 1 for all i,j,k? No.
    # The identity is: C = A B => C[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor M[i,j,k] = 1 if we want to compute sum_{i,j,k} M[i,j,k] A[i,j] B[j,k] W[k,i] to get something related to C.
    # Usually, the factorization is: M[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
    # And the "trace convention" for W is that W is indexed by (k,i) such that when you sum over k, you get the output C[i,k]?
    # Let's look at the seed program:
    # U[i*m+j][column] = 1
    # V[j*p+k][column] = 1
    # W[k*n+i][column] = 1
    # Sum over column c: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # In schoolbook, for a specific (i,j,k), there is one column c = i*m*p + j*p + k.
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    # So the term is 1.
    # The total sum over c gives the number of c's such that U[i*m+j,c]=V[j*p+k,c]=W[k*n+i,c]=1.
    # In schoolbook, for fixed i,j,k, there is exactly one c.
    # But we want to compute C[i,k] = sum_j A[i,j] B[j,k].
    # The expression sum_{i,j,k} (sum_c U[i,j,c] V[j,k,c] W[k,i,c]) A[i,j] B[j,k]
    # = sum_c (sum_i A[i,j] U[i,j,c]) (sum_k B[j,k] V[j,k,c]) W[k,i,c]? No, indices don't match.
    
    # Let's re-read the tensor definition implied by the factors.
    # We are decomposing the tensor M where M[i,j,k] = 1? No.
    # The condition "exactly decompose the matrix multiplication tensor" means:
    # For all i, j, k: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Where M[i,j,k] is the matrix multiplication tensor.
    # What is M[i,j,k]?
    # The matrix multiplication tensor is defined by M[i,j,k] = 1 if ...?
    # Actually, the standard definition is that the tensor T satisfies:
    # T[i, j, k] = 1 for all i, j, k is NOT correct.
    # The tensor T for matrix multiplication M_{n,p,q} is such that:
    # T[i, j, k] = 1 if i, j, k are valid indices?
    # Let's check the schoolbook.
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    # This implies that for the schoolbook decomposition, the tensor entry T[i,j,k] (reconstructed) is:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # For a fixed (i,j,k), in schoolbook, there is exactly one column c (the one corresponding to i,j,k) where all three are 1.
    # So T[i,j,k] = 1.
    # Wait, if T[i,j,k] = 1 for all i,j,k, then the tensor is just the all-ones tensor.
    # Is the matrix multiplication tensor the all-ones tensor?
    # Let's check the definition.
    # The multiplication tensor $\mathcal{M}_{n,p,q}$ is defined such that
    # $\mathcal{M}_{n,p,q} = \sum_{i=0}^{n-1} \sum_{j=0}^{p-1} \sum_{k=0}^{q-1} \mathbf{e}_i \otimes \mathbf{e}_j \otimes \mathbf{e}_k$?
    # If so, then indeed T[i,j,k] = 1.
    # Let's verify with the formula: C = AB.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # The bilinear form <T, A, B, W> = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] W[k,i].
    # If T[i,j,k] = 1, then sum_{i,j,k} A[i,j] B[j,k] W[k,i].
    # This equals sum_i sum_k (sum_j A[i,j] B[j,k]) W[k,i] = sum_{i,k} C[i,k] W[k,i] = Tr(C W^T) ?
    # The problem says "output factor follows the official trace convention W[k*n+i,r]".
    # This usually means that the contraction sum_{i,j,k} T[i,j,k] U[i,j] V[j,k] W[k,i] computes the trace of the product or similar.
    # If T is the all-ones tensor, then the schoolbook decomposition works with T[i,j,k]=1.
    # And Strassen's algorithm decomposes the same tensor (all-ones) with rank 7 for n=p=q=2.
    # So, if I use the tensor product of Strassen factors, I am decomposing the tensor product of the all-ones tensor with itself?
    # The tensor product of two tensors T1 and T2 has entries (T1 x T2)[(i1,i2),(j1,j2),(k1,k2)] = T1[i1,j1,k1] T2[i2,j2,k2].
    # If T1 and T2 are all-ones, then T1 x T2 is all-ones.
    # So the tensor product of the Strassen decomposition (rank 7) gives a decomposition of the 4x4 multiplication tensor (which is all-ones) with rank 7*7=49.
    
    # So, the plan is:
    # 1. Construct U_4, V_4, W_4 as Kronecker products (with appropriate index permutations) of U_2, V_2, W_2.
    # 2. The row indices for U_4 are i*4+j. The "i" corresponds to the first dimension of the 2x2 block, and "j" to the second?
    #    Wait, 4x4 matrix A can be viewed as 2x2 blocks of 2x2 matrices.
    #    Let A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    #    Then A[i,j] for i,j in 0..3.
    #    i = 2*i2 + i1, j = 2*j2 + j1.
    #    So i1 is the "inner" index, i2 is the "outer" index.
    #    The Kronecker product structure usually maps (i2, i1) to i2*2 + i1.
    #    So U_4[2*i2 + i1, 2*j2 + j1, c2*7 + c1] = U_2[i2, j2, c2] * U_2[i1, j1, c1]?
    #    Let's check the dimensions.
    #    U_2 has shape (4, 7). Rows are (i,j) for 2x2.
    #    U_4 should have shape (16, 49). Rows are (I,J) for 4x4.
    #    If we use Kronecker product U_4 = U_2 \otimes U_2, then
    #    U_4[ (r1, r2), (c1, c2) ] = U_2[r1, c1] U_2[r2, c2].
    #    Here r1, r2 are indices for U_2 rows (0..3), c1, c2 for U_2 cols (0..6).
    #    Row index of U_4 is r1*4 + r2? Or r1*2 + r2?
    #    Standard Kronecker product A (m x n) and B (p x q) gives C (mp x nq).
    #    C[ i*p + k, j*q + l ] = A[i, j] B[k, l].
    #    So if U_2 is 4x7, U_4 = U_2 \otimes U_2 is 16x49.
    #    Row index R = i*4 + k, where i, k are U_2 row indices.
    #    Col index C = j*7 + l, where j, l are U_2 col indices.
    #    U_4[R, C] = U_2[i, j] U_2[k, l].
    #    
    #    Now, how do U_2 row indices (i, k) map to 4x4 matrix indices (I, J)?
    #    U_2 rows correspond to (i, j) of the 2x2 matrix.
    #    Let U_2 row index r correspond to (i, j) such that r = i*2 + j.
    #    Then for U_4, the row index R = i*4 + k.
    #    Here i is the row index of the first U_2 (outer), k is the row index of the second U_2 (inner).
    #    So i = 2*i2 + i1? No.
    #    Let's define the mapping carefully.
    #    For 4x4 matrix A, A[I, J].
    #    We can view A as a 2x2 block matrix where each block is 2x2.
    #    Let I = 2*i2 + i1, J = 2*j2 + j1.
    #    Then the "outer" indices are (i2, j2) and "inner" are (i1, j1).
    #    The Kronecker product U_2 \otimes U_2 has row indices (r_outer, r_inner).
    #    r_outer corresponds to (i2, j2) in U_2?
    #    U_2 row r_outer = i2*2 + j2?
    #    U_2 row r_inner = i1*2 + j1?
    #    So R = r_outer * 4 + r_inner = (i2*2 + j2)*4 + (i1*2 + j1).
    #    But the required row index for U is I*4 + J = (2*i2 + i1)*4 + (2*j2 + j1) = 8*i2 + 4*i1 + 8*j2 + 4*j1? No.
    #    I*4 + J = 4*(2*i2 + i1) + (2*j2 + j1) = 8*i2 + 4*i1 + 2*j2 + j1.
    #    This is NOT the same as (i2*2 + j2)*4 + (i1*2 + j1) = 8*i2 + 4*j2 + 4*i1 + 2*j1.
    #    So a simple Kronecker product U_2 \otimes U_2 does NOT have the row indices in the order I*4+J.
    #    We need to permute the rows.
    
    #    Let's construct U_4 directly.
    #    U_4[I*4+J, c] where c = c_outer*7 + c_inner.
    #    U_4[I*4+J, c_outer*7 + c_inner] = U_2[i2*2+j2, c_outer] * U_2[i1*2+j1, c_inner]
    #    where I = 2*i2 + i1, J = 2*j2 + j1.
    
    #    Similarly for V_4.
    #    V_4[J*4+K, c] = V_2[j2*2+k2, c_outer] * V_2[j1*2+k1, c_inner]
    #    where J = 2*j2 + j1, K = 2*k2 + k1.
    
    #    And W_4.
    #    W_4[K*4+I, c] = W_2[k2*2+i2, c_outer] * W_2[k1*2+i1, c_inner]
    #    where K = 2*k2 + k1, I = 2*i2 + i1.
    
    #    Let's verify the indices for W.
    #    W row index is K*4 + I.
    #    K = 2*k2 + k1, I = 2*i2 + i1.
    #    W_2 row index is k*2 + i.
    #    So W_2 outer row index is k2*2 + i2.
    #    W_2 inner row index is k1*2 + i1.
    #    So W_4[K*4+I, c_outer*7 + c_inner] = W_2[k2*2+i2, c_outer] * W_2[k1*2+i1, c_inner].
    
    #    This looks consistent.
    
    U_4 = np.zeros((16, 49), dtype=int)
    V_4 = np.zeros((16, 49), dtype=int)
    W_4 = np.zeros((16, 49), dtype=int)
    
    for i2 in range(2):
        for i1 in range(2):
            I = 2 * i2 + i1
            for j2 in range(2):
                for j1 in range(2):
                    J = 2 * j2 + j1
                    r_U_outer = i2 * 2 + j2
                    r_U_inner = i1 * 2 + j1
                    for k2 in range(2):
                        for k1 in range(2):
                            K = 2 * k2 + k1
                            r_V_outer = j2 * 2 + k2
                            r_V_inner = j1 * 2 + k1
                            r_W_outer = k2 * 2 + i2
                            r_W_inner = k1 * 2 + i1
                            
                            for c_outer in range(7):
                                for c_inner in range(7):
                                    c = c_outer * 7 + c_inner
                                    val_U = U_2[r_U_outer, c_outer] * U_2[r_U_inner, c_inner]
                                    val_V = V_2[r_V_outer, c_outer] * V_2[r_V_inner, c_inner]
                                    val_W = W_2[r_W_outer, c_outer] * W_2[r_W_inner, c_inner]
                                    
                                    U_4[I * 4 + J, c] += val_U
                                    V_4[J * 4 + K, c] += val_V
                                    W_4[K * 4 + I, c] += val_W
    
    # The entries are integers, so we can return them as lists of lists of integers.
    return {
        "U": U_4.tolist(),
        "V": V_4.tolist(),
        "W": W_4.tolist()
    }
# EVOLVE-BLOCK-END
