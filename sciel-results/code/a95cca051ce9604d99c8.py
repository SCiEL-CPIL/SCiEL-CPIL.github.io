import numpy as np
import time

def construct():
    start_time = time.time()
    
    # Define 2x2 Strassen matrices
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    # Kronecker product for 4x4
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Construct permutation for row-major to standard tensor index mapping
    # The Kronecker product produces indices in a specific order.
    # We need to map them to the standard row-major (i*4+j) for A and B,
    # and column-major (k*4+i) for C as per prompt "W[k*n+i,r]"
    
    # Indices in Kron product:
    # If U2 is indexed by (i2, j2) and U2 by (i2', j2'), 
    # Kron index is i2 * 4 + i2' ... no, standard np.kron(A, B) 
    # A is n x m, B is p x q. Result np x mq.
    # Index (I, J) = A[i, k] * B[j, l] where I = i*m + j, J = k*q + l.
    # Here U2 is 4x7. U4 is 16x49.
    # Row index of U4: (r1, c1) of U2 -> r1*4 + r2.
    # We want Row index (i, j) -> i*4 + j.
    # In Kron, if we map i = i1*2 + i0, j = j1*2 + j0.
    # Then Kron row is i1*4 + i0.
    # We want standard row i*4 + j = (i1*2+i0)*4 + (j1*2+j0) = i1*8 + i0*4 + j1*2 + j0.
    # So we need to permute rows of U4 such that the row for (i1, i0, j1, j0)
    # moves to the position corresponding to i*4+j.
    
    perm = np.zeros(16, dtype=int)
    for S in range(16):
        # S is the target standard index: i*4 + j
        i = S // 4
        j = S % 4
        i1, i0 = divmod(i, 2)
        j1, j0 = divmod(j, 2)
        # S is the target. What is the source index in Kron?
        # Kron row index K_src is based on the ordering of the sub-blocks.
        # np.kron(A, B) puts A[0,0]*B at top left, A[0,1]*B at top right, etc.
        # Row index in Kron result for element (i1, i0) of A and (j1, j0) of B?
        # No, U is one matrix.
        # The rows of U2 are indexed 0..3. Let's call them (r1, r0).
        # U4 row r is formed by U2 row r1, col r0? No.
        # U2 row i1, element at position corresponding to block (i1, i0)?
        # In np.kron(U2, U2), the rows are ordered:
        # U2[0,0]*U2_row, U2[0,1]*U2_row, ...
        # Specifically, row R in U4 corresponds to (i1, i0) where R = i1*4 + i0.
        # Wait, U2 is 4x7. U2 row 0, 1, 2, 3.
        # Let's say U2 row index k corresponds to (k//2, k%2).
        # Then U4 row index K corresponds to (K//4, K%4) -> (i1, i0).
        # But we want the row corresponding to standard (i, j).
        # This implies we are mixing the "row" indices of the two 2x2 levels.
        # Actually, U4 row index K = i1*4 + i0.
        # This K does NOT depend on j1, j0.
        # BUT, U4 also has columns.
        # The decomposition is for the TENSOR.
        # The Kronecker product of the tensors T_2 \otimes T_2 = T_4.
        # The indices of T_4 are (i, j, k).
        # i = i1*2 + i0. j = j1*2 + j0. k = k1*2 + k0.
        # The rank-49 decomposition sums over r1, r2 in 0..6.
        # U_4[i, r1, r2] = U_2[i1, r1] * U_2[i0, r2].
        # The matrix U4 has rows indexed by i (0..15) and cols by (r1, r2) -> 0..48.
        # In numpy, U4[i, R] where R = r1*7 + r2.
        # We just need to ensure that the row i in U4 corresponds to the element A_{i, j}?
        # NO. U is the matrix for Input A.
        # A is flattened to 16 elements.
        # U is 16x49.
        # U[row_A, col_rank] = coefficient.
        # In the Kronecker construction, U4[row, col] = U2[row_1, col_1] * U2[row_2, col_2].
        # row_1, row_2 come from the decomposition of the index of A.
        # Index of A: a_{ij}.
        # In T_2 \otimes T_2, the index i is split into i1, i0.
        # But j is ALSO part of the index of A?
        # NO. A is 4x4. Flattened a_{00}, a_{01}, a_{10}, a_{11}.
        # T_4 = sum_{i,j,k} A_{ij} B_{jk} C_{ki} e_i \otimes e_j \otimes e_k?
        # Standard Tensor: $M = \sum A_{ij} B_{jk} C_{ki}$.
        # The Kronecker product works on the indices.
        # The "Row" of U corresponds to the pair (i, j) of A?
        # NO. The tensor is $A \otimes B \otimes C$.
        # The multiplication is $\langle U, A \rangle \langle V, B \rangle \to W C$.
        # $C = \sum_r (\sum_{i,j} U_{ij,r} A_{ij}) (\sum_{j,k} V_{jk,r} B_{jk}) W_r$.
        # So U is indexed by (i, j).
        # In Kronecker: U_4[(i1,i0, j1,j0), r1, r2] = U_2[(i1,j1), r1] * U_2[(i0,j0), r2].
        # Here, the index into U_2 is a PAIR of indices from the two levels.
        # So we need a permutation that maps the flattened standard index $S = i*4+j$
        # to the index used in the Kronecker construction $K = (i1,j1)*4 + (i0,j0)$?
        # Let's check:
        # i = i1*2 + i0. j = j1*2 + j0.
        # S = i*4 + j = (i1*2+i0)*4 + j1*2 + j0 = i1*8 + i0*4 + j1*2 + j0.
        # K (Kron index) = Index of U2 row for level 1, Index of U2 row for level 2.
        # U2 row index for level 1 (top-left 2x2 block of indices) corresponds to (i1, j1).
        # U2 row index for level 2 (bottom-right 2x2 block) corresponds to (i0, j0).
        # So K = (i1*2+j1)*4 + (i0*2+j0) ?
        # Wait, U2 is 4x7. The rows of U2 are indexed 0..3.
        # Let's assume U2 row p corresponds to (p//2, p%2).
        # Then for level 1, we need U2 row index P1 = i1*2 + j1.
        # For level 2, we need U2 row index P2 = i0*2 + j0.
        # The Kronecker product U4 = np.kron(U2, U2) has rows indexed by P1*4 + P2.
        # So K = P1*4 + P2.
        
        P1 = i1 * 2 + j1
        P2 = i0 * 2 + j0
        K = P1 * 4 + P2
        perm[S] = K
        
    U4_fixed = U4[perm, :]
    V4_fixed = V4[perm, :]
    
    # W handling
    # W is for C. Index is k*n + i = k*4 + i.
    # C = sum_r (U A) (V B) W_r.
    # W is 16x49. W[row_C, col_rank].
    # row_C corresponds to index of C element.
    # Standard C element is c_{ik}. Index = i*4 + k.
    # But prompt says W[k*n+i] -> k*4 + i.
    # So the "row" of W in the output matrix corresponds to C_{ik} but ordered by k*4+i.
    # In Kronecker: W_4[(k1,i1, k0,i0), r1, r2] = W_2[(k1,i1), r1] * W_2[(k0,i0), r2].
    # Level 1 index: Q1 = k1*2 + i1.
    # Level 2 index: Q2 = k0*2 + i0.
    # K = Q1*4 + Q2.
    # Target Index S_c = k*4 + i.
    # k = k1*2 + k0. i = i1*2 + i0.
    # S_c = (k1*2+k0)*4 + (i1*2+i0) = k1*8 + k0*4 + i1*2 + i0.
    # We need perm_c[S_c] = K.
    
    perm_w = np.zeros(16, dtype=int)
    for S_c in range(16):
        k = S_c // 4
        i = S_c % 4
        k1, k0 = divmod(k, 2)
        i1, i0 = divmod(i, 2)
        Q1 = k1 * 2 + i1
        Q2 = k0 * 2 + i0
        K = Q1 * 4 + Q2
        perm_w[S_c] = K
        
    W4_fixed = W4[perm_w, :]
    
    return {
        "U": U4_fixed.tolist(),
        "V": V4_fixed.tolist(),
        "W": W4_fixed.tolist()
    }
