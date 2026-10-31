import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin well under 120s

    n, m, p = DIMENSIONS
    
    # Strassen 2x2 coefficients (7 terms each for A, B, C linear forms)
    # These are the standard Strassen bilinear forms for 2x2 matrices
    a_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0: A00 + A11
        [[0, 0], [1, 1]],   # s=1: A10 + A11
        [[1, 0], [0, 0]],   # s=2: A00
        [[0, 0], [0, 1]],   # s=3: A11
        [[1, 1], [0, 0]],   # s=4: A00 + A01
        [[-1, 0], [1, 0]],  # s=5: A10 - A00
        [[0, 1], [0, -1]],  # s=6: A01 - A11
    ], dtype=int)

    b_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0: B00 + B11
        [[1, 0], [0, 0]],   # s=1: B00
        [[0, 1], [0, -1]],  # s=2: B01 - B11
        [[-1, 0], [1, 0]],  # s=3: B10 - B00
        [[0, 0], [0, 1]],   # s=4: B11
        [[1, 1], [0, 0]],   # s=5: B00 + B01
        [[0, 0], [1, 1]],   # s=6: B10 + B11
    ], dtype=int)

    c_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0: C00 + C11
        [[0, 0], [1, -1]],  # s=1: C10 - C11
        [[0, 1], [0, 1]],   # s=2: C01 + C11
        [[1, 0], [1, 0]],   # s=3: C00 + C10
        [[-1, 1], [0, 0]],  # s=4: -C00 + C01
        [[0, 0], [0, 1]],   # s=5: C11
        [[1, 0], [0, 0]],   # s=6: C00
    ], dtype=int)

    rank = 49  # 7 * 7
    
    # We construct U, V, W using the tensor product structure.
    # The indices for the 4x4 matrix are decomposed into (block_index, inner_index).
    # i = 2*S + i2, j = 2*T + j2, etc.
    
    # Precompute index mappings
    # For U: row = i*4 + j = (2S + i2)*4 + (2T + j2) = 8S + 2T + 4i2 + j2
    # For V: row = j*4 + k = (2T + j2)*4 + (2K + k2) = 8T + 2K + 4j2 + k2
    # For W: row = k*4 + i = (2K + k2)*4 + (2S + i2) = 8K + 2S + 4k2 + i2
    
    # Shapes:
    # a_coeffs: (7, 2, 2) -> (s, S, T)
    # b_coeffs: (7, 2, 2) -> (s, T, K)
    # c_coeffs: (7, 2, 2) -> (s, S, K)
    
    # We want to fill U[row, r] where r = 7*s + t
    # U[row, r] = sum over S,T,i2,j2 of a_coeffs[s,S,T] * a_coeffs[t,i2,j2] 
    #             where row = 8S + 2T + 4i2 + j2
    #
    # This is essentially a Kronecker product of the coefficient matrices with respect to the index structure.
    
    # Let's build the full coefficient tensors for the 4x4 case directly.
    # A_4x4 linear form for term r=(s,t) is:
    # L_A[i,j] = sum_{S,T,i2,j2} a_s[S,T] * a_t[i2,j2] * delta_{i, 2S+i2} * delta_{j, 2T+j2}
    # This can be computed by broadcasting.
    
    # a_coeffs shape: (7, 2, 2)
    # We need to expand this to (7, 4, 4)
    # a_full[s, i, j] = a_coeffs[s, i//2, i%2] * a_coeffs[s, j//2, j%2] ? No.
    # The tensor product of two linear forms L1 and L2 on 2x2 matrices gives a linear form on 4x4.
    # If L1 = sum a_ij A_ij and L2 = sum b_kl B_kl, then L1 * L2 is not a linear form.
    # Wait, the bilinear form for matrix mult is sum_r U_r(A) V_r(B) W_r(C).
    # U_r(A) is a linear form in A.
    # For the tensor product decomposition:
    # U_{(s,t)}(A) = U_s(A_block) * U_t(A_inner)? No, that's not right either.
    #
    # Let's look at the loop in the previous code:
    # U[row_u, c_idx] += outer_a * inner_a
    # where outer_a = a_coeffs[s][S][T] and inner_a = a_coeffs[t][i2][j2]
    # and row_u corresponds to i=2S+i2, j=2T+j2.
    #
    # This means U_{s,t}[i,j] = a_s[S,T] * a_t[i2,j2] where i=2S+i2, j=2T+j2.
    # This is the element-wise product of the "block" coefficient matrix and the "inner" coefficient matrix,
    # but mapped to the 4x4 grid.
    
    # Let's construct U_full of shape (7, 7, 4, 4)
    # U_full[s, t, i, j] = a_coeffs[s, i//2, i%2] * a_coeffs[t, j//2, j%2]
    # Wait, the previous code uses:
    # outer_a = a_coeffs[s][S][T]
    # inner_a = a_coeffs[t][i2][j2]
    # i = 2*S + i2
    # j = 2*T + j2
    #
    # So U[s,t,i,j] = a_coeffs[s, i//2, i%2] * a_coeffs[t, j//2, j%2] is INCORRECT based on the code.
    # The code says:
    # S = i//2, i2 = i%2
    # T = j//2, j2 = j%2
    # U[s,t,i,j] += a_coeffs[s, S, T] * a_coeffs[t, i2, j2]
    #
    # So U[s,t,i,j] = a_coeffs[s, i//2, j//2] * a_coeffs[t, i%2, j%2]
    
    # Let's verify this interpretation.
    # a_coeffs[s] is a 2x2 matrix M_s.
    # M_s[S,T] is the coefficient for A[S,T] in the s-th Strassen term.
    # The 4x4 matrix A has blocks.
    # The term (s,t) corresponds to the product of the s-th linear form on the "outer" 2x2 block structure
    # and the t-th linear form on the "inner" 2x2 elements?
    #
    # Actually, the matrix multiplication of block matrices:
    # C = A * B
    # C_{ij} = sum_k A_{ik} B_{kj}
    # If we group indices into (S, i2) and (T, j2):
    # C_{S i2, T j2} = sum_{K, k2} A_{S i2, K k2} B_{K k2, T j2}
    #
    # The Strassen decomposition for 2x2 says:
    # C_{S,T} = sum_s M_s[S,T] * (sum_{S',T'} a_s[S',T'] A_{S',T'}) * (sum_{K',K''} b_s[K',K''] B_{K',K''}) ... this is getting confusing.
    
    # Let's stick to the code's logic which was verified to work (score 1.3061).
    # The code logic:
    # U[row, r] = sum_{S,T,i2,j2} (if i=2S+i2, j=2T+j2) a_s[S,T] * a_t[i2,j2]
    # This is exactly:
    # U[s,t] (as a 4x4 matrix) = np.kron(a_coeffs[s], a_coeffs[t]) ?
    # Let's check np.kron(A, B) for 2x2 A and 2x2 B.
    # kron(A, B)[2i + i', 2j + j'] = A[i, j] * B[i', j']
    #
    # In our case:
    # i = 2S + i2 -> S = i//2, i2 = i%2
    # j = 2T + j2 -> T = j//2, j2 = j%2
    #
    # Code: U[s,t,i,j] = a_coeffs[s, S, T] * a_coeffs[t, i2, j2]
    # Kron: U[s,t,i,j] = a_coeffs[s, S, T] * a_coeffs[t, i2, j2]
    #
    # YES! It is exactly the Kronecker product.
    
    # So:
    # U_full = np.einsum('sij,tij->stij', a_coeffs, a_coeffs) ? No.
    # U[s,t] = np.kron(a_coeffs[s], a_coeffs[t])
    
    U_4d = np.zeros((7, 7, 4, 4), dtype=int)
    V_4d = np.zeros((7, 7, 4, 4), dtype=int)
    W_4d = np.zeros((7, 7, 4, 4), dtype=int)
    
    for s in range(7):
        for t in range(7):
            U_4d[s, t] = np.kron(a_coeffs[s], a_coeffs[t])
            V_4d[s, t] = np.kron(b_coeffs[s], b_coeffs[t])
            W_4d[s, t] = np.kron(c_coeffs[s], c_coeffs[t])
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Now flatten to (16, 49)
    # U[row, r] where row = i*4+j, r = 7*s+t
    # U_4d shape (7,7,4,4). We want U_flat[row, r].
    # row = i*4+j.
    # U_4d[s,t,i,j] maps to U_flat[i*4+j, 7*s+t].
    
    # Reshape U_4d to (7, 7, 16) by flattening the last two dims in row-major order (i then j).
    # U_4d[s,t] is 4x4. Flattening gives index i*4+j.
    # So U_flat[s,t, row] = U_4d[s,t,i,j]
    
    U_flat = U_4d.reshape(7, 7, 16)
    # We want shape (16, 49).
    # Current: (s, t, row).
    # Target: (row, s*7+t).
    
    # Permute to (row, s, t)
    U_perm = np.transpose(U_flat, (2, 0, 1))
    # Reshape to (16, 49)
    U = U_perm.reshape(16, 49)
    
    V_flat = V_4d.reshape(7, 7, 16)
    V_perm = np.transpose(V_flat, (2, 0, 1))
    V = V_perm.reshape(16, 49)
    
    W_flat = W_4d.reshape(7, 7, 16)
    W_perm = np.transpose(W_flat, (2, 0, 1))
    W = W_perm.reshape(16, 49)
    
    # The task requires W[k*n+i, r].
    # In our construction, W_4d[s,t] is a 4x4 matrix C.
    # C[i,j] corresponds to output element at row i, col j.
    # The tensor convention is C[i,k] (row i, col k).
    # The evaluator expects W indexed by [k*n + i].
    # Wait, the task says: "output factor follows the official trace convention W[k*n+i,r]"
    # Usually C[i,k] = sum_j A[i,j]B[j,k].
    # The output matrix C has rows i and cols k.
    # The linear form for C is usually associated with the output index (i,k).
    # However, the prompt says W[k*n+i]. This implies the index is (k, i).
    # Let's check the previous code's W construction.
    # Previous code:
    # row_w = k * 4 + i
    # W[row_w, c_idx] += ...
    # So the previous code *did* use W[k*n+i].
    #
    # In my Kronecker construction:
    # W_4d[s,t] = kron(c_coeffs[s], c_coeffs[t]).
    # c_coeffs[s] is a 2x2 matrix M_c.
    # M_c[S,K] is the coefficient for C[S,K] in the s-th term.
    # So W_4d[s,t,i,j] = c_coeffs[s, i//2, j//2] * c_coeffs[t, i%2, j%2].
    # Here i is the row of C, j is the col of C.
    # So W_4d[s,t,i,j] corresponds to C[i,j].
    # The index in the W matrix should be based on the output index.
    # If the convention is W[k*n+i], then the "row" of W corresponds to the pair (k,i).
    # In standard matrix mult, C[i,k]. So k is col, i is row.
    # So the index is (col, row).
    # My W_4d has indices (row, col).
    # So I need to transpose the 4x4 matrix before flattening, or map indices carefully.
    
    # Let's re-verify the "official trace convention".
    # Often in tensor decompositions, the output tensor is indexed by (i, k).
    # If the vector is flattened as [i, k], the index is i*p + k.
    # If the vector is flattened as [k, i], the index is k*n + i.
    # The prompt explicitly says W[k*n+i].
    # My W_4d[s,t] has shape (4,4) where first dim is row (i) and second is col (k).
    # So W_4d[s,t, i, k].
    # I need to map this to index idx = k*4 + i.
    
    # So I should transpose the 4x4 matrix to (k, i) before flattening.
    W_4d_transposed = np.transpose(W_4d, (0, 1, 3, 2)) # (s, t, k, i)
    
    W_flat_t = W_4d_transposed.reshape(7, 7, 16)
    W_perm_t = np.transpose(W_flat_t, (2, 0, 1))
    W = W_perm_t.reshape(16, 49)
    
    # U and V do not need transposition because:
    # U index: i*m + j. i is row, j is col. My U_4d is (i, j). Correct.
    # V index: j*p + k. j is row, k is col. My V_4d is (j, k). Correct.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
