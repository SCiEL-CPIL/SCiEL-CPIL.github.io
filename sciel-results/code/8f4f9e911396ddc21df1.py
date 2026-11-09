"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = 120.0 - 5.0  # 5 second safety margin

    n, m, p = DIMENSIONS
    
    # Schoolbook rank is n*m*p = 120
    rank_schoolbook = n * m * p
    
    # Initialize best with schoolbook
    U_best = [[0] * rank_schoolbook for _ in range(n * m)]
    V_best = [[0] * rank_schoolbook for _ in range(m * p)]
    W_best = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_best[i * m + j][col] = 1
                V_best[j * p + k][col] = 1
                W_best[k * n + i][col] = 1
                col += 1
    best_rank = rank_schoolbook

    # Strassen-based block decomposition strategy
    # We partition indices into blocks and use Strassen's algorithm for 2x2x2 blocks.
    # n=4, m=5, p=6.
    # We can partition n into 2 blocks of 2.
    # We can partition m into 2 blocks of 2, leaving 1 remainder.
    # We can partition p into 3 blocks of 2.
    
    # Let's define the Strassen terms for 2x2x2.
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = AB
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 + M7
    
    # We represent the decomposition as a sum of rank-1 terms.
    # Each Strassen term M_r corresponds to a "virtual" multiplication.
    # However, M_r is a sum of products of A and B elements.
    # For example, M1 = A00B00 + A00B11 + A11B00 + A11B11.
    # This means M1 itself is not a single rank-1 term in the tensor decomposition sense if we look at the global indices,
    # BUT in the context of the Strassen algorithm, we compute M1 as a scalar (or sub-matrix) product.
    # The tensor decomposition of the *block* multiplication using Strassen has rank 7 for the block.
    # This implies that the tensor for the 2x2x2 block can be decomposed into 7 rank-1 tensors.
    
    # Let's construct the 7 rank-1 tensors for a 2x2x2 block explicitly.
    # Let the local indices be i_l, j_l, k_l in {0,1}.
    # The tensor T[i_l, j_l, k_l] = 1 for all i_l, j_l, k_l? No.
    # The multiplication tensor for 2x2x2 is T = sum_{i,j,k} e_i \otimes e_j \otimes e_{i,k} ?
    # No, the standard tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes f_{i,k} where f_{i,k} is the basis for C_{i,k}.
    # In our flat indexing: U[i*m+j], V[j*p+k], W[k*n+i].
    # For a 2x2x2 block, if we map local (i,j,k) to global, the structure is preserved.
    
    # The 7 rank-1 components of the 2x2x2 multiplication tensor are given by the Strassen coefficients.
    # Let's define them as triples (u_vec, v_vec, w_vec) where u_vec is length 4 (2x2 A), v_vec length 4 (2x2 B), w_vec length 4 (2x2 C).
    # Note: In our global system, W is indexed by (k, i). In the block, C is (i, k).
    # So local W index is k*2 + i.
    
    # Let's derive the 7 vectors.
    # We need sum_{r=1}^7 u_r[i,j] v_r[j,k] w_r[k,i] = 1 if (i,j,k) is a valid term?
    # Actually, the identity is:
    # sum_{i,j,k} A[i,j] B[j,k] C_basis[i,k] = sum_{r} (sum_{i,j} u_r[i,j] A[i,j]) (sum_{j,k} v_r[j,k] B[j,k]) (sum_{k,i} w_r[k,i] C_basis[i,k])?
    # No, the tensor decomposition T = sum_r u_r \otimes v_r \otimes w_r means:
    # T[i,j,k] = sum_r u_r[i,j] v_r[j,k] w_r[k,i]
    # For matrix multiplication, T[i,j,k] is 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # In the standard basis, the tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_{i,k} (where e_{i,k} is the basis vector for C_{i,k}).
    # Wait, the third index in the tensor is the output index.
    # If we view C as a vector of length n*p, indexed by (i,k) -> i*p+k? Or k*n+i?
    # The problem states: "output factor follows the official trace convention W[k*n+i,r]".
    # So the output index is k*n + i.
    # Thus, the tensor element T[i, j, k_out] where k_out corresponds to (k, i)?
    # Let's re-read carefully: "W[k*n+i,r]".
    # The tensor equation is:
    # C[k*n+i] = sum_{j} A[i*m+j] B[j*p+k] ?
    # No, standard matrix multiplication: C[i, k] = sum_j A[i, j] B[j, k].
    # If C is stored as C[k*n + i], then C[k*n+i] corresponds to row i, column k.
    # So the tensor T has indices (i, j, output_idx).
    # T[i, j, k*n+i] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # Since C[i,k] = sum_j A[i,j]B[j,k], every term A[i,j]B[j,k] contributes to C[i,k] with coefficient 1.
    # So T[i, j, k*n+i] = 1 for all i, j, k.
    # And T is zero otherwise.
    
    # So we need to find rank-1 decompositions of this tensor.
    # For a 2x2x2 block (local indices i', j', k' in {0,1}):
    # T_local[i', j', k'*2+i'] = 1.
    
    # Let's define the 7 Strassen vectors for this local tensor.
    # u_r is a vector of length 4 (indices 0..3 corresponding to i'*2+j').
    # v_r is a vector of length 4 (indices 0..3 corresponding to j'*2+k').
    # w_r is a vector of length 4 (indices 0..3 corresponding to k'*2+i').
    
    # Let's compute them.
    # M1 = (A00+A11)(B00+B11)
    # u_1: A00=1, A11=1 -> [1, 0, 0, 1] (indices: 00, 01, 10, 11)
    # v_1: B00=1, B11=1 -> [1, 0, 0, 1] (indices: 00, 01, 10, 11)
    # w_1: Contributes to C00 and C11.
    # C00 = M1 - M4 + M6 - M7. So w_1[0] (C00) = 1.
    # C11 = M1 - M3 + M5 + M7. So w_1[3] (C11) = 1.
    # w_1: [1, 0, 0, 1] (indices: 00, 01, 10, 11)
    
    # M2 = (A10+A11)B00
    # u_2: A10=1, A11=1 -> [0, 0, 1, 1]
    # v_2: B00=1 -> [1, 0, 0, 0]
    # w_2: Contributes to C10.
    # C10 = M2 + M4. So w_2[2] (C10) = 1.
    # w_2: [0, 0, 1, 0]
    
    # M3 = A00(B10-B11)
    # u_3: A00=1 -> [1, 0, 0, 0]
    # v_3: B10=1, B11=-1 -> [0, 0, 1, -1]
    # w_3: Contributes to C01 and C11.
    # C01 = M3 + M5. So w_3[1] (C01) = 1.
    # C11 = M1 - M3 + M5 + M7. So w_3[3] (C11) = -1.
    # w_3: [0, 1, 0, -1]
    
    # M4 = A11(B01-B00)
    # u_4: A11=1 -> [0, 0, 0, 1]
    # v_4: B01=1, B00=-1 -> [-1, 1, 0, 0]
    # w_4: Contributes to C00 and C10.
    # C00 = M1 - M4 + M6 - M7. So w_4[0] (C00) = -1.
    # C10 = M2 + M4. So w_4[2] (C10) = 1.
    # w_4: [-1, 0, 1, 0]
    
    # M5 = (A00+A01)B11
    # u_5: A00=1, A01=1 -> [1, 1, 0, 0]
    # v_5: B11=1 -> [0, 0, 0, 1]
    # w_5: Contributes to C01 and C11.
    # C01 = M3 + M5. So w_5[1] (C01) = 1.
    # C11 = M1 - M3 + M5 + M7. So w_5[3] (C11) = 1.
    # w_5: [0, 1, 0, 1]
    
    # M6 = (A10-A00)(B00+B01)
    # u_6: A10=1, A00=-1 -> [-1, 0, 1, 0]
    # v_6: B00=1, B01=1 -> [1, 1, 0, 0]
    # w_6: Contributes to C00.
    # C00 = M1 - M4 + M6 - M7. So w_6[0] (C00) = 1.
    # w_6: [1, 0, 0, 0]
    
    # M7 = (A01-A11)(B10+B11)
    # u_7: A01=1, A11=-1 -> [0, 1, 0, -1]
    # v_7: B10=1, B11=1 -> [0, 0, 1, 1]
    # w_7: Contributes to C00 and C11.
    # C00 = M1 - M4 + M6 - M7. So w_7[0] (C00) = -1.
    # C11 = M1 - M3 + M5 + M7. So w_7[3] (C11) = 1.
    # w_7: [-1, 0, 0, 1]
    
    strassen_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    strassen_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_w = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 0, 0],
        [-1, 0, 0, 1]
    ]
    
    # Verify Strassen decomposition for 2x2x2
    # T[i,j,k] = sum_r u_r[i*2+j] v_r[j*2+k] w_r[k*2+i]
    # Should be 1 for all i,j,k in {0,1}
    
    def verify_strassen():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    val = 0
                    for r in range(7):
                        val += strassen_u[r][i*2+j] * strassen_v[r][j*2+k] * strassen_w[r][k*2+i]
                    if val != 1:
                        return False
        return True
    
    if not verify_strassen():
        # Fallback to schoolbook if Strassen is broken (should not happen)
        return {"U": U_best, "V": V_best, "W": W_best}
    
    # Now construct the global decomposition.
    # We partition:
    # i: 0..3 into blocks of 2: bi=0 (i=0,1), bi=1 (i=2,3)
    # j: 0..4 into blocks of 2: bj=0 (j=0,1), bj=1 (j=2,3), remainder j=4
    # k: 0..5 into blocks of 2: bk=0 (k=0,1), bk=1 (k=2,3), bk=2 (k=4,5)
    
    # The main part covers i in {0,1,2,3}, j in {0,1,2,3}, k in {0,1,2,3,4,5}.
    # Rank for this part: 2 (i_blocks) * 2 (j_blocks) * 3 (k_blocks) * 7 (Strassen) = 84.
    
    # The remainder covers j=4.
    # For j=4, we have terms A[i,4] B[4,k] contributing to C[i,k].
    # This is a 4x1x6 sub-tensor.
    # Rank of this sub-tensor is 4*6 = 24 (schoolbook).
    # Total rank = 84 + 24 = 108.
    
    total_rank = 84 + 24
    
    if total_rank >= best_rank:
        return {"U": U_best, "V": V_best, "W": W_best}
    
    U_new = np.zeros((n * m, total_rank), dtype=int)
    V_new = np.zeros((m * p, total_rank), dtype=int)
    W_new = np.zeros((p * n, total_rank), dtype=int)
    
    col = 0
    
    # Add Strassen blocks
    for bi in range(2):
        for bj in range(2):
            for bk in range(3):
                for r in range(7):
                    # Local indices
                    # i_l in {0,1} -> i_g = 2*bi + i_l
                    # j_l in {0,1} -> j_g = 2*bj + j_l
                    # k_l in {0,1} -> k_g = 2*bk + k_l
                    
                    # Update U
                    for i_l in range(2):
                        for j_l in range(2):
                            i_g = 2 * bi + i_l
                            j_g = 2 * bj + j_l
                            U_new[i_g * m + j_g, col] += strassen_u[r][i_l * 2 + j_l]
                            
                    # Update V
                    for j_l in range(2):
                        for k_l in range(2):
                            j_g = 2 * bj + j_l
                            k_g = 2 * bk + k_l
                            V_new[j_g * p + k_g, col] += strassen_v[r][j_l * 2 + k_l]
                            
                    # Update W
                    # W is indexed by k*n + i.
                    # Local W index is k_l * 2 + i_l.
                    # Global W index is k_g * n + i_g.
                    for k_l in range(2):
                        for i_l in range(2):
                            k_g = 2 * bk + k_l
                            i_g = 2 * bi + i_l
                            W_new[k_g * n + i_g, col] += strassen_w[r][k_l * 2 + i_l]
                            
                    col += 1
    
    # Add remainder for j=4
    # j_g = 4
    for i in range(n):
        for k in range(p):
            U_new[i * m + 4, col] = 1
            V_new[4 * p + k, col] = 1
            W_new[k * n + i, col] = 1
            col += 1
    
    # Verify the new decomposition
    def verify_decomp(U, V, W, n, m, p, tol=1e-9):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = len(U_arr[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    if verify_decomp(U_new, V_new, W_new, n, m, p):
        return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    else:
        # If verification fails, return schoolbook
        return {"U": U_best, "V": V_best, "W": W_best}

# EVOLVE-BLOCK-END
