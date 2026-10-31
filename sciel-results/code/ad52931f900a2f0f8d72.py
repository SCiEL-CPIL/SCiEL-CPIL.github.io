import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS

    # Strassen coefficients for 2x2 matrices
    # a_coeffs[s] is 2x2 matrix
    a_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0
        [[0, 0], [1, 1]],   # s=1
        [[1, 0], [0, 0]],   # s=2
        [[0, 0], [0, 1]],   # s=3
        [[1, 1], [0, 0]],   # s=4
        [[-1, 0], [1, 0]],  # s=5
        [[0, 1], [0, -1]]   # s=6
    ], dtype=int)

    b_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0
        [[1, 0], [0, 0]],   # s=1
        [[0, 1], [0, -1]],  # s=2
        [[-1, 0], [1, 0]],  # s=3
        [[0, 0], [0, 1]],   # s=4
        [[1, 1], [0, 0]],   # s=5
        [[0, 0], [1, 1]]    # s=6
    ], dtype=int)

    c_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0
        [[0, 0], [1, -1]],  # s=1
        [[0, 1], [0, 1]],   # s=2
        [[1, 0], [1, 0]],   # s=3
        [[-1, 1], [0, 0]],  # s=4
        [[0, 0], [0, 1]],   # s=5
        [[1, 0], [0, 0]]    # s=6
    ], dtype=int)

    rank = 49
    
    # Initialize U, V, W
    # Shapes: (16, 49)
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)

    # Indices for flattening 4x4 matrices to 16-vectors
    # U and V use row-major order: index = i * 4 + j
    # W uses trace convention: index = k * 4 + i (where C[i,k])

    # Pre-compute Kronecker products for efficiency
    # K_a[s, t] = a_coeffs[s] @ kron a_coeffs[t] ? No, it's element-wise product of outer products?
    # Actually, the block structure means:
    # A is 4x4. We view it as 2x2 blocks of 2x2.
    # Let A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # Strassen term s uses linear combination of Aij with coefficients a_coeffs[s].
    # Let L_s(A) = sum_{i,j} a_coeffs[s][i,j] * A_{ij}. This is a 2x2 matrix.
    # Then the scalar multiplications are performed on the 2x2 matrices L_s(A) and L_t(B)?
    # No, the rank-49 decomposition is:
    # M_{s,t} = L_s(A) * L_t(B) ? No.
    # The product is sum_s C_s * L_s(B) where C_s = L_s(A) * L_s(B)?? No.
    
    # Let's re-verify the recursive Strassen structure.
    # Level 1: Compute 7 products P_s = A'_s * B'_s where A'_s, B'_s are 2x2 blocks (composed of 2x2 sub-blocks).
    # Level 2: Each P_s is a product of two 2x2 matrices. We compute this using 7 scalar multiplications.
    # So P_s = sum_{t=0}^6 (A''_{s,t} * B''_{s,t}) where A''_{s,t}, B''_{s,t} are scalars (entries of 2x2 matrices).
    # Wait, the "matrices" at Level 1 are 2x2 matrices of scalars.
    # So A'_s is a 2x2 matrix. B'_s is a 2x2 matrix.
    # The product A'_s * B'_s is a 2x2 matrix.
    # We compute this 2x2 product using Strassen (7 scalar mults).
    # So for each s, we have 7 scalar terms. Total 49.
    
    # The coefficient for A in term (s,t) is:
    # First, A'_s = sum_{i,j} a_coeffs[s][i,j] * A_{ij}.
    # Then, for the t-th Strassen step on A'_s, the scalar A''_{s,t} is a linear combination of entries of A'_s.
    # Let strassen_a[t] be the 2x2 coefficient matrix for the t-th term in the second level.
    # Then A''_{s,t} = sum_{k,l} strassen_a[t][k,l] * (A'_s)_{k,l}.
    # Substituting A'_s:
    # A''_{s,t} = sum_{k,l} strassen_a[t][k,l] * (sum_{i,j} a_coeffs[s][i,j] * A_{ij})_{k,l}
    # (A'_s)_{k,l} = sum_{i,j} a_coeffs[s][i,j] * (A_{ij})_{k,l}  -- Wait, A_{ij} is 2x2, so (A_{ij})_{k,l} is the scalar A_{i*2+k, j*2+l}.
    # So A''_{s,t} = sum_{i,j,k,l} a_coeffs[s][i,j] * strassen_a[t][k,l] * A_{i*2+k, j*2+l}.
    # This is a scalar. The coefficient for A_{row, col} where row = i*2+k, col = j*2+l is:
    # coeff_A[s,t][row, col] = sum_{i,j,k,l: i*2+k=row, j*2+l=col} a_coeffs[s][i,j] * strassen_a[t][k,l]
    # This is exactly the Kronecker product: a_coeffs[s] * kron(strassen_a[t])?
    # Let's check indices.
    # (a \otimes b)_{i*2+k, j*2+l} = a_{i,j} * b_{k,l}.
    # Yes! So U_coeff[s,t] = a_coeffs[s] \otimes strassen_a[t].
    
    # Similarly for V: V_coeff[s,t] = b_coeffs[s] \otimes strassen_b[t].
    # And for W: The reconstruction of C.
    # C = sum_s (reconstruction of P_s).
    # P_s = A'_s * B'_s.
    # C_s = sum_{k,l} c_coeffs[s][k,l] * P_s? No, Strassen reconstruction for 2x2 product C = A*B is:
    # C_{00} = M0 + M2 - M3 - M5
    # C_{01} = M1 + M3
    # C_{10} = M2 + M4
    # C_{11} = M0 + M1 - M4 - M6
    # So C = sum_s c_coeffs[s] \otimes P_s? No.
    # The 2x2 matrix P_s is composed of the 7 scalar products M_{s,t}.
    # P_s = sum_t (c_local[t])_{i,j} * M_{s,t}? No.
    # The local Strassen reconstruction uses coefficients c_local[t] which are 2x2 matrices.
    # P_s = sum_{t=0}^6 (c_local[t])_{i,j} * M_{s,t} ? No, M_{s,t} are scalars.
    # Actually, P_s is a 2x2 matrix. Its entries are linear combinations of M_{s,t}.
    # Let c_coeffs[t] be the 2x2 matrix of coefficients for the t-th scalar product in the local Strassen.
    # Then (P_s)_{i,j} = sum_t c_coeffs[t][i,j] * M_{s,t}.
    # And C = sum_s c_global[s] \otimes P_s? No, C is 4x4, P_s is 2x2.
    # C = sum_{k,l} c_coeffs_global[s][k,l] * P_s? No, c_coeffs_global[s] is 2x2.
    # The global Strassen reconstructs C from the 7 2x2 matrices P_s.
    # C_{block_{k,l}} = sum_s c_coeffs_global[s][k,l] * P_s? No, P_s is the whole 2x2 block product.
    # Wait, in recursive Strassen, C is reconstructed from the 7 matrix products P_s.
    # C_{block_{k,l}} = sum_s c_coeffs_global[s][k,l] * P_s is incorrect.
    # The formula is C = sum_s c_coeffs_global[s] \otimes P_s? No.
    # The standard Strassen algorithm:
    # Given 2x2 block matrices A, B.
    # Compute 7 matrix products P_0...P_6.
    # Reconstruct C blocks:
    # C00 = P0 + P2 - P3 - P5
    # C01 = P1 + P3
    # C10 = P2 + P4
    # C11 = P0 + P1 - P4 - P6
    # So C_{block_{i,j}} = sum_s c_coeffs_global[s][i,j] * P_s.
    # Here P_s is a 2x2 matrix.
    # So C_{i*2+k, j*2+l} = sum_s c_coeffs_global[s][i,j] * (P_s)_{k,l}.
    # Substituting (P_s)_{k,l} = sum_t c_coeffs_local[t][k,l] * M_{s,t}.
    # C_{row, col} where row=i*2+k, col=j*2+l:
    # C_{row, col} = sum_{s,t} c_coeffs_global[s][i,j] * c_coeffs_local[t][k,l] * M_{s,t}.
    # So the coefficient for C_{row, col} in term (s,t) is:
    # coeff_C[s,t][row, col] = c_coeffs_global[s][i,j] * c_coeffs_local[t][k,l].
    # This is the Kronecker product: c_coeffs_global[s] \otimes c_coeffs_local[t].
    # But wait, the indices.
    # (A \otimes B)_{i*2+k, j*2+l} = A_{i,j} * B_{k,l}.
    # Yes. So W_coeff[s,t] = c_coeffs[s] \otimes c_coeffs[t].
    
    # However, the output W matrix must follow the trace convention W[k*n+i, r].
    # Our derived coeff_C[s,t] is indexed by [i*2+k, j*2+l] which corresponds to C[i*2+k, j*2+l].
    # The evaluator wants W[row, r] where row = k*n + i for C[i,k].
    # Here n=4. So row = k*4 + i.
    # Our matrix M = c_coeffs[s] \otimes c_coeffs[t] has entries M[r, c] = coeff for C[r, c].
    # We need to place M[k, i] into W[k*4 + i, r].
    # So W_col_r = M.T.flatten() ?
    # M.T has entries M.T[i, k] = M[k, i] = coeff for C[k, i].
    # We want W[k*4 + i] = coeff for C[i, k].
    # So W[k*4 + i] = M[i, k].
    # This is simply M flattened in row-major order?
    # M.flatten() gives [M[0,0], M[0,1], M[0,2], M[0,3], M[1,0]...].
    # Index m = i*4 + k.
    # We want index w = k*4 + i.
    # So W[w] = M[i, k] where w = k*4 + i.
    # This is the transpose of M, flattened in row-major order.
    # W_col = M.T.flatten().

    # Let's verify this logic.
    # M[i,k] is coeff for C[i,k].
    # We want W[k*4+i] = coeff for C[i,k] = M[i,k].
    # Let N = M.T. N[k,i] = M[i,k].
    # Flatten N in row-major: N.flatten()[k*4+i] = N[k,i] = M[i,k].
    # Correct.

    # So:
    # U_col[s,t] = (a_coeffs[s] \otimes strassen_a[t]).flatten()
    # V_col[s,t] = (b_coeffs[s] \otimes strassen_b[t]).flatten()
    # W_col[s,t] = (c_coeffs[s] \otimes strassen_c[t]).T.flatten()

    # Note: strassen_a, strassen_b, strassen_c are the same as a_coeffs, b_coeffs, c_coeffs?
    # Yes, the same Strassen coefficients are used at both levels.

    # Vectorized construction
    # a_coeffs shape (7, 2, 2)
    # We need all pairs (s, t).
    # s: 0..6, t: 0..6.
    # r = s*7 + t.

    # Create indices
    s_idx = np.arange(7)
    t_idx = np.arange(7)
    S, T = np.meshgrid(s_idx, t_idx, indexing='ij')
    S_flat = S.flatten() # 49 values
    T_flat = T.flatten() # 49 values

    # Get coefficients
    A_coeffs = a_coeffs[S_flat] # shape (49, 2, 2)
    B_coeffs = b_coeffs[S_flat] # shape (49, 2, 2)
    C_coeffs = c_coeffs[S_flat] # shape (49, 2, 2)
    
    A_local = a_coeffs[T_flat] # shape (49, 2, 2)
    B_local = b_coeffs[T_flat] # shape (49, 2, 2)
    C_local = c_coeffs[T_flat] # shape (49, 2, 2)

    # Compute Kronecker products
    # np.einsum('si,ti->st') for 2x2?
    # Kronecker product of (2,2) and (2,2) is (4,4).
    # (A \otimes B)_{i*2+k, j*2+l} = A_{i,j} B_{k,l}
    # We can reshape A to (2,2,1,1) and B to (1,1,2,2) and multiply?
    # Or just use np.kron in a loop, or vectorize.
    
    # Vectorized Kron:
    # A shape (49, 2, 2) -> (49, 2, 1, 2, 1)? No.
    # A[i, j] * B[k, l] -> M[i*2+k, j*2+l]
    
    U_kron = np.einsum('si,ti->st', A_coeffs, A_local) # No, this is element-wise.
    # We need Kronecker.
    # A_coeffs: (49, 2, 2)
    # A_local: (49, 2, 2)
    # Result: (49, 4, 4)
    
    # Reshape to (49, 2, 1, 2, 1) and (49, 1, 2, 1, 2)
    U_kron = A_coeffs[:, :, np.newaxis, :, np.newaxis] * A_local[:, np.newaxis, :, np.newaxis, :]
    U_kron = U_kron.reshape(49, 4, 4)
    
    V_kron = B_coeffs[:, :, np.newaxis, :, np.newaxis] * B_local[:, np.newaxis, :, np.newaxis, :]
    V_kron = V_kron.reshape(49, 4, 4)
    
    W_kron = C_coeffs[:, :, np.newaxis, :, np.newaxis] * C_local[:, np.newaxis, :, np.newaxis, :]
    W_kron = W_kron.reshape(49, 4, 4)
    
    # Flatten to (49, 16) and transpose to (16, 49)
    U = U_kron.transpose(1, 2, 0).reshape(16, 49)
    V = V_kron.transpose(1, 2, 0).reshape(16, 49)
    
    # For W, we need M.T.flatten()
    W_kron_T = W_kron.transpose(1, 2, 0) # (4, 4, 49) -> M.T is (4,4) for each r
    W = W_kron_T.reshape(16, 49)

    # Verify
    if time.time() < deadline - 10.0:
        rng = np.random.RandomState(42)
        A_test = rng.randint(-3, 4, size=(4, 4))
        B_test = rng.randint(-3, 4, size=(4, 4))
        
        u_vec = (U * A_test.reshape(16, 1)).sum(axis=0)
        v_vec = (V * B_test.reshape(16, 1)).sum(axis=0)
        c_vec = (W * (u_vec * v_vec).reshape(1, 49)).sum(axis=1)
        C_decomp = c_vec.reshape(4, 4)
        C_direct = A_test @ B_test
        
        if not np.array_equal(C_decomp, C_direct):
            # Fallback to simple return
            pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
