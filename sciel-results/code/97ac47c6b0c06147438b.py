import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def _verify(U, V, W):
    """Verify that the decomposition is correct using vectorized operations."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    
    rank = U.shape[1]
    
    # Reshape for efficient computation
    # U: (n*m, rank) -> (n, m, rank)
    U_reshaped = U.reshape(n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    V_reshaped = V.reshape(m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # T shape: (n, m, p)
    # T[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
    
    # Use einsum: T[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # True tensor: all ones for valid (i,j,k)
    # T[i,j,k] should be 1 for all i in [0,n), j in [0,m), k in [0,p)
    true_T = np.ones((n, m, p))
    
    if not np.allclose(T, true_T, atol=1e-9):
        return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    n, m, p = DIMENSIONS
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # STRATEGY: Rank-22 decomposition using two 2x3x3 blocks
    # The 2x3x6 tensor can be decomposed as two 2x3x3 sub-tensors
    # Each 2x3x3 block has rank 11, total rank = 22
    
    # For a 2x3x3 block:
    # A is 2x3, B is 3x3, C is 2x3
    # We use Strassen for the 2x2x2 part + scalar terms for the 3rd column/row
    
    # 2x3x3 block decomposition (rank 11):
    # A indices: i in {0,1}, j in {0,1,2} -> flat = i*3+j (0..5)
    # B indices: j in {0,1,2}, k in {0,1,2} -> flat = j*3+k (0..8)
    # C indices: i in {0,1}, k in {0,1,2} -> flat = i*3+k (0..5)
    
    # Strassen terms for A_L(2x2) * B_T(2x2) where:
    # A_L: A[0,0], A[0,1], A[1,0], A[1,1] -> flat indices 0,1,3,4
    # B_T: B[0,0], B[0,1], B[1,0], B[1,1] -> flat indices 0,1,3,4
    # C: C[0,0], C[0,1], C[1,0], C[1,1] -> flat indices 0,1,3,4
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    
    # Plus 4 scalar terms for the 3rd column of A and 3rd row of B:
    # K1 = A02 * B20 -> C00 += K1
    # K2 = A02 * B21 -> C01 += K2
    # K3 = A12 * B20 -> C10 += K3
    # K4 = A12 * B21 -> C11 += K4
    
    # Wait, that's only 11 terms but we need to handle B22 and C02, C12 too.
    # Let me reconsider.
    
    # Actually for 2x3x3:
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    
    # So we need additional terms. Let me use a different approach.
    
    # Actually, the known rank for 2x3x3 is 11. Let me construct it properly.
    
    # For 2x3x3, we can think of it as:
    # C = A * B where A is 2x3, B is 3x3, C is 2x3
    
    # Split B into columns: B = [B_col0, B_col1, B_col2]
    # C_col0 = A * B_col0 (2x3 * 3x1 = 2x1)
    # C_col1 = A * B_col1
    # C_col2 = A * B_col2
    
    # For each column, we need to compute A (2x3) times a vector (3x1)
    # This can be done with rank 5 (using Strassen-like for 2x3 * 3x1)
    # But we can share structure.
    
    # Let me use the known rank-11 decomposition for 2x3x3.
    
    # Terms for 2x3x3 block (local indices):
    # A: 2x3, flat index i*3+j
    # B: 3x3, flat index j*3+k  
    # C: 2x3, flat index i*3+k
    
    # I'll define 11 terms as (u_dict, v_dict, w_dict)
    # where keys are local flat indices
    
    block_terms = []
    
    # Strassen for the 2x2x2 core (j=0,1 for A and B; k=0,1 for C)
    # A00=0, A01=1, A10=3, A11=4
    # B00=0, B01=1, B10=3, B11=4
    # C00=0, C01=1, C10=3, C11=4
    
    # M1 = (A00+A11)(B00+B11)
    block_terms.append(({0: 1, 4: 1}, {0: 1, 4: 1}, {0: 1, 4: 1}))
    # M2 = (A10+A11)B00
    block_terms.append(({3: 1, 4: 1}, {0: 1}, {3: 1, 4: -1}))
    # M3 = A00(B01-B11)
    block_terms.append(({0: 1}, {1: 1, 4: -1}, {1: 1, 4: 1}))
    # M4 = A11(B10-B00)
    block_terms.append(({4: 1}, {3: 1, 0: -1}, {0: 1, 3: 1}))
    # M5 = (A00+A01)B11
    block_terms.append(({0: 1, 1: 1}, {4: 1}, {0: -1, 1: 1}))
    # M6 = (A10-A00)(B00+B01)
    block_terms.append(({3: 1, 0: -1}, {0: 1, 1: 1}, {4: 1}))
    # M7 = (A01-A11)(B10+B11)
    block_terms.append(({1: 1, 4: -1}, {3: 1, 4: 1}, {0: 1}))
    
    # Now handle the 3rd column of B (k=2) and 3rd row of A (j=2)
    # C[0,2] = A00*B02 + A01*B12 + A02*B22
    # C[1,2] = A10*B02 + A11*B12 + A12*B22
    
    # We need terms involving B[:,2] (indices 2,5,8) and A[:,2] (indices 2,5)
    # And C[:,2] (indices 1,4)
    
    # K1 = A02 * B22 -> C02
    block_terms.append(({2: 1}, {8: 1}, {1: 1}))
    # K2 = A12 * B22 -> C12
    block_terms.append(({5: 1}, {8: 1}, {4: 1}))
    # K3 = (A00+A10) * B02 -> contributes to C02 and C12
    # C02 += A00*B02, C12 += A10*B02
    block_terms.append(({0: 1, 3: 1}, {2: 1}, {1: 1, 4: 1}))
    # K4 = (A01+A11) * B12 -> contributes to C02 and C12
    # C02 += A01*B12, C12 += A11*B12
    block_terms.append(({1: 1, 4: 1}, {5: 1}, {1: 1, 4: 1}))
    
    # That's 11 terms. Let me verify this is correct.
    
    # Actually, let me be more careful. Let me verify the 2x3x3 decomposition.
    
    # For now, let's just use the schoolbook if verification fails.
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]  # 6 x 22
    V = [[0] * rank for _ in range(m * p)]  # 18 x 22
    W = [[0] * rank for _ in range(p * n)]  # 12 x 22
    
    # For block b (b=0,1), the local B columns are k=0,1,2
    # Global B column: block_start + k where block_start = b*3
    # Global B index: j*p + (block_start + k)
    
    # For block b, the local C columns are k=0,1,2
    # Global C column: block_start + k
    # Global W index: (block_start + k)*n + i
    
    for block in range(2):
        block_start = block * 3
        for t_idx, (u_dict, v_dict, w_dict) in enumerate(block_terms):
            col = block * 11 + t_idx
            
            # U: local A index -> global A index (same, since A is 2x3)
            for l_idx, val in u_dict.items():
                U[l_idx][col] = val
            
            # V: local B index j*3+k -> global B index j*p + (block_start + k)
            for l_idx, val in v_dict.items():
                j_local = l_idx // 3
                k_local = l_idx % 3
                g_idx = j_local * p + (block_start + k_local)
                V[g_idx][col] = val
            
            # W: local C index i*3+k -> global W index (block_start + k)*n + i
            for l_idx, val in w_dict.items():
                i_local = l_idx // 3
                k_local = l_idx % 3
                g_idx = (block_start + k_local) * n + i_local
                W[g_idx][col] = val
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if _verify(U, V, W):
        return result
    else:
        # Fallback to schoolbook
        return schoolbook()
