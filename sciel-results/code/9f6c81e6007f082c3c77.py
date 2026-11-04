"""Program for matrix multiplication 4x4 x 4x6 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    
    Returns (U, V, W) where:
      U: (4, 7) - for A[i,j], row index = i*2+j
      V: (4, 7) - for B[j,k], row index = j*2+k
      W: (4, 7) - for C[i,k], row index = k*2+i  (trace convention: W[k*n+i] with n=2)
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # W: C[i,k] contributions, W row = k*2 + i
    # C[0,0]: i=0,k=0 -> W[0]; C[0,1]: i=0,k=1 -> W[2]
    # C[1,0]: i=1,k=0 -> W[1]; C[1,1]: i=1,k=1 -> W[3]
    
    # C[0,0] = M1 + M4 - M5 + M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    
    # C[0,1] = M3 + M5
    W[2, 2] = 1; W[2, 4] = 1
    
    # C[1,0] = M2 + M4
    W[1, 1] = 1; W[1, 3] = 1
    
    # C[1,1] = M1 - M2 + M3 + M6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W


def _build_block_decomposition(n, m, p, block_a, block_b, block_c):
    """
    Build a block-based decomposition using 2x2 Strassen for inner blocks.
    
    A is n x m: split into (n/block_a) x (m/block_b) blocks of size block_a x block_b
    B is m x p: split into (m/block_b) x (p/block_c) blocks of size block_b x block_c
    C is n x p: split into (n/block_a) x (p/block_c) blocks of size block_a x block_c
    
    For this to work with 2x2 Strassen, we need block_a=block_b=block_c=2.
    """
    if block_a != 2 or block_b != 2 or block_c != 2:
        return None
    
    na = n // block_a  # number of row blocks of A
    nb = m // block_b  # number of col blocks of A / row blocks of B
    nc = p // block_c  # number of col blocks of B
    
    if na * block_a != n or nb * block_b != m or nc * block_c != p:
        return None
    
    sU, sV, sW = _strassen_2x2()
    
    # For each pair of block row (ai) and block col (ck) of C,
    # we sum over block col (bj) of A / block row (bj) of B:
    # C[ai][ck] = sum_bj A[ai][bj] * B[bj][ck]
    # Each product is 2x2 * 2x2 -> 2x2, using Strassen (7 terms)
    # Total terms: na * nb * nc * 7
    
    total_rank = na * nb * nc * 7
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    col_idx = 0
    for ai in range(na):
        for ck in range(nc):
            for bj in range(nb):
                for t in range(7):
                    col = col_idx
                    col_idx += 1
                    
                    # U part: A[ai][bj] block, row index in A is (ai*2+ai2)*m + (bj*2+bj2)
                    for ai2 in range(2):
                        for bj2 in range(2):
                            coeff_inner = sU[ai2*2 + bj2, t]
                            if coeff_inner != 0:
                                u_row = (ai*2 + ai2) * m + (bj*2 + bj2)
                                U[u_row, col] += coeff_inner
                    
                    # V part: B[bj][ck] block, row index in B is (bj*2+bj2)*p + (ck*2+ck2)
                    for bj2 in range(2):
                        for ck2 in range(2):
                            coeff_inner = sV[bj2*2 + ck2, t]
                            if coeff_inner != 0:
                                v_row = (bj*2 + bj2) * p + (ck*2 + ck2)
                                V[v_row, col] += coeff_inner
                    
                    # W part: C[ai][ck] block, row index in W is (ck*2+ck2)*n + (ai*2+ai2)
                    for ai2 in range(2):
                        for ck2 in range(2):
                            coeff_inner = sW[ck2*2 + ai2, t]
                            if coeff_inner != 0:
                                w_row = (ck*2 + ck2) * n + (ai*2 + ai2)
                                W[w_row, col] += coeff_inner
    
    return U, V, W


def _build_mixed_decomposition(n, m, p):
    """
    Mixed decomposition: use Strassen for the first 4 columns of B (4x4x4 part),
    and schoolbook for the last 2 columns (4x4x2 part).
    
    4x4x4 via 2x2 Strassen blocks: 2*2*2*7 = 56 terms
    4x4x2 via schoolbook: 4*4*2 = 32 terms
    Total: 88 terms
    
    Alternative: use Strassen for all 6 columns:
    2*2*3*7 = 84 terms
    """
    # Option 1: All Strassen (2x2 blocks)
    result_strassen = _build_block_decomposition(n, m, p, 2, 2, 2)
    
    # Option 2: Strassen for first 4 cols, schoolbook for last 2
    # 4x4x4 part: 56 terms
    # 4x4x2 part: 32 terms
    # Total: 88 terms (worse)
    
    # Option 3: Try to find a better outer decomposition
    # For 2x2 x 2x3 -> 2x3, can we do better than 11 terms?
    # 7 (Strassen for first 2 cols) + 4 (schoolbook for last col) = 11
    # Each term costs 7 (inner Strassen)
    # Total: 11 * 7 = 77
    
    # Let's try a different approach: treat B as 4x6 = 4x(2+2+2)
    # Use Strassen for each 2-column strip
    # A is 4x4, B_strip is 4x2
    # C_strip is 4x2
    # Block structure: A is 2x2 blocks of 2x2, B_strip is 2x1 blocks of 2x2
    # For each pair of A row block and B_strip col block:
    #   2x2 * 2x2 -> 2x2 via Strassen (7 terms)
    # Number of such products: 2 (A row blocks) * 1 (B_strip col block) = 2
    # Per strip: 2 * 7 = 14 terms
    # 3 strips: 3 * 14 = 42 terms
    # Wait, that's not right. Let me reconsider.
    
    # Actually, for A (4x4) * B (4x6):
    # A has 2 row blocks (each 2 rows) and 2 col blocks (each 2 cols)
    # B has 2 row blocks (each 2 rows) and 3 col blocks (each 2 cols)
    # C has 2 row blocks (each 2 rows) and 3 col blocks (each 2 cols)
    #
    # C[ai][ck] = sum_bj A[ai][bj] * B[bj][ck]
    # For fixed ai, ck: sum over bj in {0,1}
    # Each product A[ai][bj] * B[bj][ck] is 2x2 * 2x2 -> 2x2
    # Using Strassen: 7 terms per product
    # Total: 2 (ai) * 3 (ck) * 2 (bj) * 7 = 84 terms
    
    # This is the same as before. Let's just use the block decomposition.
    
    if result_strassen is not None:
        return result_strassen
    
    # Fallback: schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1
                V[j*p + k, idx] = 1
                W[k*n + i, idx] = 1
                idx += 1
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    deadline = time.time() + 120
    
    # Try the block Strassen decomposition
    U, V, W = _build_block_decomposition(n, m, p, 2, 2, 2)
    
    if U is not None:
        rank = U.shape[1]
        # Verify correctness
        if time.time() < deadline - 30:
            # Quick verification with random matrices
            A = np.random.randint(-10, 10, size=(n, m))
            B = np.random.randint(-10, 10, size=(m, p))
            C_expected = A @ B
            
            U_mat = np.array(U)
            V_mat = np.array(V)
            W_mat = np.array(W)
            
            # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This is a tensor contraction
            C_computed = np.zeros((n, p))
            for r in range(rank):
                u_col = U_mat[:, r]
                v_col = V_mat[:, r]
                w_col = W_mat[:, r]
                
                # u_col has shape (n*m,), reshape to (n, m)
                u_reshaped = u_col.reshape(n, m)
                v_reshaped = v_col.reshape(m, p)
                w_reshaped = w_col.reshape(p, n)
                
                # Contribution: sum_{i,j,k} u[i,j] * v[j,k] * w[k,i] * delta
                # Actually, C[i,k] += sum_j u[i,j] * v[j,k] * w[k,i]
                # But w[k,i] is the same for all j, so:
                # C[i,k] += w[k,i] * sum_j u[i,j] * v[j,k]
                # = w[k,i] * (u_reshaped @ v_reshaped)[i,k]
                
                prod = u_reshaped @ v_reshaped
                for i in range(n):
                    for k in range(p):
                        C_computed[i, k] += w_reshaped[k, i] * prod[i, k]
            
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                # Verification failed, try schoolbook
                U, V, W = _build_mixed_decomposition(n, m, p)
                rank = U.shape[1]
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    U, V, W = _build_mixed_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
