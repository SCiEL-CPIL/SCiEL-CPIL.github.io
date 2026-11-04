import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START

def _a_gidx(i, j):
    return i * 4 + j

def _b_gidx(i, j):
    return i * 8 + j

def _w_gidx(i, k):
    return k * 2 + i

def _construct_rank56():
    """
    Construct a rank-56 decomposition using 8 independent 2x2x2 Strassen sub-problems.
    The 2x4 * 4x8 multiplication is decomposed into 8 blocks of 2x2 * 2x2.
    Each block uses the standard 7-term Strassen algorithm.
    """
    n, m, p = 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Define the 8 sub-problems.
    # We partition the 4x8 B matrix into 8 blocks of size 2x2.
    # B is 4 rows, 8 cols.
    # Rows 0-1, Cols 0-1 -> Block 0
    # Rows 0-1, Cols 2-3 -> Block 1
    # Rows 0-1, Cols 4-5 -> Block 2
    # Rows 0-1, Cols 6-7 -> Block 3
    # Rows 2-3, Cols 0-1 -> Block 4
    # Rows 2-3, Cols 2-3 -> Block 5
    # Rows 2-3, Cols 4-5 -> Block 6
    # Rows 2-3, Cols 6-7 -> Block 7
    
    # For each block, we use the 2x2 Strassen algorithm.
    # A is 2x4. We use the full A for each block?
    # Wait, C[i, k] = sum_j A[i, j] B[j, k].
    # If we fix k in a specific block, say k in {0,1}, then:
    # C[i, 0] = A[i,0]B[0,0] + A[i,1]B[1,0] + A[i,2]B[2,0] + A[i,3]B[3,0]
    # This couples A[0,0..1] with B[0,0..1] and A[0,2..3] with B[2,0..3].
    # This is NOT a simple 2x2 block multiplication of A_block * B_block.
    # It is a sum of two 2x2 multiplications:
    # C_block = A_left * B_top + A_right * B_bottom
    # Where A_left = A[:, 0:2], A_right = A[:, 2:4]
    # B_top = B[0:2, k_block], B_bottom = B[2:4, k_block]
    # C_block = C[:, k_block]
    
    # So for each k_block (4 blocks of width 2), we have:
    # C[:, k_block] = A[:, 0:2] @ B[0:2, k_block] + A[:, 2:4] @ B[2:4, k_block]
    
    # This is the sum of two 2x2 matrix multiplications.
    # Rank of sum of two 2x2 multiplications?
    # It is 14 (using Strassen for each).
    # Can we do better?
    # The tensor for sum of two 2x2 multiplications is T = T1 + T2.
    # T1 is 2x2x2, T2 is 2x2x2.
    # They are on disjoint sets of A and B indices?
    # T1 uses A[0:2, 0:2] and B[0:2, 0:2].
    # T2 uses A[0:2, 2:4] and B[2:4, 0:2].
    # The A indices are disjoint (cols 0-1 vs 2-3).
    # The B indices are disjoint (rows 0-1 vs 2-3).
    # So the supports are disjoint.
    # Therefore, rank(T1 + T2) = rank(T1) + rank(T2) = 7 + 7 = 14.
    
    # So for each k_block, rank is 14.
    # There are 4 k_blocks.
    # Total rank = 4 * 14 = 56.
    
    # This confirms the rank 56 construction.
    # Let's implement it.
    
    # Strassen terms for 2x2:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b21-b22)
    # m4 = a22(b12-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    strassen_terms = [
        # (u_indices, v_indices, w_indices, w_coeffs)
        # u: A indices, v: B indices, w: C indices
        # Local indices 0..3 for 2x2 matrices.
        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        
        # m1
        {'u': [(0, 1), (3, 1)], 'v': [(0, 1), (3, 1)], 'w': [(0, 1), (3, 1)]},
        # m2
        {'u': [(2, 1), (3, 1)], 'v': [(0, 1)], 'w': [(2, 1), (3, -1)]},
        # m3
        {'u': [(0, 1)], 'v': [(1, 1), (3, -1)], 'w': [(1, 1), (3, 1)]},
        # m4
        {'u': [(3, 1)], 'v': [(2, 1), (0, -1)], 'w': [(0, 1), (2, 1)]},
        # m5
        {'u': [(0, 1), (1, 1)], 'v': [(3, 1)], 'w': [(0, -1), (1, 1)]},
        # m6
        {'u': [(2, 1), (0, -1)], 'v': [(0, 1), (1, 1)], 'w': [(3, 1)]},
        # m7
        {'u': [(1, 1), (3, -1)], 'v': [(2, 1), (3, 1)], 'w': [(0, 1)]}
    ]
    
    # 4 k-blocks, 2 A-blocks (left, right), 2 B-blocks (top, bottom)
    # For each k_block (4 of them):
    #   We compute C[:, k_block] = A_left @ B_top + A_right @ B_bottom
    #   This is two 2x2 multiplications.
    #   We run Strassen for A_left @ B_top and A_right @ B_bottom.
    #   The results are added to the same C[:, k_block].
    
    col = 0
    for k_block in range(4):
        k_off = k_block * 2
        # A_left @ B_top
        a_col_off = 0
        b_row_off = 0
        for term in strassen_terms:
            for (local_idx, coeff) in term['u']:
                lr = local_idx // 2
                lc = local_idx % 2
                gi = lr
                gj = a_col_off + lc
                g_idx = _a_gidx(gi, gj)
                U[g_idx, col] += coeff
            
            for (local_idx, coeff) in term['v']:
                lr = local_idx // 2
                lc = local_idx % 2
                gi = b_row_off + lr
                gj = k_off + lc
                g_idx = _b_gidx(gi, gj)
                V[g_idx, col] += coeff
            
            for (local_idx, coeff) in term['w']:
                lr = local_idx // 2
                lc = local_idx % 2
                gi = lr
                gk = k_off + lc
                g_idx = _w_gidx(gi, gk)
                W[g_idx, col] += coeff
            
            col += 1
        
        # A_right @ B_bottom
        a_col_off = 2
        b_row_off = 2
        for term in strassen_terms:
            for (local_idx, coeff) in term['u']:
                lr = local_idx // 2
                lc = local_idx % 2
                gi = lr
                gj = a_col_off + lc
                g_idx = _a_gidx(gi, gj)
                U[g_idx, col] += coeff
            
            for (local_idx, coeff) in term['v']:
                lr = local_idx // 2
                lc = local_idx % 2
                gi = b_row_off + lr
                gj = k_off + lc
                g_idx = _b_gidx(gi, gj)
                V[g_idx, col] += coeff
            
            for (local_idx, coeff) in term['w']:
                lr = local_idx // 2
                lc = local_idx % 2
                gi = lr
                gk = k_off + lc
                g_idx = _w_gidx(gi, gk)
                W[g_idx, col] += coeff
            
            col += 1
    
    return U, V, W

def _matmul_check(U, V, W, n, m, p, num_tests=10):
    """
    Verify that the decomposition correctly computes matrix multiplication.
    """
    rng = np.random.default_rng(42)
    for _ in range(num_tests):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        expected = A @ B
        computed = np.zeros((n, p))
        for c in range(U.shape[1]):
            u_vec = U[:, c]
            v_vec = V[:, c]
            w_vec = W[:, c]
            u_reshaped = u_vec.reshape(n, m)
            v_reshaped = v_vec.reshape(m, p)
            w_reshaped = w_vec.reshape(p, n)
            uv = u_reshaped @ v_reshaped
            w_transposed = w_reshaped.T
            computed += uv * w_transposed
        if not np.allclose(computed, expected, atol=1e-10, rtol=1e-10):
            return False
    return True

def _to_half_integer_list(mat):
    result = []
    for row in mat:
        row_list = []
        for val in row:
            rounded = round(val * 2) / 2
            row_list.append(rounded)
        result.append(row_list)
    return result

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Try rank 48 first (if we had a construction for it)
    # For now, we stick with rank 56 as it is proven correct.
    # Future work: Implement rank 48 using sum of two 2x2 products with shared multiplications.
    
    U, V, W = _construct_rank56()
    
    if _matmul_check(U, V, W, n, m, p):
        return {
            "U": _to_half_integer_list(U),
            "V": _to_half_integer_list(V),
            "W": _to_half_integer_list(W)
        }
    else:
        # Fallback: Schoolbook (should not happen)
        rank = 64
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[_a_gidx(i, j), col] = 1.0
                    V[_b_gidx(j, k), col] = 1.0
                    W[_w_gidx(i, k), col] = 1.0
                    col += 1
        return {
            "U": _to_half_integer_list(U),
            "V": _to_half_integer_list(V),
            "W": _to_half_integer_list(W)
        }

# EVOLVE-BLOCK-END
