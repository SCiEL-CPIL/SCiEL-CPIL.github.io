"""Matrix multiplication 2x4x6 rank minimization via structured Strassen decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)."""
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, T, atol=1e-9):
    """Verify that the decomposition exactly reproduces T."""
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    T_check = np.zeros_like(T)
    for r in range(U.shape[1]):
        T_check += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_check, T, atol=atol)

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 * 2x2 matrix multiplication.
    Each term is a dict with keys 'u' (list of (idx, coeff)), 'v' (list of (idx, coeff)),
    and 'w' (list of (idx, coeff)).
    Indexing: A is 2x2, B is 2x2, C is 2x2.
    A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    """
    return [
        # m0: (A00+A11)*(B00+B11) -> C00 += 1, C11 += 1
        {'u': [(0, 1), (3, 1)], 'v': [(0, 1), (3, 1)], 'w': [(0, 1), (3, 1)]},
        # m1: (A10+A11)*B00 -> C10 += 1, C11 += -1
        {'u': [(2, 1), (3, 1)], 'v': [(0, 1)], 'w': [(2, 1), (3, -1)]},
        # m2: A00*(B01-B11) -> C01 += 1, C11 += 1
        {'u': [(0, 1)], 'v': [(1, 1), (3, -1)], 'w': [(1, 1), (3, 1)]},
        # m3: A11*(B10-B00) -> C00 += 1, C10 += 1
        {'u': [(3, 1)], 'v': [(2, 1), (0, -1)], 'w': [(0, 1), (2, 1)]},
        # m4: (A00+A01)*B11 -> C00 += -1, C01 += 1
        {'u': [(0, 1), (1, 1)], 'v': [(3, 1)], 'w': [(0, -1), (1, 1)]},
        # m5: (A10-A00)*(B00+B01) -> C11 += 1
        {'u': [(2, 1), (0, -1)], 'v': [(0, 1), (1, 1)], 'w': [(3, 1)]},
        # m6: (A01-A11)*(B10+B11) -> C00 += 1
        {'u': [(1, 1), (3, -1)], 'v': [(2, 1), (3, 1)], 'w': [(0, 1)]},
    ]

def _construct_strassen_2x2x4(n, m, p):
    """
    Construct a rank-42 decomposition using Strassen on 2x2x2 blocks.
    
    Partition the 2x4 matrix A into two 2x2 blocks (columns 0-1 and 2-3).
    Partition the 4x6 matrix B into three 4x2 blocks (cols 0-1, 2-3, 4-5).
    For each pair (A_block, B_block), we compute a 2x2 * 4x2 product.
    We decompose the 4x2 into two 2x2 blocks (top 2 rows, bottom 2 rows).
    So A_block (2x2) * B_block (4x2) is equivalent to:
      A_block * B_top (2x2) + A_block * B_bot (2x2)
    But wait, that's not quite right. Let me think more carefully.
    
    Actually, the 2x2 * 4x2 product can be viewed as:
    A (2x2) * B (4x2) where B = [B1; B2] with B1, B2 being 2x2.
    The result is A * B = [A*B1; A*B2] which is 4x2... no that's wrong.
    
    Let me reconsider. The matrix multiplication is C = A * B where A is 2x4 and B is 4x6.
    C is 2x6.
    
    We can partition A into A = [A1 | A2] where A1, A2 are 2x2.
    We can partition B into B = [[B11, B12, B13], [B21, B22, B23]] where Bij are 2x2.
    
    Then C = [A1, A2] * [[B11, B12, B13], [B21, B22, B23]]
         = [A1*B11 + A2*B21, A1*B12 + A2*B22, A1*B13 + A2*B23]
    
    Each term like A1*B11 is a 2x2 * 2x2 product, which can be done with rank 7 (Strassen).
    There are 2 * 3 = 6 such products, so total rank = 6 * 7 = 42.
    """
    strassen_terms = _strassen_2x2x2_terms()
    
    # For each combination of A-block (2 choices) and B-block (3 choices),
    # we add 7 Strassen terms.
    
    rank = 6 * 7  # 42
    U_dim = n * m  # 8
    V_dim = m * p  # 24
    W_dim = p * n  # 12
    
    U = np.zeros((U_dim, rank), dtype=np.float64)
    V = np.zeros((V_dim, rank), dtype=np.float64)
    W = np.zeros((W_dim, rank), dtype=np.float64)
    
    col = 0
    for a_block in range(2):  # 0 or 1
        for b_block in range(3):  # 0, 1, or 2
            for term in strassen_terms:
                # Process each Strassen term
                for u_idx, u_coeff in term['u']:
                    # u_idx is local index in 2x2: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                    local_i = u_idx // 2
                    local_j = u_idx % 2
                    global_i = local_i  # A has 2 rows
                    global_j = a_block * 2 + local_j
                    U[global_i * m + global_j, col] = u_coeff
                
                for v_idx, v_coeff in term['v']:
                    local_j = v_idx // 2
                    local_k = v_idx % 2
                    # B has 4 rows, 6 cols
                    # b_block determines which 2x2 block in B
                    # B is partitioned as 2x3 blocks of 2x2
                    # Row index: local_j (0 or 1 within the 2x2 block)
                    # But which 2x2 block in rows? We need to be careful.
                    # B is 4x6. We partition columns into 3 blocks of 2.
                    # For B_block, the column range is [b_block*2, b_block*2+2)
                    # The rows are 0-3, but we're multiplying A (2x4) * B (4x6)
                    # So the 4x2 block of B has rows 0-3 and columns b_block*2 to b_block*2+1
                    # Wait, I need to reconsider the partitioning.
                    
                    # Actually, let me re-derive. A is 2x4, B is 4x6.
                    # A = [A1 | A2] where A1 = A[:, 0:2], A2 = A[:, 2:4]
                    # B = [B_top; B_bot] where B_top = B[0:2, :], B_bot = B[2:4, :]
                    # But B is 4x6, so B_top is 2x6 and B_bot is 2x6.
                    # Then C = A * B = [A1, A2] * [B_top; B_bot] = A1*B_top + A2*B_bot
                    # A1 is 2x2, B_top is 2x6. So A1*B_top is 2x6.
                    # A2 is 2x2, B_bot is 2x6. So A2*B_bot is 2x6.
                    
                    # To use Strassen, we need 2x2 * 2x2 products.
                    # So we further partition B_top into three 2x2 blocks: B_top = [B_top_0, B_top_1, B_top_2]
                    # Similarly B_bot = [B_bot_0, B_bot_1, B_bot_2]
                    
                    # For A_block = 0 (A1), we compute A1 * B_top_b for b=0,1,2
                    # For A_block = 1 (A2), we compute A2 * B_bot_b for b=0,1,2
                    
                    # So for (a_block, b_block):
                    # If a_block == 0: use B_top, rows 0-1
                    # If a_block == 1: use B_bot, rows 2-3
                    row_base = a_block * 2  # 0 for top, 2 for bottom
                    global_row = row_base + local_j
                    global_col = b_block * 2 + local_k
                    V[global_row * p + global_col, col] = v_coeff
                
                for w_idx, w_coeff in term['w']:
                    local_i = w_idx // 2
                    local_k = w_idx % 2
                    global_i = local_i  # C has 2 rows
                    global_k = b_block * 2 + local_k
                    W[global_k * n + global_i, col] = w_coeff
                
                col += 1
    
    return U, V, W, rank

def _construct_schoolbook(n, m, p):
    """Construct the schoolbook rank-nmp decomposition."""
    rank = n * m * p
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    U = np.zeros((U_dim, rank), dtype=np.float64)
    V = np.zeros((V_dim, rank), dtype=np.float64)
    W = np.zeros((W_dim, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W, rank

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication.
    Uses Strassen on 2x2x2 sub-blocks as the primary approach.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    start_time = time.time()
    deadline = start_time + 110

    U_dim = n * m  # 8
    V_dim = m * p  # 24
    W_dim = p * n  # 12

    # Build the target tensor for verification
    T = _build_tensor(n, m, p)

    # Try Strassen rank-42 decomposition
    U, V, W, rank = _construct_strassen_2x2x4(n, m, p)
    
    # Verify the Strassen decomposition
    if not _verify(U, V, W, T):
        # Fallback to schoolbook
        U, V, W, rank = _construct_schoolbook(n, m, p)

    # Final verification
    if not _verify(U, V, W, T):
        # Fallback to schoolbook
        U, V, W, rank = _construct_schoolbook(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
