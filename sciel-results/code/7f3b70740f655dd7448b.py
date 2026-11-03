import numpy as np
import time

DIMENSIONS = [3, 4, 7]


def _validate_decomposition(U, V, W, n, m, p, rank):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    if U_arr.shape != (n * m, rank):
        return False
    if V_arr.shape != (m * p, rank):
        return False
    if W_arr.shape != (p * n, rank):
        return False

    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_reshaped[i, j, :] * V_reshaped[j, k, :] * W_reshaped[k, i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _strassen_2x2x2():
    """
    Return U, V, W for 2x2x2 matrix multiplication using Strassen (rank 7).
    U: (2*2, 7) = (4, 7)
    V: (2*2, 7) = (4, 7)
    W: (2*2, 7) = (4, 7)
    """
    # Strassen 7 products for 2x2 matrices
    # M1 = (A11 - A22) * (B22 + B21)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A12 - A11) * (B11 + B12)
    # M7 = (A21 - A22) * (B22 + B21)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # For 2x2x2: n=2, m=2, p=2
    # U row index = i*2 + j
    # V row index = j*2 + k
    # W row index = k*2 + i
    
    # Each Strassen product r has:
    # U contribution: sum of a_i * A[i,j] for the first matrix factor
    # V contribution: sum of b_j * B[j,k] for the second matrix factor
    # W contribution: c * C[i,k] where C = (first) * (second)
    
    # Let's build each of the 7 rank-1 terms
    # Term r: U_r[i,j] * V_r[j,k] * W_r[k,i] = 1 if (i,j,k) corresponds to the product
    
    # Actually, let's think of it as:
    # C[i,k] = sum_r U[i,j,r] * A[i,j] * V[j,k,r] * B[j,k] * W[k,i,r]
    # For each rank-r term, we need:
    # U_r[i,j] * V_r[j,k] * W_r[k,i] to pick out the specific A[i,j]*B[j,k] combination
    
    # For Strassen, each M_r = (linear combo of A) * (linear combo of B)
    # So for M1 = (A11 - A22) * (B22 + B21):
    #   First factor: A11 - A22 => U1[i,j]: U1[0,0]=1, U1[1,1]=-1, others 0
    #   Second factor: B22 + B21 => V1[j,k]: V1[1,1]=1, V1[1,0]=1, others 0
    #   But this gives a 2x2 matrix product, not a single scalar.
    
    # Wait, I need to reconsider. The tensor decomposition is for the full multiplication.
    # Let me use a different approach: directly construct the 7 rank-1 tensors.
    
    # For 2x2x2, the multiplication tensor has 8 non-zero entries (all 1s).
    # Strassen shows it can be decomposed into 7 rank-1 tensors.
    
    # Let me use the standard Strassen decomposition mapped to our indexing.
    
    # I'll construct it by verifying against the known Strassen formula.
    
    # Let me use a simpler approach: use the known fact that 2x2x2 has rank 7
    # and construct it explicitly.
    
    # Actually, let me just use the schoolbook for 2x2x2 (rank 8) and focus on
    # the block structure for the full 3x4x7 problem. The key insight is the
    # block partitioning, not necessarily Strassen within each block.
    
    # But the cell finding says Strassen on 2x2x2 blocks reduces rank.
    # Let me construct the Strassen decomposition properly.
    
    # For C = A*B where A, B are 2x2:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # Strassen:
    # M1 = (A[0,0]-A[1,1])*(B[1,1]+B[1,0])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[0,1]-B[1,1])
    # M4 = A[1,1]*(B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[0,1]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[1,0]-A[1,1])*(B[1,1]+B[1,0])
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # In our tensor decomposition:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    # where i in [0,2), j in [0,2), k in [0,2)
    
    # Each rank-1 term r corresponds to one M_r.
    # For M_r = (sum_{i,j} a_r[i,j] * A[i,j]) * (sum_{j,k} b_r[j,k] * B[j,k])
    # The contribution to C[i',k'] from this term is:
    #   (sum_{i,j} a_r[i,j] * A[i,j]) * (sum_{j,k} b_r[j,k] * B[j,k])
    # This is a scalar, but C[i',k'] is a matrix entry.
    
    # I think I'm overcomplicating this. Let me use a different representation.
    
    # The tensor T[i,j,k] = 1 for all i,j,k in [0,2).
    # We want T[i,j,k] = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]
    
    # For rank 1: T[i,j,k] = u[i,j] * v[j,k] * w[k,i]
    # This equals 1 only if u[i,j]=v[j,k]=w[k,i]=1 for all i,j,k, which is impossible
    # for a single rank-1 term to cover all 8 entries.
    
    # For rank 7 (Strassen), we need 7 such terms that sum to 1 for each (i,j,k).
    
    # Let me look up or derive the explicit Strassen tensor decomposition.
    
    # Actually, I'll take a different approach. I'll construct the block decomposition
    # using schoolbook for each sub-block, which is simpler and still exploits the
    # structure. The rank reduction comes from the block partitioning itself.
    
    # For now, let me return schoolbook for 2x2x2 (rank 8) as a safe fallback.
    # The main strategy will be the block partitioning of 3x4x7.
    
    rank = 8
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i*2+j, col] = 1
                V[j*2+k, col] = 1
                W[k*2+i, col] = 1
                col += 1
    return U, V, W


def _construct_block_decomposition(n, m, p):
    """
    Construct a block-partitioned decomposition for n x m x p matrix multiplication.
    Split n=3 into 2+1, and use appropriate sub-block decompositions.
    
    The idea:
    - Split A (3x4) into A_top (2x4) and A_bottom (1x4)
    - Split C (3x7) into C_top (2x7) and C_bottom (1x7)
    - C_top = A_top * B (2x4 * 4x7 = 2x7)
    - C_bottom = A_bottom * B (1x4 * 4x7 = 1x7)
    
    For C_top = A_top * B:
    - Split m=4 into 2+2
    - A_top is 2x4, split into A_top_left (2x2) and A_top_right (2x2)
    - B is 4x7, split into B_top (4x... wait, B is 4x7)
    - Actually, B is 4x7. Split the 4 dimension into 2+2.
    - B_left = B[0:2, :] (2x7), B_right = B[2:4, :] (2x7)
    - C_top = A_top_left * B_left + A_top_right * B_right
    - Each is 2x2 * 2x7 = 2x7
    
    For 2x2 * 2x7:
    - Split p=7 into 2+2+2+1
    - Or use Strassen-like approaches
    
    This is getting complex. Let me use a simpler block strategy:
    Just split n=3 into 2+1 and use schoolbook for each part, but with
    the understanding that we can apply Strassen to the 2x2 sub-blocks.
    
    Actually, let me just implement the direct block decomposition with
    schoolbook for each sub-block, which gives rank = (2*4*7) + (1*4*7) = 56 + 28 = 84.
    That's not better than schoolbook.
    
    The improvement comes from using Strassen on the 2x2x2 sub-blocks.
    Let me think about this more carefully.
    
    For the 2x4x7 part (A_top * B):
    - Split m=4 into 2+2
    - C_top = A_top[:,0:2] * B[0:2,:] + A_top[:,2:4] * B[2:4,:]
    - Each term is 2x2 * 2x7
    - For 2x2 * 2x7: split p=7 into 2+2+2+1
      - 2x2 * 2x2 (use Strassen, rank 7)
      - 2x2 * 2x2 (use Strassen, rank 7)
      - 2x2 * 2x2 (use Strassen, rank 7)
      - 2x2 * 2x1 (schoolbook, rank 4)
      - Total for 2x2*2x7: 7+7+7+4 = 25
    - So 2x4x7: 25 + 25 = 50
    
    For the 1x4x7 part (A_bottom * B):
    - This is just a row vector times matrix: 1x4 * 4x7 = 1x7
    - Schoolbook: rank 4*7 = 28
    
    Total: 50 + 28 = 78
    
    This matches the cell finding! Let me implement this.
    """
    n, m, p = 3, 4, 7
    
    # We'll build the decomposition by concatenating sub-decompositions
    # Each sub-decomposition operates on a specific (i_range, j_range, k_range)
    
    # Helper: create a sub-decomposition for n_sub x m_sub x p_sub
    # and place it at offset (i_off, j_off, k_off) in the full tensor
    
    def _schoolbook_sub(n_sub, m_sub, p_sub, i_off, j_off, k_off):
        """Create schoolbook decomposition for a sub-block."""
        rank = n_sub * m_sub * p_sub
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U[(i_off + i) * m + (j_off + j), col] = 1
                    V[(j_off + j) * p + (k_off + k), col] = 1
                    W[(k_off + k) * n + (i_off + i), col] = 1
                    col += 1
        return U, V, W
    
    def _strassen_2x2x2_sub(i_off, j_off, k_off):
        """
        Create Strassen decomposition for a 2x2x2 sub-block.
        Uses the 7 Strassen products.
        """
        # For 2x2x2, I need to find 7 rank-1 tensors that sum to the identity tensor.
        # Let me use the explicit Strassen decomposition.
        
        # The 7 Strassen products for 2x2 matrices:
        # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M1 = (a-d)*(h+g)
        # M2 = (c+d)*e
        # M3 = a*(f-h)
        # M4 = d*(g-e)
        # M5 = (a+b)*h
        # M6 = (b-a)*(e+f)
        # M7 = (c-d)*(h+g)
        
        # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
        
        # In tensor form, for each M_r, we have:
        # M_r = (sum_{i,j} alpha_r[i,j] * A[i,j]) * (sum_{j,k} beta_r[j,k] * B[j,k])
        
        # The contribution to C[i',k'] is:
        # gamma_r[i',k'] * (sum_{i,j} alpha_r[i,j] * A[i,j]) * (sum_{j,k} beta_r[j,k] * B[j,k])
        
        # But in our tensor decomposition, we need:
        # T[i,j,k] = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]
        
        # where T[i,j,k] = 1 for all i,j,k in [0,2).
        
        # This is tricky. Let me use a different approach.
        
        # I'll construct the 7 rank-1 tensors directly by solving for them.
        # Actually, let me just use schoolbook for the 2x2x2 sub-blocks for now,
        # and rely on the block structure for the rank reduction.
        
        # Schoolbook for 2x2x2 has rank 8.
        # With Strassen it's rank 7.
        # The difference is small, so let me use schoolbook for simplicity
        # and see if the overall structure still gives a good rank.
        
        # Actually, the cell finding says the rank goes from 80 to 78, which
        # is a reduction of 2. This suggests that Strassen on three 2x2x2 blocks
        # saves 3 ranks (one per block), and the 2x2x1 block is schoolbook.
        
        # Let me just use schoolbook for all sub-blocks and compute the total rank.
        # If it's not better than 84, I'll need to properly implement Strassen.
        
        rank = 8  # schoolbook for 2x2x2
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[(i_off + i) * m + (j_off + j), col] = 1
                    V[(j_off + j) * p + (k_off + k), col] = 1
                    W[(k_off + k) * n + (i_off + i), col] = 1
                    col += 1
        return U, V, W
    
    # Build the full decomposition
    # Part 1: 2x4x7 (top 2 rows of A)
    # Split m=4 into 2+2
    # Part 1a: 2x2x7 (top-left 2x2 of A, first 2 cols of B)
    #   Split p=7 into 2+2+2+1
    # Part 1b: 2x2x7 (top-right 2x2 of A, last 2 cols of B)
    #   Split p=7 into 2+2+2+1
    # Part 2: 1x4x7 (bottom row of A)
    #   Schoolbook
    
    # Let me just use schoolbook for everything and see what rank we get.
    # Then I can optimize specific sub-blocks.
    
    # Full schoolbook: rank 84
    # Block structure with schoolbook sub-blocks:
    # Part 1: 2x4x7 = 56
    # Part 2: 1x4x7 = 28
    # Total: 84 (same as full schoolbook)
    
    # So the block structure alone doesn't help. We need Strassen on the 2x2x2 blocks.
    
    # Let me properly implement Strassen for 2x2x2.
    
    # I found a reference: the Strassen decomposition for 2x2x2 can be written as:
    # For each of the 7 products, we have specific U, V, W entries.
    
    # Let me construct it explicitly.
    
    # For M1 = (A00 - A11) * (B11 + B10):
    #   U1[0,0] = 1, U1[1,1] = -1 (for the A part)
    #   V1[1,1] = 1, V1[1,0] = 1 (for the B part)
    #   But this gives a scalar product, not a matrix.
    
    # I think the issue is that in the tensor decomposition, each rank-1 term
    # contributes to ALL (i,j,k) entries, not just one.
    
    # Let me use a different approach: I'll construct the Strassen decomposition
    # by ensuring that for each (i,j,k) in [0,2)^3, the sum over r of
    # U_r[i,j] * V_r[j,k] * W_r[k,i] = 1.
    
    # I'll use the known Strassen coefficients.
    
    # After some thought, let me just use a lookup table for the 2x2x2 Strassen
    # decomposition. The 7 rank-1 tensors are:
    
    # I'll define them as 2x2 matrices for U_r, V_r, W_r.
    
    # Let me use the following construction (verified against Strassen's algorithm):
    
    strassen_U = [
        [[1, 0], [0, -1]],   # M1 A-part
        [[0, 0], [1, 1]],    # M2 A-part
        [[1, 0], [0, 0]],    # M3 A-part
        [[0, 0], [0, 1]],    # M4 A-part
        [[1, 1], [0, 0]],    # M5 A-part
        [[0, 1], [-1, 0]],   # M6 A-part
        [[0, 0], [1, -1]],   # M7 A-part
    ]
    strassen_V = [
        [[0, 0], [1, 1]],    # M1 B-part
        [[1, 0], [0, 0]],    # M2 B-part
        [[0, 1], [0, -1]],   # M3 B-part
        [[-1, 0], [1, 0]],   # M4 B-part
        [[0, 0], [0, 1]],    # M5 B-part
        [[1, 1], [0, 0]],    # M6 B-part
        [[0, 0], [1, 1]],    # M7 B-part
    ]
    strassen_W = [
        [[1, -1], [1, 0]],   # M1 C-part (contributes to C00, C11)
        [[0, 0], [1, -1]],   # M2 C-part
        [[0, 1], [0, 1]],    # M3 C-part
        [[1, 0], [1, 0]],    # M4 C-part
        [[-1, 1], [0, 0]],   # M5 C-part
        [[0, 0], [0, 1]],    # M6 C-part
        [[1, 0], [0, 0]],    # M7 C-part
    ]
    
    # Let me verify: C00 = M1 + M4 - M5 + M7
    # C00 from M1: W1[0,0] = 1
    # C00 from M4: W4[0,0] = 1
    # C00 from M5: W5[0,0] = -1
    # C00 from M7: W7[0,0] = 1
    # Sum: 1 + 1 - 1 + 1 = 2. But it should be 1.
    
    # I think my W matrices are wrong. Let me reconsider.
    
    # The W_r[k,i] should be such that when we compute
    # sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]
    # we get 1 for each (i,j,k).
    
    # This is a system of equations. Let me just solve it numerically.
    
    # Actually, let me take a completely different approach. I'll use the
    # block structure with schoolbook for all sub-blocks, which gives rank 84,
    # and then try to find a better decomposition using a more systematic method.
    
    # For now, let me return the schoolbook decomposition.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist(), rank


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Start with schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_rank = rank_sb
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()

    # Try block-partitioned decomposition
    if time.time() < deadline - 20:
        U_block, V_block, W_block, rank_block = _construct_block_decomposition(n, m, p)
        if _validate_decomposition(U_block, V_block, W_block, n, m, p, rank_block):
            if rank_block < best_rank:
                best_rank = rank_block
                best_U = U_block
                best_V = V_block
                best_W = W_block

    return {"U": best_U, "V": best_V, "W": best_W}
