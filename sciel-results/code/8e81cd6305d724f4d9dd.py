import numpy as np
import time
import random

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p  # 40
    
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s limit
    
    # Build schoolbook decomposition as fallback
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Strassen-based decomposition for 2x4x5
    # Partition A (2x4) into 2x2 blocks of size 1x2
    # Partition B (4x5) into 2x2 blocks of size 2x2
    # Partition C (2x5) into 2x2 blocks of size 1x2
    # 
    # A = [[A00, A01], [A10, A11]] where each is 1x2
    # B = [[B00, B01], [B10, B11]] where each is 2x2
    # C = [[C00, C01], [C10, C11]] where each is 1x2
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Apply Strassen to each 2x2x2 sub-problem:
    # For multiplying 1x2 by 2x2 to get 1x2:
    # This is equivalent to two 1x1x2 multiplications
    # We can use a rank-2 decomposition for each 1x1x2
    # 
    # Actually, let's think differently. The 2x4x5 matmul can be seen as:
    # C = A * B where A is 2x4, B is 4x5
    # Split columns of A and rows of B into two groups of 2:
    # A = [A_L | A_R] where A_L, A_R are 2x2
    # B = [B_L; B_R] where B_L, B_R are 2x5
    # C = A_L * B_L + A_R * B_R
    #
    # Each A_L*B_L is 2x2 * 2x5 = 2x5
    # Each A_R*B_R is 2x2 * 2x5 = 2x5
    #
    # For 2x2 * 2x5, we can use Strassen-like approach:
    # Split 2x5 into two 2x2 and one 2x1
    # This gets complicated. Let's use a different approach.
    #
    # Alternative: Use the fact that 2x4x5 = 2*(2x2x5) + 2*(2x2x5)
    # For 2x2x5, we can use rank 7 (Strassen for 2x2x2 gives rank 7, 
    # and 2x2x5 = 2x2x2 + 2x2x2 + 2x2x1, each with rank 7,7,4 = 18 total)
    # That's not better than 40.
    #
    # Let me try a more direct approach. The key insight is:
    # 2x4x5 matmul has 2*4*5 = 40 scalar multiplications in schoolbook.
    # We can group these into Strassen-like patterns.
    #
    # Consider splitting the 4 middle dimension into two parts of size 2:
    # C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    #
    # First sum: 2x2 * 2x5 matmul (A_left * B_top)
    # Second sum: 2x2 * 2x5 matmul (A_right * B_bottom)
    #
    # For each 2x2 * 2x5:
    # Split the 5 columns into 2+2+1
    # C = A * [B1 | B2 | B3] = A*B1 + A*B2 + A*B3
    # where B1, B2 are 2x2 and B3 is 2x1
    #
    # A*B1 (2x2*2x2): rank 7 via Strassen
    # A*B2 (2x2*2x2): rank 7 via Strassen  
    # A*B3 (2x2*2x1): rank 4 (schoolbook)
    # Total per 2x2*2x5: 7+7+4 = 18
    # Total for 2x4x5: 18 + 18 = 36
    #
    # This gives rank 36! Let's implement it.
    
    def strassen_2x2x2(U, V, W, offset):
        """
        Add Strassen 2x2x2 decomposition to U, V, W starting at column offset.
        U has shape (4, rank), V has shape (4, rank), W has shape (4, rank)
        for a 2x2 * 2x2 multiplication.
        """
        # Strassen's 7 multiplications for 2x2x2:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        strassen_patterns = [
            # (U_pattern, V_pattern, W_pattern) for each of the 7 multiplications
            # U is for A (2x2), V is for B (2x2), W is for C (2x2)
            # Indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            
            # M1 = (A00+A11)*(B00+B11) -> contributes to C00(+1), C11(+1)
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            
            # M2 = (A10+A11)*B00 -> contributes to C10(+1), C11(-1)
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            
            # M3 = A00*(B01-B11) -> contributes to C00(-1), C01(+1)
            ([1, 0, 0, 0], [0, 1, 0, -1], [-1, 1, 0, 0]),
            
            # M4 = A11*(B10-B00) -> contributes to C00(+1), C10(+1)
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            
            # M5 = (A00+A01)*B11 -> contributes to C00(-1), C01(+1)
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            
            # M6 = (A10-A00)*(B00+B01) -> contributes to C11(+1)
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            
            # M7 = (A01-A11)*(B10+B11) -> contributes to C00(+1), C11(-1)
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]),
        ]
        
        for u_pat, v_pat, w_pat in strassen_patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    U[idx, offset] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    V[idx, offset] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    W[idx, offset] = val
            offset += 1
        
        return offset
    
    def schoolbook_2x2x1(U, V, W, offset):
        """
        Add schoolbook 2x2x1 decomposition to U, V, W starting at column offset.
        For multiplying 2x2 by 2x1 to get 2x1.
        U has shape (4, rank), V has shape (4, rank), W has shape (2, rank)
        """
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        #
        # 4 terms:
        # Term 1: A00*B0 -> C0
        # U: [1, 0, 0, 0], V: [1, 0], W: [1, 0]
        # Term 2: A01*B1 -> C0
        # U: [0, 1, 0, 0], V: [0, 1], W: [1, 0]
        # Term 3: A10*B0 -> C1
        # U: [0, 0, 1, 0], V: [1, 0], W: [0, 1]
        # Term 4: A11*B1 -> C1
        # U: [0, 0, 0, 1], V: [0, 1], W: [0, 1]
        
        patterns = [
            ([1, 0, 0, 0], [1, 0], [1, 0]),
            ([0, 1, 0, 0], [0, 1], [1, 0]),
            ([0, 0, 1, 0], [1, 0], [0, 1]),
            ([0, 0, 0, 1], [0, 1], [0, 1]),
        ]
        
        for u_pat, v_pat, w_pat in patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    U[idx, offset] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    V[idx, offset] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    W[idx, offset] = val
            offset += 1
        
        return offset
    
    # Now build the full 2x4x5 decomposition
    # A is 2x4: A[i, j] for i in 0..1, j in 0..3
    # B is 4x5: B[j, k] for j in 0..3, k in 0..4
    # C is 2x5: C[i, k] for i in 0..1, k in 0..4
    #
    # Split j into two groups: j in {0,1} and j in {2,3}
    # A_left = A[:, 0:2] (2x2), A_right = A[:, 2:4] (2x2)
    # B_top = B[0:2, :] (2x5), B_bottom = B[2:4, :] (2x5)
    # C = A_left * B_top + A_right * B_bottom
    #
    # For A_left * B_top (2x2 * 2x5):
    # Split B_top into B_top_left (2x2), B_top_mid (2x2), B_top_right (2x1)
    # A_left * B_top_left: 2x2*2x2 -> rank 7 (Strassen)
    # A_left * B_top_mid: 2x2*2x2 -> rank 7 (Strassen)
    # A_left * B_top_right: 2x2*2x1 -> rank 4 (schoolbook)
    # Total: 18
    #
    # Similarly for A_right * B_bottom: 18
    # Grand total: 36
    
    total_rank = 36
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)
    
    # Helper to map 2x2x2 indices to global indices
    # For A (2x4): local (i, j_local) -> global (i, j_offset + j_local)
    # For B (4x5): local (j_local, k_local) -> global (j_offset + j_local, k_offset + k_local)
    # For C (2x5): local (i, k_local) -> global (i, k_offset + k_local)
    
    def add_2x2x2_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset, use_strassen=True):
        """
        Add a 2x2x2 block decomposition.
        A block is at rows [a_i_offset, a_i_offset+1], cols [a_j_offset, a_j_offset+1]
        B block is at rows [b_j_offset, b_j_offset+1], cols [b_k_offset, b_k_offset+1]
        C block is at rows [c_i_offset, c_i_offset+1], cols [c_k_offset, c_k_offset+1]
        """
        if use_strassen:
            # Strassen for 2x2x2
            strassen_patterns = [
                ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
                ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
                ([1, 0, 0, 0], [0, 1, 0, -1], [-1, 1, 0, 0]),
                ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
                ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
                ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
                ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]),
            ]
            
            for u_pat, v_pat, w_pat in strassen_patterns:
                # U: A indices (i, j) -> (a_i_offset + i, a_j_offset + j)
                for idx, val in enumerate(u_pat):
                    if val != 0:
                        i_local = idx // 2
                        j_local = idx % 2
                        i_global = a_i_offset + i_local
                        j_global = a_j_offset + j_local
                        U[i_global * m + j_global, col_offset] = val
                
                # V: B indices (j, k) -> (b_j_offset + j, b_k_offset + k)
                for idx, val in enumerate(v_pat):
                    if val != 0:
                        j_local = idx // 2
                        k_local = idx % 2
                        j_global = b_j_offset + j_local
                        k_global = b_k_offset + k_local
                        V[j_global * p + k_global, col_offset] = val
                
                # W: C indices (i, k) -> (c_i_offset + i, c_k_offset + k)
                for idx, val in enumerate(w_pat):
                    if val != 0:
                        i_local = idx // 2
                        k_local = idx % 2
                        i_global = c_i_offset + i_local
                        k_global = c_k_offset + k_local
                        W[k_global * n + i_global, col_offset] = val
                
                col_offset += 1
        
        return col_offset
    
    def add_2x2x1_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        """
        Add a 2x2x1 block decomposition (schoolbook).
        A block is 2x2, B block is 2x1, C block is 2x1.
        """
        # 4 terms
        patterns = [
            # (u_pattern, v_pattern, w_pattern)
            # u: A (2x2) -> 4 elements
            # v: B (2x1) -> 2 elements
            # w: C (2x1) -> 2 elements
            
            # Term 1: A[0,0] * B[0,0] -> C[0,0]
            ([1, 0, 0, 0], [1, 0], [1, 0]),
            # Term 2: A[0,1] * B[1,0] -> C[0,0]
            ([0, 1, 0, 0], [0, 1], [1, 0]),
            # Term 3: A[1,0] * B[0,0] -> C[1,0]
            ([0, 0, 1, 0], [1, 0], [0, 1]),
            # Term 4: A[1,1] * B[1,0] -> C[1,0]
            ([0, 0, 0, 1], [0, 1], [0, 1]),
        ]
        
        for u_pat, v_pat, w_pat in patterns:
            # U: A indices
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = a_i_offset + i_local
                    j_global = a_j_offset + j_local
                    U[i_global * m + j_global, col_offset] = val
            
            # V: B indices (2x1)
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx // 1  # 0 or 1
                    k_local = 0
                    j_global = b_j_offset + j_local
                    k_global = b_k_offset + k_local
                    V[j_global * p + k_global, col_offset] = val
            
            # W: C indices (2x1)
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx // 1  # 0 or 1
                    k_local = 0
                    i_global = c_i_offset + i_local
                    k_global = c_k_offset + k_local
                    W[k_global * n + i_global, col_offset] = val
            
            col_offset += 1
        
        return col_offset
    
    # Build the decomposition
    col = 0
    
    # A_left * B_top: A[:, 0:2] * B[0:2, :]
    # B_top is 2x5, split into 2x2, 2x2, 2x1
    
    # A_left * B_top[:, 0:2] (2x2 * 2x2)
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 0, 0, 0, col)  # 7 cols
    
    # A_left * B_top[:, 2:4] (2x2 * 2x2)
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 2, 0, 2, col)  # 7 cols
    
    # A_left * B_top[:, 4:5] (2x2 * 2x1)
    col = add_2x2x1_block(U, V, W, 0, 0, 0, 4, 0, 4, col)  # 4 cols
    
    # A_right * B_bottom: A[:, 2:4] * B[2:4, :]
    # B_bottom is 2x5, split into 2x2, 2x2, 2x1
    
    # A_right * B_bottom[:, 0:2] (2x2 * 2x2)
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 0, 0, 0, col)  # 7 cols
    
    # A_right * B_bottom[:, 2:4] (2x2 * 2x2)
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 2, 0, 2, col)  # 7 cols
    
    # A_right * B_bottom[:, 4:5] (2x2 * 2x1)
    col = add_2x2x1_block(U, V, W, 0, 2, 2, 4, 0, 4, col)  # 4 cols
    
    # Total: 7+7+4+7+7+4 = 36
    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    
    # Try to reduce rank by searching for linear dependencies among the 36 rank-1 terms
    # If we can find a linear combination that reduces the effective rank, we win
    # Check if the 36 vectors in the tensor space are linearly independent
    # The tensor lives in R^(nm * mp * pn) = R^(8*20*10) = R^1600
    # 36 vectors in 1600-dim space are almost certainly independent
    # But let's check for near-dependencies that might allow rank reduction
    
    # Alternative: try different partition strategies
    # Strategy 2: Split middle dim as 1+3 instead of 2+2
    # A_left (2x1) * B_top (1x5) + A_right (2x3) * B_bottom (3x5)
    # 2x1*1x5: rank 10 (schoolbook)
    # 2x3*3x5: can we do better than 30?
    #   Split 2x3*3x5 as 2x1*1x5 + 2x2*2x5 + 2x0*... no
    #   2x3*3x5: split B into 3 groups of 5... 
    #   Actually 2x3*3x5: split A cols into 1+2, B rows into 1+2
    #   = (2x1*1x5) + (2x1*2x5) + (2x2*1x5) + (2x2*2x5)
    #   = 10 + 10 + 10 + 18 = 48. Worse.
    #   Or use Strassen on 2x2*2x5 part: still 18
    #   Total: 10+48 = 58. Worse than 36.
    
    # Strategy 3: Split middle dim as 1+1+2
    # A[:,0]*B[0,:] + A[:,1]*B[1,:] + A[:,2:4]*B[2:4,:]
    # 2x1*1x5: rank 10
    # 2x1*1x5: rank 10  
    # 2x2*2x5: rank 18
    # Total: 38. Slightly worse than 36.
    
    # Strategy 4: Try to share Strassen terms between blocks
    # The two 2x2x2 Strassen blocks for A_left use the same A but different B columns.
    # Can we combine? M1_left uses (A00+A11)*(B00+B11) for cols 0,1
    # M1_right uses (A00+A11)*(B22+B33) for cols 2,3
    # These are different B vectors, so they can't be merged.
    # But what about M3_left = A00*(B01-B11) and M3_right = A00*(B23-B33)?
    # Still different B vectors. No merging possible.
    
    # The rank 36 decomposition appears to be the best with this approach.
    # Let's verify it passes.
    
    # Verify the decomposition
    def verify_decomposition(U, V, W):
        """Verify that U, V, W form an exact decomposition of the matmul tensor."""
        rank = U.shape[1]
        
        # Check all "one" entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        
        # Check all "zero" entries using vectorized approach
        # Build the full tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # and check it matches the matmul tensor
        
        # Efficient: for each rank term r, add outer product
        T = np.zeros((nm, mp, pn), dtype=np.float64)
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        # Expected tensor: T[i*m+j, j*p+k, k*n+i] = 1, else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T[i*m+j, j*p+k, k*n+i] - 1.0) > 1e-9:
                        return False
        
        # Check zeros
        # Create expected tensor
        expected = np.zeros((nm, mp, pn), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected[i*m+j, j*p+k, k*n+i] = 1.0
        
        if np.max(np.abs(T - expected)) > 1e-9:
            return False
        
        return True
    
    if time.time() > deadline:
        return best_result
    
    if verify_decomposition(U, V, W):
        # Convert to list format
        U_list = [[int(round(U[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(nm)]
        V_list = [[int(round(V[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(mp)]
        W_list = [[int(round(W[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(pn)]
        
        best_result = {"U": U_list, "V": V_list, "W": W_list}
        best_rank = total_rank
    else:
        # Fallback to schoolbook
        best_result = {"U": U_school, "V": V_school, "W": W_school}
        best_rank = rank_school
    
    return best_result
# EVOLVE-BLOCK-END
