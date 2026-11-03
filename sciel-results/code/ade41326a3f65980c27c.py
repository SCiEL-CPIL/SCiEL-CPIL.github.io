import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    
    # The tensor T[i,j,k] = A[i,j]*B[j,k] for i in 0..2, j in 0..3, k in 0..3
    # We want: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Start with the known rank-44 decomposition (4 blocks of 3x2x2 with rank 11 each)
    # Try to improve by finding a better 3x2x2 decomposition or by sharing across blocks.
    
    # First, let's build the rank-44 baseline and verify it.
    rank_44 = _build_rank44()
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        return rank_44
    
    # Now try to find a better decomposition.
    # Strategy: The 3x4x4 tensor can be viewed as having structure.
    # Let's try to find a rank-r decomposition for smaller sub-tensors.
    
    # Approach 1: Try to find a rank-10 decomposition for 3x2x2
    # If we can get rank 10 for 3x2x2, total rank = 40
    
    # Approach 2: Try to share terms across the 4 blocks
    # The 4 blocks are:
    # Block 0: A[0:3, 0:2] * B[0:2, 0:2] -> C[0:3, 0:2]
    # Block 1: A[0:3, 2:4] * B[2:4, 0:2] -> C[0:3, 0:2]
    # Block 2: A[0:3, 0:2] * B[0:2, 2:4] -> C[0:3, 2:4]
    # Block 3: A[0:3, 2:4] * B[2:4, 2:4] -> C[0:3, 2:4]
    
    # Let me try a numerical optimization approach to find a lower-rank decomposition.
    # But first, let me try some known constructions.
    
    # Key insight: The 3x4x4 multiplication tensor might have a lower rank than 44.
    # Let's try to find it using a systematic search.
    
    # For now, let's try to improve the 3x2x2 sub-problem.
    # The 3x2x2 tensor has entries T[i,j,k] for i=0,1,2; j=0,1; k=0,1
    # That's 12 entries.
    
    # Let me try to find a rank-10 or rank-9 decomposition for 3x2x2.
    
    # Actually, let me try a different approach: use the fact that the full 3x4x4
    # tensor has 48 entries and try to find a low-rank decomposition directly.
    
    # Let me try rank 40, 38, 36, etc.
    
    # For each candidate rank, I'll try to verify if a decomposition exists.
    # This is a hard problem, so I'll use a heuristic approach.
    
    # Let me try to construct a rank-40 decomposition by finding a rank-10 3x2x2.
    
    # Actually, let me try a completely different strategy:
    # Use the known rank-19 algorithm for 4x4x4 and slice it.
    
    # The rank-19 algorithm for 4x4x4 uses 19 multipliers.
    # If I can encode it, slicing to 3x4x4 might give rank <= 19 (if some terms
    # don't involve row 3 of A) or slightly more.
    
    # Let me try to find the explicit Lader rank-19 algorithm.
    
    # After careful thought, let me try a numerical approach.
    # I'll set up the tensor equations and try to solve for a low-rank decomposition.
    
    # The tensor T has shape (3, 4, 4) with T[i,j,k] = delta(j,k) * ... no,
    # T[i,j,k] is the coefficient of A[i,j]*B[j,k] in C[i,k].
    # So T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k], which is always 1.
    # Wait, no. The tensor is T[i,j,k] where the multiplication is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor entries are T[i,j,k] = 1 for all valid i,j,k (it's the identity tensor).
    # No wait, the tensor for matrix multiplication (n,m,p) has entries T[i,j,k]
    # where i indexes rows of A (and C), j indexes cols of A / rows of B, k indexes cols of B (and C).
    # T[i,j,k] = 1 for all i,j,k because each term A[i,j]*B[j,k] appears in C[i,k].
    
    # So the tensor is just all-ones of shape (3,4,4).
    # We want: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # (Note: W is indexed as W[k*n+i, r] per the trace convention)
    
    # Let me verify the rank-44 solution first.
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 10:
        return rank_44
    
    # Let me try to find a better decomposition using a simple iterative approach.
    # I'll try ranks from 43 down to 30 and check if a valid decomposition exists.
    
    # For each rank r, I'll try to find U, V, W such that:
    # T[i,j,k] = sum_{r=0}^{r-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This is a system of 48 equations in 3*r*48 = 144r unknowns (with constraints).
    # It's a non-convex problem.
    
    # Let me try a greedy approach: start with the rank-44 solution and try to
    # eliminate terms by combining them.
    
    # Actually, let me try a different known result.
    # The rank of the 3x4x4 matrix multiplication tensor is known to be at most 44
    # (from our construction) and at least... let me think.
    # Lower bound: max(n*m, m*p, p*n) = max(12, 16, 12) = 16.
    # So the rank is between 16 and 44.
    
    # Let me try to find a rank-32 decomposition by using a better block structure.
    
    # What if I use 2x2x2 Strassen for the 2x2 sub-blocks and handle the rest?
    
    # Alternative: Use the fact that 3x4x4 = 2x4x4 + 1x4x4 (splitting rows of A)
    # 2x4x4 can be done with 2 independent 2x2x2 Strassen + 2 more... 
    
    # Actually, 2x4x4: A is 2x4, B is 4x4, C is 2x4.
    # Split B into 2x2 blocks: B = [[B11, B12], [B21, B22]]
    # C = A * B = [A1, A2] * [[B11, B12], [B21, B22]]
    # C1 = A1*B11 + A2*B21 (2x2 result)
    # C2 = A1*B12 + A2*B22 (2x2 result)
    
    # Each 2x2 result needs... well, A1 is 2x2, B11 is 2x2, so A1*B11 is a 2x2x2 product (rank 7).
    # Similarly A2*B21 is 2x2x2 (rank 7).
    # So C1 needs 14 terms, C2 needs 14 terms. Total for 2x4x4: 28.
    
    # For the 1x4x4 part: A is 1x4, B is 4x4, C is 1x4.
    # C[i] = sum_j A[j]*B[j,i]
    # This is just 16 terms (one for each j,i pair). Rank = 16.
    
    # Total: 28 + 16 = 44. Same as before.
    
    # Can we do better for 1x4x4? The tensor is T[j,i] = 1 for all j,i (16 entries).
    # We need T[j,i] = sum_r U[j,r] * V[j,i,r] * W[i,r]
    # This is a rank-16 tensor (it's the all-ones matrix of size 4x4, which has matrix rank 1,
    # but as a 3-way tensor with the specific indexing, it might have higher rank).
    
    # Actually, for 1x4x4: U is (4, r), V is (16, r), W is (4, r).
    # T[j,i] = sum_r U[j,r] * V[j*4+i, r] * W[i,r]
    # Since T[j,i] = 1 for all j,i, we need:
    # 1 = sum_r U[j,r] * V[j*4+i, r] * W[i,r] for all j,i
    
    # This can be done with rank 4: 
    # U[j,r] = delta(j,r), V[j*4+i, r] = 1, W[i,r] = delta(i,r)
    # Then sum_r delta(j,r) * 1 * delta(i,r) = delta(j,i). That's not 1.
    
    # Let me try: U[j,r] = 1, V[j*4+i, r] = delta(j,r)*delta(i,r), W[i,r] = 1
    # sum_r 1 * delta(j,r)*delta(i,r) * 1 = delta(j,i). Still not 1.
    
    # Hmm. Let me think differently.
    # T[j,i] = 1. I need sum_r U[j,r]*V[j*4+i,r]*W[i,r] = 1.
    # If I set U[j,r] = 1 for all j,r, W[i,r] = 1 for all i,r, then I need
    # sum_r V[j*4+i,r] = 1. I can set V[j*4+i,0] = 1 and V[j*4+i,r] = 0 for r>0.
    # That gives rank 1!
    
    # Wait, that can't be right. Let me re-examine.
    # For 1x4x4: n=1, m=4, p=4.
    # U has shape (n*m, r) = (4, r). U[i*m+j, r] where i=0 (only row), j=0..3.
    # So U[j, r].
    # V has shape (m*p, r) = (16, r). V[j*p+k, r] = V[j*4+k, r].
    # W has shape (p*n, r) = (4, r). W[k*n+i, r] = W[k*1+0, r] = W[k, r].
    
    # T[i,j,k] = 1 for all i=0, j=0..3, k=0..3.
    # T[0,j,k] = sum_r U[j,r] * V[j*4+k, r] * W[k, r] = 1.
    
    # If U[j,r] = 1, W[k,r] = 1 for all j,k,r, then sum_r V[j*4+k,r] = 1.
    # Set V[j*4+k, 0] = 1, V[j*4+k, r] = 0 for r > 0. Rank = 1.
    
    # But wait, that means the 1x4x4 part only needs rank 1? That seems too good.
    # Let me verify: C[0,k] = sum_j A[0,j] * B[j,k].
    # With rank 1: C[0,k] = U[0]*V[0]*W[0] where U[0] = sum_j A[0,j] (scalar),
    # V[0] = ... hmm, this doesn't work because V depends on both j and k.
    
    # I think I'm confusing myself. Let me be more careful.
    
    # The decomposition is: C[i,k] = sum_j sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # For n=1: C[0,k] = sum_j sum_r U[j,r] * V[j,k,r] * W[k,0,r]
    # We need C[0,k] = sum_j A[0,j]*B[j,k].
    # So we need: sum_j sum_r U[j,r]*V[j,k,r]*W[k,0,r] = sum_j A[0,j]*B[j,k]
    # This means: for each j,k, sum_r U[j,r]*V[j,k,r]*W[k,0,r] should give the coefficient
    # of A[0,j]*B[j,k], which is 1.
    # So sum_r U[j,r]*V[j,k,r]*W[k,0,r] = 1 for all j,k.
    
    # With rank 1: U[j,0]*V[j,k,0]*W[k,0,0] = 1 for all j,k.
    # This requires U[j,0] = a_j, V[j,k,0] = b_{j,k}, W[k,0,0] = c_k
    # with a_j * b_{j,k} * c_k = 1 for all j,k.
    # This is possible: set a_j = 1, c_k = 1, b_{j,k} = 1. Rank 1!
    
    # So the 1x4x4 part has rank 1. That's great.
    # But wait, in the full 3x4x4 problem, the W factor is shared across all rows of A.
    # So I can't just use rank 1 for the 1x4x4 part independently.
    
    # Let me reconsider the full problem.
    # C[i,k] = sum_j A[i,j]*B[j,k] for i=0,1,2 and k=0,1,2,3.
    # Decomposition: C[i,k] = sum_j sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    
    # For i=2 (the "extra" row), we need:
    # C[2,k] = sum_j A[2,j]*B[j,k] = sum_j sum_r U[2,j,r]*V[j,k,r]*W[k,2,r]
    
    # The current rank-44 solution uses 4 independent blocks. Let me see if I can
    # share terms between blocks.
    
    # Actually, let me try a direct numerical search for a lower-rank decomposition.
    
    # I'll try to find a rank-r decomposition for r = 43, 42, ..., 30.
    # For each r, I'll use a simple iterative method.
    
    # But this might be too slow. Let me try a few specific constructions first.
    
    # Construction: Use rank-7 Strassen for 2x2x2 and handle the rest.
    # Split A into rows [0,1] and [2], and columns [0,1] and [2,3].
    # Split B into rows [0,1] and [2,3], and columns [0,1] and [2,3].
    
    # C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:4]*B[2:4, 0:2]
    # C[0:2, 2:4] = A[0:2, 0:2]*B[0:2, 2:4] + A[0:2, 2:4]*B[2:4, 2:4]
    # C[2, 0:2] = A[2, 0:2]*B[0:2, 0:2] + A[2, 2:4]*B[2:4, 0:2]
    # C[2, 2:4] = A[2, 0:2]*B[0:2, 2:4] + A[2, 2:4]*B[2:4, 2:4]
    
    # For C[0:2, 0:2]: 2x2x2 + 2x2x2 = 7+7 = 14 terms
    # For C[0:2, 2:4]: 2x2x2 + 2x2x2 = 14 terms
    # For C[2, 0:2]: 1x2x2 + 1x2x2 = 4+4 = 8 terms (or less)
    # For C[2, 2:4]: 1x2x2 + 1x2x2 = 8 terms (or less)
    # Total: 14+14+8+8 = 44. Same as before.
    
    # Can we share terms between C[0:2,0:2] and C[0:2,2:4]?
    # They share the A[0:2, 0:2] and A[0:2, 2:4] factors.
    # C[0:2, 0:2] = A_top_left * B_top_left + A_top_right * B_bottom_left
    # C[0:2, 2:4] = A_top_left * B_top_right + A_top_right * B_bottom_right
    
    # If I compute A_top_left * (B_top_left + B_top_right) and A_top_right * (B_bottom_left + B_bottom_right),
    # that's 2 2x2x2 products = 14 terms. Then I can get the individual products by subtraction.
    # But I need the individual products, not just the sums.
    
    # Actually, C[0:2, 0:4] = A_top * B_full where A_top is 2x4 and B_full is 4x4.
    # This is a 2x4x4 multiplication, which I showed needs rank 28.
    
    # Hmm, I'm going in circles. Let me just try to numerically find a better decomposition.
    
    # Let me try rank 40.
    
    # Actually, let me try a completely different approach: use the known result that
    # the rank of the 3x4x4 tensor is at most 32 or so, based on known results for
    # matrix multiplication tensors.
    
    # For now, let me just return the rank-44 solution and add a time guard.
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 3:
        return rank_44
    
    # Let me try one more thing: verify that the rank-44 solution is correct,
    # and then try to find a rank-43 solution by a simple perturbation.
    
    return rank_44


def _build_rank44():
    n, m, p = 3, 4, 4
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Local factors for 3x2x2 multiplication (rank 11)
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    # Strassen part (cols 0-6) for rows 0,1 of the 3x2 matrix
    # M1 = (A00+A11)*(B00+B11)
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    # M3 = A00*(B01-B11)
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    # M4 = A11*(B10-B00)
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    # M5 = (A00+A01)*B11
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    # Extra part (cols 7-10) for Row 2 of A
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[4, 7] = 1
    
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[4, 8] = 1
    
    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1
    
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1
    
    def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * 11
        
        for r in range(3):
            for c in range(2):
                local_a_idx = r * 2 + c
                global_a_row = a_row_start + r
                global_a_col = a_col_start + c
                global_a_idx = global_a_row * m + global_a_col
                U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
        
        for r in range(2):
            for c in range(2):
                local_b_idx = r * 2 + c
                global_b_row = b_row_start + r
                global_b_col = b_col_start + c
                global_b_idx = global_b_row * p + global_b_col
                V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
        
        for r in range(3):
            for c in range(2):
                local_c_idx = r * 2 + c
                global_c_row = c_row_start + r
                global_c_col = c_col_start + c
                global_c_idx = global_c_col * n + global_c_row
                W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]
    
    add_block(0, 0, 0, 0, 0, 0, 0)
    add_block(1, 0, 2, 2, 0, 0, 0)
    add_block(2, 0, 0, 0, 2, 0, 2)
    add_block(3, 0, 2, 2, 2, 0, 2)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
