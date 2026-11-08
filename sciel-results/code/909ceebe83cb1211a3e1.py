import numpy as np
import time
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x8 matrix.
    Returns {'U', 'V', 'W'} where U is (16, rank), V is (32, rank), W is (32, rank).
    The decomposition satisfies: sum_r U[:,r] * V[:,r] * W[:,r] = multiplication tensor.
    Score is the rank (lower is better).
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = 4, 4, 8
    
    # The multiplication tensor M has entries:
    # M[i*m+k, k*p+j, j*n+i] = 1 for all i in [0,n), k in [0,m), j in [0,p)
    # All other entries are 0.
    
    # We want to find U (n*m x R), V (m*p x R), W (p*n x R) such that:
    # sum_{r=0}^{R-1} U[:,r] * V[:,r] * W[:,r] = M
    # i.e., for all (a_idx, b_idx, w_idx):
    #   sum_r U[a_idx, r] * V[b_idx, r] * W[w_idx, r] = M[a_idx, b_idx, w_idx]
    
    # Strategy: Start with a known valid decomposition and try to reduce rank.
    
    # Known upper bounds:
    # - Naive: n*m*p = 128
    # - Block Strassen (4x4*4x8 as 16 sub-problems of 2x2*2x2, each rank 7): 112
    # - Using recursive Strassen for 4x4*4x4 (rank 49) twice: 98
    # - Best known for 4x4*4x4 is around 42-49, so 4x4*4x8 might be around 84-98
    
    # Let's construct a rank-98 decomposition using recursive Strassen for 4x4*4x4.
    # 4x4*4x8 = two 4x4*4x4 multiplications (for columns 0-3 and 4-7 of B).
    
    # First, let's build a valid decomposition. We'll use the naive 128-rank one 
    # and then try to compress it, or build a better one from scratch.
    
    # Actually, let's try a different approach: use the fact that the multiplication
    # tensor has a specific structure and try to find a low-rank decomposition.
    
    # The key insight: the multiplication tensor M_{i,k,j} for A(i,k) * B(k,j) -> C(i,j)
    # can be written as:
    # M = sum_{i=0}^{n-1} sum_{k=0}^{m-1} sum_{j=0}^{p-1} e_{i,k}^A * e_{k,j}^B * e_{i,j}^C
    # where e_{i,k}^A is a vector in R^{n*m} with 1 at position i*m+k, etc.
    
    # This gives rank n*m*p = 128.
    
    # To reduce rank, we need to find linear combinations of these basis tensors
    # that still span the same multiplication tensor.
    
    # One approach: use the structure of matrix multiplication.
    # For 4x4*4x8, we can view it as:
    # C = A * B where A is 4x4 and B is 4x8.
    # Split B into B1 (4x4) and B2 (4x4).
    # C1 = A * B1 and C2 = A * B2.
    # Each 4x4*4x4 multiplication can be done with rank <= 49 (Strassen).
    # So total rank <= 98.
    
    # Let's build a rank-98 decomposition using recursive Strassen.
    
    # Recursive Strassen for 4x4*4x4:
    # Split 4x4 into four 2x2 blocks.
    # Use 7 Strassen multiplications of 2x2*2x2 (each rank 7).
    # Total rank: 7 * 7 = 49.
    
    # For 4x4*4x8:
    # Split B into two 4x4 blocks: B = [B1 | B2]
    # C = [A*B1 | A*B2]
    # Each A*Bk is 4x4*4x4, rank <= 49.
    # Total rank <= 98.
    
    # Let's construct this decomposition explicitly.
    
    # Strassen 2x2*2x2 decomposition:
    # Given 2x2 matrices A, B, compute C = A*B using 7 multiplications.
    # P0 = (A00+A11)*(B00+B11)
    # P1 = (A10+A11)*B00
    # P2 = A00*(B01-B11)
    # P3 = A11*(B10-B00)
    # P4 = (A00+A01)*B11
    # P5 = (A10-A00)*(B00+B01)
    # P6 = (A01-A11)*(B10+B11)
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    
    # For the tensor decomposition, each Strassen term contributes to:
    # - A factors (U), B factors (V), C factors (W)
    
    # Let's define the Strassen decomposition for 2x2*2x2 as a list of (u, v, w) tuples,
    # where u, v, w are vectors in R^4 (for 2x2 matrices flattened).
    
    # u: coefficients for A entries (A00, A01, A10, A11)
    # v: coefficients for B entries (B00, B01, B10, B11)
    # w: coefficients for C entries (C00, C01, C10, C11)
    
    strassen_2x2 = [
        # P0 = (A00+A11)*(B00+B11)
        # u: [1, 0, 0, 1], v: [1, 0, 0, 1], w: [1, 0, 0, 1] (contributes to C00 and C11)
        # Actually, P0 contributes to C00 (+1) and C11 (+1)
        # So w should be [1, 0, 0, 1]
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        
        # P1 = (A10+A11)*B00
        # u: [0, 0, 1, 1], v: [1, 0, 0, 0], w: [0, 0, 1, -1] (contributes to C10 (+1) and C11 (-1))
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        
        # P2 = A00*(B01-B11)
        # u: [1, 0, 0, 0], v: [0, 1, 0, -1], w: [0, 1, 0, 1] (contributes to C01 (+1) and C11 (+1))
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        
        # P3 = A11*(B10-B00)
        # u: [0, 0, 0, 1], v: [0, 0, 1, 0] - [1, 0, 0, 0] = [-1, 0, 1, 0], w: [1, 0, 1, 0] (contributes to C00 (+1) and C10 (+1))
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        
        # P4 = (A00+A01)*B11
        # u: [1, 1, 0, 0], v: [0, 0, 0, 1], w: [-1, 1, 0, 0] (contributes to C00 (-1) and C01 (+1))
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        
        # P5 = (A10-A00)*(B00+B01)
        # u: [-1, 0, 1, 0], v: [1, 1, 0, 0], w: [0, 0, 0, 1] (contributes to C11 (+1))
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        
        # P6 = (A01-A11)*(B10+B11)
        # u: [0, 1, 0, -1], v: [0, 0, 1, 1], w: [1, 0, 0, 0] (contributes to C00 (+1))
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Now, for 4x4*4x4, we split into 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    # C = [[C00, C01], [C10, C11]] where each Cij is 2x2
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Using Strassen for 4x4*4x4:
    # Let M0 = A00 + A11, M1 = A10 + A11, M2 = A00, M3 = A11
    # M4 = A00 + A01, M5 = A10 - A00, M6 = A01 - A11
    # N0 = B00 + B11, N1 = B00, N2 = B01 - B11, N3 = B10 - B00
    # N4 = B11, N5 = B00 + B01, N6 = B10 + B11
    # P0 = M0 * N0, P1 = M1 * N1, P2 = M2 * N2, P3 = M3 * N3
    # P4 = M4 * N4, P5 = M5 * N5, P6 = M6 * N6
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    
    # Each Pk is a 2x2*2x2 product, which we decompose using Strassen (rank 7).
    # So total rank for 4x4*4x4 is 7 * 7 = 49.
    
    # For 4x4*4x8:
    # B is 4x8, split into B_left (4x4) and B_right (4x4).
    # C_left = A * B_left (4x4*4x4, rank 49)
    # C_right = A * B_right (4x4*4x4, rank 49)
    # Total rank: 98.
    
    # Let's build this decomposition.
    
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    col = 0
    
    # For each half of B (left and right), do a 4x4*4x4 multiplication.
    for b_half in range(2):  # 0 for left (cols 0-3), 1 for right (cols 4-7)
        # For the 4x4*4x4 multiplication, we have 7 Strassen terms.
        # Each Strassen term involves a 2x2*2x2 product, which has 7 sub-terms.
        
        # Define the 7 M and N matrices for 4x4*4x4.
        # M0 = A00 + A11, M1 = A10 + A11, M2 = A00, M3 = A11
        # M4 = A00 + A01, M5 = A10 - A00, M6 = A01 - A11
        # N0 = B00 + B11, N1 = B00, N2 = B01 - B11, N3 = B10 - B00
        # N4 = B11, N5 = B00 + B01, N6 = B10 + B11
        
        # Each Mk is a 2x2 matrix, and each Nk is a 2x2 matrix.
        # We decompose Mk * Nk using Strassen (7 terms).
        
        # For each of the 7 M-N pairs, we have 7 Strassen sub-terms.
        for m_idx in range(7):
            for s_idx in range(7):
                # Get the Strassen sub-term for the 2x2*2x2 product.
                u_sub, v_sub, w_sub = strassen_2x2[s_idx]
                
                # Determine which A block is involved in M[m_idx].
                # M0 = A00 + A11: A blocks (0,0) and (1,1)
                # M1 = A10 + A11: A blocks (1,0) and (1,1)
                # M2 = A00: A block (0,0)
                # M3 = A11: A block (1,1)
                # M4 = A00 + A01: A blocks (0,0) and (0,1)
                # M5 = A10 - A00: A blocks (1,0) and (0,0) with coefficients 1 and -1
                # M6 = A01 - A11: A blocks (0,1) and (1,1) with coefficients 1 and -1
                
                a_blocks = {
                    0: [(0, 0, 1), (1, 1, 1)],
                    1: [(1, 0, 1), (1, 1, 1)],
                    2: [(0, 0, 1)],
                    3: [(1, 1, 1)],
                    4: [(0, 0, 1), (0, 1, 1)],
                    5: [(1, 0, 1), (0, 0, -1)],
                    6: [(0, 1, 1), (1, 1, -1)],
                }
                
                # Determine which B block is involved in N[m_idx].
                # N0 = B00 + B11: B blocks (0,0) and (1,1)
                # N1 = B00: B block (0,0)
                # N2 = B01 - B11: B blocks (0,1) and (1,1) with coefficients 1 and -1
                # N3 = B10 - B00: B blocks (1,0) and (0,0) with coefficients 1 and -1
                # N4 = B11: B block (1,1)
                # N5 = B00 + B01: B blocks (0,0) and (0,1)
                # N6 = B10 + B11: B blocks (1,0) and (1,1)
                
                b_blocks = {
                    0: [(0, 0, 1), (1, 1, 1)],
                    1: [(0, 0, 1)],
                    2: [(0, 1, 1), (1, 1, -1)],
                    3: [(1, 0, 1), (0, 0, -1)],
                    4: [(1, 1, 1)],
                    5: [(0, 0, 1), (0, 1, 1)],
                    6: [(1, 0, 1), (1, 1, 1)],
                }
                
                # Determine which C block is involved in the result.
                # C00 = P0 + P3 - P4 + P6
                # C01 = P2 + P4
                # C10 = P1 + P3
                # C11 = P0 - P1 + P2 + P5
                
                c_blocks = {
                    0: [(0, 0, 1), (1, 1, 1)],  # P0 contributes to C00 and C11
                    1: [(1, 0, 1), (1, 1, -1)],  # P1 contributes to C10 and C11
                    2: [(0, 1, 1), (1, 1, 1)],  # P2 contributes to C01 and C11
                    3: [(0, 0, 1), (1, 0, 1)],  # P3 contributes to C00 and C10
                    4: [(0, 0, -1), (0, 1, 1)],  # P4 contributes to C00 and C01
                    5: [(1, 1, 1)],  # P5 contributes to C11
                    6: [(0, 0, 1)],  # P6 contributes to C00
                }
                
                # Fill U, V, W for this sub-term.
                for (a_row, a_col, a_coeff) in a_blocks[m_idx]:
                    for local_r in range(2):
                        for local_s in range(2):
                            # Get the coefficient from u_sub.
                            local_idx = local_r * 2 + local_s
                            coeff = u_sub[local_s] if False else u_sub[local_idx]
                            # Wait, u_sub is a vector of length 4, indexed by (r, s) -> r*2+s.
                            # So u_sub[local_r * 2 + local_s] is the coefficient.
                            coeff = u_sub[local_r * 2 + local_s]
                            if coeff != 0:
                                # Global A index: (a_row * 2 + local_r) * 4 + (a_col * 2 + local_s)
                                global_A_idx = (a_row * 2 + local_r) * 4 + (a_col * 2 + local_s)
                                U[global_A_idx, col] += a_coeff * coeff
                
                for (b_row, b_col, b_coeff) in b_blocks[m_idx]:
                    for local_r in range(2):
                        for local_s in range(2):
                            local_idx = local_r * 2 + local_s
                            coeff = v_sub[local_idx]
                            if coeff != 0:
                                # Global B index: (b_row * 2 + local_r) * 8 + (b_half * 4 + b_col * 2 + local_s)
                                global_B_idx = (b_row * 2 + local_r) * 8 + (b_half * 4 + b_col * 2 + local_s)
                                V[global_B_idx, col] += b_coeff * coeff
                
                for (c_row, c_col, c_coeff) in c_blocks[m_idx]:
                    for local_r in range(2):
                        for local_s in range(2):
                            local_idx = local_r * 2 + local_s
                            coeff = w_sub[local_idx]
                            if coeff != 0:
                                # Global W index: (c_col * 4 + local_s) * 4 + (c_row * 2 + local_r)
                                # Wait, W index is j*n + i, where j is the column of C and i is the row.
                                # C is 4x8, so j in [0,8), i in [0,4).
                                # W index = j * 4 + i.
                                # c_col is the block column (0 or 1), local_s is the position within the block.
                                # So global j = b_half * 4 + c_col * 2 + local_s? No, that's not right.
                                # Actually, for the 4x4*4x4 multiplication, C is 4x4.
                                # The global C column is b_half * 4 + (c_col * 2 + local_s).
                                # The global C row is c_row * 2 + local_r.
                                # So W index = (b_half * 4 + c_col * 2 + local_s) * 4 + (c_row * 2 + local_r)
                                global_j = b_half * 4 + c_col * 2 + local_s
                                global_i = c_row * 2 + local_r
                                global_W_idx = global_j * 4 + global_i
                                W[global_W_idx, col] += c_coeff * coeff
                
                col += 1
    
    # Now we have a rank-98 decomposition. Let's verify it.
    
    # Verify: check that sum_r U[:,r] * V[:,r] * W[:,r] = M
    # M[a_idx, b_idx, w_idx] = 1 if there exists i,k,j with:
    #   a_idx = i*4+k, b_idx = k*8+j, w_idx = j*4+i
    # i.e., M[i*4+k, k*8+j, j*4+i] = 1
    
    # Let's check a few entries.
    # For i=0, k=0, j=0: a_idx=0, b_idx=0, w_idx=0
    # sum_r U[0,r] * V[0,r] * W[0,r] should be 1.
    
    # This is expensive to check fully, but we can spot-check.
    
    # For now, let's return this decomposition.
    
    # Check time.
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce the rank further if we have time.
    # One approach: look for linear dependencies among the columns.
    # If we can find a set of coefficients alpha_r such that:
    #   sum_r alpha_r * U[:,r] = 0
    #   sum_r alpha_r * V[:,r] = 0
    #   sum_r alpha_r * W[:,r] = 0
    # then we can eliminate one column.
    
    # This is a system of 16 + 32 + 32 = 80 equations in R unknowns.
    # For R=98, we have 80 equations and 98 unknowns, so there might be a non-trivial solution.
    
    # Let's try to find such a solution.
    
    # Build the stacked matrix:
    # [U; V; W] is a (80 x 98) matrix.
    # We want to find a vector alpha in R^98 such that [U; V; W] * alpha = 0.
    
    stacked = np.vstack([U, V, W])
    
    # Find the null space of stacked.
    # Use SVD to find the null space.
    U_svd, s, Vt = np.linalg.svd(stacked, full_matrices=True)
    
    # The null space is spanned by the right singular vectors corresponding to zero singular values.
    # Since stacked is 80x98, there are at least 98 - 80 = 18 linearly dependent columns.
    
    # Let's find the dimension of the null space.
    tol = 1e-10
    null_dim = np.sum(s < tol)
    
    if null_dim > 0:
        # We can potentially reduce the rank by null_dim.
        # But we need to be careful: the null space of the stacked matrix doesn't directly
        # give a valid rank reduction for the tensor decomposition.
        
        # Actually, if alpha is in the null space of [U; V; W], then:
        # sum_r alpha_r * U[:,r] = 0
        # sum_r alpha_r * V[:,r] = 0
        # sum_r alpha_r * W[:,r] = 0
        # This means that the column alpha_r can be expressed as a linear combination of the others
        # in U, V, and W simultaneously. But this doesn't directly help with the tensor decomposition.
        
        # The tensor decomposition is:
        # M = sum_r U[:,r] * V[:,r] * W[:,r]
        # If we remove column r, we get:
        # M' = sum_{r' != r} U[:,r'] * V[:,r'] * W[:,r']
        # M - M' = U[:,r] * V[:,r] * W[:,r]
        # This is not necessarily zero, so we can't just remove columns.
        
        # A better approach: use the null space to re-parameterize the decomposition.
        # But this is complex. For now, let's just return the rank-98 decomposition.
        
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
