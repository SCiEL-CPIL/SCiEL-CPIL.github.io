"""Matrix multiplication 4x5x5 rank-minimal decomposition."""
import numpy as np
import time
import sys

DIMENSIONS = [4, 5, 5]

def construct():
    """
    Construct a tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    Returns U, V, W with the W convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    # Build schoolbook baseline
    rank = n * m * p  # 4*5*5 = 100
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

    # Try Strassen-based block decomposition for the 2x2 sub-blocks
    # Split A (4x5) into 2 blocks of 2x5
    # Split B (5x5) into 2 blocks of 2x2 plus a 1x5 remainder
    # Actually, let's think about this differently.
    #
    # A is 4x5, B is 5x5, C is 4x5
    # C[i,k] = sum_{j=0}^{4} A[i,j] B[j,k]
    #
    # Split j into two groups: j in {0,1,2} and j in {3,4}
    # Then C[i,k] = sum_{j=0}^{2} A[i,j] B[j,k] + sum_{j=3}^{4} A[i,j] B[j,k]
    #
    # For the second part (j=3,4), we have a 4x2 * 2x5 multiplication
    # For the first part (j=0,1,2), we have a 4x3 * 3x5 multiplication
    #
    # Alternatively, split into 2x2 blocks where possible.
    # Let's use: j in {0,1} and j in {2,3,4}
    # Part 1: 4x2 * 2x5 (schoolbook rank 40)
    # Part 2: 4x3 * 3x5 (schoolbook rank 60)
    # Total: 100, no savings.
    #
    # Better: use Strassen on 2x2 blocks.
    # A is 4x5, B is 5x5
    # Think of A as [A_left | A_right] where A_left is 4x2, A_right is 4x3
    # Think of B as [B_top; B_bottom] where B_top is 2x5, B_bottom is 3x5
    #
    # Hmm, this doesn't directly give 2x2x2 blocks.
    #
    # Let me try a different approach: treat the 4x5 * 5x5 as follows:
    # View A as 2 blocks of 2x5 (top 2 rows, bottom 2 rows)
    # View B as having 5 rows, split into 2 groups: first 2 rows, last 3 rows
    #
    # Actually, the cleanest approach for Strassen savings:
    # We need 2x2x2 blocks. For that we need 2 rows of A, 2 columns of A 
    # (which are 2 rows of B), and 2 columns of B (which are 2 columns of C).
    #
    # Let's partition:
    # Rows of A/C: {0,1} and {2,3}  (two groups of 2)
    # Columns of A / Rows of B: {0,1}, {2,3}, {4}  (three groups: 2,2,1)
    # Columns of B/C: {0,1}, {2,3}, {4}  (three groups: 2,2,1)
    #
    # This gives a block structure. The 2x2x2 blocks occur where both the 
    # A-column group and B-column group have size 2.
    #
    # Block (i-group, j-group, k-group) where all have size 2:
    # i-group: {0,1} or {2,3}  (2 choices)
    # j-group: {0,1} or {2,3}  (2 choices)  
    # k-group: {0,1} or {2,3}  (2 choices)
    # That's 2*2*2 = 8 blocks of 2x2x2, each with Strassen rank 7 instead of 8.
    # Savings: 8 * (8-7) = 8, so rank = 100 - 8 = 92.
    #
    # The remaining terms involve groups of size 1:
    # j-group {4} with any i-group, k-group: 2*2*1 = 4 terms each of rank 1... 
    # Actually let me count more carefully.
    #
    # Total terms = sum over all block combinations of (size_i * size_j * size_k)
    # = (2+2) * (2+2+1) * (2+2+1) = 4 * 5 * 5 = 100. ✓
    #
    # With Strassen on 2x2x2 blocks:
    # For blocks where i-size=2, j-size=2, k-size=2: rank 7 instead of 8
    # Number of such blocks: 2 * 2 * 2 = 8, saving 8 total.
    # All other blocks: rank = size_i * size_j * size_k (schoolbook)
    #
    # Let's enumerate:
    # i-sizes: [2, 2]
    # j-sizes: [2, 2, 1]
    # k-sizes: [2, 2, 1]
    #
    # Blocks with all size 2: i in {0,1}, j in {0,1,2}, k in {0,1,2} -> 8 blocks, each rank 7
    # Blocks with at least one size 1:
    #   j-size=1: 2*1*3 = 6 terms (i has 2 groups, k has 3 groups) -> rank 2*1*3=6... 
    #   Actually each such block has rank = size_i * 1 * size_k
    #   Sum: for j-group {4}: sum over i-groups (2,2) and k-groups (2,2,1): 2*2+2*2+2*1 = 4+4+2=10
    #   Similarly for k-group {4}: 10
    #   For both j-group {4} and k-group {4}: 2+2 = 4 (counted in both above, subtract once)
    #   Wait, let me just sum: total = sum_{ig,jg,kg} size_i(ig)*size_j(jg)*size_k(kg) = 100
    #   Strassen savings only on 2x2x2 blocks.
    #
    # So total rank = 100 - 8 = 92.
    #
    # Let me implement this.

    def strassen_decompose_2x2x2(i0, j0, k0, n, m, p):
        """
        Return list of (U_dict, V_dict, W_dict) for Strassen 2x2x2 block
        starting at row i0, col j0 (of A), col k0 (of B).
        Flat indices: U: i*m+j, V: j*p+k, W: k*n+i
        """
        terms = []
        # Strassen products M1..M7
        # a00=A[i0,j0], a01=A[i0,j0+1], a10=A[i0+1,j0], a11=A[i0+1,j0+1]
        # b00=B[j0,k0], b01=B[j0,k0+1], b10=B[j0+1,k0], b11=B[j0+1,k0+1]
        
        # M1 = (a00+a11)(b00+b11)
        u1 = {(i0,j0): 1, (i0+1,j0+1): 1}
        v1 = {(j0,k0): 1, (j0+1,k0+1): 1}
        w1 = {(k0,i0): 1, (k0+1,i0+1): 1}
        terms.append((u1, v1, w1))
        
        # M2 = (a10+a11)(b00)
        u2 = {(i0+1,j0): 1, (i0+1,j0+1): 1}
        v2 = {(j0,k0): 1}
        w2 = {(k0,i0+1): 1, (k0+1,i0+1): 1}  # C[i0+1,k0] += M2, C[i0+1,k0+1] += M2
        terms.append((u2, v2, w2))
        
        # M3 = (a00)(b01-b11)
        u3 = {(i0,j0): 1}
        v3 = {(j0,k0+1): 1, (j0+1,k0+1): -1}
        w3 = {(k0+1,i0): 1, (k0+1,i0+1): 1}  # C[i0,k0+1] += M3, C[i0+1,k0+1] += M3
        terms.append((u3, v3, w3))
        
        # M4 = (a11)(b10-b00)
        u4 = {(i0+1,j0+1): 1}
        v4 = {(j0+1,k0): 1, (j0,k0): -1}
        w4 = {(k0,i0): 1, (k0,i0+1): 1}  # C[i0,k0] += M4, C[i0+1,k0] += M4
        terms.append((u4, v4, w4))
        
        # M5 = (a00+a01)(b11)
        u5 = {(i0,j0): 1, (i0,j0+1): 1}
        v5 = {(j0+1,k0+1): 1}
        w5 = {(k0,i0): -1, (k0+1,i0): 1}  # C[i0,k0] -= M5, C[i0,k0+1] += M5
        terms.append((u5, v5, w5))
        
        # M6 = (a10-a00)(b00+b01)
        u6 = {(i0+1,j0): 1, (i0,j0): -1}
        v6 = {(j0,k0): 1, (j0,k0+1): 1}
        w6 = {(k0+1,i0+1): 1}  # C[i0+1,k0+1] += M6
        terms.append((u6, v6, w6))
        
        # M7 = (a01-a11)(b10+b11)
        u7 = {(i0,j0+1): 1, (i0+1,j0+1): -1}
        v7 = {(j0+1,k0): 1, (j0+1,k0+1): 1}
        w7 = {(k0,i0): 1}  # C[i0,k0] += M7
        terms.append((u7, v7, w7))
        
        return terms

    def schoolbook_term(i, j, k, n, m, p):
        """Return (u_dict, v_dict, w_dict) for a single schoolbook term."""
        u = {(i, j): 1}
        v = {(j, k): 1}
        w = {(k, i): 1}
        return (u, v, w)

    # Partition
    i_groups = [(0, 2), (2, 4)]  # groups of size 2
    j_groups = [(0, 2), (2, 4), (4, 5)]  # sizes 2, 2, 1
    k_groups = [(0, 2), (2, 4), (4, 5)]  # sizes 2, 2, 1

    all_terms = []

    for ig_start, ig_end in i_groups:
        for jg_start, jg_end in j_groups:
            for kg_start, kg_end in k_groups:
                size_i = ig_end - ig_start
                size_j = jg_end - jg_start
                size_k = kg_end - kg_start

                if size_i == 2 and size_j == 2 and size_k == 2:
                    # Use Strassen
                    terms = strassen_decompose_2x2x2(ig_start, jg_start, kg_start, n, m, p)
                    all_terms.extend(terms)
                else:
                    # Schoolbook for this block
                    for i in range(ig_start, ig_end):
                        for j in range(jg_start, jg_end):
                            for k in range(kg_start, kg_end):
                                all_terms.append(schoolbook_term(i, j, k, n, m, p))

    # Convert to U, V, W matrices
    rank_new = len(all_terms)
    U_new = np.zeros((n * m, rank_new))
    V_new = np.zeros((m * p, rank_new))
    W_new = np.zeros((p * n, rank_new))

    for r, (u_dict, v_dict, w_dict) in enumerate(all_terms):
        for (i, j), val in u_dict.items():
            U_new[i * m + j, r] = val
        for (j, k), val in v_dict.items():
            V_new[j * p + k, r] = val
        for (k, i), val in w_dict.items():
            W_new[k * n + i, r] = val

    # Verify the decomposition
    # The matmul tensor T has T[i,j,k] = 1 for all valid (i,j,k)
    # Decomposition: sum_r U[:,r] * V[:,r] * W[:,r] should equal T
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Check a sample
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_new[i*m+j, r] * V_new[j*p+k, r] * W_new[k*n+i, r] for r in range(rank_new))
                if abs(val - 1.0) > 1e-9:
                    error += 1

    if error == 0:
        # Convert to lists for the interface
        U_list = U_new.tolist()
        V_list = V_new.tolist()
        W_list = W_new.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fall back to schoolbook
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
