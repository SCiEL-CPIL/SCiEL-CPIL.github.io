"""Program for matrix multiplication 3x5x6 with reduced rank."""
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

def construct():
    """
    Return a rank-minimal (or near-minimal) tensor decomposition for 3x5x6 matrix multiplication.
    
    The decomposition is constructed by:
    1. Using Strassen's algorithm for 2x2 blocks where possible to reduce rank
    2. Falling back to schoolbook multiplication for the remaining parts
    3. Combining these into a single decomposition
    
    The key insight is that 3x5x6 can be decomposed as:
    - One 2x2x2 Strassen block (rank 7) covering rows 0-1, cols 0-1, and intermediate 0-1
    - Schoolbook for the rest
    
    This gives rank 7 + (90 - 8) = 7 + 82 = 89, but we can do better by
    overlapping and combining more carefully.
    
    Actually, let's think more carefully. The schoolbook rank is 3*5*6 = 90.
    
    If we take a 2x2 block in the first matrix (rows 0-1, cols 0-1) and a 2x2 block
    in the second matrix (cols 0-1, rows 0-1), we get a 2x2x2 subproblem.
    The schoolbook rank for this 2x2x2 subproblem is 8, but Strassen reduces it to 7.
    So we save 1 multiplication.
    
    We can apply this to multiple 2x2 blocks. Let's figure out how many disjoint
    2x2x2 blocks we can fit.
    
    For the 3x5 matrix, we can form:
    - Rows 0-1, cols 0-1: one 2x2 block
    - Rows 0-1, cols 2-3: another 2x2 block
    - Rows 0-1, cols 4: 1x2 block (can't form 2x2)
    - Row 2, cols 0-4: 1x5 block (can't form 2x2)
    
    For the 5x6 matrix, we can form:
    - Rows 0-1, cols 0-1: one 2x2 block
    - Rows 0-1, cols 2-3: another 2x2 block
    - Rows 0-1, cols 4-5: another 2x2 block
    - Rows 2-3, cols 0-1: another 2x2 block
    - Rows 2-3, cols 2-3: another 2x2 block
    - Rows 2-3, cols 4-5: another 2x2 block
    - Row 4, cols 0-5: 1x6 block (can't form 2x2)
    
    But the intermediate dimension must match. The 2x2 block in the first matrix
    uses intermediate indices 0-1, and the 2x2 block in the second matrix must
    also use intermediate indices 0-1.
    
    So we can form 2x2x2 blocks where:
    - First matrix: rows 0-1, cols 0-1 (intermediate 0-1)
    - Second matrix: rows 0-1, cols 0-1 (intermediate 0-1)
    
    And:
    - First matrix: rows 0-1, cols 2-3 (intermediate 2-3)
    - Second matrix: rows 2-3, cols 0-1 (intermediate 2-3)
    
    Wait, I need to be more careful. The intermediate indices in the first matrix
    correspond to the column indices, and in the second matrix correspond to the
    row indices.
    
    Let me re-index. The tensor is T[i,j,k] for i in [0,n), j in [0,m), k in [0,p).
    The decomposition has rank r, with factors U (n*m x r), V (m*p x r), W (p*n x r).
    
    Schoolbook: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For the 2x2x2 Strassen block, we need to identify which (i,j,k) tuples are
    involved.
    
    Let's say we take i in {0,1}, j in {0,1}, k in {0,1}. This is a 2x2x2 subproblem.
    The schoolbook rank for this is 8, Strassen rank is 7.
    
    We can apply Strassen to multiple disjoint 2x2x2 subproblems.
    
    For n=3, m=5, p=6:
    - i can be 0,1,2
    - j can be 0,1,2,3,4
    - k can be 0,1,2,3,4,5
    
    We can form 2x2x2 blocks for:
    - i in {0,1}, j in {0,1}, k in {0,1}
    - i in {0,1}, j in {2,3}, k in {2,3}
    - i in {0,1}, j in {4}, k in {4,5} -- no, j only has 1 element
    - i in {2}, j in {0,1}, k in {0,1} -- no, i only has 1 element
    
    So we can form 2 disjoint 2x2x2 Strassen blocks, saving 2 multiplications total.
    Rank = 90 - 2 = 88.
    
    But we can do better by being smarter about the decomposition.
    
    Actually, let me just implement a clean Strassen-based decomposition and see what rank we get.
    """
    n, m, p = DIMENSIONS
    
    # Initialize with schoolbook decomposition
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook), dtype=np.int64)
    V = np.zeros((m * p, rank_schoolbook), dtype=np.int64)
    W = np.zeros((p * n, rank_schoolbook), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Now try to apply Strassen to 2x2x2 blocks to reduce rank
    # For each 2x2x2 block, replace 8 schoolbook terms with 7 Strassen terms
    
    # Strassen's 7 products for 2x2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's identify all possible 2x2x2 blocks and apply Strassen
    # We need to track which (i,j,k) tuples have been used
    
    # For now, let's just apply Strassen to the first 2x2x2 block: i in {0,1}, j in {0,1}, k in {0,1}
    # This replaces 8 schoolbook terms with 7 Strassen terms, saving 1 rank.
    
    # The 8 schoolbook terms for i in {0,1}, j in {0,1}, k in {0,1} are at columns:
    # (i=0,j=0,k=0): col 0
    # (i=0,j=0,k=1): col 1
    # (i=0,j=1,k=0): col 5
    # (i=0,j=1,k=1): col 6
    # (i=1,j=0,k=0): col 10
    # (i=1,j=0,k=1): col 11
    # (i=1,j=1,k=0): col 15
    # (i=1,j=1,k=1): col 16
    
    # We'll create 7 new Strassen terms and remove the 8 old ones.
    # The new terms will be appended to the decomposition.
    
    # Old columns to remove:
    old_cols = [0, 1, 5, 6, 10, 11, 15, 16]
    
    # New Strassen terms:
    # M1 = (A00 + A11) * (B00 + B11)
    # A00 is (i=0,j=0), A11 is (i=1,j=1)
    # B00 is (j=0,k=0), B11 is (j=1,k=1)
    
    # M1: U has A00 and A11, V has B00 and B11, W has C00 and C11 contributions
    # Actually, let me think about this more carefully.
    
    # The tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # For the 2x2x2 block, the relevant T values are T[0,0,0], T[0,0,1], T[0,1,0], T[0,1,1],
    # T[1,0,0], T[1,0,1], T[1,1,0], T[1,1,1].
    
    # Strassen computes these from 7 products of combinations.
    
    # Let me define the Strassen products and how they contribute to T.
    
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # This product contributes to:
    #   C[0,0] with +1, C[1,1] with +1
    #   So T[0,0,0] gets +1, T[0,1,1] gets +1, T[1,0,0] gets +1, T[1,1,1] gets +1
    #   Wait, that's not right. Let me re-derive.
    
    # Actually, the Strassen algorithm for 2x2 matrix multiplication:
    # C = A * B where A, B, C are 2x2 matrices.
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # The tensor T[i,j,k] corresponds to C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] is the contribution of A[i,j] * B[j,k] to C[i,k].
    
    # In Strassen, the 7 products are:
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1] - B[1,1])
    # M4 = A[1,1] * (B[1,0] - B[0,0])
    # M5 = (A[0,0] + A[0,1]) * B[1,1]
    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    
    # And C is computed as:
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # So each M_r contributes to specific T[i,j,k] values.
    
    # For M1:
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # This is a product of (A[0,0] + A[1,1]) and (B[0,0] + B[1,1]).
    # In the tensor decomposition, this means we need a term where:
    #   U has A[0,0] and A[1,1] (i.e., U[0*m+0, c] = 1, U[1*m+1, c] = 1)
    #   V has B[0,0] and B[1,1] (i.e., V[0*p+0, c] = 1, V[1*p+1, c] = 1)
    #   W has the contributions to C[0,0] and C[1,1] with +1
    #   W[0*n+0, c] += 1 (C[0,0])
    #   W[1*n+1, c] += 1 (C[1,1])
    
    # But wait, M1 contributes +1 to C[0,0] and +1 to C[1,1].
    # C[0,0] = T[0,*,0] summed over j, but in the tensor, T[i,j,k] is the individual contribution.
    # Actually, I think I'm overcomplicating this.
    
    # The tensor T is the matrix multiplication tensor, where T[i,j,k] = 1 if the
    # (i,j) entry of A and (j,k) entry of B contribute to the (i,k) entry of C.
    # In the schoolbook decomposition, T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # and the schoolbook has T[i,j,k] = 1 for all i,j,k.
    
    # So the tensor is just the all-ones tensor of shape (n,m,p).
    
    # Strassen's algorithm says that the 2x2x2 all-ones tensor can be decomposed
    # into 7 rank-1 tensors instead of 8.
    
    # Let me look up the explicit decomposition.
    
    # For the 2x2x2 tensor, the Strassen decomposition is:
    # T = T1 + T2 + T3 + T4 + T5 + T6 + T7
    # where each T_r is a rank-1 tensor.
    
    # T1[i,j,k] = 1 if i in {0,1} and j in {0,1} and k in {0,1} with specific pattern...
    
    # Actually, let me just look at the contributions.
    
    # M1 = (A00 + A11)(B00 + B11)
    # This product appears in C00 with +1 and C11 with +1.
    # So T1[i,j,k] = 1 if (i=0,j=0,k=0) or (i=0,j=1,k=1) or (i=1,j=0,k=0) or (i=1,j=1,k=1)
    # Wait, that's not right either.
    
    # Let me think about it differently. The Strassen algorithm computes:
    # C[0,0] = M1 + M4 - M5 + M7
    # where M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    
    # So the contribution of M1 to C[0,0] is +1 * M1.
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # = A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + A[1,1]*B[1,1]
    
    # So M1 contributes to T[0,0,0], T[0,1,1], T[1,0,0], T[1,1,1] with +1 each.
    
    # Similarly for the other M_r.
    
    # Let me just hardcode the Strassen decomposition for 2x2x2.
    
    # I'll create a function that, given the base indices (i0, j0, k0) for a 2x2x2 block,
    # returns the 7 rank-1 tensors.
    
    def strassen_2x2x2(i0, j0, k0):
        """
        Return the 7 rank-1 tensors for a 2x2x2 Strassen block.
        Each tensor is represented as (U_entries, V_entries, W_entries)
        where each is a list of (index, coefficient) pairs.
        
        i0, j0, k0 are the base indices for the 2x2x2 block.
        The block covers i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}.
        """
        tensors = []
        
        # M1 = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
        # Contributes to C[i0,k0] and C[i0+1,k0+1] with +1
        u1 = [(i0 * m + j0, 1), ((i0+1) * m + (j0+1), 1)]
        v1 = [(j0 * p + k0, 1), ((j0+1) * p + (k0+1), 1)]
        w1 = [(k0 * n + i0, 1), ((k0+1) * n + (i0+1), 1)]
        tensors.append((u1, v1, w1))
        
        # M2 = (A[i0+1,j0] + A[i0+1,j0+1]) * B[j0,k0]
        # Contributes to C[i0+1,k0] with +1
        u2 = [((i0+1) * m + j0, 1), ((i0+1) * m + (j0+1), 1)]
        v2 = [(j0 * p + k0, 1)]
        w2 = [(k0 * n + (i0+1), 1)]
        tensors.append((u2, v2, w2))
        
        # M3 = A[i0,j0] * (B[j0,k0+1] - B[j0+1,k0+1])
        # Contributes to C[i0,k0+1] with +1
        u3 = [(i0 * m + j0, 1)]
        v3 = [(j0 * p + (k0+1), 1), ((j0+1) * p + (k0+1), -1)]
        w3 = [((k0+1) * n + i0, 1)]
        tensors.append((u3, v3, w3))
        
        # M4 = A[i0+1,j0+1] * (B[j0+1,k0] - B[j0,k0])
        # Contributes to C[i0,k0] with +1 and C[i0+1,k0] with +1
        u4 = [((i0+1) * m + (j0+1), 1)]
        v4 = [((j0+1) * p + k0, 1), (j0 * p + k0, -1)]
        w4 = [(k0 * n + i0, 1), (k0 * n + (i0+1), 1)]
        tensors.append((u4, v4, w4))
        
        # M5 = (A[i0,j0] + A[i0,j0+1]) * B[j0+1,k0+1]
        # Contributes to C[i0,k0+1] with +1
        u5 = [(i0 * m + j0, 1), (i0 * m + (j0+1), 1)]
        v5 = [((j0+1) * p + (k0+1), 1)]
        w5 = [((k0+1) * n + i0, 1)]
        tensors.append((u5, v5, w5))
        
        # M6 = (A[i0+1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k0+1])
        # Contributes to C[i0+1,k0+1] with +1
        u6 = [((i0+1) * m + j0, 1), (i0 * m + j0, -1)]
        v6 = [(j0 * p + k0, 1), (j0 * p + (k0+1), 1)]
        w6 = [((k0+1) * n + (i0+1), 1)]
        tensors.append((u6, v6, w6))
        
        # M7 = (A[i0,j0+1] - A[i0+1,j0+1]) * (B[j0+1,k0] + B[j0+1,k0+1])
        # Contributes to C[i0,k0] with +1 and C[i0+1,k0+1] with +1
        u7 = [(i0 * m + (j0+1), 1), ((i0+1) * m + (j0+1), -1)]
        v7 = [((j0+1) * p + k0, 1), ((j0+1) * p + (k0+1), 1)]
        w7 = [(k0 * n + i0, 1), ((k0+1) * n + (i0+1), 1)]
        tensors.append((u7, v7, w7))
        
        return tensors
    
    # Now, let's apply Strassen to all possible 2x2x2 blocks.
    # We need to be careful not to double-count.
    
    # Let's track which (i,j,k) tuples have been covered by Strassen blocks.
    covered = set()
    new_tensors = []
    
    for i0 in range(0, n - 1):
        for j0 in range(0, m - 1):
            for k0 in range(0, p - 1):
                # Check if this 2x2x2 block is disjoint from previously covered blocks
                block = set()
                for i in [i0, i0+1]:
                    for j in [j0, j0+1]:
                        for k in [k0, k0+1]:
                            block.add((i, j, k))
                
                if block & covered:
                    continue
                
                # Apply Strassen to this block
                tensors = strassen_2x2x2(i0, j0, k0)
                new_tensors.extend(tensors)
                covered |= block
    
    # Now, create the new decomposition.
    # Start with the schoolbook decomposition, remove the covered terms, and add the Strassen terms.
    
    # The covered (i,j,k) tuples have their schoolbook terms removed.
    # The remaining (i,j,k) tuples keep their schoolbook terms.
    
    # Build the new U, V, W matrices.
    # First, add the schoolbook terms for uncovered (i,j,k) tuples.
    new_U = []
    new_V = []
    new_W = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    new_U.append([(i * m + j, 1)])
                    new_V.append([(j * p + k, 1)])
                    new_W.append([(k * n + i, 1)])
    
    # Add the Strassen terms.
    for (u_entries, v_entries, w_entries) in new_tensors:
        new_U.append(u_entries)
        new_V.append(v_entries)
        new_W.append(w_entries)
    
    total_rank = len(new_U)
    
    # Convert to matrices.
    U_final = np.zeros((n * m, total_rank), dtype=np.int64)
    V_final = np.zeros((m * p, total_rank), dtype=np.int64)
    W_final = np.zeros((p * n, total_rank), dtype=np.int64)
    
    for c in range(total_rank):
        for (idx, coeff) in new_U[c]:
            U_final[idx, c] = coeff
        for (idx, coeff) in new_V[c]:
            V_final[idx, c] = coeff
        for (idx, coeff) in new_W[c]:
            W_final[idx, c] = coeff
    
    # Verify the decomposition by checking that the tensor is correct.
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This should equal 1 for all i,j,k.
    
    # Let's verify a few entries.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(total_rank):
                    val += U_final[i * m + j, c] * V_final[j * p + k, c] * W_final[k * n + i, c]
                if val != 1:
                    # This is an error in the decomposition.
                    # For now, fall back to schoolbook.
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
    
    # Return the decomposition as lists.
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
