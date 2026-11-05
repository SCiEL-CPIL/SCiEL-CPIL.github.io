"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Schoolbook rank as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                U_school[u_idx, idx] = 1
                V_school[v_idx, idx] = 1
                W_school[w_idx, idx] = 1
                idx += 1
    
    # Strategy: Use Strassen's rank-8 algorithm for 3x3x3,
    # replicate for the 8 output columns (with zero-padding to 9)
    # This gives rank 8*9 = 72, same as schoolbook but let's try
    # a direct approach: use the 3x3x3 rank-8 and tile it.
    
    # Actually, for 3x3x8, we can think of it as 8 independent 3x3x1 multiplications
    # or as a 3x3x3 multiplied by 3x3x8.
    
    # Let's try: decompose 3x3x8 by viewing it as (3x3x3) * (3x3x8/3)
    # No, that doesn't work directly.
    
    # Better approach: The tensor for M_{3,3,8} can be decomposed by
    # using the rank-8 decomposition of M_{3,3,3} and applying it
    # to each "block" of the output.
    
    # Actually, the simplest improvement: try to find a rank < 72 decomposition
    # by exploiting structure.
    
    # For 3x3x8: we can use the fact that the middle dimension is 3.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Let's try a Kronecker-product-like approach:
    # If we had M_{3,3,3} with rank 8, then M_{3,3,8} can be seen as
    # M_{3,3,3} ⊗ M_{1,1,8} but that's not standard.
    
    # Alternative: Use the group algebra approach mentioned in the directive.
    # Embed 3x3x8 into Z_9 structure.
    
    # For now, let's implement a concrete rank-72 decomposition that might
    # be improvable, and add a search for better decompositions.
    
    # Try: Can we do better than 72?
    # Lower bound: max(n*m, m*p, n*p) = max(9, 24, 24) = 24
    # Upper bound: 72 (schoolbook)
    
    # Let's try a structured approach:
    # Use the 3x3x3 rank-8 algorithm and extend to 3x3x8.
    # 
    # The 3x3x3 Strassen-like rank-8 decomposition:
    # M_8 = (M_1, M_2, M_3, M_4, M_5, M_6, M_7, M_8)
    # where each M_r = u_r ⊗ v_r ⊗ w_r
    
    # For 3x3x8, we can write:
    # T_{3,3,8} = sum over the 8 output positions
    # This is equivalent to having 8 "layers" of 3x3 multiplication.
    
    # A key insight: if we use the same rank-8 factors for all 8 output columns
    # but with different V factors, we might get rank 8*8 = 64? No, that's not right.
    
    # Let me think more carefully. The tensor is:
    # T[i,j,k] = delta_{ij} * delta_{jk} (for the multiplication)
    # No, T[i,j,k] = 1 means C[i,k] += A[i,j]*B[j,k]
    
    # So T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the multiplication is
    # C = A*B, i.e., C[i,k] = sum_j A[i,j]B[j,k]
    
    # The tensor decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r[i,j] = U[i*m+j, r], V_r[j,k] = V[j*p+k, r], W_r[k,i] = W[k*n+i, r]
    
    # For schoolbook: U_r = e_{i,j}, V_r = e_{j,k}, W_r = e_{k,i}
    
    # Can we do better? For 3x3x3, rank 8 is known.
    # For 3x3x8, one approach:
    # Write 8 = 3 + 3 + 2
    # Use rank-8 for the first 3 columns, rank-8 for next 3, rank-8 for last 2?
    # That gives 24, but we need to be careful about the U factors.
    
    # Actually, a cleaner approach:
    # M_{3,3,8} can be decomposed as:
    # First, note that 3x3x8 = 3x3x(3+3+2)
    # We can use the rank-8 decomposition for 3x3x3 three times (for 9 columns)
    # and then remove the 9th column.
    
    # Rank-8 for 3x3x3 (Strassen-like):
    # Let me define the 8 terms:
    # s1 = (A00+A11+A22) * (B00+B11+B22)  -> but this isn't quite right for 3x3
    
    # Actually, for 3x3x3, the known rank-8 decomposition is more complex.
    # Let me use a different approach: try to find a rank < 72 via optimization.
    
    # Given the time constraint, let's implement a simple structured decomposition
    # that achieves rank 72 (same as schoolbook) but with a path to improvement.
    
    # Actually, let's try a concrete better decomposition:
    # Use the fact that we can share U factors across output columns.
    # 
    # For each (i,j) pair in A, we need to multiply A[i,j] with B[j,:] (a row of B)
    # and add to C[i,:].
    # 
    # So for fixed i,j: C[i,:] += A[i,j] * B[j,:]
    # This is a rank-1 update of the i-th row of C using the j-th row of B.
    # 
    # In tensor form: for each (i,j), we have a term:
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1 for all k, W[k*n+i, r] = 1 for all k
    # But this would require p terms per (i,j) pair, giving n*m*p = 72.
    
    # To do better, we need to share factors.
    # 
    # Key idea: Use the rank-8 decomposition of 3x3x3.
    # The 3x3x3 tensor has rank 8. If we can express 3x3x8 as a combination
    # of 3x3x3 tensors, we might get rank < 72.
    
    # 3x3x8: output has 8 columns. We can write:
    # C[:, 0:3] = A * B[:, 0:3]  (3x3x3, rank 8)
    # C[:, 3:6] = A * B[:, 3:6]  (3x3x3, rank 8)  
    # C[:, 6:8] = A * B[:, 6:8]  (3x3x2, rank <= 6)
    # Total: 8 + 8 + 6 = 22? But the U factors are shared!
    
    # Wait, the U factors for all three blocks are the same (they depend on A only).
    # So we can share them!
    # 
    # For the first 3 output columns (k=0,1,2):
    # Use rank-8 decomposition of 3x3x3:
    # C[i,k] = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] for k=0,1,2
    # 
    # For the next 3 (k=3,4,5):
    # C[i,k] = sum_r U_r[i,j] * V'_r[j,k-3] * W'_r[k-3,i] for k=3,4,5
    # 
    # For the last 2 (k=6,7):
    # C[i,k] = sum_r U_r[i,j] * V''_r[j,k-6] * W''_r[k-6,i] for k=6,7
    
    # But the U factors are the same for all blocks! So total rank = 8 + 8 + rank(3x3x2)
    # 
    # What's the rank of 3x3x2? Lower bound: max(9, 6, 6) = 9.
    # Upper bound: 3*3*2 = 18.
    # 
    # So total rank <= 8 + 8 + 18 = 34? But we need to verify the U sharing works.
    
    # Actually, the issue is that the U factors for the 3x3x3 decomposition are specific.
    # Let me define them properly.
    
    # Rank-8 decomposition of 3x3x3 (one known decomposition):
    # This is non-trivial. Let me use a simpler approach for now.
    
    # Given the complexity, let's implement the schoolbook (rank 72) as the baseline
    # and add a structured decomposition that might achieve lower rank.
    
    # For now, return the schoolbook decomposition.
    if time.time() > deadline - 10:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Try a structured decomposition using shared U factors:
    # Split output into blocks of size 3, 3, 2
    # Use rank-8 for 3x3x3 blocks and rank-6 for 3x3x2 block (if possible)
    
    # For 3x3x2, the minimal rank is at least 9 (since n*m=9).
    # Let's use rank 9 for the 3x3x2 part (schoolbook for that block).
    
    # Total rank: 8 + 8 + 9 = 25
    # But we need to verify this works with shared U factors.
    
    # The key question: can the same U factors be used for different V,W factors?
    # Yes! The tensor decomposition is:
    # T = sum_r U_r ⊗ V_r ⊗ W_r
    # If we have T1 = sum_{r=1}^{8} U_r ⊗ V_r ⊗ W_r (for 3x3x3)
    # and T2 = sum_{r=1}^{8} U_r ⊗ V'_r ⊗ W'_r (for another 3x3x3)
    # then T1 + T2 = sum_{r=1}^{8} U_r ⊗ (V_r + V'_r) ⊗ ... NO, that's not right.
    # 
    # Actually, T1 and T2 have different W factors (different output indices).
    # So they can't be combined that way.
    
    # Let me reconsider. The full tensor for 3x3x8 is:
    # T[i,j,k] = 1 for all i,j,k (with the multiplication structure)
    # 
    # A decomposition is: T = sum_{r=1}^{R} U_r ⊗ V_r ⊗ W_r
    # where U_r is 3x3, V_r is 3x8, W_r is 8x3.
    
    # For the schoolbook: R = 72, with U_r = e_{i,j}, V_r = e_{j,k}, W_r = e_{k,i}
    
    # To get a better decomposition, we need to find U_r, V_r, W_r with R < 72.
    
    # One approach: use the rank-8 decomposition of 3x3x3 as a building block.
    # The 3x3x3 tensor T_{3,3,3}[i,j,k] = 1 can be written as:
    # T_{3,3,3} = sum_{r=1}^{8} u_r ⊗ v_r ⊗ w_r
    # 
    # For 3x3x8, we can write:
    # T_{3,3,8}[i,j,k] = T_{3,3,3}[i,j,k mod 3] * (1 if k < 6) + ... 
    # This doesn't quite work because the structure is different.
    
    # Let me try a different approach: direct optimization.
    # Given the time constraint, I'll implement the schoolbook and add a note
    # that future improvements could use tensor decomposition algorithms.
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
