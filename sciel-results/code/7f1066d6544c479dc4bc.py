import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Block decomposition: 5 = 2 + 2 + 1
    # For 2x2x2 blocks use Strassen (7 terms)
    # For all other blocks use schoolbook
    # For 2x2x1 blocks use a 5-term decomposition (improvement over schoolbook's 4? Actually 4 is better for 2x2x1)
    # Wait, schoolbook for 2x2x1 is 4 terms. Can we do 3? 
    # The tensor rank of 2x2x1 (which is just 2 linear forms) is at most 2 (if we split the 1-dim)
    # Actually a 2x2x1 tensor is T[i,j,k] = A[i,j] * B[j,k] where k is fixed? No.
    # The tensor is T_{i,j,k} = A_{i,j} B_{j,k}. 
    # If the block is 2x2x1, then i in {0,1}, j in {0,1}, k in {0}.
    # T[i, j, 0] = A[i,j] * B[j, 0].
    # This is sum over j of A[i,j] * (B[j,0]).
    # We can decompose this as:
    # Term 1: A[0,0] * B[0,0] -> T[0,0,0]
    # Term 2: A[1,0] * B[0,0] -> T[1,0,0]
    # Term 3: A[0,1] * B[1,0] -> T[0,1,0]
    # Term 4: A[1,1] * B[1,0] -> T[1,1,0]
    # Rank 4. Is rank 3 possible?
    # The tensor is 2x2x1. This is a matricization of size 2x2 for index k=0.
    # T[:, :, 0] is a 2x2 matrix M where M[i,j] = A[i,j] B[j,0].
    # This matrix has rank at most 2.
    # The tensor rank is the minimum number of simple tensors.
    # A simple tensor is u v w^T. Here w is fixed (1-dim). So it's u v * 1.
    # So it's the matrix rank. Rank of M.
    # M[i,j] = A[i,j] B[j,0].
    # Generally rank 2. So rank 2?
    # Let's check.
    # We need c1, c2 such that:
    # A[0,0]B[0,0] = c1[0,0,0] * u1[0] v1[0] + c2[0,0,0] u2[0] v2[0]
    # This is exactly representing the matrix M as sum of 2 rank-1 matrices.
    # Since rank(M) <= 2, we can always do this with 2 terms.
    # Can we always do it with 1 term? Only if rank(M)=1, i.e. A[0,0]B[0,0] / A[0,1]B[1,0] == A[1,0]B[0,0] / A[1,1]B[1,0].
    # Not generally.
    # So 2 terms for 2x2x1 is possible!
    # Let's verify the indices.
    # M[i,j] = A[i,j] * B[j,0] (where 0 is the k-index in the block).
    # Wait, in the block, B is indexed by (j, k) locally. If block is 2x2x1, j is 0,1 and k is 0.
    # So B[0,0], B[1,0].
    # The term is A[i,j] * B[j,k].
    # For k=0: A[i,0]B[0,0] + A[i,1]B[1,0].
    # This is A[:,0]*B[0,0] + A[:,1]*B[1,0].
    # This is a linear combination of 2 vectors in i-space?
    # The tensor T[i,j,k] for k=0 is a 2x2 matrix.
    # T[i,j,0] = A[i,j] * B[j,0].
    # This can be written as:
    # T = u1 v1 + u2 v2
    # where u1, v1, u2, v2 are vectors in R^2 (for i and j).
    # T[i,j] = u1[i] v1[j] + u2[i] v2[j].
    # We want u1, v1, u2, v2 such that u1[i] v1[j] + u2[i] v2[j] = A[i,j] B[j,0].
    # Let B_col = [B[0,0], B[1,0]].
    # T[i,j] = sum_k' A[i,k'] * B[k',0] ... wait j is the inner index.
    # T[i,j] is NOT a matrix product. It's element-wise product A[i,j] * B[j,0]?
    # NO. The tensor is T[i,j,k] = sum_l A[i,l] B[l,k].
    # In a block, we are isolating a specific (i,j,k).
    # The schoolbook term for (i,j,k) is A[i,j] * B[j,k] -> C[i,k].
    # So for a 2x2x1 block (i in {0,1}, j in {0,1}, k=0):
    # We need to reproduce C[i,0] = A[i,0]B[0,0] + A[i,1]B[1,0] for i=0,1.
    # This is C[0,0] and C[1,0].
    # The tensor entries involved are T[0,0,0], T[0,1,0], T[1,0,0], T[1,1,0].
    # Wait, the tensor is T_{i,j,k} which equals 1 if we are at the schoolbook term?
    # No, the matrix multiplication tensor is defined as M_{ijk} = 1? No.
    # The tensor is the structure of the multiplication.
    # T_{(i,k), j} = delta_{i,k}? No.
    # Let's stick to the decomposition logic.
    # We want to represent the contribution of the block.
    # Block 2x2x1: A is 2x2, B is 2x1, C is 2x1.
    # C = A * B.
    # C[0] = A[0,0]B[0] + A[0,1]B[1]
    # C[1] = A[1,0]B[0] + A[1,1]B[1]
    # This is a 2x1 vector result.
    # Can we do this with rank 2?
    # Rank 1: u v w^T. C[i] = u[i] * v[j] * w[k]?
    # The term in the sum is prod U[i,j] * V[j,k] * W[k,i]?
    # No, W[k*n+i, r].
    # The contribution to C[i,k] is sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    # For the block, we fix i in {0,1}, j in {0,1}, k=0.
    # We need:
    # sum_r U_block[i,j] V_block[j,0] W_block[0,i] = A_block[i,j] B_block[j,0]
    # (where U_block, etc are the submatrices of the factor matrices for this block).
    # Note that the RHS depends on i and j.
    # Let x[j] = B_block[j,0].
    # We need sum_r (U_block[:,j] * W_block[j] ) [i] = A_block[i,j] x[j].
    # This looks like we are summing over j? No, r is the rank index. j is fixed in the term?
    # No, the decomposition is sum_r (u_r . a) (v_r . b) w_r^T.
    # For a specific C entry (i,k): sum_r (sum_j U[r,i,j] A[i,j]) (sum_k' V[r,j,k'] B[j,k']) W[r,k,i].
    # This is the standard tensor network contraction.
    # If we restrict to block i in {0,1}, j in {0,1}, k=0:
    # C[i,0] = sum_r (sum_{j in 0,1} U[r,i,j] A[i,j]) * (sum_{k' in 0} V[r,j,k'] B[j,k']) * W[r,0,i]
    # This is coupled across j.
    # Schoolbook uses 4 terms:
    # r1: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> A[0,0]B[0,0] contributes to C[0,0].
    # r2: U[0,1]=1, V[1,0]=1, W[0,0]=1 -> A[0,1]B[1,0] contributes to C[0,0].
    # r3: U[1,0]=1, V[0,0]=1, W[1,0]=1 -> A[1,0]B[0,0] contributes to C[1,0].
    # r4: U[1,1]=1, V[1,0]=1, W[1,0]=1 -> A[1,1]B[1,0] contributes to C[1,0].
    # Total 4.
    
    # Can we combine?
    # We need 2 outputs C[0,0], C[1,0].
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    # Let's try rank 3? Or rank 2?
    # If rank 2, we need 2 terms r1, r2.
    # C[i,0] = (U[r1,i,0]A[i,0] + U[r1,i,1]A[i,1]) (V[r1,0,0]B[0,0] + V[r1,1,0]B[1,0]) W[r1,0,i]
    #            + (U[r2,i,0]A[i,0] + U[r2,i,1]A[i,1]) (V[r2,0,0]B[0,0] + V[r2,1,0]B[1,0]) W[r2,0,i]
    
    # This seems hard to match exactly for arbitrary A, B with integer coefficients.
    # The "2x2x1" tensor rank is actually 4 for the general case?
    # Let's check the rank of the tensor T where T[i,j,k] = delta_{i, j'} delta_{j, k'}?
    # No, the multiplication tensor.
    # The tensor for 2x2x1 multiplication is T(i,j,k) = 1 if it contributes?
    # The multiplication is C_i = sum_j A_{ij} B_{jk}.
    # For 2x2 matrix A, 2x1 B.
    # Tensor has indices i, j, k. i in 0,1; j in 0,1; k in 0.
    # T_{i,j,k} = 1 if we are at that slot? No, the tensor is the identity on the basis.
    # Actually, the tensor rank of 2x2x1 matrix multiplication is 4.
    # (It is the rank of the 2x2x1 slice, which is a 2x2 matrix of coefficients? No.)
    # Reference: The rank of n x n matrix multiplication is known.
    # For 2x2x2 it is 7.
    # For 2x2x1 it is 4.
    # So schoolbook is optimal for 2x2x1.
    
    # What about 1x1x1? Rank 1. Schoolbook is 1. Optimal.
    # What about 1x2x1? 
    # C[0] = A[0,0]B[0,0] + A[0,1]B[1,0].
    # Rank 2? 
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0].
    # Term 1: A[0,0]B[0,0]. Term 2: A[0,1]B[1,0].
    # Can we combine? 
    # It's a sum of 2 products. Rank 2 is optimal. Schoolbook is 2.
    
    # So schoolbook seems optimal for all blocks except 2x2x2?
    # Where Strassen gives 7 vs schoolbook 8.
    
    # Are there other decompositions for 2x2x2?
    # Rank 7 is optimal for 2x2x2.
    
    # So the current construction:
    # 2x2x2: 7 terms. (8 blocks of this size? No, 3x3x3 blocks = 27 blocks.
    # Splits: 2,2,1.
    # Combinations:
    # (2,2,2): 2*2*2 = 8 blocks? No.
    # Blocks are defined by ranges.
    # (0,2), (2,4), (4,5).
    # r=0, c=0, k=0 -> 2x2x2 (7)
    # r=1, c=0, k=0 -> 2x2x2 (7)
    # r=2, c=0, k=0 -> 1x2x2 (schoolbook: 1*2*2 = 4)
    # r=0, c=1, k=0 -> 2x2x2 (7)
    # r=1, c=1, k=0 -> 2x2x2 (7)
    # r=2, c=1, k=0 -> 1x2x2 (4)
    # r=0, c=2, k=0 -> 2x1x2 (schoolbook: 2*1*2 = 4)
    # r=1, c=2, k=0 -> 2x1x2 (4)
    # r=2, c=2, k=0 -> 1x1x2 (2)
    # ... same for k=1, k=2.
    
    # Total:
    # k=0:
    # r,c=0,0 (2x2x2): 7
    # r,c=1,0 (2x2x2): 7
    # r,c=2,0 (1x2x2): 4
    # r,c=0,1 (2x2x2): 7
    # r,c=1,1 (2x2x2): 7
    # r,c=2,1 (1x2x2): 4
    # r,c=0,2 (2x1x2): 4
    # r,c=1,2 (2x1x2): 4
    # r,c=2,2 (1x1x2): 2
    # Sum k=0: 7+7+4+7+7+4+4+4+2 = 46.
    
    # k=1:
    # Same as k=0 because block sizes are same?
    # r,c,k blocks:
    # (0,0,1): 2x2x2 -> 7
    # (1,0,1): 2x2x2 -> 7
    # (2,0,1): 1x2x2 -> 4
    # (0,1,1): 2x2x2 -> 7
    # (1,1,1): 2x2x2 -> 7
    # (2,1,1): 1x2x2 -> 4
    # (0,2,1): 2x1x2 -> 4
    # (1,2,1): 2x1x2 -> 4
    # (2,2,1): 1x1x2 -> 2
    # Sum k=1: 46.
    
    # k=2:
    # k block is (4,5) so size 1.
    # (0,0,2): 2x2x1 -> 4
    # (1,0,2): 2x2x1 -> 4
    # (2,0,2): 1x2x1 -> 2
    # (0,1,2): 2x2x1 -> 4
    # (1,1,2): 2x2x1 -> 4
    # (2,1,2): 1x2x1 -> 2
    # (0,2,2): 2x1x1 -> 2
    # (1,2,2): 2x1x1 -> 2
    # (2,2,2): 1x1x1 -> 1
    # Sum k=2: 4+4+2+4+4+2+2+2+1 = 25.
    
    # Total Rank = 46 + 46 + 25 = 117.
    
    # This is the "schoolbook 125" minus "8 savings from Strassen" = 117?
    # 125 - 8 = 117. Yes.
    
    # Can we do better?
    # Maybe different splits?
    # 5 = 3 + 2?
    # Blocks: 3,2.
    # 2x2x2 Strassen (7) vs 8.
    # 3x3x3?
    # Rank of 3x3x3 is at most 27 (schoolbook).
    # Is there a faster way? 
    # Coppersmith-Winograd? Rank 23.8 for asymptotic. For fixed 3?
    # Maybe not integer.
    
    # What if we use 5x5 decomposition directly?
    # Current rank 117.
    
    # Let's verify the code logic.
    # The code produces rank 117?
    # If score is 1.0684, and base is 125 (schoolbook), then 125/117 = 1.06837.
    # Yes, 1.0684.
    
    # So we are at rank 117.
    # Can we beat 117?
    # Maybe use Laderman's algorithm for 4x4?
    # 5x5 is not 4x4.
    
    # Maybe we can share Strassen terms across blocks?
    # The blocks are disjoint in indices.
    # U, V, W indices are unique for each block (i, j, k ranges are disjoint).
    # So terms cannot be shared directly unless we use a more global decomposition.
    
    # If we use a global 5x5x5 decomposition.
    # Is there a known rank < 117?
    # Le Gall (1987) upper bound for 5x5x5?
    # Maybe we can try to find a 116 or 115 rank decomposition?
    
    # Given the constraints and time, it is risky to search for a global decomposition.
    # However, the "Directive" suggests a MIPS search or refinement.
    # But 5x5x5 is a fixed small case.
    
    # Let's check if we can optimize the 1x1x2 or 1x2x1 blocks.
    # 1x1x2: C[0,0] = A[0,0]B[0,0], C[0,1] = A[0,0]B[0,1].
    # T[0,0,0] = 1, T[0,0,1] = 1? No, T[i,j,k].
    # C[i,k] = A[i,j]B[j,k].
    # 1x1x2: i=0, j=0, k=0,1.
    # C[0,0] = A[0,0]B[0,0].
    # C[0,1] = A[0,0]B[0,1].
    # Rank 2 schoolbook.
    # Can we do rank 1?
    # C[0,0] = u A[0,0] v B[0,0] w.
    # C[0,1] = u A[0,0] v B[0,1] w.
    # Ratio C[0,1]/C[0,0] = B[0,1]/B[0,0].
    # The output ratio depends on B. The input to rank 1 term is fixed structure.
    # A rank 1 term produces C[i,k] proportional to A[i,j]B[j,k] for a FIXED j?
    # No, rank 1 term is a single j?
    # In the decomposition sum_r U_r V_r W_r.
    # If rank 1, U, V, W are vectors.
    # C[i,k] = U[i,j?] V[j?,k] W?
    # The indices are fixed.
    # For 1x1x2 block, i=0, j=0 is the only j.
    # So the term is U[0,0] V[0,k] W[k,0].
    # C[0,k] = U[0,0] V[0,k] W[k,0] * A[0,0] * B[0,k].
    # We want C[0,0] = A[0,0]B[0,0] and C[0,1] = A[0,0]B[0,1].
    # So we need:
    # U[0,0] V[0,0] W[0,0] = 1
    # U[0,0] V[0,1] W[1,0] = 1
    # This is 2 equations, 6 variables. Easy to solve.
    # BUT, we must ensure this term does not contribute to OTHER blocks.
    # U[0,0] is in row 0, col 0 of U.
    # If we use this U[0,0] in the 2x2x2 block as well, we might interfere?
    # The blocks are disjoint. U indices are (r, c).
    # Block 1x1x2 at (0,0,2) uses U[0,0].
    # Block 2x2x2 at (0,0,0) uses U[0,0], U[0,1], U[1,0], U[1,1].
    # So U[0,0] is shared.
    # In the Strassen decomposition, U[0,0] is used in multiple terms.
    # If we add a new term that uses U[0,0] with a non-zero value, it adds to the sum.
    # So we CANNOT just add a term; we must modify the existing ones or ensure the new term has 0 overlap with existing support?
    # No, the tensor is linear. We can change coefficients.
    # But we want to REDUCE rank.
    # Currently 1x1x2 uses 2 terms:
    # t1: U[0,0]=1, V[0,0]=1, W[0,0]=1.
    # t2: U[0,0]=1, V[0,1]=1, W[1,0]=1.
    # We want to replace these with 1 term?
    # If we replace, we must change the existing decomposition.
    # This requires solving a system of equations for the whole tensor.
    
    # Given the complexity and risk, and that 117 is a solid score, I will stick with the current block strategy but verify the implementation details.
    # Wait, I noticed `add_schoolbook` for 1x1x1 uses 1 term.
    # 1x1x2 uses 2 terms.
    # 2x1x1 uses 2 terms.
    # 1x2x1 uses 2 terms.
    # 1x2x2 uses 4 terms.
    # 2x1x2 uses 4 terms.
    # 2x2x1 uses 4 terms.
    # 2x2x2 uses 7 terms.
    
    # Is there any 1-term optimization for 1x1x2?
    # If I can prove 1x1x2 can be rank 1, I save 1 per block.
    # Number of 1x1x2 blocks:
    # k=2 (size 1).
    # r=0 (size 2), c=2 (size 1) -> 2x1x2? No.
    # r=2 (size 1), c=2 (size 1) -> 1x1x2. (1 block)
    # r=0 (size 2), c=2 (size 1) -> 2x1x1? No, k=2 is size 1.
    # Let's re-list 1x1x2 blocks (ar=1, ac=1, bc=2).
    # r=2 (1), c=2 (1), k=0 (2) -> 1x1x2.
    # r=2 (1), c=2 (1), k=1 (2) -> 1x1x2.
    # r=2 (1), c=2 (1), k=2 (1) -> 1x1x1.
    # So 2 blocks of 1x1x2.
    # If rank 1 is possible, save 2. Total 115.
    
    # Let's check 1x1x2 rank again.
    # T[i,j,k] for i=0,j=0,k=0,1.
    # We want C[0,0] = A[0,0]B[0,0], C[0,1] = A[0,0]B[0,1].
    # Rank 1 term: U[0,0] V[0,k] W[k,0].
    # Contribution to C[0,k] is U[0,0] V[0,k] W[k,0] * A[0,0] * B[0,k].
    # We need this to equal A[0,0]B[0,k].
    # So U[0,0] V[0,k] W[k,0] = 1 for k=0,1.
    # We can set U[0,0]=1, V[0,0]=1, W[0,0]=1.
    # And V[0,1]=1, W[1,0]=1.
    # This IS the 2-term schoolbook.
    # Can we merge?
    # We need a SINGLE term r such that U_r, V_r, W_r satisfy the conditions for k=0 AND k=1 simultaneously?
    # No, V_r and W_r are vectors over k.
    # V_r[0] = 1, W_r[0] = 1 -> term for k=0.
    # V_r[1] = 1, W_r[1] = 1 -> term for k=1.
    # This is ONE term with support on V[0], V[1], W[0], W[1].
    # Yes!
    # The term is:
    # U: {(0,0): 1}
    # V: {(0,0): 1, (0,1): 1}
    # W: {(0,0): 1, (1,0): 1}
    # This single term contributes:
    # k=0: U[0,0] V[0,0] W[0,0] * A[0,0] B[0,0] = 1 * A[0,0] B[0,0]. Correct.
    # k=1: U[0,0] V[0,1] W[1,0] * A[0,0] B[0,1] = 1 * A[0,0] B[0,1]. Correct.
    # So 1x1x2 has rank 1!
    # Schoolbook used 2. We save 1 per block.
    
    # Similarly for 1x2x1 (ar=1, ac=2, bc=1).
    # C[0,k] = A[0,j] B[j,0] sum over j? No.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # 1x2x1: i=0, k=0. j=0,1.
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0].
    # Rank 1?
    # Term: U[i,j] V[j,0] W[0,i].
    # We need sum_j ... no, the term is product.
    # A rank 1 term corresponds to a single path i->j->k?
    # No, U is a matrix i x j. V is j x k. W is k x i.
    # The term is trace(U A V B W^T)? No.
    # The contribution is sum_{i,j,k} U_{ij} V_{jk} W_{ki} A_{ij} B_{jk}.
    # This is NOT sum over j. It is a specific (i,j,k) tuple?
    # No, the indices i,j,k are summed in the reconstruction?
    # Reconstruction: C_{ik} = sum_r sum_j U_{r,ij} V_{r,jk} W_{r,ki} A_{ij} B_{jk}.
    # If we pick a single r, and set U_{r,0,0}=1, others 0.
    # Then term is 1 * V_{r,jk} W_{r,k0} A_{0j} B_{jk}.
    # We need this to sum to C_{0k} = A_{00}B_{0k} + A_{01}B_{1k}.
    # If we set U_{r,0,0}=1, U_{r,0,1}=1.
    # Then term is V_{r,0k} W_{r,k0} A_{00} B_{0k} + V_{r,1k} W_{r,k0} A_{01} B_{1k}.
    # We want:
    # k=0: V_{r,00} W_{r,00} = 1, V_{r,10} W_{r,00} = 1.
    # This works with W_{r,00}=1, V_{r,00}=1, V_{r,10}=1.
    # So 1x2x1 is rank 1!
    
    # Similarly 2x1x1 (ar=2, ac=1, bc=1).
    # C_{i0} = A_{i0} B_{00}.
    # Rank 1?
    # U_{r,00}=1, U_{r,10}=1. V_{r,00}=1. W_{r,00}=1, W_{r,10}=1.
    # Term: A_{00}B_{00} * 1*1*1 + A_{10}B_{00} * 1*1*1.
    # Works. Rank 1.
    
    # So all 1xa xb or ax1xb or abx1 blocks can be rank 1?
    # Let's check 1x2x2 (ar=1, ac=2, bc=2).
    # C_{0k} = A_{00}B_{0k} + A_{01}B_{1k}.
    # Rank 1?
    # U_{r,00}=1, U_{r,01}=1.
    # V_{r,jk} W_{r,k0}.
    # We need sum_j V_{r,jk} W_{r,k0} A_{0j} B_{jk} = A_{00}B_{0k} + A_{01}B_{1k}.
    # This requires V_{r,0k} W_{r,k0} = 1 for all k.
    # And V_{r,1k} W_{r,k0} = 1 for all k.
    # And W_{r,k0} same for both?
    # If W_{r,00}=1, W_{r,10}=1.
    # Then V_{r,00}=1, V_{r,01}=1.
    # V_{r,10}=1, V_{r,11}=1.
    # Term: (A_{00}B_{00} + A_{01}B_{10}) * 1 + (A_{00}B_{01} + A_{01}B_{11}) * 1.
    # This matches C_{00} and C_{01}.
    # So 1x2x2 is rank 1!
    
    # Wow. Any block where one dimension is 1 can be rank 1?
    # If ar=1, we set U to 1 for all j.
    # Then we need V, W to handle the j,k sum.
    # The term is sum_j V_{jk} W_{ki} A_{ij} B_{jk}.
    # If we set U_{ij} = 1 for all j.
    # Then we need sum_j V_{jk} W_{ki} B_{jk} = B_{jk} (effectively identity)?
    # No, we need the result to be sum_j A_{ij} B_{jk}.
    # Since A_{ij} is factored out (if U=1), we need the rest to produce sum_j B_{jk}?
    # No, A_{ij} is NOT factored out if U is not constant?
    # U_{r,ij}. If we set U_{r,00}=1, U_{r,01}=1.
    # The term is V_{0k} W_{k0} A_{00} B_{0k} + V_{1k} W_{k0} A_{01} B_{1k}.
    # We want this to equal 1 * A_{00} B_{0k} + 1 * A_{01} B_{1k}.
    # So we need V_{0k} W_{k0} = 1 and V_{1k} W_{k0} = 1.
    # We can set W_{k0} = 1 for all k.
    # Then V_{0k} = 1, V_{1k} = 1.
    # So yes, if one dimension is 1, we can always use 1 term by setting the factor on that dimension to 1, and the others to 1?
    # Wait, what if ar=2, ac=1, bc=1?
    # U_{i0}. Set U_{00}=1, U_{10}=1.
    # V_{00}=1.
    # W_{00}=1, W_{10}=1.
    # Term: A_{00}B_{00} * 1*1*1 + A_{10}B_{00} * 1*1*1.
    # Matches.
    
    # So ANY block with at least one dimension 1 can be rank 1?
    # Let's check 1x1x1. Rank 1.
    # 1x1x2. Rank 1.
    # 1x2x1. Rank 1.
    # 2x1x1. Rank 1.
    # 1x2x2. Rank 1.
    # 2x1x2. Rank 1.
    # 2x2x1. Rank 1?
    # C_{i0} = A_{i0}B_{00} + A_{i1}B_{10}.
    # U_{i0}=1, U_{i1}=1.
    # V_{00}=1, V_{10}=1.
    # W_{00}=1, W_{10}=1.
    # Term: (A_{00}B_{00} + A_{01}B_{10}) * 1 + (A_{10}B_{00} + A_{11}B_{10}) * 1.
    # Matches C_{00} and C_{10}.
    # Yes!
    
    # So only 2x2x2 blocks need 7 terms.
    # All others need 1 term.
    
    # Let's recalculate the rank.
    # Blocks 27.
    # 8 blocks are 2x2x2. Rank 7 each -> 56.
    # 19 blocks have at least one dim 1. Rank 1 each -> 19.
    # Total Rank = 56 + 19 = 75.
    
    # Score = 125 / 75 = 1.666.
    # Huge improvement!
    
    # Let's implement this.
    # For any block:
    # If ar=2, ac=2, bc=2: Strassen.
    # Else: 1 term.
    # The term is:
    # U: {(r1+i, c1+j): 1 for all i,j}
    # V: {(c1+j, k1+k): 1 for all j,k}
    # W: {(k1+k, r1+i): 1 for all k,i}
    
    # Let's verify this doesn't break the 2x2x2 blocks.
    # The 2x2x2 blocks are disjoint.
    # The "1 term" blocks are disjoint from each other and from 2x2x2.
    # So it should work.

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    blocks = [(0, 2), (2, 4), (4, 5)]
    terms = []

    def add_term(u_dict, v_dict, w_dict):
        if time.time() > deadline:
            return
        terms.append((u_dict, v_dict, w_dict))

    def add_strassen_2x2(r1, c1, k1):
        A00 = (r1) * m + c1
        A01 = (r1) * m + (c1 + 1)
        A10 = (r1 + 1) * m + c1
        A11 = (r1 + 1) * m + (c1 + 1)
        
        B00 = c1 * p + k1
        B01 = c1 * p + (k1 + 1)
        B10 = (c1 + 1) * p + k1
        B11 = (c1 + 1) * p + (k1 + 1)
        
        C00 = k1 * n + r1
        C01 = (k1 + 1) * n + r1
        C10 = k1 * n + (r1 + 1)
        C11 = (k1 + 1) * n + (r1 + 1)
        
        add_term({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00: 1, C11: 1})
        add_term({A10: 1, A11: 1}, {B00: 1}, {C10: 1, C11: -1})
        add_term({A00: 1}, {B01: 1, B11: -1}, {C01: 1, C11: 1})
        add_term({A11: 1}, {B10: 1, B00: -1}, {C00: 1, C10: 1})
        add_term({A00: 1, A01: 1}, {B11: 1}, {C00: -1, C01: 1})
        add_term({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11: 1})
        add_term({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00: 1})

    def add_simple_block(r1, c1, k1, ar, ac, bc):
        u_dict = {}
        v_dict = {}
        w_dict = {}
        for ii in range(ar):
            for jj in range(ac):
                u_dict[(r1 + ii) * m + (c1 + jj)] = 1
        for jj in range(ac):
            for kk in range(bc):
                v_dict[(c1 + jj) * p + (k1 + kk)] = 1
        for kk in range(bc):
            for ii in range(ar):
                w_dict[(k1 + kk) * n + (r1 + ii)] = 1
        add_term(u_dict, v_dict, w_dict)

    for r_idx, (r1, r2) in enumerate(blocks):
        for c_idx, (c1, c2) in enumerate(blocks):
            for k_idx, (k1, k2) in enumerate(blocks):
                if time.time() > deadline:
                    break
                ar = r2 - r1
                ac = c2 - c1
                bc = k2 - k1
                
                if ar == 2 and ac == 2 and bc == 2:
                    add_strassen_2x2(r1, c1, k1)
                else:
                    add_simple_block(r1, c1, k1, ar, ac, bc)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        u_dict, v_dict, w_dict = terms[r]
        for idx, val in u_dict.items():
            U[idx][r] = val
        for idx, val in v_dict.items():
            V[idx][r] = val
        for idx, val in w_dict.items():
            W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}

