"""
Optimized matrix multiplication decomposition for 2x4 * 4x8.
Uses structured low-rank decomposition with careful rank minimization.
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # For multiplying A (n x m) by B (m x p) to get C (n x p)
    # The tensor T_{i,j,k} = sum_l A_{i,l} B_{l,k}
    # We want rank-minimal decomposition: T = sum_r U_{i,j,r} * V_{j,k,r} * W_{i,k,r}
    # Note: W uses trace convention W[k*n+i, r]
    
    # Schoolbook rank is n*m*p = 2*4*8 = 64
    # Let's try to find a lower rank decomposition.
    
    # For matrix multiplication, the known lower bounds and constructions:
    # - Strassen's algorithm for 2x2 gives rank 7 instead of 8
    # - For general n x m * m x p, we can use block decomposition
    
    # Let's try a systematic approach:
    # 1. Start with schoolbook (rank 64)
    # 2. Try to find linear dependencies in the tensor
    
    # The multiplication tensor for n x m * m x p has:
    # - Input dimensions: n*m (for A) and m*p (for B)
    # - Output dimension: n*p (for C)
    # - Tensor entries: T[i, j, k, l] where A has index (i,l), B has index (l,k), C has index (i,k)
    
    # Actually, let me reconsider the tensor structure:
    # C_{i,k} = sum_l A_{i,l} * B_{l,k}
    # So the tensor is T[i, k, l] = 1 (it's just the identity in the contraction index)
    # Wait, that's not right either.
    
    # Let me think about this more carefully.
    # The matrix multiplication map is: (A, B) -> A*B
    # This is a bilinear map from R^{n*m} x R^{m*p} -> R^{n*p}
    # The tensor T in R^{n*m} x R^{m*p} x R^{n*p} has entries:
    # T[a, b, c] = 1 if a = (i, l), b = (l, k), c = (i, k) for some i, l, k
    # i.e., T[(i,l), (l,k), (i,k)] = 1
    
    # So for each (i, l, k) triple, we have one non-zero entry.
    # Total number of non-zero entries = n * m * p = 2 * 4 * 8 = 64
    
    # The rank of this tensor is what we want to minimize.
    
    # Known results:
    # - For 2x2 * 2x2: rank 7 (Strassen)
    # - For larger matrices, various constructions exist
    
    # Let's try to use a block-based approach:
    # Partition the matrices into blocks and use Strassen-like techniques
    
    # For 2x4 * 4x8:
    # We can view this as multiplying a 2x4 matrix by a 4x8 matrix.
    # One approach: split the 4x8 matrix into two 4x4 blocks: B = [B1, B2]
    # Then A*B = [A*B1, A*B2]
    # Each A*B1 and A*B2 is a 2x4 * 4x4 multiplication.
    
    # For 2x4 * 4x4:
    # We can further decompose. Let A be 2x4, B be 4x4.
    # Split A into two 2x2 blocks: A = [A1, A2]
    # Then A*B = [A1, A2] * B = A1*B_left + A2*B_right (not quite)
    
    # Actually, let me try a different approach. Let's use the fact that:
    # The multiplication tensor can sometimes be decomposed using
    # bilinear forms that share structure.
    
    # Let's try to construct a decomposition with rank < 64.
    
    # Strategy: Use the "border rank" or "rank" constructions from literature.
    # For now, let's try a simple optimization:
    # - Start with the 64-term schoolbook decomposition
    # - Try to find linear combinations that reduce the number of terms
    
    # Actually, let me just try to find the minimum rank by construction.
    
    # For n=2, m=4, p=8:
    # One known construction uses the fact that we can group terms.
    
    # Let's try: for each i in {0,1}, k in {0,...,7}:
    # C[i,k] = sum_l A[i,l] * B[l,k]
    # 
    # We can write this as:
    # For fixed i, the vector C[i,:] = A[i,:] * B
    # This is a linear combination of rows of B.
    
    # The tensor rank is related to the number of independent bilinear forms needed.
    
    # Let me try a concrete construction:
    # Use the "Kronecker" or "block" decomposition.
    
    # Partition indices:
    # i in {0, 1}
    # l in {0, 1, 2, 3}  
    # k in {0, 1, ..., 7}
    
    # Group l into pairs: l = 2a + b where a in {0,1}, b in {0,1}
    # Group k into groups of 4: k = 4c + d where c in {0,1}, d in {0,1,2,3}
    
    # Then:
    # C[i, 4c+d] = sum_{a=0}^{1} sum_{b=0}^{1} A[i, 2a+b] * B[2a+b, 4c+d]
    
    # = sum_{a=0}^{1} (A[i, 2a] * B[2a, 4c+d] + A[i, 2a+1] * B[2a+1, 4c+d])
    
    # This doesn't immediately help reduce rank.
    
    # Let me try another approach: use the known result that for
    # multiplying an n x m matrix by an m x p matrix, the rank is at most
    # n*p*ceil(m/2) or similar constructions.
    
    # Actually, I recall that for rectangular matrix multiplication,
    # the rank can be significantly lower than n*m*p.
    
    # Let's try a systematic search for small rank decompositions.
    
    # For rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that for all i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # (since T[(i,j), (j,k), (i,k)] = 1)
    
    # Wait, I need to be more careful with the indices.
    
    # Let me redefine:
    # U has shape (n*m, r) with indices (i, j) where i in [0,n), j in [0,m)
    # V has shape (m*p, r) with indices (j, k) where j in [0,m), k in [0,p)
    # W has shape (p*n, r) with indices (k, i) where k in [0,p), i in [0,n)
    #   (using trace convention W[k*n + i, r])
    
    # The condition is:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1
    
    # This is because the tensor entry T[(i,j), (j,k), (i,k)] = 1.
    
    # So we need to solve this system of equations for U, V, W with minimal r.
    
    # This is a hard problem in general, but for small dimensions we can try.
    
    # Let's start with r = 64 (schoolbook) and try to reduce.
    
    # Schoolbook: r = 64
    # U[i*m + j, c] = 1 if c = i*m*p + j*p + k... no, let me re-derive.
    
    # In the schoolbook decomposition, each term corresponds to a unique (i, j, k) triple.
    # For term c = i*p + j (wait, that doesn't work either).
    
    # Let me just use the original schoolbook:
    # c = i * m * p + j * p + k
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    
    # This gives rank 64.
    
    # Now, can we do better? Let's try r = 32, 24, 16, etc.
    
    # For a given r, we have:
    # Unknowns: n*m*r + m*p*r + p*n*r = (8 + 32 + 16) * r = 56r variables
    # Equations: n*m*p = 64 equations
    # 
    # For r = 2: 112 variables, 64 equations - underdetermined, might have solutions
    # But the equations are nonlinear (cubic in the variables).
    
    # This is a system of polynomial equations, which is hard to solve in general.
    
    # Let me try a different approach: use known constructions from the literature.
    
    # For 2x4 * 4x8, one possible construction:
    # Use the fact that 2x4 * 4x8 can be embedded in a larger structure.
    
    # Actually, let me try to use the "laser method" or similar techniques,
    # but those are for asymptotic bounds and may not give exact decompositions.
    
    # For now, let's try to find a decomposition with rank as low as possible
    # by using a heuristic search.
    
    # Start with schoolbook and try to merge terms.
    
    # Two terms (i1, j1, k1) and (i2, j2, k2) can be merged if there exists
    # a bilinear form that captures both. This happens when the tensors
    # T1 and T2 are "compatible" in some sense.
    
    # Let me just try a few specific constructions.
    
    # Construction 1: Group by i
    # For each i in {0, 1}:
    #   C[i, k] = sum_j A[i, j] * B[j, k]
    #   This is a linear combination of rows of B.
    #   We can write: C[i, :] = A[i, :] * B
    #   This is a matrix-vector product for each i.
    
    # The rank contribution for fixed i is m*p = 4*8 = 32 (schoolbook).
    # Total: 2 * 32 = 64.
    
    # Can we share structure between i=0 and i=1?
    # If A[0, :] and A[1, :] are linearly related, maybe, but in general they're independent.
    
    # Construction 2: Group by k
    # For each k in {0, ..., 7}:
    #   C[i, k] = sum_j A[i, j] * B[j, k]
    #   This is a linear combination of columns of A.
    #   Rank contribution per k: n*m = 2*4 = 8
    #   Total: 8 * 8 = 64.
    
    # Same as before.
    
    # Construction 3: Use Strassen-like ideas
    # For 2x2 * 2x2, Strassen uses 7 multiplications instead of 8.
    # Can we adapt this to our case?
    
    # Let's try to partition the problem into 2x2 * 2x2 blocks.
    # A is 2x4, B is 4x8.
    # Split A into [A11 A12; A21 A22] where each block is 1x2? No, that's not 2x2.
    
    # Let me try: split the m=4 dimension into two parts of size 2.
    # A = [A_left, A_right] where A_left is 2x2, A_right is 2x2.
    # B = [B_top; B_bottom] where B_top is 2x8, B_bottom is 2x8.
    # Then A*B = A_left * B_top + A_right * B_bottom.
    
    # Each term is a 2x2 * 2x8 multiplication.
    # For 2x2 * 2x8:
    #   C[i, k] = sum_{j=0}^{1} A[i, j] * B[j, k]
    #   This is a 2x2 matrix times a 2x8 matrix.
    #   Schoolbook rank: 2*2*8 = 32.
    #   Can we do better?
    
    # For 2x2 * 2x8, we can think of it as:
    #   C[:, k] = A * B[:, k] for each k.
    #   Each column of C is A times a column of B.
    #   A is 2x2, so A * v for a vector v in R^2.
    #   The map v -> A*v is linear, so we need 2 independent linear forms.
    #   But A is fixed in the tensor sense...
    
    # Actually, in the tensor decomposition, A and B are variables, so we can't fix A.
    
    # Let me try yet another approach. Let's just compute the schoolbook decomposition
    # and return it, since finding a lower rank decomposition is non-trivial.
    
    # But wait, the task says "rank-minimal exact tensor decomposition", so we should
    # try to find the minimum rank.
    
    # Let me check if there's a known result for 2x4 * 4x8.
    
    # I don't recall a specific result, so let's try to find a good decomposition
    # using a heuristic.
    
    # Heuristic: Start with schoolbook (rank 64) and try to find linear dependencies.
    
    # The tensor T can be viewed as a matrix in different ways (matricizations).
    # If a matricization has rank r, then the tensor rank is at least r.
    
    # Let's compute the matricizations to get lower bounds.
    
    # Matricization 1: Fix i, view as matrix with rows (j, k) and columns... hmm, this is getting complicated.
    
    # Let me just try a few specific rank values and see if we can find solutions.
    
    # For simplicity, let's try rank = 32 first (half of schoolbook).
    
    # Actually, let me just return the schoolbook decomposition for now, since
    # finding a lower rank decomposition requires more sophisticated methods.
    
    # But the task asks for rank-minimal, so let's at least try to improve.
    
    # One idea: use the fact that for fixed i, the vectors A[i, :] are independent
    # of each other, so we can't share much structure between different i.
    
    # Similarly for fixed k.
    
    # Let me try to use a "block Strassen" approach:
    # Partition the m=4 dimension into two blocks of size 2.
    # A = [A1, A2] (each 2x2)
    # B = [B1; B2] (each 2x8)
    # C = A1*B1 + A2*B2
    
    # For each term A1*B1 and A2*B2, we have a 2x2 * 2x8 multiplication.
    # If we can decompose 2x2 * 2x8 with rank r, then total rank is 2r.
    
    # What's the rank of 2x2 * 2x8?
    # Schoolbook: 2*2*8 = 32.
    # Can we do better?
    
    # For 2x2 * 2x2, Strassen gives rank 7.
    # For 2x2 * 2x8, we can think of it as 4 independent 2x2 * 2x2 multiplications
    # (one for each pair of columns in the 8-column matrix), but that gives rank 4*7=28.
    # Actually, that's not right because the columns are not independent in the tensor sense.
    
    # Let me try: 2x2 * 2x8 can be viewed as:
    # C[i, k] = A[i, 0] * B[0, k] + A[i, 1] * B[1, k]
    # For each k, this is a linear combination of two rows of B.
    # The tensor has entries T[(i, j), (j, k), (i, k)] = 1 for i in {0,1}, j in {0,1}, k in {0,...,7}.
    # Total 2*2*8 = 32 non-zero entries.
    
    # Can we find a rank < 32 decomposition?
    
    # Let's try rank 16:
    # We need U (4x16), V (16x16)... no, U is (2*2=4) x r, V is (2*8=16) x r, W is (8*2=16) x r.
    # For r=16: U is 4x16, V is 16x16, W is 16x16.
    # Number of unknowns: 4*16 + 16*16 + 16*16 = 64 + 256 + 256 = 576.
    # Number of equations: 32.
    # Underdetermined, so solutions likely exist.
    
    # But the equations are nonlinear, so it's not obvious.
    
    # Let me try a constructive approach for 2x2 * 2x8:
    # Use the Strassen idea:
    # For 2x2 * 2x2, Strassen defines:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # And then:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 2x2 * 2x8, we can apply this to each pair of columns.
    # But the B matrix has 8 columns, so we'd have 4 pairs.
    # If we use Strassen for each pair, we get 4*7 = 28 multiplications.
    # But we can share some of the M's if the structure allows.
    
    # Actually, for 2x2 * 2x8, let's denote B as having columns b0, b1, ..., b7.
    # C[i, k] = A[i, 0] * b_k[0] + A[i, 1] * b_k[1]
    
    # We can group columns into pairs: (b0, b1), (b2, b3), (b4, b5), (b6, b7).
    # For each pair, apply Strassen-like decomposition.
    # This gives 4 * 7 = 28 terms.
    
    # So 2x2 * 2x8 has rank at most 28.
    
    # Then 2x4 * 4x8 has rank at most 2 * 28 = 56.
    
    # Can we do better? Maybe by sharing structure between the two A blocks.
    
    # Let's try to implement this.
    
    # First, let's implement the 2x2 * 2x8 decomposition with rank 28.
    
    def decompose_2x2_2x8():
        """
        Decompose 2x2 * 2x8 matrix multiplication.
        Returns U (4x28), V (16x28), W (16x28)
        where U[i*2+j, r], V[j*8+k, r], W[k*2+i, r]
        """
        n, m, p = 2, 2, 8
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # For each pair of columns (c0, c1) in B, apply Strassen-like decomposition.
        # There are 4 pairs: (0,1), (2,3), (4,5), (6,7)
        
        for pair_idx in range(4):
            k0 = pair_idx * 2
            k1 = pair_idx * 2 + 1
            
            base = pair_idx * 7
            
            # M1 = (A00 + A11)(B00 + B11)
            # U: A00 and A11 contribute
            U[0][base + 0] = 1  # A00
            U[3][base + 0] = 1  # A11
            # V: B0k0 + B1k0 and B0k1 + B1k1... wait, this doesn't work directly.
            
            # Actually, Strassen works for 2x2 * 2x2, where B is also 2x2.
            # Here, B is 2x8, so we need to adapt.
            
            # Let me reconsider. For 2x2 * 2x2:
            # C00 = A00*B00 + A01*B10
            # C01 = A00*B01 + A01*B11
            # C10 = A10*B00 + A11*B10
            # C11 = A10*B01 + A11*B11
            
            # Strassen computes:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # For 2x2 * 2x8, B has columns b0, ..., b7.
            # Let's group into 4 pairs: (b0,b1), (b2,b3), (b4,b5), (b6,b7).
            # For each pair (bk, bk+1), we have a 2x2 * 2x2 subproblem.
            
            # So for pair (k0, k1):
            # B_sub = [B[:, k0], B[:, k1]] which is 2x2.
            # Apply Strassen to this subproblem.
            
            # M1 = (A00+A11)(B0k0 + B1k1)... no, that's not right.
            
            # Let me be more careful. For the subproblem with columns k0, k1:
            # B_sub[0, 0] = B[0, k0], B_sub[0, 1] = B[0, k1]
            # B_sub[1, 0] = B[1, k0], B_sub[1, 1] = B[1, k1]
            
            # C_sub[0, 0] = A00*B0k0 + A01*B1k0
            # C_sub[0, 1] = A00*B0k1 + A01*B1k1
            # C_sub[1, 0] = A10*B0k0 + A11*B1k0
            # C_sub[1, 1] = A10*B0k1 + A11*B1k1
            
            # Strassen for this subproblem:
            # M1 = (A00+A11)(B0k0 + B1k1)... 
            # Wait, Strassen uses:
            # M1 = (A00+A11)(B00+B11) where B00, B11 are entries of the 2x2 B.
            # In our case, B00 = B[0,k0], B11 = B[1,k1], B01 = B[0,k1], B10 = B[1,k0].
            
            # So:
            # M1 = (A00+A11)(B[0,k0] + B[1,k1])
            # M2 = (A10+A11)B[0,k0]
            # M3 = A00(B[0,k1] - B[1,k1])
            # M4 = A11(B[1,k0] - B[0,k0])
            # M5 = (A00+A01)B[1,k1]
            # M6 = (A10-A00)(B[0,k0] + B[0,k1])
            # M7 = (A01-A11)(B[1,k0] + B[1,k1])
            
            # And:
            # C[0,k0] = M1 + M4 - M5 + M7
            # C[0,k1] = M3 + M5
            # C[1,k0] = M2 + M4
            # C[1,k1] = M1 - M2 + M3 + M6
            
            # Now, let's encode this in U, V, W.
            # For each M, we have a term U_term * V_term * W_term.
            
            # M1: (A00+A11) * (B[0,k0] + B[1,k1])
            # U: A00 and A11 -> U[0, r] = 1, U[3, r] = 1
            # V: B[0,k0] and B[1,k1] -> V[0*8+k0, r] = 1, V[1*8+k1, r] = 1
            # W: C[0,k0] and C[0,k1] and C[1,k0] and C[1,k1] get contributions from M1.
            #   C[0,k0] += M1 -> W[k0*2+0, r] += 1
            #   C[0,k1] += 0 (M1 doesn't contribute to C[0,k1] directly in the formula above)
            #   Wait, let me re-check the Strassen formulas.
            
            # Standard Strassen:
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # So M1 contributes to C00 and C11.
            # In our notation: C[0,k0] and C[1,k1].
            
            # Let me redo this carefully.
            
            # For the subproblem with B columns k0, k1:
            # B_sub = [[B[0,k0], B[0,k1]],
            #          [B[1,k0], B[1,k1]]]
            
            # Strassen:
            # M1 = (A00+A11) * (B_sub[0,0] + B_sub[1,1]) = (A00+A11)*(B[0,k0]+B[1,k1])
            # M2 = (A10+A11) * B_sub[0,0] = (A10+A11)*B[0,k0]
            # M3 = A00 * (B_sub[0,1] - B_sub[1,1]) = A00*(B[0,k1]-B[1,k1])
            # M4 = A11 * (B_sub[1,0] - B_sub[0,0]) = A11*(B[1,k0]-B[0,k0])
            # M5 = (A00+A01) * B_sub[1,1] = (A00+A01)*B[1,k1]
            # M6 = (A10-A00) * (B_sub[0,0] + B_sub[0,1]) = (A10-A00)*(B[0,k0]+B[0,k1])
            # M7 = (A01-A11) * (B_sub[1,0] + B_sub[1,1]) = (A01-A11)*(B[1,k0]+B[1,k1])
            
            # C_sub[0,0] = M1 + M4 - M5 + M7
            # C_sub[0,1] = M3 + M5
            # C_sub[1,0] = M2 + M4
            # C_sub[1,1] = M1 - M2 + M3 + M6
            
            # Now encode each M:
            
            # M1:
            # U: A00=1, A11=1 -> indices 0 and 3
            # V: B[0,k0]=1, B[1,k1]=1 -> indices 0*8+k0 and 1*8+k1
            # W: C[0,k0]+=1, C[1,k1]+=1 -> indices k0*2+0 and k1*2+1
            U[0][base + 0] = 1
            U[3][base + 0] = 1
            V[0 * 8 + k0][base + 0] = 1
            V[1 * 8 + k1][base + 0] = 1
            W[k0 * 2 + 0][base + 0] = 1
            W[k1 * 2 + 1][base + 0] = 1
            
            # M2:
            # U: A10=1, A11=1 -> indices 2 and 3
            # V: B[0,k0]=1 -> index 0*8+k0
            # W: C[1,k0]+=1 -> index k0*2+1
            U[2][base + 1] = 1
            U[3][base + 1] = 1
            V[0 * 8 + k0][base + 1] = 1
            W[k0 * 2 + 1][base + 1] = 1
            
            # M3:
            # U: A00=1 -> index 0
            # V: B[0,k1]=1, B[1,k1]=-1 -> indices 0*8+k1 and 1*8+k1
            # W: C[0,k1]+=1 -> index k1*2+0
            U[0][base + 2] = 1
            V[0 * 8 + k1][base + 2] = 1
            V[1 * 8 + k1][base + 2] = -1
            W[k1 * 2 + 0][base + 2] = 1
            
            # M4:
            # U: A11=1 -> index 3
            # V: B[1,k0]=1, B[0,k0]=-1 -> indices 1*8+k0 and 0*8+k0
            # W: C[0,k0]+=1, C[1,k0]+=1 -> indices k0*2+0 and k0*2+1
            U[3][base + 3] = 1
            V[1 * 8 + k0][base + 3] = 1
            V[0 * 8 + k0][base + 3] = -1
            W[k0 * 2 + 0][base + 3] = 1
            W[k0 * 2 + 1][base + 3] = 1
            
            # M5:
            # U: A00=1, A01=1 -> indices 0 and 1
            # V: B[1,k1]=1 -> index 1*8+k1
            # W: C[0,k0]-=1, C[0,k1]+=1 -> indices k0*2+0 and k1*2+0
            U[0][base + 4] = 1
            U[1][base + 4] = 1
            V[1 * 8 + k1][base + 4] = 1
            W[k0 * 2 + 0][base + 4] = -1
            W[k1 * 2 + 0][base + 4] = 1
            
            # M6:
            # U: A10=1, A00=-1 -> indices 2 and 0
            # V: B[0,k0]=1, B[0,k1]=1 -> indices 0*8+k0 and 0*8+k1
            # W: C[1,k1]+=1 -> index k1*2+1
            U[2][base + 5] = 1
            U[0][base + 5] = -1
            V[0 * 8 + k0][base + 5] = 1
            V[0 * 8 + k1][base + 5] = 1
            W[k1 * 2 + 1][base + 5] = 1
            
            # M7:
            # U: A01=1, A11=-1 -> indices 1 and 3
            # V: B[1,k0]=1, B[1,k1]=1 -> indices 1*8+k0 and 1*8+k1
            # W: C[0,k0]+=1 -> index k0*2+0
            U[1][base + 6] = 1
            U[3][base + 6] = -1
            V[1 * 8 + k0][base + 6] = 1
            V[1 * 8 + k1][base + 6] = 1
            W[k0 * 2 + 0][base + 6] = 1
        
        return U, V, W
    
    # Now, for 2x4 * 4x8:
    # A is 2x4, B is 4x8.
    # Split A into A1 (2x2) and A2 (2x2).
    # Split B into B1 (2x8) and B2 (2x8).
    # C = A1*B1 + A2*B2.
    
    # For A1*B1: use the 2x2*2x8 decomposition above.
    # For A2*B2: same decomposition.
    
    # Total rank: 2 * 28 = 56.
    
    # Let's construct this.
    
    U1, V1, W1 = decompose_2x2_2x8()
    U2, V2, W2 = decompose_2x2_2x8()
    
    # Now combine:
    # U has shape (2*4=8) x 56
    # V has shape (4*8=32) x 56
    # W has shape (8*2=16) x 56
    
    U = [[0] * 56 for _ in range(8)]
    V = [[0] * 56 for _ in range(32)]
    W = [[0] * 56 for _ in range(16)]
    
    # A1*B1:
    # A1 is the left half of A: A[0, 0:2] and A[1, 0:2]
    # In U1, indices 0,1,2,3 correspond to A1[0,0], A1[0,1], A1[1,0], A1[1,1]
    # In the full A, these are A[0,0], A[0,1], A[1,0], A[1,1]
    # U index: i*m + j = i*4 + j
    # So A[0,0] -> U[0], A[0,1] -> U[1], A[1,0] -> U[4], A[1,1] -> U[5]
    
    # V1 corresponds to B1 which is the top half of B: B[0:2, :]
    # In V1, indices j*8+k correspond to B1[j, k] = B[j, k]
    # In the full B, B[j, k] -> V[j*8 + k]
    
    # W1 corresponds to C, which is the same for both terms.
    # In W1, indices k*2+i correspond to C[i, k]
    # In the full C, C[i, k] -> W[k*2 + i]
    
    for r in range(28):
        for idx_u, val in enumerate(U1[r] if isinstance(U1[r], list) else [0]*4):
            pass
    
    # Let me just copy the values directly.
    for r in range(28):
        # U1[r] has 4 entries for A1
        # Map A1 indices to full A indices:
        # A1[0,0] -> A[0,0] -> U index 0
        # A1[0,1] -> A[0,1] -> U index 1
        # A1[1,0] -> A[1,0] -> U index 4
        # A1[1,1] -> A[1,1] -> U index 5
        u_map = [0, 1, 4, 5]
        for local_idx, val in enumerate(U1[r]):
            if val != 0:
                U[u_map[local_idx]][r] = val
        
        # V1[r] has 16 entries for B1
        # B1[j, k] -> V[j*8 + k]
        for local_idx, val in enumerate(V1[r]):
            if val != 0:
                j = local_idx // 8
                k = local_idx % 8
                V[j * 8 + k][r] = val
        
        # W1[r] has 16 entries for C
        for local_idx, val in enumerate(W1[r]):
            if val != 0:
                W[local_idx][r] = val
    
    for r in range(28):
        base2 = r + 28
        # U2[r] has 4 entries for A2
        # A2[0,0] -> A[0,2] -> U index 2
        # A2[0,1] -> A[0,3] -> U index 3
        # A2[1,0] -> A[1,2] -> U index 6
        # A2[1,1] -> A[1,3] -> U index 7
        u_map2 = [2, 3, 6, 7]
        for local_idx, val in enumerate(U2[r]):
            if val != 0:
                U[u_map2[local_idx]][base2 + local_idx - local_idx] = val  # This is wrong
    
    # Let me redo this more carefully.
    
    U = [[0] * 56 for _ in range(8)]
    V = [[0] * 56 for _ in range(32)]
    W = [[0] * 56 for _ in range(16)]
    
    for r in range(28):
        # First half (A1*B1)
        u_map1 = [0, 1, 4, 5]
        for local_idx in range(4):
            U[u_map1[local_idx]][r] = U1[local_idx][r]
        
        for local_idx in range(16):
            V[local_idx][r] = V1[local_idx][r]
        
        for local_idx in range(16):
            W[local_idx][r] = W1[local_idx][r]
        
        # Second half (A2*B2)
        base2 = r + 28
        u_map2 = [2, 3, 6, 7]
        for local_idx in range(4):
            U[u_map2[local_idx]][base2] = U2[local_idx][r]
        
        for local_idx in range(16):
            j = local_idx // 8
            k = local_idx % 8
            V[(j + 2) * 8 + k][base2] = V2[local_idx][r]
        
        for local_idx in range(16):
            W[local_idx][base2] = W2[local_idx][r]
    
    return {"U": U, "V": V, "W": W}
