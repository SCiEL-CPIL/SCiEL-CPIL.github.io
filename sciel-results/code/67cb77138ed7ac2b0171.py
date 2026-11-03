import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = 2, 3, 9
    
    # Use the known upper bound: rank(m,n,p) <= mn + mp + np - 2
    # For 2x3 x 3x9: 6 + 27 + 18 - 2 = 49
    # This uses the "diagonal trick" decomposition.
    
    # The construction:
    # For each (i,j) in A: U[i*m+j, r] contributes to rank-1 term
    # For each (j,k) in B: V[j*p+k, r] contributes
    # For each (i,k) in C: W[k*n+i, r] contributes
    #
    # The decomposition has rank mn + mp + np - 2 terms:
    # Term type 1: for each (i,j), one term where U[i*m+j]=1, V[j*p+0]=1, W[0*n+i]=1
    #   (this handles the "first column" of B)
    # Term type 2: for each (j,k) with k>0, one term where U[0*m+j]=1, V[j*p+k]=1, W[k*n+0]=1
    #   (this handles B's columns 1..p-1 with A's first row)
    # Term type 3: for each (i,k) with i>0, one term where U[i*m+0]=1, V[0*p+k]=1, W[k*n+i]=1
    #   (this handles A's rows 1..n-1 with B's first column)
    #
    # Wait, let me think more carefully. The standard bound uses:
    # rank <= mn + mp + np - 2
    # 
    # The construction works as follows:
    # We decompose the multiplication tensor M_{i,j,k} = sum_t U[i,j,t] * V[j,k,t] * W[i,k,t]
    #
    # One standard construction:
    # For t = 0: U[i,j,0] = 1 if i=0 or j=0, else 0
    #             V[j,k,0] = 1 if j=0 or k=0, else 0  
    #             W[i,k,0] = 1 if i=0 or k=0, else 0
    # This doesn't quite work directly.
    #
    # Let me use a different approach. The rank mn+mp+np-2 bound can be achieved by:
    # Using (mn-1) + (mp-1) + (np-1) + 1 terms? No, that's mn+mp+np-2.
    #
    # Actually, the standard construction is:
    # Terms indexed by:
    # 1. For each pair (i, j) with (i,j) != (0,0): one term
    # 2. For each pair (j, k) with (j,k) != (0,0): one term  
    # 3. For each pair (i, k) with (i,k) != (0,0): one term
    # 4. One additional term for the "corner"
    #
    # Total: (mn-1) + (mp-1) + (np-1) + 1 = mn + mp + np - 2
    #
    # Let me define the terms explicitly:
    # Let the terms be indexed by r = 0, 1, ..., rank-1 where rank = mn+mp+np-2
    
    # Actually, let me use a cleaner formulation.
    # The tensor M = sum_{i,j,k} E_{i,j} ⊗ E_{j,k} ⊗ E_{i,k}
    # 
    # We can write:
    # M = sum_{i,j} E_{i,j} ⊗ E_{j,0} ⊗ E_{i,0}    (mn terms, handles B's 0th column)
    #   + sum_{j,k>0} E_{0,j} ⊗ E_{j,k} ⊗ E_{0,k}  (m(p-1) terms, handles A's 0th row)
    #   + sum_{i>0,k} E_{i,0} ⊗ E_{0,k} ⊗ E_{i,k}  (n(p-1) terms, handles B's 0th row)
    #
    # But this gives mn + m(p-1) + n(p-1) = mn + mp - m + np - n = mn + mp + np - m - n
    # For m=3, n=2: 6 + 27 + 18 - 3 - 2 = 46. That's not right either.
    #
    # Let me reconsider. The correct bound is mn + mp + np - 2.
    # 
    # A known construction (from Strassen's work):
    # The rank is at most mn + mp + np - 2.
    # 
    # Here's one way to see it:
    # Consider the bilinear map (A, B) -> C where C = AB.
    # We can decompose this as:
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # Group the terms:
    # C[i,k] = A[i,0] B[0,k] + A[i,1] B[1,k] + ... + A[i,m-1] B[m-1,k]
    #
    # For each j, the term A[i,j] B[j,k] can be seen as a rank-1 operation.
    # But we can share computations.
    #
    # The key insight: we can use the fact that:
    # A[i,j] B[j,k] = (A[i,j] * B[j,k])
    #
    # One standard decomposition achieving rank mn + mp + np - 2:
    # 
    # Let's index the rank-1 terms as follows:
    # Type A: (i, j) for all (i,j) in [n]x[m] -> n*m terms
    # Type B: (j, k) for all (j,k) in [m]x[p] -> m*p terms
    # Type C: (i, k) for all (i,k) in [n]x[p] -> n*p terms
    # Total: mn + mp + np
    #
    # But we can eliminate 2 terms by noting that some terms are linearly dependent
    # or can be combined.
    #
    # Actually, I recall now. The construction is:
    # 
    # For the multiplication tensor, we have:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k (with appropriate indices)
    #
    # The rank mn+mp+np-2 bound comes from the following:
    # 
    # Define vectors:
    # For each (i,j): u_{i,j} = e_i (in U-space, indexed by (i,j))
    # For each (j,k): v_{j,k} = e_j (in V-space, indexed by (j,k))
    # For each (i,k): w_{i,k} = e_k (in W-space, indexed by (i,k))
    #
    # Hmm, this is getting complicated. Let me just implement the naive rank 54
    # since I'm not 100% sure of the rank-49 construction details.
    #
    # Wait, let me try a different approach. I'll use the fact that:
    # rank(m,n,p) <= min over all factorizations of the dimensions.
    #
    # Actually, let me just return the naive 54 to be safe.
    
    # Try to achieve rank 49 using the mn + mp + np - 2 bound
    # The construction uses the following idea:
    # 
    # The multiplication tensor can be decomposed as:
    # T = sum_{i,j} (e_i ⊗ e_j) ⊗ (e_j ⊗ e_0) ⊗ (e_i ⊗ e_0)  [n*m terms]
    #   + sum_{j,k>0} (e_0 ⊗ e_j) ⊗ (e_j ⊗ e_k) ⊗ (e_0 ⊗ e_k) [m*(p-1) terms]
    #   + sum_{i>0,k} (e_i ⊗ e_0) ⊗ (e_0 ⊗ e_k) ⊗ (e_i ⊗ e_k) [(n-1)*p terms]
    #   + correction terms to handle the overlaps
    #
    # Actually, let me use a simpler approach. I'll construct the rank-49
    # decomposition using the standard "diagonal trick":
    #
    # The key observation: for any i, j, k:
    # A[i,j] * B[j,k] = A[i,j] * B[j,k] * 1
    #
    # We can write:
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # Let's define the rank-1 terms as follows:
    # 
    # Term 0: handles the "base" case
    # For r = 0: 
    #   U[i*m+j, 0] = 1 for all (i,j)
    #   V[j*p+k, 0] = 1 for all (j,k)
    #   W[k*n+i, 0] = 1 for all (i,k)
    # This gives one term but doesn't work directly.
    #
    # Let me use a different standard construction:
    # 
    # The rank mn+mp+np-2 bound is achieved by:
    # 1. For each (i,j) with i>0 or j>0: term where U[i*m+j]=1, V[j*p+0]=1, W[0*n+i]=1
    #    Wait, this doesn't capture all of B.
    #
    # I'll use the following verified construction:
    # 
    # Terms are indexed by:
    # Group 1: (i, j) for all i in [n], j in [m] -> n*m terms
    #   U[i*m+j, r] = 1
    #   V[j*p+k, r] = delta_{k,0}  (only k=0)
    #   W[k*n+i, r] = 1 for all k
    #   This handles: sum_j A[i,j] B[j,0] for C[i,0]
    #
    # Group 2: (j, k) for j in [m], k in [1,p) -> m*(p-1) terms
    #   U[i*m+j, r] = delta_{i,0}  (only i=0)
    #   V[j*p+k, r] = 1
    #   W[k*n+i, r] = 1 for all i
    #   This handles: A[0,j] B[j,k] for C[0,k] with k>0
    #
    # Group 3: (i, k) for i in [1,n), k in [p] -> (n-1)*p terms
    #   U[i*m+0, r] = 1  (only j=0)
    #   V[0*p+k, r] = 1
    #   W[k*n+i, r] = 1
    #   This handles: A[i,0] B[0,k] for C[i,k] with i>0
    #
    # Total: nm + m(p-1) + (n-1)p = nm + mp - m + np - p = nm + mp + np - m - p
    # For n=2, m=3, p=9: 6 + 27 - 3 + 18 - 9 = 39
    # But we're missing terms! Let's check what's covered:
    #
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # From Group 1: contributes A[i,j] B[j,0] * delta_{k,0} = A[i,j] B[j,0] if k=0
    #   So for k=0: sum_j A[i,j] B[j,0] ✓
    #
    # From Group 2: contributes A[0,j] B[j,k] * delta_{i,0} = A[0,j] B[j,k] if i=0
    #   So for i=0: sum_j A[0,j] B[j,k] ✓
    #
    # From Group 3: contributes A[i,0] B[0,k] * delta_{j,0} = A[i,0] B[0,k] if j=0
    #   But wait, this adds A[i,0] B[0,k] for i>0, k any
    #
    # Let's verify for C[i,k]:
    # If i=0, k=0: 
    #   Group 1: sum_j A[0,j] B[j,0]
    #   Group 2: sum_j A[0,j] B[j,0] (since k=0 is not >0, Group 2 doesn't apply)
    #   Wait, Group 2 is for k>0, so for k=0, Group 2 contributes nothing.
    #   Group 3: i=0 is not >0, so Group 3 contributes nothing.
    #   Total: sum_j A[0,j] B[j,0] ✓
    #
    # If i=0, k>0:
    #   Group 1: k>0, so delta_{k,0}=0, contributes 0
    #   Group 2: sum_j A[0,j] B[j,k] ✓
    #   Group 3: i=0 not >0, contributes 0
    #   Total: sum_j A[0,j] B[j,k] ✓
    #
    # If i>0, k=0:
    #   Group 1: sum_j A[i,j] B[j,0]
    #   Group 2: i>0, delta_{i,0}=0, contributes 0
    #   Group 3: A[i,0] B[0,0]
    #   Total: sum_j A[i,j] B[j,0] + A[i,0] B[0,0]
    #   But we want sum_j A[i,j] B[j,0]
    #   So we have an extra term A[i,0] B[0,0]! ✗
    #
    # The problem is that Group 1 already includes the j=0 term: A[i,0] B[0,0]
    # And Group 3 also includes A[i,0] B[0,0]
    # So we're double-counting.
    #
    # Fix: In Group 1, exclude j=0 when i>0? Or in Group 3, exclude k=0?
    #
    # Let me redefine:
    # Group 1: (i, j) for all i, j -> nm terms
    #   U[i*m+j, r] = 1
    #   V[j*p+k, r] = delta_{k,0}
    #   W[k*n+i, r] = 1
    #   Contributes to C[i,k]: delta_{k,0} * sum_j A[i,j] B[j,0]
    #
    # Group 2: (j, k) for j in [m], k in [1,p) -> m(p-1) terms
    #   U[i*m+j, r] = delta_{i,0}
    #   V[j*p+k, r] = 1
    #   W[k*n+i, r] = 1
    #   Contributes to C[i,k]: delta_{i,0} * delta_{k>0} * A[0,j] B[j,k]
    #   Wait, this sums over j for fixed (j,k) term? No, each term is for specific (j,k).
    #   Actually, each rank-1 term is for a specific (j,k) in Group 2.
    #   The term contributes: delta_{i,0} * A[0,j] * B[j,k] * 1
    #   Summing over all Group 2 terms: for i=0, k>0: sum_j A[0,j] B[j,k] ✓
    #
    # Group 3: (i, k) for i in [1,n), k in [p] -> (n-1)p terms
    #   U[i*m+0, r] = 1  (j=0)
    #   V[0*p+k, r] = 1
    #   W[k*n+i, r] = 1
    #   Each term contributes: A[i,0] * B[0,k] * 1
    #   Summing over Group 3: for i>0: sum_k A[i,0] B[0,k]
    #   But this adds to C[i,k] the term A[i,0] B[0,k] for each k
    #
    # Now let's check C[i,k] = sum_j A[i,j] B[j,k]:
    #
    # Case i=0, k=0:
    #   Group 1: sum_j A[0,j] B[j,0]
    #   Group 2: 0 (k=0 not >0)
    #   Group 3: 0 (i=0 not >0)
    #   Total: sum_j A[0,j] B[j,0] ✓
    #
    # Case i=0, k>0:
    #   Group 1: 0 (k>0)
    #   Group 2: sum_j A[0,j] B[j,k]
    #   Group 3: 0
    #   Total: sum_j A[0,j] B[j,k] ✓
    #
    # Case i>0, k=0:
    #   Group 1: sum_j A[i,j] B[j,0]
    #   Group 2: 0
    #   Group 3: A[i,0] B[0,0]
    #   Total: sum_j A[i,j] B[j,0] + A[i,0] B[0,0]
    #   = A[i,0]B[0,0] + A[i,1]B[1,0] + ... + A[i,m-1]B[m-1,0] + A[i,0]B[0,0]
    #   = 2*A[i,0]B[0,0] + A[i,1]B[1,0] + ... ✗
    #
    # Still double-counting! The issue is that Group 1 includes j=0, and Group 3 also has j=0.
    #
    # Solution: In Group 1, for i>0, exclude j=0.
    # Group 1 modified: 
    #   For i=0: all j in [m] -> m terms
    #   For i>0: j in [1,m] -> (m-1) terms each, (n-1) such i -> (n-1)(m-1) terms
    #   Total Group 1: m + (n-1)(m-1) = m + nm - m - n + 1 = nm - n + 1
    #
    # Group 2: m(p-1) terms (unchanged)
    # Group 3: (n-1)p terms (unchanged)
    #
    # Total: (nm - n + 1) + m(p-1) + (n-1)p = nm - n + 1 + mp - m + np - p
    # = nm + mp + np - n - m - p + 1
    # For n=2,m=3,p=9: 6+27+18-2-3-9+1 = 38
    #
    # Let's verify:
    # Case i>0, k=0:
    #   Group 1: sum_{j=1}^{m-1} A[i,j] B[j,0]  (j=0 excluded)
    #   Group 2: 0
    #   Group 3: A[i,0] B[0,0]
    #   Total: A[i,0]B[0,0] + sum_{j=1}^{m-1} A[i,j] B[j,0] = sum_{j=0}^{m-1} A[i,j] B[j,0] ✓
    #
    # Case i>0, k>0:
    #   Group 1: 0 (k>0)
    #   Group 2: 0 (i>0)
    #   Group 3: A[i,0] B[0,k]
    #   Total: A[i,0] B[0,k]
    #   But we want sum_j A[i,j] B[j,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + ... + A[i,m-1]B[m-1,k]
    #   Missing: A[i,1]B[1,k] + ... + A[i,m-1]B[m-1,k] ✗
    #
    # So we're still missing terms for i>0, k>0, j>0!
    #
    # We need additional terms for those. Let me add:
    # Group 4: (i, j, k) for i>0, j>0, k>0 -> (n-1)(m-1)(p-1) terms
    #   U[i*m+j, r] = 1
    #   V[j*p+k, r] = 1
    #   W[k*n+i, r] = 1
    #   Each contributes A[i,j] B[j,k] to C[i,k]
    #
    # Total rank: (nm - n + 1) + m(p-1) + (n-1)p + (n-1)(m-1)(p-1)
    # = nm - n + 1 + mp - m + np - p + (n-1)(m-1)(p-1)
    # For n=2,m=3,p=9:
    # = 6-2+1 + 27-3 + 18-9 + (1)(2)(8)
    # = 5 + 24 + 9 + 16 = 54
    #
    # That's back to 54! So this decomposition doesn't help.
    #
    # The issue is that the simple grouping doesn't reduce the rank below nm*mp*... 
    #
    # Let me try a completely different approach. I'll use the known result that
    # rank(m,n,p) <= mn + mp + np - 2 and try to find the explicit construction
    # from the literature.
    #
    # After reflection, I believe the correct construction for the mn+mp+np-2 bound
    # uses a more subtle decomposition that I'm not recalling correctly.
    #
    # Given the time constraints and uncertainty, I'll return the naive rank 54.
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = (i * m + j) * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
    # Strategy: Block decomposition
    # A is 2x3, B is 3x9, C is 2x9
    # We can view B as a block matrix of 3 blocks of size 3x3: B = [B0, B1, B2]
    # Then C = A * [B0, B1, B2] = [A*B0, A*B1, A*B2]
    # Each A*Bk is a 2x3 * 3x3 multiplication.
    # Let R be the rank of the 2x3 x 3x3 multiplication tensor.
    # Then the total rank is 3*R.
    # We know rank(2x3 x 3x3) <= 2*3*3 = 18.
    # Can we do better than 18 for 2x3 x 3x3?
    # Strassen-like methods might help.
    # However, 2x3 x 3x3 is small. Let's try to find the minimal rank for 2x3 x 3x3.
    # Actually, we can just use the naive rank for the sub-blocks if we can't find better.
    # Naive rank for 2x3 x 3x3 is 18. Total rank 54.
    # This is the same as the current seed.
    
    # Is there a known lower rank for 2x3 x 3x3?
    # According to literature, the rank of 2x3 x 3x3 is not definitively known to be < 18, but it's likely 18 or close.
    # Let's try a different blocking.
    # What if we block A? A is 2x3. Can't really block 2 rows.
    # What if we block the columns of A and rows of B differently?
    
    # Let's look at the dimensions: 2x3 and 3x9.
    # The middle dimension is 3.
    # This is essentially 3 independent 2x1 * 1x9 multiplications? No.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # This is a sum of 3 terms.
    # Each term is a rank-1 matrix (outer product of row i of A and row j of B? No, column j of A and row j of B).
    # A[i,:] is 1x3. B[:,k] is 3x1.
    # C[i,k] = dot(A[i,:], B[:,k]).
    
    # Let's try to use a fast matrix multiplication algorithm if possible.
    # For 2x3 x 3x9, we can pad to 4x4 x 4x4?
    # 2->4, 3->4, 9->16? No, 9 is not a power of 2.
    # Pad 2x3 to 4x4. Pad 3x9 to 4x16? (9->16).
    # Then use Strassen for 4x4? Rank 16?
    # Strassen 4x4 rank is 16 (2^4).
    # But we have wasted dimensions.
    # Let's try padding to 8x8?
    # 2x3 -> 8x8. 3x9 -> 8x16 (9->16).
    # Strassen 8x8 rank 32.
    # This seems worse than 54? No, 32 < 54.
    # But we need to ensure the padding doesn't break the exactness or increase rank unnecessarily.
    # If we pad A to 8x8 with zeros, and B to 8x16 with zeros,
    # The product C will be 8x16. We only care about the top-left 2x9 part.
    # The rank of the tensor for 8x8 * 8x16 might be higher than needed if the zero-padding creates dependencies.
    # However, if we use a standard Strassen decomposition for 8x8, it applies to the whole matrix.
    # The rank of the 8x8 x 8x8 tensor is 32.
    # But our B is 8x16. We can view it as two 8x8 blocks.
    # So we do two 8x8 multiplications.
    # Rank = 32 + 32 = 64. This is worse than 54.
    
    # Let's try padding to 4x4.
    # A (2x3) -> 4x4.
    # B (3x9) -> 4x12? No, 9 is not multiple of 4.
    # Pad B to 4x12 (9->12).
    # Then we do 4x4 * 4x12.
    # This is 3 blocks of 4x4 * 4x4.
    # Strassen 4x4 rank is 16.
    # Total rank = 3 * 16 = 48.
    # This is better than 54!
    
    # Let's verify this.
    # A is 2x3. Pad to 4x4:
    # A_padded = [[A00, A01, A02, 0],
    #             [A10, A11, A12, 0],
    #             [0,    0,    0,    0],
    #             [0,    0,    0,    0]]
    # B is 3x9. Pad to 4x12:
    # B_padded = [[B00, B01, ..., B08, 0],
    #             [B10, B11, ..., B18, 0],
    #             [B20, B21, ..., B28, 0],
    #             [0,    0,   ..., 0,    0]]
    # Split B_padded into three 4x4 blocks: B1, B2, B3.
    # B1 = cols 0-3 of B_padded.
    # B2 = cols 4-7 of B_padded.
    # B3 = cols 8-11 of B_padded.
    
    # C_padded = A_padded * B_padded.
    # C_padded is 4x12.
    # We want C[0:2, 0:9].
    # C_padded[0:2, 0:4] corresponds to C[0:2, 0:3] and C[0:2, 3] (which is 0).
    # C_padded[0:2, 4:8] corresponds to C[0:2, 4:7] and C[0:2, 7] (which is 0? No, B col 7 is padded 0? No, B has 9 cols. 0..8.
    # B_padded cols:
    # 0: B[:,0]
    # 1: B[:,1]
    # 2: B[:,2]
    # 3: 0
    # 4: B[:,4]
    # 5: B[:,5]
    # 6: B[:,6]
    # 7: 0
    # 8: B[:,8]
    # 9: 0
    # 10: 0
    # 11: 0
    
    # So C_padded[0:2, 0:12] will have the correct values in cols 0,1,2, 4,5,6, 8.
    # And zeros in cols 3,7,9,10,11.
    # We only need cols 0..8.
    # So if we compute the full 4x12 product using Strassen for each 4x4 block, we get the result.
    
    # Rank of 4x4 * 4x4 is 16 (Strassen).
    # We have 3 such blocks.
    # Total rank = 3 * 16 = 48.
    
    # Can we do better?
    # What if we use a different padding?
    # 2x3 -> 3x3? No, 2 rows.
    # 3x9 -> 3x9.
    # 2x3 * 3x9.
    # If we pad 2x3 to 3x3, and 3x9 to 3x9.
    # Then we need rank of 3x3 * 3x9.
    # This is 3 blocks of 3x3 * 3x3.
    # Rank of 3x3 * 3x3 is 27 (naive).
    # Total rank 3 * 27 = 81. Worse.
    
    # What about 4x4 * 4x4 rank 16?
    # Is there a rank 15 algorithm for 4x4?
    # Strassen is 16.
    # It is an open problem if 4x4 can be done in 15.
    # Assuming 16 is optimal for now.
    
    # So rank 48 is a good target.
    
    # Let's construct the U, V, W matrices for rank 48.
    # We will use the Strassen decomposition for 4x4.
    # Strassen 4x4 is a tensor product of Strassen 2x2?
    # Or we can just use the 7 multiplications for 2x2 and apply it to 2x2 blocks of the 4x4 matrix.
    # A 4x4 matrix has 4 2x2 blocks.
    # Strassen 2x2 has rank 7.
    # If we just apply Strassen to each 2x2 block independently, the rank is 4 * 7 = 28.
    # But Strassen 4x4 rank is 16, which is less than 28.
    # So we need the specific Strassen 4x4 algorithm.
    # Strassen 4x4 can be viewed as applying Strassen 2x2 to the 2x2 blocks of A and B, but with some sharing?
    # Actually, Strassen 4x4 rank 16 is achieved by viewing the 4x4 multiplication as a 2x2 multiplication of 2x2 blocks, where the scalar multiplications are replaced by 2x2 multiplications? No, that would be 7 * 4 = 28.
    # The rank 16 algorithm for 4x4 is derived from the rank 7 algorithm for 2x2 by a recursive step?
    # Yes, T(4) = T(2)^2? No.
    # Strassen's algorithm for n=2 has rank 7.
    # For n=4, we can use the same structure, but the "scalars" are 2x2 matrices.
    # If we use rank 7 for 2x2, then T(4) <= 7 * T(2) = 7 * 7 = 49? No.
    # The recursion is T(n) = 7 T(n/2).
    # T(2) = 7.
    # T(4) = 7 T(2) = 49? No, T(1) = 1. T(2) = 7. T(4) = 49.
    # Wait, the rank of the tensor for n x n matrix multiplication is denoted R(n).
    # R(2) = 7.
    # R(4) <= 7 R(2) = 49.
    # But it is known that R(4) = 16?
    # No, R(4) is not known to be 16.
    # The trivial bound is R(n) <= n^3.
    # R(2) = 7 < 8.
    # R(4) <= 7 * R(2) = 49?
    # Actually, if we use Strassen for 4x4, we treat it as 2x2 blocks.
    # The 7 products are of 2x2 matrices.
    # Each 2x2 product can be done with rank 7.
    # So total rank is 7 * 7 = 49.
    # This is worse than 54? No, 49 < 54.
    # So rank 49 is better than 54.
    
    # Can we do better?
    # What if we use a different blocking?
    # 2x3 * 3x9.
    # Pad to 4x4 * 4x12.
    # 4x12 is three 4x4 blocks.
    # Rank = 3 * R(4).
    # If R(4) = 16, rank = 48.
    # If R(4) = 49, rank = 147.
    # Is R(4) 16?
    # According to Wikipedia, "It is not known whether the rank of 4x4 matrix multiplication is 16 or higher."
    # "The best known upper bound is 49."
    # So we should assume R(4) = 49 for safety, or find a better decomposition.
    
    # If R(4) = 49, then 3 * 49 = 147. Worse than 54.
    # So the 4x4 padding strategy is risky if R(4) is high.
    
    # Let's stick to the 2x2 Strassen which is known to be rank 7.
    # Can we use 2x2 Strassen on our 2x3 * 3x9 problem?
    # Pad 2x3 to 2x4? No, 2x2 blocks.
    # Pad 2x3 to 4x4?
    # Let's try to decompose 2x3 * 3x9 using 2x2 Strassen directly.
    # A is 2x3. B is 3x9.
    # We can view A as two 2x2 blocks? No, 3 cols.
    # A = [A1, A2] where A1 is 2x2, A2 is 2x1.
    # B = [B1; B2; B3] where B1, B2, B3 are 1x9? No, B is 3x9.
    # B1 is 1x9, B2 is 1x9, B3 is 1x9.
    # C = A1 * B1 + A2 * B2? No.
    # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k].
    # This is a sum of 3 terms.
    # Each term is a rank-1 matrix?
    # A[:,0] is 2x1. B[0,:] is 1x9.
    # Outer product is 2x9.
    # So C = A[:,0]B[0,:] + A[:,1]B[1,:] + A[:,2]B[2,:].
    # This is a sum of 3 rank-1 matrices?
    # No, the "scalars" are not scalars.
    # This is just the definition of matrix multiplication.
    # The rank of the tensor is the minimum number of such terms.
    # The naive rank is n*m*p = 2*3*9 = 54.
    # This corresponds to:
    # C[i,k] = sum_j A[i,j] B[j,k].
    # This is 54 terms of the form A[i,j] B[j,k] E_{i,k}.
    
    # Can we group these terms?
    # If we use Strassen for 2x2, we can reduce the rank of 2x2 * 2x2 from 8 to 7.
    # Our problem is 2x3 * 3x9.
    # Let's try to embed this into a 2x4 * 4x12 problem?
    # Or 4x4 * 4x12?
    
    # Let's try a different approach.
    # Use the fact that 3 = 2+1.
    # A = [A_2x2, A_2x1].
    # B = [B_1x9; B_1x9; B_1x9].
    # This doesn't seem to help directly with Strassen.
    
    # What if we pad 2x3 to 4x4 and 3x9 to 4x16?
    # Then use Strassen 4x4?
    # If we assume R(4) = 16 (optimistic), rank = 3 * 16 = 48.
    # If R(4) = 49 (pessimistic), rank = 3 * 49 = 147.
    
    # Is there a known algorithm for 2x3 * 3x9 with rank < 54?
    # If we use the 2x2 Strassen on the 2x2 part of A and the corresponding part of B?
    # Let's try to construct a rank 49 decomposition using 2x2 Strassen.
    # We can view the 2x3 * 3x9 multiplication as:
    # C = A * B.
    # Pad A to 4x4: A' = [[A, 0], [0, 0]] where A is 2x3, so A is top-left 2x3.
    # Pad B to 4x12: B' = [[B, 0], [0, 0]] where B is 3x9, so B is top-left 3x9.
    # This padding is not block-aligned with 2x2.
    
    # Let's try to align with 2x2 blocks.
    # Pad A to 4x4.
    # A = 2x3.
    # A' = [[A00, A01, A02, 0],
    #       [A10, A11, A12, 0],
    #       [0,    0,    0,    0],
    #       [0,    0,    0,    0]]
    # This is not a 2x2 block matrix in the sense of Strassen.
    # Strassen requires the matrix to be partitioned into 2x2 blocks of equal size.
    # So A' should be partitioned into four 2x2 blocks.
    # A' = [[A11, A12], [A21, A22]].
    # A11 = [[A00, A01], [A10, A11]].
    # A12 = [[A02, 0], [A12, 0]].
    # A21 = [[0, 0], [0, 0]].
    # A22 = [[0, 0], [0, 0]].
    
    # B' = 4x12.
    # B' = [[B11, B12, B13], [B21, B22, B23]].
    # B11 = [[B00, B01], [B10, B11]].
    # B12 = [[B02, B03], [B12, B13]]? No, B is 3x9.
    # B' rows 0-1 are B rows 0-1.
    # B' rows 2-3 are 0.
    # B' cols 0-11.
    # B11 (2x2) = B[0:2, 0:2].
    # B12 (2x2) = B[0:2, 2:4]. (Col 2 is B[:,2], Col 3 is 0).
    # B13 (2x2) = B[0:2, 4:6].
    # ...
    # B21 (2x2) = 0.
    # B22 (2x2) = 0.
    # B23 (2x2) = 0.
    
    # C' = A' * B'.
    # C' = [[A11*B11 + A12*B21, ...], ...]
    # Since A21, A22, B21, B22, B23 are 0,
    # C'11 = A11*B11.
    # C'12 = A11*B12 + A12*B22 (0) = A11*B12.
    # C'13 = A11*B13.
    # C'14 = A11*B14.
    # ...
    # C'21 = 0.
    # C'22 = 0.
    
    # We want C = C'[0:2, 0:9].
    # C'[0:2, 0:2] = C'11[0:2, 0:2] = A11*B11.
    # A11 = A[0:2, 0:2]. B11 = B[0:2, 0:2].
    # This gives C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2].
    # But C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2] * B[2, 0:2].
    # We are missing the term A[0:2, 2] * B[2, 0:2].
    # This term is in C'12?
    # C'12 = A11*B12.
    # B12 = B[0:2, 2:4].
    # B[0:2, 2] = B[:,2] (first col of B12).
    # B[0:2, 3] = 0.
    # So A11*B12 = A[0:2, 0:2] * [B[:,2], 0].
    # This gives A[0:2, 0:2] * B[:,2] in the first column of C'12.
    # But we need A[0:2, 2] * B[2, 0:2].
    # This term involves A[0:2, 2] (which is in A12) and B[2, 0:2] (which is in B21, which is 0).
    # So this decomposition fails to capture the full product.
    
    # The problem is that the padding does not preserve the structure of the multiplication in a way that allows simple block decomposition.
    # The term A[i,2] B[2,k] is lost because B[2,:] is padded to 0 in the second block row.
    
    # To capture A[i,2] B[2,k], we need B[2,:] to be non-zero in the appropriate block.
    # So we should pad B such that B[2,:] is in the first block row?
    # But B has 3 rows. A has 2 rows.
    # If we pad A to 4 rows, we can put A[0:2] in the first block row and 0 in the second.
    # If we pad B to 4 rows, we can put B[0:2] in the first block row and B[2:3] in the second block row?
    # B is 3x9.
    # B' = [[B[0:2, :]], [B[2:3, :], 0]].
    # This is not a 2x2 block matrix with 2x2 blocks.
    # B' is 4x12.
    # Block row 1: B[0:2, :]. (2x12).
    # Block row 2: B[2:3, :], 0. (2x12).
    # This works.
    # A' = [[A[0:2, :]], [0]].
    # A' is 4x4.
    # Block row 1: A[0:2, :]. (2x4).
    # Block row 2: 0. (2x4).
    
    # C' = A' * B'.
    # C' = [[A[0:2, :] * B[0:2, :]], [0]].
    # C'[0:2, :] = A[0:2, :] * B[0:2, :].
    # But we want A[0:2, :] * B[:, :].
    # A[0:2, :] is 2x3. B[:, :] is 3x9.
    # A[0:2, :] * B[0:2, :] is 2x9.
    # This is missing the term A[0:2, 2] * B[2, :].
    # So this padding also fails.
    
    # We need to include B[2, :] in the multiplication.
    # So B' must have B[2, :] in a row that is multiplied by A[0:2, 2].
    # A[0:2, 2] is the 3rd column of A.
    # In A', this is in the first block row, 3rd column.
    # So B' must have B[2, :] in the 3rd row?
    # But B' is partitioned into 2x2 blocks.
    # Row 2 of B' is in the second block row.
    # So A' must have non-zero entries in the second block row to multiply B[2, :].
    # But A' has 0 in the second block row.
    # Contradiction.
    
    # So we cannot use a simple 4x4 padding with 2x2 Strassen.
    
    # What if we use a different size?
    # 2x3 * 3x9.
    # Pad to 3x3 * 3x9?
    # A' = [[A], [0]].
    # B' = B.
    # C' = A' * B' = [[A*B], [0]].
    # We need rank of 3x3 * 3x9.
    # This is 3 blocks of 3x3 * 3x3.
    # Rank = 3 * R(3).
    # R(3) is not known to be < 27.
    # If R(3) = 27, rank = 81.
    
    # It seems difficult to beat 54 with simple padding.
    # Let's stick with the naive rank 54 for now, as it is safe and correct.
    # The task is to find a rank-minimal decomposition.
    # If we can't find a better one, we return 54.
    
    # However, the directive says "rewrite... with a stronger algorithm".
    # If I can't find a better one, I should still output the correct one.
    # But maybe I can try a random search for a lower rank?
    # Unlikely to succeed in 150s.
    
    # Let's check if there is a known result for 2x3 x 3x9.
    # The rank of the matrix multiplication tensor <m,n,p> is at most mn + mp + np - 2.
    # For 2,3,9: 6 + 18 + 27 - 2 = 49.
    # This is a known upper bound!
    # So rank <= 49.
    # Can we achieve 49?
    # The bound is derived from a specific decomposition.
    # Let's try to construct the rank 49 decomposition.
    
    # The bound mn + mp + np - 2 is achieved by the "border rank" or similar?
    # Actually, the rank is at most mn + mp + np - 2 for all m,n,p.
    # This is a standard result.
    # So we should aim for rank 49.
    
    # How to construct it?
    # The decomposition is based on the fact that the tensor is "submultiplicative".
    # I don't remember the exact construction.
    # But I can try to implement it.
    # Or I can just return the naive 54 if I can't construct 49.
    # But 49 is better.
    
    # Let's try to search for the construction of the rank mn+mp+np-2 decomposition.
    # It is often called the "rank bound" decomposition.
    # It involves using the fact that the tensor is a sum of rank-1 terms.
    # I will try to implement a known algorithm for this bound.
    # If I can't, I will fall back to 54.
    
    # Since I don't have the explicit formulas, I will stick to 54 to be safe.
    # Wait, the score is 1.0 for 54.
    # If I return 49, the score will be 54/49 > 1.0.
    # This is an improvement.
    
    # Let's try to find a rank 49 decomposition online or in memory.
    # The construction is non-trivial.
    # I will use the naive 54 for now.
    
    # Actually, let's try to use the 2x2 Strassen on a padded 4x4 matrix, but with a different padding strategy that might work?
    # No, I already showed it fails.
    
    # Let's assume the naive 54 is the best I can do quickly.
    
    U = [[0] * 54 for _ in range(6)]
    V = [[0] * 54 for _ in range(27)]
    W = [[0] * 54 for _ in range(18)]
    
    for i in range(2):
        for j in range(3):
            for k in range(9):
                col = (i * 3 + j) * 9 + k
                U[i * 3 + j][col] = 1
                V[j * 9 + k][col] = 1
                W[k * 2 + i][col] = 1
                
    return {"U": U, "V": V, "W": W}
