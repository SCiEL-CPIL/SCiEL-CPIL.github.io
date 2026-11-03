"""Matrix multiplication 3x5x5 rank-minimal decomposition."""
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 3x5x5 matrix multiplication using Strassen's algorithm for 2x2x2 sub-blocks.
    
    Strategy: Partition the 5x5 middle dimension into 2+2+1 and the 3x5 output into 2+1 rows.
    - The 2x2x2 core (rows 0-1, cols 0-3) uses Strassen (rank 7 instead of 8)
    - Cross terms use schoolbook
    - The remaining 1x1x1 and 1x1x5 and 5x1x1 terms use schoolbook
    
    Total rank: 7 (Strassen core) + 2*5 (row 2 x cols 0-3) + 2*1 (rows 0-1 x col 4) 
              + 1*1 (row 2 x col 4) + 2*5 (rows 0-1 x cols 4... wait, let me recompute)
    
    Actually, let me think more carefully:
    - A is 3x5, B is 5x5, C = A*B is 3x5
    - Split A into rows: A_top (2x5), A_bot (1x5)
    - Split B into cols: B_left (5x4), B_right (5x1)
    - C_top (2x4) = A_top * B_left  -> use Strassen on 2x2x2 blocks? 
      But B_left is 5x4, not 4x4. Split B_left into B_left_top (4x4) and B_left_bot (1x4)
      C_top = A_top[:, :4] * B_left_top + A_top[:, 4] * B_left_bot
      A_top[:, :4] is 2x4, split into 2x2 blocks: A00, A01 (top row), A10, A11 (bot row)
      B_left_top is 4x4, split into 2x2 blocks: B00, B01, B10, B11
      
      C00 = A00*B00, C01 = A00*B01, C10 = A10*B10, C11 = A11*B11
      Each 2x2x2 uses Strassen: 7 multiplications each = 28 total
      
      Wait, that's 4 separate 2x2x2 multiplications, each rank 7 = 28.
      Schoolbook would be 4*4 = 16. So Strassen is worse here!
      
    Let me reconsider. The benefit of Strassen comes when you have a 2x2x2 tensor, not when you have multiple independent 2x2x2 tensors.
    
    Better approach: The 2x2x2 Strassen trick applies to computing a 2x2 product from 2x2 inputs.
    If we have C (2x2) = A (2x2) * B (2x2), Strassen uses 7 mults instead of 8.
    
    For 3x5x5:
    - Split A (3x5) into A1 (2x5) and A2 (1x5)
    - Split B (5x5) into B1 (5x2), B2 (5x2), B3 (5x1)
    - C1 (2x2) = A1 * B1: This is 2x5x2. Split A1 into A1a (2x2), A1b (2x2), A1c (2x1)
      C1 = A1a*B1 + A1b*B2 + A1c*B3
      A1a*B1 is 2x2x2 -> Strassen rank 7
      A1b*B2 is 2x2x2 -> Strassen rank 7  
      A1c*B3 is 2x1x1 -> rank 2
      Subtotal: 7+7+2 = 16 vs schoolbook 2*5*2 = 20. Save 4.
    
    - C2 (2x2) = A1 * B2: Wait, I already used B2 above.
    
    Let me redo this more carefully.
    
    C (3x5) = A (3x5) * B (5x5)
    Split rows of A: A_top (2x5), A_bot (1x5)
    Split cols of B: B_c0 (5x1), B_c1 (5x1), B_c2 (5x1), B_c3 (5x1), B_c4 (5x1)
    
    C_top (2x5) = A_top * B
    C_bot (1x5) = A_bot * B
    
    For C_top: A_top is 2x5, B is 5x5
    Split B into B_left (5x4) and B_right (5x1)
    C_top_left (2x4) = A_top * B_left
    C_top_right (2x1) = A_top * B_right
    
    For C_top_left (2x4): 
    Split A_top into A_t1 (2x2), A_t2 (2x2), A_t3 (2x1)
    Split B_left into B_l1 (5x2), B_l2 (5x2)
    C_top_left = A_t1*B_l1 + A_t2*B_l2 + A_t3*B_l3 where B_l3 doesn't exist...
    
    Hmm, this is getting complicated. Let me just use the known result:
    The 3x5x5 can be decomposed with rank 71 using Strassen on appropriate 2x2x2 subproblems.
    
    Known from literature: For multiplying an m x n matrix by an n x p matrix,
    if n is even, you can use Strassen on the 2x2x2 blocks.
    
    For 3x5x5: Split the 5 into 2+2+1.
    - The 2x2x2 part (first 2 rows of A, first 4 cols of B, first 2 rows of C) 
      Actually this doesn't work cleanly.
    
    Let me just implement the straightforward Strassen-based decomposition:
    Rank 71 decomposition for 3x5x5.
    """
    n, m, p = DIMENSIONS  # n=3 rows of A, m=5 cols of A / rows of B, p=5 cols of B
    
    # We'll build U, V, W with rank 71
    # Strategy: 
    # - Use Strassen for the 2x2x2 core where A has 2 rows, B has 2 cols, C has 2 entries
    # - Use schoolbook for the rest
    
    # Partition:
    # A rows: [0,1] (top2), [2] (bot1)
    # B cols: [0,1] (left2), [2,3] (mid2), [4] (right1)
    # B rows: [0,1] (top2), [2,3] (mid2), [4] (bot1)
    
    # C entries:
    # C[0,0] = sum_k A[0,k]*B[k,0]
    # C[0,1] = sum_k A[0,k]*B[k,1]
    # C[0,2] = sum_k A[0,k]*B[k,2]
    # C[0,3] = sum_k A[0,k]*B[k,3]
    # C[0,4] = sum_k A[0,k]*B[k,4]
    # C[1,0..4] similar
    # C[2,0..4] similar
    
    # For C[0:2, 0:2] (2x2 block): 
    #   = A[0:2, 0:5] * B[0:5, 0:2]
    #   Split A[0:2, 0:5] = [A[0:2,0:2], A[0:2,2:4], A[0:2,4:5]]
    #   Split B[0:5, 0:2] = [B[0:2,0:2], B[2:4,0:2], B[4:5,0:2]]
    #   C[0:2,0:2] = A00*B00 + A01*B10 + A02*B20
    #   Where A00 is 2x2, B00 is 2x2 -> Strassen 7 mults
    #   A01 is 2x2, B10 is 2x2 -> Strassen 7 mults
    #   A02 is 2x1, B20 is 1x2 -> 2 mults
    #   Subtotal: 7+7+2 = 16 (vs schoolbook 2*5*2=20, save 4)
    
    # Similarly for C[0:2, 2:4]: 16 mults (save 4)
    # C[0:2, 4:5]: A[0:2,0:5]*B[0:5,4:5] = 2*5*1 = 10 mults (no savings)
    # C[2:3, 0:2]: A[2:3,0:5]*B[0:5,0:2] = 1*5*2 = 10 mults
    # C[2:3, 2:4]: 1*5*2 = 10 mults
    # C[2:3, 4:5]: 1*5*1 = 5 mults
    
    # Total: 16+16+10+10+10+5 = 67? That's less than 71. Let me recheck.
    
    # Wait, I think I'm double counting. Let me just implement a clean decomposition.
    
    rank = 71
    U = [[0.0] * rank for _ in range(n * m)]  # 15 x 71
    V = [[0.0] * rank for _ in range(m * p)]  # 25 x 71
    W = [[0.0] * rank for _ in range(p * n)]  # 15 x 71
    
    col = 0
    
    # Helper: add a schoolbook term
    def add_schoolbook(i, j, k):
        nonlocal col
        U[i * m + j][col] = 1.0
        V[j * p + k][col] = 1.0
        W[k * n + i][col] = 1.0
        col += 1
    
    # Helper: add Strassen terms for 2x2x2
    # Computes C[i:i+2, k:k+2] += A[i:i+2, j:j+2] * B[j:j+2, k:k+2]
    # Using Strassen's 7 multiplications
    def add_strassen(i, j, k):
        nonlocal col
        # A block: rows i,i+1; cols j,j+1
        # B block: rows j,j+1; cols k,k+1
        # C block: rows i,i+1; cols k,k+1
        
        # Strassen:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each Mx is a product of linear combinations of A and B entries.
        # We need to expand each into rank-1 terms.
        
        # M1 = (A[i,j] + A[i+1,j+1]) * (B[j,k] + B[j+1,k+1])
        # = A[i,j]*B[j,k] + A[i,j]*B[j+1,k+1] + A[i+1,j+1]*B[j,k] + A[i+1,j+1]*B[j+1,k+1]
        # That's 4 terms, but we can keep it as one "virtual" multiplication
        # In the tensor decomposition, each rank-1 term is one column.
        # M1 contributes to C[i,k], C[i,k+1], C[i+1,k], C[i+1,k+1]
        
        # Actually, in the tensor decomposition framework, each column of U,V,W
        # represents one scalar multiplication. The Strassen algorithm groups
        # these multiplications cleverly.
        
        # Let me just enumerate the 7 Strassen multiplications and their contributions.
        
        # For simplicity and correctness, let me use the standard approach:
        # Each of the 7 Strassen "multiplications" M1..M7 is itself a product
        # of two linear forms, which in the bilinear tensor decomposition
        # corresponds to multiple rank-1 terms unless the linear forms are single variables.
        
        # This is getting complex. Let me use a different, cleaner approach.
        
        # Actually, the standard way to express Strassen in the (U,V,W) format:
        # Each rank-1 term in the decomposition is: U[.,c] * V[.,c] * W[.,c]
        # where the product U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 
        # contributes to the (i,k) entry of C for the (j) summation.
        
        # For Strassen, we have 7 "virtual" multiplications, but each one is
        # a product of sums, so each expands to multiple rank-1 terms.
        
        # Let me just do schoolbook for simplicity and get rank 75, 
        # then see if I can improve.
        
        # Actually, let me implement it properly. The 7 Strassen multiplications
        # M1..M7 each involve products of linear combinations. In the tensor
        # decomposition, each such product of two linear forms (each a sum of 
        # variables) requires as many rank-1 terms as the product of the 
        # number of nonzero coefficients.
        
        # This is too complex to do by hand. Let me use a known result:
        # The rank of 2x2x2 matrix multiplication is 7 (Strassen).
        # For 3x5x5, if we can identify disjoint 2x2x2 sub-tensors, we can
        # apply Strassen to each.
        
        # A 2x2x2 sub-tensor involves 2 rows of A, 2 cols of A (=2 rows of B),
        # and 2 cols of B (=2 entries of C per row of A).
        
        # In our 3x5x5 problem:
        # We can form floor(3/2)*floor(5/2)*floor(5/2) = 1*2*2 = 4 disjoint 2x2x2 sub-tensors
        # Each saves 1 multiplication (8 -> 7), so total savings = 4
        # Rank = 75 - 4 = 71
        
        # The 4 sub-tensors are:
        # 1. A rows [0,1], B rows [0,1], B cols [0,1] -> C[0:2, 0:2]
        # 2. A rows [0,1], B rows [0,1], B cols [2,3] -> C[0:2, 2:4]
        # 3. A rows [0,1], B rows [2,3], B cols [0,1] -> C[0:2, 0:2] (adds to same C entries)
        # 4. A rows [0,1], B rows [2,3], B cols [2,3] -> C[0:2, 2:4] (adds to same C entries)
        
        # Wait, these aren't disjoint in C. They all contribute to C[0:2, 0:4].
        # But they ARE disjoint in the sense that each involves different (i,j,k) triples.
        
        # Sub-tensor 1: i in {0,1}, j in {0,1}, k in {0,1}
        # Sub-tensor 2: i in {0,1}, j in {0,1}, k in {2,3}
        # Sub-tensor 3: i in {0,1}, j in {2,3}, k in {0,1}
        # Sub-tensor 4: i in {0,1}, j in {2,3}, k in {2,3}
        
        # These are disjoint sets of (i,j,k) triples, so we can apply Strassen to each.
        
        # Remaining triples (schoolbook):
        # i=2, all j, all k: 1*5*5 = 25
        # i in {0,1}, j=4, all k: 2*1*5 = 10
        # i in {0,1}, all j, k=4: 2*5*1 = 10 (but j=4,k=4 already counted)
        # i in {0,1}, j=4, k=4: 1 (already in both above)
        
        # Let me count: total triples = 3*5*5 = 75
        # Strassen sub-tensors cover: 4 * (2*2*2) = 4*8 = 32 triples
        # Remaining: 75 - 32 = 43 triples, each needs 1 rank-1 term = 43
        # Strassen sub-tensors: 4 * 7 = 28 rank-1 terms
        # Total: 43 + 28 = 71. Correct!
        
        pass  # We'll implement below
    
    # Let me implement the 4 Strassen sub-tensors and the remaining schoolbook terms.
    
    # Strassen for a 2x2x2 sub-tensor with base indices (i0, j0, k0):
    # The 8 triples are: (i0+i, j0+j, k0+k) for i,j,k in {0,1}
    # Strassen replaces 8 rank-1 terms with 7.
    
    def strassen_block(i0, j0, k0):
        nonlocal col
        # The 7 Strassen multiplications:
        # M1 = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
        # M2 = (A[i0+1,j0] + A[i0+1,j0+1]) * B[j0,k0]
        # M3 = A[i0,j0] * (B[j0,k0+1] - B[j0+1,k0+1])
        # M4 = A[i0+1,j0+1] * (B[j0+1,k0] - B[j0,k0])
        # M5 = (A[i0,j0] + A[i0,j0+1]) * B[j0+1,k0+1]
        # M6 = (A[i0+1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k0+1])
        # M7 = (A[i0,j0+1] - A[i0+1,j0+1]) * (B[j0+1,k0] + B[j0+1,k0+1])
        
        # Each Mx is a product of two linear forms. In the tensor decomposition,
        # we need to express each as a sum of rank-1 terms.
        
        # M1 = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
        # This is 4 rank-1 terms:
        # A[i0,j0]*B[j0,k0] -> contributes to C[i0,k0]
        # A[i0,j0]*B[j0+1,k0+1] -> contributes to C[i0,k0+1]
        # A[i0+1,j0+1]*B[j0,k0] -> contributes to C[i0+1,k0]
        # A[i0+1,j0+1]*B[j0+1,k0+1] -> contributes to C[i0+1,k0+1]
        # But with Strassen, these are combined into ONE multiplication of sums.
        
        # In the (U,V,W) tensor format, each column c has:
        # U[i*m+j, c] = coefficient of A[i,j] in the first linear form
        # V[j*p+k, c] = coefficient of B[j,k] in the second linear form  
        # W[k*n+i, c] = coefficient in the output
        # The product U*V*W gives the contribution to C[i,k] from term j.
        
        # For M1: first form = A[i0,j0] + A[i0+1,j0+1], second form = B[j0,k0] + B[j0+1,k0+1]
        # This single "multiplication" actually requires 4 rank-1 terms in the naive expansion.
        # The magic of Strassen is that the 7 M-values can be combined to give C with only 7 mults
        # instead of 8, but each M is still a product that may need multiple rank-1 terms.
        
        # I think I'm overcomplicating this. Let me look at it differently.
        
        # In the standard bilinear algorithm for matrix multiplication, each rank-1 term
        # corresponds to one scalar multiplication. Strassen's algorithm for 2x2x2 uses
        # 7 scalar multiplications. Each scalar multiplication is of the form:
        # Mx = (linear combination of A entries) * (linear combination of B entries)
        
        # In the (U,V,W) representation, each such Mx corresponds to ONE column if
        # the linear combinations are represented as vectors. But U, V, W are matrices
        # where each column represents one rank-1 tensor.
        
        # I believe the correct interpretation is:
        # For each Strassen multiplication Mx = f_x(A) * g_x(B), we add ONE column to U,V,W
        # where U[.,c] represents f_x (as a vector over A's entries),
        # V[.,c] represents g_x (as a vector over B's entries),
        # and W[.,c] represents how Mx contributes to C's entries.
        
        # So each Strassen Mx is ONE rank-1 term in the decomposition!
        # That's why 2x2x2 has rank 7: 7 columns in U,V,W.
        
        # Let me verify: For M1 = (A00+A11)*(B00+B11):
        # U column: U[0,0]=1, U[1,1]=1, rest 0 (for the 2x2 A block)
        # V column: V[0,0]=1, V[1,1]=1, rest 0 (for the 2x2 B block)
        # W column: W[0,0]=1, W[0,1]=1, W[1,0]=1, W[1,1]=1 (M1 contributes to all 4 C entries)
        
        # Check: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1 -> contributes to C[0,0] from A[0,0]*B[0,0] ✓
        # U[0,0]*V[1,1]*W[0,1] = 1*1*1 = 1 -> contributes to C[0,1] from A[0,0]*B[1,1] ✓
        # U[1,1]*V[0,0]*W[1,0] = 1*1*1 = 1 -> contributes to C[1,0] from A[1,1]*B[0,0] ✓
        # U[1,1]*V[1,1]*W[1,1] = 1*1*1 = 1 -> contributes to C[1,1] from A[1,1]*B[1,1] ✓
        
        # But we also need: A[0,0]*B[1,1] contributes to C[0,1], and A[1,1]*B[0,0] to C[1,0]
        # These are captured. And A[0,0]*B[0,0] to C[0,0], A[1,1]*B[1,1] to C[1,1]. ✓
        
        # Great, so each Strassen Mx is indeed one rank-1 term.
        
        # Now let me implement the 7 Strassen terms for block (i0, j0, k0):
        
        # M1 = (A[i0,j0]+A[i0+1,j0+1])*(B[j0,k0]+B[j0+1,k0+1])
        #   -> C[i0,k0] += 1, C[i0,k0+1] += 1, C[i0+1,k0] += 1, C[i0+1,k0+1] += 1
        U[i0*m+j0][col] = 1.0
        U[(i0+1)*m+(j0+1)][col] = 1.0
        V[j0*p+k0][col] = 1.0
        V[(j0+1)*p+(k0+1)][col] = 1.0
        W[k0*n+i0][col] = 1.0
        W[(k0+1)*n+i0][col] = 1.0
        W[k0*n+(i0+1)][col] = 1.0
        W[(k0+1)*n+(i0+1)][col] = 1.0
        col += 1
        
        # M2 = (A[i0+1,j0]+A[i0+1,j0+1])*B[j0,k0]
        #   -> C[i0+1,k0] += 1
        U[(i0+1)*m+j0][col] = 1.0
        U[(i0+1)*m+(j0+1)][col] = 1.0
        V[j0*p+k0][col] = 1.0
        W[k0*n+(i0+1)][col] = 1.0
        col += 1
        
        # M3 = A[i0,j0]*(B[j0,k0+1]-B[j0+1,k0+1])
        #   -> C[i0,k0+1] += 1, C[i0,k0+1] -= 1 (net 0 for C[i0,k0+1] from this pair? No...)
        #   Actually: A[i0,j0]*B[j0,k0+1] -> C[i0,k0+1] += 1
        #              A[i0,j0]*B[j0+1,k0+1] -> C[i0,k0+1] -= 1
        #   Wait, B[j0+1,k0+1] with j=j0+1 contributes to C[i0,k0+1]
        U[i0*m+j0][col] = 1.0
        V[j0*p+(k0+1)][col] = 1.0
        V[(j0+1)*p+(k0+1)][col] = -1.0
        W[(k0+1)*n+i0][col] = 1.0
        col += 1
        
        # M4 = A[i0+1,j0+1]*(B[j0+1,k0]-B[j0,k0])
        #   -> C[i0+1,k0] += 1 (from B[j0+1,k0]), C[i0+1,k0] -= 1 (from B[j0,k0])
        U[(i0+1)*m+(j0+1)][col] = 1.0
        V[(j0+1)*p+k0][col] = 1.0
        V[j0*p+k0][col] = -1.0
        W[k0*n+(i0+1)][col] = 1.0
        col += 1
        
        # M5 = (A[i0,j0]+A[i0,j0+1])*B[j0+1,k0+1]
        #   -> C[i0,k0+1] += 1
        U[i0*m+j0][col] = 1.0
        U[i0*m+(j0+1)][col] = 1.0
        V[(j0+1)*p+(k0+1)][col] = 1.0
        W[(k0+1)*n+i0][col] = 1.0
        col += 1
        
        # M6 = (A[i0+1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k0+1])
        #   -> C[i0+1,k0] += 1, C[i0+1,k0+1] += 1
        #      C[i0,k0] -= 1, C[i0,k0+1] -= 1
        U[(i0+1)*m+j0][col] = 1.0
        U[i0*m+j0][col] = -1.0
        V[j0*p+k0][col] = 1.0
        V[j0*p+(k0+1)][col] = 1.0
        W[k0*n+(i0+1)][col] = 1.0
        W[(k0+1)*n+(i0+1)][col] = 1.0
        W[k0*n+i0][col] = -1.0
        W[(k0+1)*n+i0][col] = -1.0
        col += 1
        
        # M7 = (A[i0,j0+1]-A[i0+1,j0+1])*(B[j0+1,k0]+B[j0+1,k0+1])
        #   -> C[i0,k0] += 1, C[i0,k0+1] += 1
        #      C[i0+1,k0] -= 1, C[i0+1,k0+1] -= 1
        U[i0*m+(j0+1)][col] = 1.0
        U[(i0+1)*m+(j0+1)][col] = -1.0
        V[(j0+1)*p+k0][col] = 1.0
        V[(j0+1)*p+(k0+1)][col] = 1.0
        W[k0*n+i0][col] = 1.0
        W[(k0+1)*n+i0][col] = 1.0
        W[k0*n+(i0+1)][col] = -1.0
        W[(k0+1)*n+(i0+1)][col] = -1.0
        col += 1
    
    # Apply Strassen to the 4 disjoint 2x2x2 blocks:
    strassen_block(0, 0, 0)  # i in {0,1}, j in {0,1}, k in {0,1}
    strassen_block(0, 0, 2)  # i in {0,1}, j in {0,1}, k in {2,3}
    strassen_block(0, 2, 0)  # i in {0,1}, j in {2,3}, k in {0,1}
    strassen_block(0, 2, 2)  # i in {0,1}, j in {2,3}, k in {2,3}
    
    # Now add schoolbook terms for all remaining (i,j,k) triples
    # Strassen blocks cover: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # Remaining: 
    #   i=2, j=0..4, k=0..4 (25 terms)
    #   i in {0,1}, j=4, k=0..4 (10 terms)
    #   i in {0,1}, j=0..4, k=4 (10 terms, but j=4,k=4 overlaps with above)
    #   i in {0,1}, j=4, k=4 (1 term, already in j=4 set)
    
    # Let me just iterate over all triples and skip those covered by Strassen
    covered = set()
    for i0, j0, k0 in [(0,0,0), (0,0,2), (0,2,0), (0,2,2)]:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_schoolbook(i, j, k)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
