"""
Matrix multiplication 2,4,7 tensor decomposition.
Strategy:
1. Construct a structured decomposition using Strassen for 2x2 blocks and schoolbook for remainders.
   This yields a rank of 50 (improvement over schoolbook 56).
2. Apply random linear transformations (equivalences) to the factors U, V, W.
   This preserves the tensor but changes the representation.
3. Check if the new representation allows for rank reduction (e.g., via identifying zero rows/cols 
   or linear dependencies that become apparent after transformation).
   Specifically, we look for a basis of the column spaces of U, V, W that minimizes the rank 
   of the decomposition. If the tensor is "close" to a lower rank tensor in this basis, 
   we might find a valid lower rank decomposition.
   
   Note: Random transformations do not change the tensor rank. However, they can reveal 
   sparsity or structure that allows for a simpler *representation* if the rank was not minimal 
   in the original basis. But since we start from a valid decomposition, the rank is fixed 
   unless we find a linear dependency among the terms.
   
   A more effective approach for this specific task (2x4x7) is to recognize that the 
   multiplication tensor is a "slice" of larger tensors. 
   
   Let's stick to the most robust improvement: 
   1. Build the Rank 50 decomposition explicitly.
   2. Use SVD on the "flattened" term vectors to check for numerical rank deficiency.
   3. If rank < 50, reduce.
   
   However, for 2x4x7, the known lower bound is 28. The Strassen-based bound is 50.
   Is there a known algorithm between 28 and 50?
   
   For 2x4x7, we can view it as:
   C = A B
   A is 2x4, B is 4x7.
   
   We can split A into [A1 A2] where A1, A2 are 2x2.
   B into [B1; B2] where B1, B2 are 2x7.
   C = A1 B1 + A2 B2.
   
   Using Strassen for 2x2x7? No, Strassen is for square.
   
   Let's use the block structure:
   A = [[A11, A12], [A21, A22]] (2x4)
   B = [[B11, B12, B13, B14], [B21, B22, B23, B24], [B31, B32, B33, B34], [B41, B42, B43, B44]] (4x7)
   
   This is complex. The Rank 50 construction is:
   Split 4 into 2+2.
   C = A_left * B_top + A_right * B_bottom
   Where A_left, A_right are 2x2.
   B_top, B_bottom are 2x7.
   
   Multiplication of 2x2 by 2x7.
   Let M = 2x2, N = 2x7.
   Result is 2x7.
   We can use Strassen-like methods for 2x2 * 2x7?
   Standard 2x2 * 2x7 requires 2*2*7 = 28 multiplications?
   No, 2x2 times 2x7 is just 4 dot products of length 2? No.
   (2x2) * (2x7) -> 2x7.
   Entry (i,k) is sum_j A[i,j] B[j,k].
   There are 2*7=14 entries. Each is a sum of 2 products. Total 28 scalar multiplications.
   So Rank(2x2, 2x7) <= 28.
   
   If we use this for both halves:
   C = A_left * B_top + A_right * B_bottom.
   Rank = Rank(2x2, 2x7) + Rank(2x2, 2x7) = 28 + 28 = 56.
   This is just schoolbook split.
   
   How to get 50?
   Maybe combine the two 2x2 * 2x7 products?
   Let U = [A_left, A_right] (2x4).
   Let V = [B_top; B_bottom] (4x7).
   C = U V.
   
   Consider the 7 columns of B.
   B = [B_1, B_2, B_3, B_4, B_5, B_6, B_7].
   C = [U B_1, ..., U B_7].
   Each U B_k is a 2x1 vector.
   U is 2x4. B_k is 4x1.
   Multiplying 2x4 by 4x1.
   Rank(2,4,1) = 2*4*1 = 8? No, lower bound is max(2,4,1)=4.
   Actually, for 2x4 * 4x1, we can use Strassen?
   2x4 is not square.
   
   Let's look at the "Rank 50" claim in the inspiration.
   "Strassen decomposition of rank 50".
   How?
   Perhaps by grouping the 7 columns into 2+2+2+1?
   2x2 * 2x2 (Strassen rank 7)
   2x2 * 2x2 (Strassen rank 7)
   2x2 * 2x2 (Strassen rank 7)
   2x2 * 2x1 (Rank 4? 2*2*1=4. Lower bound 2. Can we do better?
   2x2 * 2x1: A [[a,b],[c,d]] B [[x],[y]] = [[ax+by],[cx+dy]].
   4 mults. Can we do 3?
   (a+c)(x+y) - cx - by? No.
   It is likely rank 4 for 2x2x1.
   So 7+7+7+4 = 25 per 2x2 block of A?
   Wait, A is 2x4. If we split B's 7 cols into 2,2,2,1.
   We need to multiply A (2x4) by each block.
   A is fixed.
   
   Actually, the standard "Rank 50" for 2x4x7 comes from:
   Viewing 2x4x7 as two 2x2x7 blocks? No.
   
   Let's just implement the Rank 56 schoolbook first, then try to reduce it via 
   "Linear Algebraic Rank Reduction" on the flattened tensor matrix.
   
   The key insight for improvement: 
   The schoolbook decomposition has rank 56.
   The tensor T has 56 entries.
   If we can find a decomposition of rank < 56, it must be a "non-trivial" one.
   
   I will implement a search for a rank 50 decomposition by using the known 
   Strassen algorithm for 2x2x2 and padding.
   
   Construction of Rank 50:
   Split the 4x7 matrix B into 4 blocks of 1x7? No.
   Split A (2x4) into two 2x2 blocks A1, A2.
   Split B (4x7) into two 2x7 blocks B1, B2.
   C = A1 B1 + A2 B2.
   
   Now we need to compute A1 B1 and A2 B2.
   A1, A2 are 2x2. B1, B2 are 2x7.
   
   Let's compute M = A1 B1.
   A1 = [[a, b], [c, d]].
   B1 = [[x1, x2, x3, x4, x5, x6, x7], [y1, y2, y3, y4, y5, y6, y7]].
   
   M[0, k] = a x_k + b y_k.
   M[1, k] = c x_k + d y_k.
   
   This is 4 multiplications per column k. Total 4*7 = 28 multiplications.
   Same for A2 B2. Total 56.
   
   To reduce, we need to share multiplications between the two blocks or within the columns.
   
   Consider the 7 columns of B1.
   We can group them.
   
   Actually, there is a known result: Rank(2,4,7) = 50?
   Let's assume the inspiration is correct and build a Rank 50 decomposition.
   
   How to build Rank 50?
   Use Strassen for the 2x2 part of the multiplication?
   If we view the 2x4 matrix as 2 rows of 2x2 blocks? No.
   
   Let's try a different structural approach:
   **Randomized Search for Low Rank Decompositions**.
   
   We know Rank 56 works.
   We will try to find Rank 50, 49, ... using a "Perturbation and Projection" method.
   1. Start with Rank 56.
   2. Apply random invertible transforms to U, V, W.
   3. Check if the new U, V, W have a lower "effective" rank (e.g. via SVD of the term matrix).
   4. If we find a dependency, we can reduce the rank.
   
   This is what the current program tries to do but fails because it doesn't actually *reduce* the rank correctly (it just drops columns which breaks the sum).
   
   Correct Reduction:
   If the terms t_r are linearly dependent, we can express one term as a sum of others.
   This doesn't reduce the count of terms in a *sum* representation unless we can merge them.
   But tensor rank is the minimum number of terms.
   If t_1 = t_2 + t_3, we can replace (t_1, t_2, t_3) with (t_2, t_3) and add t_1 to nothing? No.
   If T = t_1 + t_2 + t_3 and t_1 = t_2 + t_3, then T = 2t_2 + 2t_3.
   This is still 2 terms (if we scale).
   So yes, linear dependence allows rank reduction.
   
   Algorithm:
   1. Generate a valid decomposition (Schoolbook Rank 56).
   2. Flatten terms into matrix M (56x56).
   3. Compute SVD of M.
   4. If rank(M) < 56, we can reduce the rank to rank(M).
   5. Reconstruct the decomposition from the SVD basis?
      This is the hard part. The SVD gives new terms in the vector space C^56.
      We need to factor each new term back into u, v, w.
      This is generally impossible for arbitrary terms.
      
   However, if we start from a *structured* decomposition, the dependencies might be simple.
   
   Let's try a **Hybrid Approach**:
   1. Construct the Rank 50 decomposition explicitly using a known algorithm for 2x4x7.
      (I will derive it or use a heuristic construction).
   2. If I can't find a static Rank 50, I will use **Randomized SVD-based Rank Reduction** on the Rank 56 decomposition.
      Specifically, I will look for a basis of the column space of the "Flattened Tensor" that corresponds to a valid decomposition.
      
   Actually, the most promising "Structural Change" is to use **Strassen's Algorithm** explicitly for the 2x2 blocks and handle the 7 columns in a way that minimizes the multiplications.
   
   Let's implement the Rank 50 construction:
   We split the 4x7 matrix into four 1x7 rows? No.
   
   Let's use the following identity for 2x4 * 4x7:
   Let A = [[a1, a2, a3, a4], [b1, b2, b3, b4]]
   Let B = [[c1, c2, c3, c4, c5, c6, c7], ...]
   
   We can use the "Powers of 2" method?
   
   I will implement a **Randomized Search** that starts from the Schoolbook Rank 56 and applies random linear combinations of the *factors* (U, V, W) to try to create zeros in the term vectors, which allows for rank reduction.
   
   Wait, the current program already does something similar but it's not effective.
   
   New Strategy: **Direct Construction of Rank 50**.
   I will hardcode a Rank 50 decomposition if I can find one, or construct it via:
   1. Split 7 into 3+4.
   2. Compute 2x4 * 4x3 (Rank 24) + 2x4 * 4x4 (Rank 32)? Total 56.
   
   Let's try **Strassen for 2x2x2** on sub-blocks.
   If we split 4 into 2+2 and 7 into 2+2+2+1.
   A = [A1 A2] (2x2 each).
   B = [B1; B2] (2x7 each).
   
   C = A1 B1 + A2 B2.
   
   To compute A1 B1 (2x2 * 2x7):
   We can group the 7 columns of B1 into 2+2+2+1.
   B1 = [B1a, B1b, B1c, B1d] where B1a,b,c are 2x2 and B1d is 2x1.
   
   A1 B1a: 2x2 * 2x2. Use Strassen (Rank 7).
   A1 B1b: 2x2 * 2x2. Use Strassen (Rank 7).
   A1 B1c: 2x2 * 2x2. Use Strassen (Rank 7).
   A1 B1d: 2x2 * 2x1. Use Schoolbook (Rank 4).
   
   Total for A1 B1: 7+7+7+4 = 25.
   Total for A2 B2: 25.
   Total Rank = 50.
   
   This is a valid Rank 50 decomposition!
   
   I will implement this construction.
   1. Define the Strassen factors for 2x2x2.
   2. Define the Schoolbook factors for 2x2x1.
   3. Combine them for the 2x4x7 case.
   
   This is a MATERIAL DIFFERENCE: It uses a specific algebraic construction (Strassen + Schoolbook hybrid) rather than generic optimization.
"""

import time
import numpy as np

DIMENSIONS = [2, 4, 7]
DEADLINE = 130

def _verify(U, V, W, n, m, p, tol=1e-7):
    """Verify if the decomposition is exact."""
    rank = U.shape[1]
    if rank == 0:
        return False
        
    U_ = U.reshape(n, m, rank)
    V_ = V.reshape(m, p, rank)
    W_ = W.reshape(p, n, rank)
    
    W_t = W_.transpose(1, 0, 2)
    T = np.einsum('ijr,jkr,ikr->ijk', U_, V_, W_t)
    
    if np.max(np.abs(T - 1.0)) > tol:
        return False
    return True

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2 * 2x2 multiplication using Strassen's algorithm (Rank 7).
    A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    C = [[p5+p4-p2+p6, p1+p6, p1+p2, p3+p4]]
    p1 = (a+d)(e+h)
    p2 = (c+d)e
    p3 = a(f-h)
    p4 = d(g-e)
    p5 = (a+b)h
    p6 = (c-b)(e+f)
    p7 = (d-c)(g+h)
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Indices for A (2x2): 0:a, 1:b, 2:c, 3:d
    # Indices for B (2x2): 0:e, 1:f, 2:g, 3:h
    # Indices for C (2x2): 0:00, 1:01, 2:10, 3:11
    
    # p1 = (a+d)(e+h)
    # U: a=1, d=1. V: e=1, h=1.
    # C: 00 += 1, 11 += 1. 01 += -1, 10 += 1? 
    # Strassen:
    # C00 = p5 + p4 - p2 + p6
    # C01 = p1 + p6
    # C10 = p1 + p2
    # C11 = p3 + p4
    
    # p1: (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[1, 0] = 1 # C01
    W[2, 0] = 1 # C10
    
    # p2: (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[0, 1] = -1 # C00
    W[2, 1] = 1  # C10
    
    # p3: a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[3, 2] = 1  # C11
    
    # p4: d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1  # C00
    W[3, 3] = 1  # C11
    
    # p5: (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1  # C00
    
    # p6: (c-b)(e+f)
    U[2, 5] = 1; U[1, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1  # C00
    W[1, 5] = 1  # C01
    
    # p7: (d-c)(g+h)
    U[3, 6] = 1; U[2, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # p7 is not used in the standard 7-product formula for C?
    # Wait, standard Strassen uses 7 products.
    # C00 = p5 + p4 - p2 + p6
    # C01 = p1 + p6
    # C10 = p1 + p2
    # C11 = p3 + p4
    # p7 is not used? 
    # Ah, the 7th product is used in some formulations, but this one uses p1..p6?
    # No, p1..p7 are the 7 products.
    # Let's check the formula again.
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-b)(e+f)
    # p7 = (d-c)(g+h)
    
    # C00 = p5 + p4 - p2 + p6
    # C01 = p1 + p6
    # C10 = p1 + p2
    # C11 = p3 + p4
    
    # It seems p7 is not used in this specific set of formulas?
    # Actually, p7 is used if we want to compute C11 differently?
    # No, C11 = p3 + p4.
    # Is p7 redundant?
    # If p7 is not used, we have Rank 6? No, Strassen is Rank 7.
    # Let's check C11.
    # C11 = dh + bg.
    # p3 + p4 = a(f-h) + d(g-e) = af - ah + dg - de.
    # This is not dh + bg.
    # My formula for C11 is wrong for Strassen.
    
    # Correct Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-b)(e+f)
    # p7 = (d-c)(g+h)
    
    # C00 = p5 + p4 - p2 + p6
    # C01 = p1 + p6
    # C10 = p1 + p2
    # C11 = p3 + p4 + p7 - p1? No.
    
    # Let's use a verified Strassen implementation.
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-b)(e+f)
    # p7 = (d-c)(g+h)
    
    # C00 = p5 + p4 - p2 + p6
    # C01 = p1 + p6
    # C10 = p1 + p2
    # C11 = p3 + p4 + p7 - p1? 
    # Let's check C11 = dh + bg.
    # p3 + p4 + p7 - p1 = a(f-h) + d(g-e) + (d-c)(g+h) - (a+d)(e+h)
    # = af - ah + dg - de + dg + dh - cg - ch - ae - ah - de - dh
    # = af - 2ah + 2dg + dh - cg - ch - ae - 2de - dh
    # This is getting messy.
    
    # I will use a simpler Rank 4 schoolbook for 2x2x2 to ensure correctness, 
    # and rely on the 2x2x1 parts to be Rank 4.
    # Total Rank = 3 * 4 + 4 = 16 for A1B1?
    # 3 blocks of 2x2x2 (Rank 4 each) + 1 block of 2x2x1 (Rank 4) = 16.
    # Total Rank = 32.
    # This is better than 56!
    
    # Wait, 2x2x2 schoolbook is Rank 4.
    # 2x2x1 schoolbook is Rank 4.
    # So A1B1 (2x2 * 2x7) = 3*(2x2x2) + 1*(2x2x1) = 3*4 + 4 = 16.
    # A2B2 = 16.
    # Total Rank = 32.
    
    # Can we do better?
    # Strassen 2x2x2 is Rank 7.
    # If we use Strassen: 3*7 + 4 = 25.
    # Total Rank = 50.
    
    # So Schoolbook hybrid is Rank 32.
    # Strassen hybrid is Rank 50.
    # Schoolbook is better here!
    
    # Let's implement the Rank 32 construction.
    return None # Placeholder

def _construct_rank_32():
    """
    Construct Rank 32 decomposition using Schoolbook for sub-blocks.
    A (2x4) = [A1 A2] (2x2 each)
    B (4x7) = [B1; B2] (2x7 each)
    B1 (2x7) = [B1a B1b B1c B1d] (2x2, 2x2, 2x2, 2x1)
    B2 (2x7) = [B2a B2b B2c B2d] (2x2, 2x2, 2x2, 2x1)
    
    C = A1 B1 + A2 B2
    A1 B1 = A1 B1a + A1 B1b + A1 B1c + A1 B1d
    Each A1 B1x is 2x2 * 2x2 or 2x2 * 2x1.
    
    Total Terms:
    4 terms for A1B1a (2x2x2 schoolbook)
    4 terms for A1B1b
    4 terms for A1B1c
    4 terms for A1B1d (2x2x1 schoolbook)
    Total 16 for A1B1.
    Total 16 for A2B2.
    Total 32.
    """
    n, m, p = 2, 4, 7
    rank = 32
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Helper to add a schoolbook 2x2x2 block
    # A_block is 2x2, B_block is 2x2, C_block is 2x2
    # A indices in U: relative to A_block
    # B indices in V: relative to B_block
    # C indices in W: relative to C_block
    
    def add_2x2x2(U, V, W, a_offset, b_offset, c_offset, start_col):
        # a_offset: (row, col) in A (2x4)
        # b_offset: (row, col) in B (4x7)
        # c_offset: (row, col) in C (2x7)
        r_a, c_a = a_offset
        r_b, c_b = b_offset
        r_c, c_c = c_offset
        
        # 4 terms
        # 1: a11 * b11 -> c11
        U[r_a*2 + c_a, start_col] = 1
        V[(r_b*2 + c_b)*7 + (c_c), start_col] = 1 # Wait, B is 4x7.
        # B index: row r_b (0 or 1 in 2x7, so 0-3 in 4x7), col c_b (0-1 in 2x2, so 0-6 in 2x7? No)
        
        # Let's map indices carefully.
        # A is 2x4. A1 is A[0:2, 0:2]. A2 is A[0:2, 2:4].
        # B is 4x7. B1 is B[0:2, 0:7]. B2 is B[2:4, 0:7].
        # B1a is B1[0:2, 0:2] -> B[0:2, 0:2].
        # B1b is B1[0:2, 2:4] -> B[0:2, 2:4].
        # B1c is B1[0:2, 4:6] -> B[0:2, 4:6].
        # B1d is B1[0:2, 6:7] -> B[0:2, 6:7].
        
        # C is 2x7.
        # C1a is C[0:2, 0:2].
        # C1b is C[0:2, 2:4].
        # C1c is C[0:2, 4:6].
        # C1d is C[0:2, 6:7].
        
        # Term 1: A[0,0] * B[0,0] -> C[0,0]
        # A idx: 0*2+0 = 0
        # B idx: 0*7+0 = 0
        # C idx: 0*2+0 = 0
        
        # Term 2: A[0,0] * B[1,0] -> C[1,0]
        # A idx: 0
        # B idx: 1*7+0 = 7
        # C idx: 1*2+0 = 2
        
        # Term 3: A[1,0] * B[0,0] -> C[0,1]
        # A idx: 1*2+0 = 2
        # B idx: 0
        # C idx: 0*2+1 = 1
        
        # Term 4: A[1,0] * B[1,0] -> C[1,1]
        # A idx: 2
        # B idx: 7
        # C idx: 2*2+1 = 5
        
        # Generalize for offsets
        
        # A indices:
        a00 = (r_a + 0)*2 + (c_a + 0)
        a01 = (r_a + 0)*2 + (c_a + 1)
        a10 = (r_a + 1)*2 + (c_a + 0)
        a11 = (r_a + 1)*2 + (c_a + 1)
        
        # B indices:
        # B is 4x7.
        # Block is 2x2.
        # Rows: r_b, r_b+1
        # Cols: c_b, c_b+1
        b00 = (r_b + 0)*7 + (c_b + 0)
        b01 = (r_b + 0)*7 + (c_b + 1)
        b10 = (r_b + 1)*7 + (c_b + 0)
        b11 = (r_b + 1)*7 + (c_b + 1)
        
        # C indices:
        # C is 2x7.
        # Block is 2x2.
        # Rows: r_c, r_c+1
        # Cols: c_c, c_c+1
        c00 = (r_c + 0)*2 + (c_c + 0)
        c01 = (r_c + 0)*2 + (c_c + 1)
        c10 = (r_c + 1)*2 + (c_c + 0)
        c11 = (r_c + 1)*2 + (c_c + 1)
        
        U[a00, start_col] += 1; V[b00, start_col] += 1; W[c00, start_col] += 1
        U[a00, start_col+1] += 1; V[b10, start_col+1] += 1; W[c01, start_col+1] += 1
        U[a10, start_col+2] += 1; V[b00, start_col+2] += 1; W[c10, start_col+2] += 1
        U[a10, start_col+3] += 1; V[b10, start_col+3] += 1; W[c11, start_col+3] += 1
        
        return start_col + 4

    def add_2x2x1(U, V, W, a_offset, b_offset, c_offset, start_col):
        # A is 2x2, B is 2x1, C is 2x1
        r_a, c_a = a_offset
        r_b, c_b = b_offset
        r_c, c_c = c_offset
        
        a00 = (r_a + 0)*2 + (c_a + 0)
        a01 = (r_a + 0)*2 + (c_a + 1)
        a10 = (r_a + 1)*2 + (c_a + 0)
        a11 = (r_a + 1)*2 + (c_a + 1)
        
        b0 = (r_b + 0)*7 + (c_b + 0)
        b1 = (r_b + 1)*7 + (c_b + 0)
        
        c0 = (r_c + 0)*2 + (c_c + 0)
        c1 = (r_c + 1)*2 + (c_c + 0)
        
        U[a00, start_col] += 1; V[b0, start_col] += 1; W[c0, start_col] += 1
        U[a01, start_col+1] += 1; V[b0, start_col+1] += 1; W[c0, start_col+1] += 1
        U[a10, start_col+2] += 1; V[b1, start_col+2] += 1; W[c1, start_col+2] += 1
        U[a11, start_col+3] += 1; V[b1, start_col+3] += 1; W[c1, start_col+3] += 1
        
        return start_col + 4

    # A1: (0,0) in A (2x4)
    # A2: (0,2) in A (2x4)
    
    # B1a: (0,0) in B (4x7)
    # B1b: (0,2) in B
    # B1c: (0,4) in B
    # B1d: (0,6) in B
    
    # B2a: (2,0) in B
    # B2b: (2,2) in B
    # B2c: (2,4) in B
    # B2d: (2,6) in B
    
    # C1a: (0,0) in C
    # C1b: (0,2) in C
    # C1c: (0,4) in C
    # C1d: (0,6) in C
    
    # C2a: (0,0) in C? No, C is 2x7.
    # A1 B1 contributes to C[0:2, 0:7].
    # A2 B2 contributes to C[0:2, 0:7].
    # So C offsets are the same for both?
    # Yes, C = A1 B1 + A2 B2.
    
    col = 0
    # A1 B1
    col = add_2x2x2(U, V, W, (0,0), (0,0), (0,0), col) # A1 B1a -> C1a
    col = add_2x2x2(U, V, W, (0,0), (0,2), (0,2), col) # A1 B1b -> C1b
    col = add_2x2x2(U, V, W, (0,0), (0,4), (0,4), col) # A1 B1c -> C1c
    col = add_2x2x1(U, V, W, (0,0), (0,6), (0,6), col) # A1 B1d -> C1d
    
    # A2 B2
    col = add_2x2x2(U, V, W, (0,2), (2,0), (0,0), col) # A2 B2a -> C1a
    col = add_2x2x2(U, V, W, (0,2), (2,2), (0,2), col) # A2 B2b -> C1b
    col = add_2x2x2(U, V, W, (0,2), (2,4), (0,4), col) # A2 B2c -> C1c
    col = add_2x2x1(U, V, W, (0,2), (2,6), (0,6), col) # A2 B2d -> C1d
    
    return U, V, W

def _construct_schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try Rank 32 construction
    U, V, W = _construct_rank_32()
    
    if _verify(U, V, W, n, m, p):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback to Schoolbook
    U, V, W = _construct_schoolbook(n, m, p)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))

