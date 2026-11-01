"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0  # Safety margin

    # Strategy: Construct a decomposition for M_{2,3,7}
    # We can decompose M_{2,3,7} into M_{2,3,6} + M_{2,3,1}
    # Rank(M_{2,3,1}) = 1 (trivial: A[0,0]*B[0,0]*1 + A[0,1]*B[1,0]*1 + A[0,2]*B[2,0]*1 ... wait)
    # Actually M_{2,3,1} is just the sum of products for the single column.
    # C[i,0] = sum_j A[i,j] B[j,0]. This is 2x3x1. Rank is 2? No, rank is 2*1? 
    # For M_{n,m,1}, rank is n. Because C[i,0] = sum_j A[i,j]B[j,0].
    # Terms: for each i, sum over j. This is not a single tensor product.
    # The tensor T[i,j,k] for k=0 is 1 for all i,j.
    # We need sum_r U[i,j,r]V[j,0,r]W[0,i,r] = 1.
    # This is just the identity on the 2x3 grid? No.
    # It's the "all ones" tensor of shape (2,3,1).
    # Rank of all-ones tensor (2,3,1) is 1? 
    # U[i,j,r]V[j,0,r]W[0,i,r]. 
    # If we set U=1, V=1, W=1 for one term, sum is 1. 
    # So Rank(M_{2,3,1}) = 1.
    
    # Total Rank = Rank(M_{2,3,6}) + 1.
    # We need a good rank for M_{2,3,6}.
    # Bläser (2010) gives Rank(M_{2,n,2n}) = 4n - 1.
    # For n=3, 2n=6. Rank = 4*3 - 1 = 11.
    # Total Rank = 11 + 1 = 12? 
    # Wait, the decomposition M_{2,3,7} = M_{2,3,6} + M_{2,3,1} is direct sum if we split the output dimension k.
    # The tensor is T[i,j,k]. 
    # If we split k into k1 (0..5) and k2 (6), the tensor is T1 (for k1) + T2 (for k2).
    # Since the supports on k are disjoint, the rank is additive.
    # So if we have Rank(M_{2,3,6})=11 and Rank(M_{2,3,1})=1, total rank is 12.
    # Is 12 possible? 
    # Lower bound for M_{2,3,7}: 
    # Omega >= n*m*p / (something). 
    # Standard lower bound: Rank >= n*m*p / (max(n,m,p) * something)?
    # For M_{2,3,7}, known best is 17? Or is 12 valid?
    # Let's check Bläser's formula again.
    # Bläser 2010: "The complexity of matrix multiplication".
    # He proved Rank(M_{2,n,2n}) = 4n - 1 for n >= 2?
    # For n=2 (2x2x4), Rank = 7? Strassen is 7 for 2x2x2. 
    # M_{2,2,4} is two 2x2 multiplications. Rank = 7+7=14? No, 4n-1 = 7. 
    # If Rank(M_{2,2,4})=7, that's better than 14.
    # For n=3 (2x3x6), Rank = 11.
    # So Rank(M_{2,3,7}) = 11 + 1 = 12.
    # This is significantly better than 39 or 42.
    
    # However, implementing Bläser's specific 11-rank decomposition for 2x3x6 is complex.
    # The current program failed to do it.
    # Let's try a different known decomposition or a simpler heuristic that might work.
    # Or, we can use the "Winograd" approach for rectangular matrices.
    
    # Alternative: Use a randomized local search to find a low rank decomposition.
    # Since we have 150s, we can try to optimize a set of random initial terms.
    # But tensor rank minimization is NP-hard.
    
    # Let's try to construct the Bläser decomposition explicitly.
    # Reference: R. Bläser, "The complexity of matrix multiplication", 2010.
    # The construction for M_{2,n,2n} uses 4n-1 terms.
    # Terms are indexed by r=0..4n-2.
    # The points x_i are chosen such that...
    
    # Given the high risk of implementing Bläser incorrectly within the time limit,
    # and the fact that the "inspiration" found a score of 1.077 (Rank 39),
    # I will attempt a **Hybrid Strassen/Schoolbook** approach that is robust,
    # but I will also try to squeeze more out of it by optimizing the split.
    
    # Actually, let's look at the dimensions: 2x3x7.
    # Split 7 into 4 + 3.
    # M_{2,3,4} and M_{2,3,3}.
    # Rank(M_{2,3,4}): 
    #   Can be seen as 2x2x4 (Rank 7) + 2x1x4 (Rank 2)? Total 9?
    #   Or 2x3x4 directly.
    # Rank(M_{2,3,3}):
    #   2x2x3 (Rank 5? 2x2x2 is 7, 2x2x3 is ?) + 2x1x3 (Rank 2).
    
    # Let's stick to the most reliable method: **Constructing a valid decomposition with known rank**.
    # The schoolbook is rank 42.
    # The "Inspiration" got 39.
    # Can we do better than 39 easily?
    # 
    # Let's try to construct a rank 34 or 35 solution using a block-wise Strassen.
    # Split A (2x3) into A1 (2x2) and A2 (2x1).
    # Split B (3x7) into B1 (2x7) and B2 (1x7).
    # C = A1*B1 + A2*B2.
    # Rank(A1*B1) is Rank(M_{2,2,7}).
    # Rank(A2*B2) is Rank(M_{2,1,7}) = 2 (since it's just 2 rows of B2 multiplied by 1 col of A2? No, A2 is 2x1, B2 is 1x7. C2 is 2x7. C2[i,k] = A2[i,0]*B2[0,k]. This is rank 2 tensor? 
    #   T[i,j=0,k] = 1. 
    #   U[i,0,r] V[0,k,r] W[k,i,r]. 
    #   We need sum_r ... = 1 for all i,k.
    #   If we use 2 terms: 
    #     r=0: U[0,0,0]=1, V[0,k,0]=1, W[k,0,0]=1 -> contributes to i=0.
    #     r=1: U[1,0,1]=1, V[0,k,1]=1, W[k,1,1]=1 -> contributes to i=1.
    #   So Rank(M_{2,1,7}) = 2.
    # 
    # Now Rank(M_{2,2,7}).
    # Split 7 into 4+3.
    # M_{2,2,4} has Rank 7 (Bläser/Strassen extension).
    # M_{2,2,3} has Rank ? 
    #   M_{2,2,3} = M_{2,2,2} + M_{2,2,1}.
    #   Rank(M_{2,2,2}) = 7 (Strassen).
    #   Rank(M_{2,2,1}) = 2 (similar to above).
    #   Total Rank(M_{2,2,3}) = 9.
    #   Total Rank(M_{2,2,7}) = 7 + 9 = 16.
    # 
    # Total Rank = Rank(M_{2,2,7}) + Rank(M_{2,1,7}) = 16 + 2 = 18.
    # 
    # This is much better than 39!
    # Let's verify the logic.
    # 
    # 1. M_{2,1,7} rank 2:
    #    Terms:
    #    r0: U[0,0]=1, V[0,k]=1, W[k,0]=1. (Contributes to C[0,k])
    #    r1: U[1,0]=1, V[0,k]=1, W[k,1]=1. (Contributes to C[1,k])
    #    Check: C[0,k] = sum_j A[0,j]B[j,k]. Here j is only 0. A[0,0]B[0,k].
    #    Our tensor term: U[0,0]V[0,k]W[k,0] = 1*1*1 = 1. Correct.
    #    C[1,k] = A[1,0]B[0,k]. Term: 1*1*1 = 1. Correct.
    #    Other entries are 0. Correct.
    # 
    # 2. M_{2,2,7} rank 16:
    #    Split k into 0..3 and 4..6.
    #    Part 1: M_{2,2,4} (k=0..3). Rank 7.
    #    Part 2: M_{2,2,3} (k=4..6). Rank 9.
    # 
    #    M_{2,2,4} Rank 7 construction:
    #    This is the "Strassen" for 2x2x4? 
    #    Actually, M_{2,2,4} is two independent 2x2 multiplications.
    #    Rank(M_{2,2,2}) = 7.
    #    So Rank(M_{2,2,4}) = 14? 
    #    Wait, Bläser's 4n-1 formula for n=2 gives 7 for M_{2,2,4}.
    #    Is that correct? 
    #    M_{2,2,4} tensor T[i,j,k]. 
    #    If we can do it in 7, that's amazing. 
    #    Standard lower bound for M_{2,2,2} is 7. 
    #    M_{2,2,4} contains two M_{2,2,2} blocks. 
    #    Usually rank is additive for disjoint supports. 
    #    So Rank(M_{2,2,4}) should be at least 14? 
    #    Unless there's a clever mixing. 
    #    Bläser's result: Rank(M_{2,n,2n}) = 4n-1.
    #    For n=2, 2n=4. Rank = 7.
    #    This implies we can multiply two 2x2 matrices (total 4x4 output? No, 2x2 input, 2x2 output per block?) 
    #    M_{2,2,4} means A is 2x2, B is 2x4, C is 2x4.
    #    This is equivalent to two 2x2 multiplications: C_left = A*B_left, C_right = A*B_right.
    #    If Rank is 7, it's better than 14.
    #    I will assume this is correct and try to construct it.
    #    However, constructing the specific 7-term decomposition for M_{2,2,4} is non-trivial.
    # 
    #    Given the complexity, I will fall back to a **robust Rank 28** solution 
    #    which is 2 * Rank(M_{2,2,7})? No.
    # 
    #    Let's use a simpler, verified decomposition: **Rank 24**.
    #    Split 2x3x7 into four 2x2x2 blocks? No.
    #    
    #    Let's go with **Rank 32** using a simple block decomposition that is easy to code and verify.
    #    Split A (2x3) into A1(2x2) + A2(2x1).
    #    Split B (3x7) into B1(2x7) + B2(1x7).
    #    C = A1*B1 + A2*B2.
    #    Rank(A2*B2) = 2.
    #    Rank(A1*B1) for 2x2x7.
    #    Split B1 (2x7) into B1a(2x4) + B1b(2x3).
    #    Rank(A1*B1a) for 2x2x4. Use Schoolbook for this part? Rank 8.
    #    Rank(A1*B1b) for 2x2x3. Use Schoolbook? Rank 6.
    #    Total Rank = 2 + 8 + 6 = 16.
    #    
    #    Wait, if I use Schoolbook for 2x2x4, rank is 2*2*4 = 16? No, 2*2*4=16 terms.
    #    Schoolbook for M_{n,m,p} is n*m*p.
    #    M_{2,2,4} -> 16 terms.
    #    M_{2,2,3} -> 12 terms.
    #    M_{2,1,7} -> 14 terms? No, Rank 2.
    #    Total = 16 + 12 + 2 = 30.
    #    
    #    Can I do better for M_{2,2,4}? 
    #    If I use Strassen for the two 2x2x2 blocks inside 2x2x4:
    #    M_{2,2,4} = M_{2,2,2} (left) + M_{2,2,2} (right).
    #    Rank = 7 + 7 = 14.
    #    So Total Rank = 14 (for 2x2x4) + 12 (for 2x2x3 schoolbook) + 2 (for 2x1x7) = 28.
    #    
    #    Can I do better for M_{2,2,3}?
    #    M_{2,2,3} = M_{2,2,2} + M_{2,2,1}.
    #    Rank = 7 (Strassen) + 2 (Schoolbook/Trivial) = 9.
    #    Total Rank = 14 + 9 + 2 = 25.
    #    
    #    Can I do better for M_{2,2,4}?
    #    If Bläser's 7 is correct, Total = 7 + 9 + 2 = 18.
    #    
    #    Let's implement the **Rank 25** solution (Strassen for 2x2x2, Schoolbook for rest).
    #    It's safer than guessing Bläser's coefficients.
    #    And 25 is much better than 39.
    # 
    #    Decomposition Plan:
    #    1. M_{2,1,7} (A2, B2): Rank 2.
    #       Terms:
    #       r0: U[0,0]=1, V[0,k]=1, W[k,0]=1
    #       r1: U[1,0]=1, V[0,k]=1, W[k,1]=1
    #       (Note: Indices in U are (i,j), V is (j,k), W is (k,i))
    #       Here j=0 for A2/B2.
    #       
    #    2. M_{2,2,4} (A1, B1a, k=0..3): Rank 14 (Two Strassen 2x2x2).
    #       Block 1 (k=0,1): Strassen 2x2x2.
    #       Block 2 (k=2,3): Strassen 2x2x2.
    #       
    #    3. M_{2,2,3} (A1, B1b, k=4..6): Rank 9 (One Strassen 2x2x2 + One Schoolbook 2x2x1).
    #       Block 3 (k=4,5): Strassen 2x2x2.
    #       Block 4 (k=6): Schoolbook 2x2x1 (Rank 4? No, M_{2,2,1} rank is 2? 
    #         T[i,j,0]=1. 
    #         U[i,j]V[j,0]W[0,i]. 
    #         Need sum=1. 
    #         Rank 2: 
    #         r0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> C[0,0]
    #         r1: U[1,1]=1, V[1,0]=1, W[0,1]=1 -> C[1,0]? No, W[k,i]. 
    #         M_{2,2,1}: C[i,0] = sum_j A[i,j]B[j,0].
    #         Terms:
    #         r0: U[0,0]=1, V[0,0]=1, W[0,0]=1. (i=0,j=0)
    #         r1: U[0,1]=1, V[1,0]=1, W[0,1]=1. (i=0,j=1)
    #         r2: U[1,0]=1, V[0,0]=1, W[0,2]=1. (i=1,j=0)
    #         r3: U[1,1]=1, V[1,0]=1, W[0,3]=1. (i=1,j=1)
    #         Rank 4.
    #       )
    #       
    #    Total Rank = 2 + 14 + (7 + 4) = 27.
    #    
    #    Let's refine M_{2,2,1} rank.
    #    Is Rank(M_{2,2,1}) = 2?
    #    T[i,j,0] = 1.
    #    U[i,j,r]V[j,0,r]W[0,i,r].
    #    If r=0: U[0,0]=1, V[0,0]=1, W[0,0]=1. Sum for (0,0) is 1.
    #    If r=1: U[1,1]=1, V[1,0]=1, W[0,1]=1. Sum for (1,1) is 1.
    #    But we need (0,1) and (1,0) to be 1 too.
    #    So Rank is at least 4? 
    #    Actually, the tensor is the "all ones" 2x2 matrix.
    #    Rank of all-ones 2x2 tensor (embedded in 2x2x1) is 4? 
    #    Yes, because it's the identity in the tensor space? No.
    #    It's the sum of 4 basis tensors. 
    #    So Rank 4.
    #    
    #    So Total Rank = 2 + 14 + 7 + 4 = 27.
    # 
    #    This is a solid Rank 27 solution.
    # 
    #    Let's code this up.

    def build_strassen_2x2x2(offset_i=0, offset_j=0, offset_k=0, rank_offset=0):
        """
        Returns U, V, W lists for a 2x2x2 Strassen decomposition.
        U shape: (2*2, 7) -> but we only fill specific indices.
        Actually, I will build the global matrices directly.
        """
        pass

    # We will build the global U, V, W matrices of size (n*m, R), (m*p, R), (p*n, R).
    # n=2, m=3, p=7.
    # U size: 6 x R
    # V size: 21 x R
    # W size: 14 x R
    
    # We will accumulate terms.
    terms = []
    
    # Helper to add a term
    def add_term(i, j, k, u_val, v_val, w_val):
        # i in [0,2), j in [0,3), k in [0,7)
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        terms.append((i*m + j, j*p + k, k*n + i, u_val, v_val, w_val))

    # 1. M_{2,1,7} part (j=2)
    # A[i,2] * B[2,k]
    # Terms for i=0, i=1
    # r=0: i=0, j=2. U[0,2]=1, V[2,k]=1, W[k,0]=1
    for k in range(7):
        add_term(0, 2, k, 1, 1, 1) # This adds 7 terms? No, Strassen/Schoolbook terms are per (i,j,k) if schoolbook.
        # Wait, my "add_term" adds a single tensor product.
        # For M_{2,1,7} rank 2, I need:
        # Term 0: Covers i=0. U[0,2]=1, V[2,k]=1 for all k? No, V is a matrix.
        # In tensor decomposition, V[j,k,r] is a scalar.
        # So for a single term r, V[2,k,r] must be defined for all k.
        # If I set V[2,k,r]=1 for all k, and W[k,0,r]=1 for all k, and U[0,2,r]=1.
        # Then sum_r U[0,2,r]V[2,k,r]W[k,0,r] = 1*1*1 = 1.
        # This works for a single term r covering all k for i=0.
        # So I don't loop over k for the term definition. I define the term once.
        
    # Let's restart the term accumulation logic.
    # A term is defined by (U_row, V_row, W_row, u_val, v_val, w_val) is NOT enough if V/W are vectors.
    # A term r is a set of scalars:
    #   U[r] is a vector of length 6.
    #   V[r] is a vector of length 21.
    #   W[r] is a vector of length 14.
    
    R = 0
    U_list = []
    V_list = []
    W_list = []
    
    def add_term_vec(u_vec, v_vec, w_vec):
        nonlocal R
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
        R += 1

    # 1. M_{2,1,7} (j=2)
    # Term 1: i=0
    u = [0]*6; v = [0]*21; w = [0]*14
    u[0*3 + 2] = 1.0 # U[0,2]
    for k in range(7):
        v[2*7 + k] = 1.0 # V[2,k]
        w[k*2 + 0] = 1.0 # W[k,0]
    add_term_vec(u, v, w)
    
    # Term 2: i=1
    u = [0]*6; v = [0]*21; w = [0]*14
    u[1*3 + 2] = 1.0 # U[1,2]
    for k in range(7):
        v[2*7 + k] = 1.0 # V[2,k]
        w[k*2 + 1] = 1.0 # W[k,1]
    add_term_vec(u, v, w)
    
    # 2. M_{2,2,4} (j=0,1; k=0..3)
    # Split into two 2x2x2 blocks.
    # Block 1: k=0,1.
    # Strassen 2x2x2 terms.
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B11)
    # M4 = A22(B22-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # We need to map these to our global indices.
    # A is 2x2 (i=0,1; j=0,1). B is 2x2 (j=0,1; k=k0,k1).
    # U[i,j,r], V[j,k,r], W[k,i,r].
    
    def add_strassen_block(k_start):
        # k_start is 0 or 2.
        # k0 = k_start, k1 = k_start+1
        # j0=0, j1=1. i0=0, i1=1.
        
        # M1
        u = [0]*6; v = [0]*21; w = [0]*14
        u[0*3+0] = 1; u[1*3+1] = 1 # A00, A11
        v[0*7+k_start] = 1; v[1*7+k_start+1] = 1 # B0k0, B1k1
        w[k_start*2+0] = 1; w[k_start*2+1] = 1; w[(k_start+1)*2+0] = 1; w[(k_start+1)*2+1] = 1
        # Wait, W[k,i]. 
        # C00 gets +M1. W[k0,0] += 1.
        # C01 gets -M1? No, C11 (0,0) is M1+M4-M5+M7.
        # C12 (0,1) is M3+M5.
        # C21 (1,0) is M2+M4.
        # C22 (1,1) is M1-M2+M3+M6.
        
        # Let's just add the 7 Strassen terms for this block.
        # Term M1
        u = [0]*6; v = [0]*21; w = [0]*14
        u[0*3+0] = 1; u[1*3+1] = 1
        v[0*7+k_start] = 1; v[1*7+k_start+1] = 1
        w[k_start*2+0] = 1 # C00
        w[(k_start+1)*2+1] = 1 # C11
        add_term_vec(u, v, w)
        
        # Term M2
        u = [0]*6; v = [0]*21; w = [0]*14
        u[1*3+0] = 1; u[1*3+1] = 1 # A10+A11
        v[0*7+k_start] = 1 # B0k0
        w[k_start*2+1] = 1 # C10
        w[(k_start+1)*2+1] = -1 # C11
        add_term_vec(u, v, w)
        
        # Term M3
        u = [0]*6; v = [0]*21; w = [0]*14
        u[0*3+0] = 1 # A00
        v[1*7+k_start] = 1; v[0*7+k_start] = -1 # B1k0 - B0k0
        w[k_start*2+0] = 1 # C00
        w[(k_start+1)*2+0] = 1 # C01
        add_term_vec(u, v, w)
        
        # Term M4
        u = [0]*6; v = [0]*21; w = [0]*14
        u[1*3+1] = 1 # A11
        v[1*7+k_start+1] = 1; v[0*7+k_start+1] = -1 # B1k1 - B0k1
        w[k_start*2+0] = 1 # C00
        w[k_start*2+1] = 1 # C10
        add_term_vec(u, v, w)
        
        # Term M5
        u = [0]*6; v = [0]*21; w = [0]*14
        u[0*3+0] = 1; u[0*3+1] = 1 # A00+A01
        v[1*7+k_start+1] = 1 # B1k1
        w[k_start*2+0] = -1 # C00
        w[(k_start+1)*2+0] = 1 # C01
        add_term_vec(u, v, w)
        
        # Term M6
        u = [0]*6; v = [0]*21; w = [0]*14
        u[1*3+0] = 1; u[0*3+0] = -1 # A10-A00
        v[0*7+k_start] = 1; v[0*7+k_start+1] = 1 # B0k0+B0k1
        w[(k_start+1)*2+1] = 1 # C11
        add_term_vec(u, v, w)
        
        # Term M7
        u = [0]*6; v = [0]*21; w = [0]*14
        u[0*3+1] = 1; u[1*3+1] = -1 # A01-A11
        v[1*7+k_start] = 1; v[1*7+k_start+1] = 1 # B1k0+B1k1
        w[k_start*2+0] = 1 # C00
        w[(k_start+1)*2+1] = 1 # C11
        add_term_vec(u, v, w)
        
    add_strassen_block(0) # k=0,1
    add_strassen_block(2) # k=2,3
    
    # 3. M_{2,2,3} (j=0,1; k=4,5,6)
    # Block 3: k=4,5 (Strassen)
    add_strassen_block(4)
    
    # Block 4: k=6 (Schoolbook 2x2x1)
    # C[i,6] = A[i,0]B[0,6] + A[i,1]B[1,6]
    # 4 terms.
    for i in range(2):
        for j in range(2):
            u = [0]*6; v = [0]*21; w = [0]*14
            u[i*3+j] = 1
            v[j*7+6] = 1
            w[6*2+i] = 1
            add_term_vec(u, v, w)
            
    # Total Rank R
    # 2 (from part 1) + 7 (block 0) + 7 (block 2) + 7 (block 4) + 4 (block 6) = 27.
    
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    # Verify
    def verify(U, V, W):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    prod = u_val * v_val * w_val
                    total = np.sum(prod)
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to Schoolbook if construction fails
        schoolbook_terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        U_sb = np.zeros((n*m, n*m*p))
        V_sb = np.zeros((m*p, n*m*p))
        W_sb = np.zeros((p*n, n*m*p))
        for c, (i, j, k) in enumerate(schoolbook_terms):
            U_sb[i*m+j, c] = 1.0
            V_sb[j*p+k, c] = 1.0
            W_sb[k*n+i, c] = 1.0
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
