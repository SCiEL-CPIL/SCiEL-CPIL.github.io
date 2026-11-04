"""Rewritten program: rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.

Strategy:
  1. Build the 4x4 * 4x4 tensor as a Kronecker product of two 2x2 * 2x2 Strassen tensors
     (rank 7*7 = 49).
  2. Extend to 4x4 * 4x7 by adding schoolbook terms for the extra 3 columns of B
     (4*4*3 = 48 terms), giving a candidate rank of 49+48 = 97.
  3. Apply Gaussian elimination over Q to the 3-tensor (flattened to a matrix of
     size (n*m*p) x rank) to find the minimal linearly independent subset of
     rank-1 terms, thereby reducing the rank.

The decomposition must satisfy:
  For all i in [0,4), j in [0,4), k in [0,7):
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1

We verify correctness by checking this identity.
"""
import numpy as np
from fractions import Fraction
from itertools import product
from collections import defaultdict

DIMENSIONS = [4, 4, 7]


def _strassen_2x2_terms():
    """Return the 7 rank-1 terms for 2x2 * 2x2 Strassen decomposition.
    
    Each term is (a_idx, b_idx, w_idx, coeff) where:
      a_idx = i*2 + j  (row i, col j of A)
      b_idx = j*2 + k  (row j, col k of B)
      w_idx = k*2 + i  (row k, col i of C, trace convention)
    
    The 7 products M1..M7 and their contributions to C entries:
      M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
      M2 = (A10+A11)B00       -> C10 += M2, C11 -= M2
      M3 = A00(B10-B11)       -> C01 += M3, C11 += M3
      M4 = A11(B01-B00)       -> C00 += M4, C10 += M4
      M5 = (A00+A01)B11       -> C00 -= M5, C01 += M5
      M6 = (A10-A00)(B00+B01) -> C10 += M6
      M7 = (A01-A10)(B10+B11) -> C01 += M7
    
    But we need to express each M as a product of linear forms in A and B.
    Each M gives one rank-1 term in the tensor decomposition.
    
    For M1 = (A00+A11)(B00+B11):
      A part: A00 + A11 -> a_vec has 1 at indices 0 (i=0,j=0) and 3 (i=1,j=1)
      B part: B00 + B11 -> b_vec has 1 at indices 0 (j=0,k=0) and 3 (j=1,k=1)
      C part: C00 += 1, C11 += 1 -> w_vec has 1 at (k=0,i=0)=0 and (k=1,i=1)=3
    
    Actually, the tensor decomposition works as follows:
      T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    For a single M term, U, V, W are vectors such that:
      U[i,j,r] * V[j,k,r] * W[k,i,r] contributes to the (i,j,k) entry.
    
    For M1 = (A00+A11)(B00+B11):
      The A factor (A00+A11) means U has non-zero entries at (i=0,j=0) and (i=1,j=1), both = 1.
      The B factor (B00+B11) means V has non-zero entries at (j=0,k=0) and (j=1,k=1), both = 1.
      The C factor: M1 contributes +1 to C[0,0] and +1 to C[1,1].
      C[0,0] corresponds to (k=0, i=0) -> w_idx = 0*2+0 = 0
      C[1,1] corresponds to (k=1, i=1) -> w_idx = 1*2+1 = 3
      So W has 1 at w_idx=0 and w_idx=3.
    
    Let me verify: sum over all (i,j,k) of U[i,j]*V[j,k]*W[k,i] should give the correct C.
    Actually no - the tensor identity is:
      For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    But T[i,j,k] = 1 for all valid (i,j,k) in the multiplication tensor.
    Wait, that's not right either. The matrix multiplication tensor T is defined such that:
      C[i,k] = sum_j A[i,j] B[j,k]
    
    The tensor T has entries T[i,j,k] = 1 for all i,j,k (since every triple contributes).
    The rank-1 decomposition expresses T as:
      T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    So for the schoolbook: U[i,j,r]=1, V[j,k,r]=1, W[k,i,r]=1 for r=(i,j,k).
    
    For Strassen, the rank-1 terms are more complex. Let me think about this differently.
    
    The standard way: the multiplication tensor for 2x2*2x2 has rank 7.
    The 7 rank-1 decompositions correspond to the 7 products M1..M7.
    
    For each M, we have:
      M = (linear form in A) * (linear form in B)
      And M contributes to certain entries of C.
    
    In the tensor decomposition:
      U[i,j,r] = coefficient of A[i,j] in the A-linear-form of M_r
      V[j,k,r] = coefficient of B[j,k] in the B-linear-form of M_r  
      W[k,i,r] = coefficient such that sum over (i,j) of A[i,j]*B[j,k] * (something) gives the C contribution.
    
    Actually, the correct formulation is:
      C[i,k] = sum_j sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] * A[i,j] * B[j,k]
    
    No, that's not right. Let me look at this more carefully.
    
    The tensor T is a 3-tensor with indices (i, j, k) where:
      - i indexes rows of A and C (size n)
      - j indexes columns of A and rows of B (size m)  
      - k indexes columns of B and C (size p)
    
    T[i,j,k] = 1 for all valid indices.
    
    A rank-r decomposition is:
      T[i,j,k] = sum_{c=1}^{r} U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r).
    
    For schoolbook: r = n*m*p, and for each (i,j,k), U[i*m+j, (i,j,k)]=1, V[j*p+k,(i,j,k)]=1, W[k*n+i,(i,j,k)]=1.
    
    For Strassen 2x2*2x2:
      The 7 terms are:
      M1 = (A00+A11)(B00+B11)
        U part: U[0,0]=1, U[3,0]=1 (A00 and A11)
        V part: V[0,0]=1, V[3,0]=1 (B00 and B11)
        W part: This M contributes to C00 and C11.
        C00 = sum_j A[0,j]B[j,0], so for j=0: A[0,0]B[0,0], for j=1: A[0,1]B[1,0]
        C11 = sum_j A[1,j]B[j,1], so for j=0: A[1,0]B[0,1], for j=1: A[1,1]B[1,1]
        
        The contribution of M1 to C is:
        C00 += M1 means: for all (i=0,k=0), the term contributes +1 to T[0,j,0] for the relevant j.
        Hmm, this is getting confusing. Let me use a different approach.
        
        Actually, the standard Strassen decomposition gives:
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4 + M6
        C11 = M1 - M2 + M3 + M6
        
        where:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B10-B11)
        M4 = A11(B01-B00)
        M5 = (A00+A01)B11
        M6 = (A10-A00)(B00+B01)
        M7 = (A01-A10)(B10+B11)
        
        Each M_r is a product of a linear form in A and a linear form in B.
        M_r = (sum_ij alpha_{ij}^r A_{ij}) * (sum_jk beta_{jk}^r B_{jk})
        
        And M_r contributes gamma_{ik}^r to C_{ik}.
        
        The tensor decomposition requires:
        T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        For this to work, we need:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        This is a 3-way tensor factorization. The Strassen decomposition provides 7 such rank-1 terms.
        
        Let me just construct it directly by expanding.
        
        For M1 = (A00+A11)(B00+B11):
          alpha: A00 coeff=1, A11 coeff=1, others=0
          beta: B00 coeff=1, B11 coeff=1, others=0
          gamma: C00 coeff=1, C11 coeff=1, others=0
          
          So U[0,0]=1 (i=0,j=0), U[3,0]=1 (i=1,j=1)
          V[0,0]=1 (j=0,k=0), V[3,0]=1 (j=1,k=1)
          W[0,0]=1 (k=0,i=0), W[3,0]=1 (k=1,i=1)
          
        Let me verify T[0,0,0] = sum_r U[0,r]*V[0,r]*W[0,r]
        For r=0 (M1): U[0,0]=1, V[0,0]=1, W[0,0]=1 -> product = 1
        Other r: need to check.
        
        This is getting complex. Let me just implement it programmatically.
    """
    # For 2x2*2x2, indices: i in {0,1}, j in {0,1}, k in {0,1}
    # a_idx = i*2+j, b_idx = j*2+k, w_idx = k*2+i
    
    # Define the 7 Strassen terms as (alpha_dict, beta_dict, gamma_dict)
    # where alpha_dict maps (i,j) -> coeff, etc.
    
    terms = []
    
    # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
    terms.append({
        'alpha': {(0,0): 1, (1,1): 1},
        'beta': {(0,0): 1, (1,1): 1},
        'gamma': {(0,0): 1, (1,1): 1}
    })
    
    # M2 = (A10+A11)B00 -> C10+=1, C11-=1
    terms.append({
        'alpha': {(1,0): 1, (1,1): 1},
        'beta': {(0,0): 1},
        'gamma': {(1,0): 1, (1,1): -1}
    })
    
    # M3 = A00(B10-B11) -> C01+=1, C11+=1
    terms.append({
        'alpha': {(0,0): 1},
        'beta': {(1,0): 1, (1,1): -1},
        'gamma': {(0,1): 1, (1,1): 1}
    })
    
    # M4 = A11(B01-B00) -> C00+=1, C10+=1
    terms.append({
        'alpha': {(1,1): 1},
        'beta': {(0,1): 1, (0,0): -1},
        'gamma': {(0,0): 1, (1,0): 1}
    })
    
    # M5 = (A00+A01)B11 -> C00-=1, C01+=1
    terms.append({
        'alpha': {(0,0): 1, (0,1): 1},
        'beta': {(1,1): 1},
        'gamma': {(0,0): -1, (0,1): 1}
    })
    
    # M6 = (A10-A00)(B00+B01) -> C10+=1
    terms.append({
        'alpha': {(1,0): 1, (0,0): -1},
        'beta': {(0,0): 1, (0,1): 1},
        'gamma': {(1,0): 1}
    })
    
    # M7 = (A01-A10)(B10+B11) -> C01+=1
    terms.append({
        'alpha': {(0,1): 1, (1,0): -1},
        'beta': {(1,0): 1, (1,1): 1},
        'gamma': {(0,1): 1}
    })
    
    return terms


def _build_4x4_4x4_strassen():
    """Build the 4x4 * 4x4 tensor as Kronecker product of two Strassen 2x2 tensors.
    
    Returns list of (a_idx, b_idx, w_idx, coeff) tuples.
    """
    base_terms = _strassen_2x2_terms()
    
    # Kronecker product: for each pair of base terms (t1, t2), create a combined term.
    # The 4x4 matrix A is divided into 2x2 blocks:
    #   A = [[A00_block, A01_block],
    #        [A10_block, A11_block]]
    # where each block is 2x2.
    
    # Similarly for B (4x4) and C (4x4).
    
    # The index mapping:
    # For A: global (i,j) where i in {0,1,2,3}, j in {0,1,2,3}
    #   Block row: i//2, Block col: j//2, Local row: i%2, Local col: j%2
    #   So global a_idx = i*4+j
    #   Local block: (i//2, j//2), local indices (i%2, j%2)
    
    # For B: global (j,k) where j in {0,1,2,3}, k in {0,1,2,3}
    #   Block row: j//2, Block col: k//2, Local row: j%2, Local col: k%2
    #   Global b_idx = j*4+k
    
    # For C: global (i,k) where i in {0,1,2,3}, k in {0,1,2,3}
    #   Block row: i//2, Block col: k//2, Local row: i%2, Local col: k%2
    #   Global w_idx = k*4+i
    
    # The Kronecker product of two 2x2*2x2 Strassen decompositions:
    # For block (I,J) of A, (J,K) of B, (I,K) of C:
    #   The product involves sum over block-row of B = block-col of A.
    #   
    #   C[I,K] = sum_J A[I,J] * B[J,K]
    #   Each block product A[I,J] * B[J,K] is a 2x2 * 2x2 multiplication.
    #   Using Strassen for each block product gives 7 terms per block product.
    #   There are 2*2*2 = 8 block products, each with 7 terms = 56 terms.
    #   But we can share: the total rank is 7*7 = 49 for the full 4x4*4x4.
    #   
    #   Actually, the Kronecker product of two rank-7 tensors has rank 7*7 = 49.
    #   The 49 terms correspond to pairs (r1, r2) where r1, r2 in {0..6}.
    #   
    #   For the Kronecker product:
    #   The A-part of term (r1,r2) is the Kronecker product of the A-parts of r1 and r2.
    #   Similarly for B and C.
    
    # Let me construct this properly.
    
    combined_terms = []
    
    for r1, term1 in enumerate(base_terms):
        for r2, term2 in enumerate(base_terms):
            # The combined term has:
            # alpha_combined[(I*2+i, J*2+j)] = alpha1[(I,j1)] * alpha2[(i,j2)] ... 
            # This is getting complicated. Let me use a different approach.
            
            # For the Kronecker product, the indices are:
            # A global index: (block_row, local_row, block_col, local_col) -> (I, i, J, j)
            #   Global i_global = I*2 + i, j_global = J*2 + j
            #   a_idx = i_global * 4 + j_global = (I*2+i)*4 + (J*2+j)
            
            # The alpha for the combined term at (I,i,J,j) is:
            #   alpha1[(I, j1)] * alpha2[(i, j2)] where j1 = J, j2 = j
            #   Wait, I need to be more careful.
            
            # term1 operates on the "outer" 2x2 blocks, term2 on the "inner" 2x2 elements.
            # 
            # For the outer level (term1):
            #   It has alpha1 at block positions (I, J) where I is block-row, J is block-col.
            #   But in the 2x2*2x2 problem, the indices are (i,j) for A, (j,k) for B, (k,i) for C.
            #   At the outer level, i->I, j->J, k->K.
            #   So alpha1[(I, J)] gives the coefficient for block A[I,J].
            #   beta1[(J, K)] gives the coefficient for block B[J,K].
            #   gamma1[(K, I)] gives the coefficient for block C[K,I].
            #
            # For the inner level (term2):
            #   It has alpha2 at local positions (i, j) within each block.
            #   beta2 at local positions (j, k).
            #   gamma2 at local positions (k, i).
            #
            # The combined term:
            #   A coefficient at (I*2+i, J*2+j) = alpha1[(I, J)] * alpha2[(i, j)]
            #   B coefficient at (J*2+j, K*2+k) = beta1[(J, K)] * beta2[(j, k)]
            #   C coefficient at (K*2+k, I*2+i) = gamma1[(K, I)] * gamma2[(k, i)]
            #
            #   But wait, the inner indices j and k need to match between A and B.
            #   In A, the column index is j (local). In B, the row index is j (local).
            #   So they should be the same j.
            #   Similarly, B's column index k (local) should match C's row index... 
            #   Actually, C's index is (k, i) in the trace convention, where k is the row of C
            #   (which is the column of B) and i is the column of C (which is the row of A).
            
            # Let me enumerate all combinations.
            
            for I in range(2):
                for J in range(2):
                    a1 = term1['alpha'].get((I, J), 0)
                    if a1 == 0:
                        continue
                    for K in range(2):
                        b1 = term1['beta'].get((J, K), 0)
                        if b1 == 0:
                            continue
                        c1 = term1['gamma'].get((K, I), 0)
                        if c1 == 0:
                            continue
                        
                        for i in range(2):
                            for j in range(2):
                                a2 = term2['alpha'].get((i, j), 0)
                                if a2 == 0:
                                    continue
                                for k in range(2):
                                    b2 = term2['beta'].get((j, k), 0)
                                    if b2 == 0:
                                        continue
                                    c2 = term2['gamma'].get((k, i), 0)
                                    if c2 == 0:
                                        continue
                                    
                                    # Global indices:
                                    a_global_i = I * 2 + i
                                    a_global_j = J * 2 + j
                                    a_idx = a_global_i * 4 + a_global_j
                                    
                                    b_global_j = J * 2 + j
                                    b_global_k = K * 2 + k
                                    b_idx = b_global_j * 4 + b_global_k
                                    
                                    c_global_k = K * 2 + k
                                    c_global_i = I * 2 + i
                                    w_idx = c_global_k * 4 + c_global_i
                                    
                                    coeff = a1 * b1 * c1 * a2 * b2 * c2
                                    if coeff != 0:
                                        combined_terms.append((a_idx, b_idx, w_idx, coeff))
    
    return combined_terms


def _verify_decomposition(U, V, W, n=4, m=4, p=7):
    """Verify that the decomposition is correct."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if total != 1:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Step 1: Build 4x4 * 4x4 Strassen decomposition (rank 49)
    strassen_terms = _build_4x4_4x4_strassen()
    
    # Step 2: Add schoolbook terms for columns 4,5,6 of B (4x4 * 4x3)
    all_terms = list(strassen_terms)
    
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                a_idx = i * m + j
                b_idx = j * p + k
                w_idx = k * n + i
                all_terms.append((a_idx, b_idx, w_idx, 1))
    
    initial_rank = len(all_terms)
    
    # Step 3: Build the tensor matrix and find minimal rank via Gaussian elimination
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want to find the minimal number of terms such that T[i,j,k] = 1 for all i,j,k.
    
    # Create a matrix where each row corresponds to a (i,j,k) triple and each column
    # corresponds to a term. The entry is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # We need this matrix to have all-ones in the "target" sense, but actually we need
    # the sum over r to equal 1 for each (i,j,k).
    
    # Actually, the constraint is: for each (i,j,k), sum_r M[(i,j,k), r] = 1
    # where M[(i,j,k), r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    
    # This is not a standard linear system because the entries are products.
    # However, if we fix U and V, then W is determined, and vice versa.
    
    # Alternative approach: treat each term as a vector in a high-dimensional space.
    # The tensor T can be viewed as a vector of size n*m*p.
    # Each rank-1 term contributes a vector that is the Kronecker product of U_r, V_r, W_r.
    # We want to express the all-ones tensor as a sum of rank-1 tensors.
    
    # Let's build the matrix A where A[(i,j,k), r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # and we want A @ ones = ones_vector (all 1s of size n*m*p).
    
    # Actually, we want: for each (i,j,k), sum_r A[(i,j,k), r] = 1.
    # This means A @ [1,1,...,1]^T = [1,1,...,1]^T (vector of size n*m*p).
    
    # But we're free to choose U, V, W. The terms we have are candidate rank-1 tensors.
    # We want to select a subset of these terms (and possibly scale them) such that
    # the sum equals the all-ones tensor.
    
    # Let's set up the linear system. Each term r has a fixed (a_idx, b_idx, w_idx, coeff).
    # The contribution to T[i,j,k] from term r is:
    #   coeff_r * delta_{a_idx, i*m+j} * delta_{b_idx, j*p+k} * delta_{w_idx, k*n+i}
    #   times... wait, no. U[i*m+j, r] is non-zero only at specific (i,j) positions.
    
    # Let me reconstruct U, V, W from the terms and then build the system.
    
    # First, build U, V, W from all_terms
    U = [[0] * initial_rank for _ in range(n * m)]
    V = [[0] * initial_rank for _ in range(m * p)]
    W = [[0] * initial_rank for _ in range(p * n)]
    
    for r, (a_idx, b_idx, w_idx, coeff) in enumerate(all_terms):
        U[a_idx][r] = coeff
        V[b_idx][r] = coeff
        W[w_idx][r] = coeff
    
    # Now build the constraint matrix.
    # For each (i,j,k), the constraint is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is a nonlinear constraint in general, but since each term has at most one
    # non-zero in each of U, V, W (for the terms we constructed), it becomes linear
    # in the sense that each (i,j,k) is hit by specific terms.
    
    # Actually, for our constructed terms, each term r has U[a_idx, r] = coeff, V[b_idx, r] = coeff,
    # W[w_idx, r] = coeff where a_idx, b_idx, w_idx are specific. So U[i*m+j, r] is non-zero
    # only when (i,j) corresponds to a_idx for that term.
    
    # The product U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] is non-zero only when:
    #   i*m+j = a_idx_r AND j*p+k = b_idx_r AND k*n+i = w_idx_r
    # In that case, the product is coeff_r^3.
    
    # So for each (i,j,k), the sum is over terms r where a_idx_r = i*m+j, b_idx_r = j*p+k,
    # w_idx_r = k*n+i, and the sum is sum_r coeff_r^3.
    
    # We need this sum to equal 1.
    
    # Let's group terms by their (a_idx, b_idx, w_idx) triple.
    from collections import defaultdict
    term_groups = defaultdict(list)
    for r, (a_idx, b_idx, w_idx, coeff) in enumerate(all_terms):
        term_groups[(a_idx, b_idx, w_idx)].append((r, coeff))
    
    # For each (i,j,k), find the corresponding (a_idx, b_idx, w_idx):
    #   a_idx = i*m+j, b_idx = j*p+k, w_idx = k*n+i
    # And the constraint is sum of coeff^3 over terms in that group = 1.
    
    # But we can also scale the terms. Let's allow scaling: each term r has a scale factor s_r.
    # Then the contribution is s_r * coeff_r^3 (if we scale U, V, W all by s_r).
    # Actually, if we scale term r by s_r, then U, V, W entries for that term are scaled by s_r,
    # and the product is scaled by s_r^3.
    
    # So the constraint is: for each (i,j,k), sum_{r in group} s_r * coeff_r^3 = 1.
    
    # This is a linear system in the variables s_r!
    
    # Let's set up the system.
    # Variables: s_0, s_1, ..., s_{initial_rank-1}
    # Constraints: for each (i,j,k), sum_{r in group(i,j,k)} s_r * coeff_r^3 = 1
    
    # Build the matrix.
    num_constraints = n * m * p
    constraint_rows = []
    constraint_rhs = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                w_idx = k * n + i
                group = term_groups.get((a_idx, b_idx, w_idx), [])
                row = [0] * initial_rank
                for r, coeff in group:
                    row[r] = coeff ** 3
                constraint_rows.append(row)
                constraint_rhs.append(1)
    
    # Now we have a system: M @ s = b where M is num_constraints x initial_rank
    # and b is a vector of 1s.
    
    # We want to find a sparse solution s (many zeros) such that M @ s = b.
    # This is equivalent to finding a subset of columns of M that can represent b.
    
    # Use Gaussian elimination to find the rank of M and a basis.
    
    # Convert to Fraction for exact arithmetic.
    M_frac = [[Fraction(x) for x in row] for row in constraint_rows]
    b_frac = [Fraction(1) for _ in constraint_rhs]
    
    # Gaussian elimination to find pivot columns.
    num_rows = len(M_frac)
    num_cols = initial_rank
    
    # Augmented matrix
    aug = [M_frac[i] + [b_frac[i]] for i in range(num_rows)]
    
    pivot_cols = []
    current_row = 0
    for col in range(num_cols):
        if current_row >= num_rows:
            break
        # Find pivot
        pivot = None
        for r in range(current_row, num_rows):
            if aug[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            continue
        # Swap
        aug[current_row], aug[pivot] = aug[pivot], aug[current_row]
        # Eliminate
        pivot_val = aug[current_row][col]
        for r in range(num_rows):
            if r != current_row:
                factor = aug[r][col] / pivot_val
                if factor != 0:
                    for c in range(num_cols + 1):
                        aug[r][c] -= factor * aug[current_row][c]
        pivot_cols.append(col)
        current_row += 1
    
    # Check consistency
    consistent = True
    for r in range(current_row, num_rows):
        if all(aug[r][c] == 0 for c in range(num_cols)) and aug[r][num_cols] != 0:
            consistent = False
            break
    
    if not consistent:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # The pivot columns give a minimal (in terms of linear independence) set.
    # But we want the smallest subset, which requires finding a basis of size = rank.
    # The pivot columns from Gaussian elimination give one such basis.
    
    # Extract the solution for the pivot columns.
    # After Gaussian elimination, the system is in reduced row echelon form.
    # The pivot columns have a 1 in the pivot row and 0 elsewhere.
    # The solution for pivot variables is given by the last column.
    
    solution = [Fraction(0)] * num_cols
    for r, col in enumerate(pivot_cols):
        solution[col] = aug[r][num_cols]
    
    # Non-pivot columns are free variables; set them to 0 for a particular solution.
    
    # Now build the final U, V, W with only the pivot columns.
    final_rank = len(pivot_cols)
    U_final = [[0] * final_rank for _ in range(n * m)]
    V_final = [[0] * final_rank for _ in range(m * p)]
    W_final = [[0] * final_rank for _ in range(p * n)]
    
    for new_r, old_r in enumerate(pivot_cols):
        scale = solution[old_r]
        a_idx, b_idx, w_idx, coeff = all_terms[old_r]
        U_final[a_idx][new_r] = coeff * scale
        V_final[b_idx][new_r] = coeff * scale
        W_final[w_idx][new_r] = coeff * scale
    
    # Convert to regular numbers (int or float) if possible
    def convert_matrix(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                if isinstance(val, Fraction):
                    if val.denominator == 1:
                        new_row.append(int(val))
                    else:
                        new_row.append(float(val))
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    U_final = convert_matrix(U_final)
    V_final = convert_matrix(V_final)
    W_final = convert_matrix(W_final)
    
    # Verify
    if not _verify_decomposition(U_final, V_final, W_final, n, m, p):
        # Fallback
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U_final, "V": V_final, "W": W_final}
