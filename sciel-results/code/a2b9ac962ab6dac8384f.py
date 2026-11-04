"""Strassen-based decomposition for 4x5 x 5x6 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Use Strassen's algorithm on 2x2 blocks.
    # Partition: A(4x5) into blocks of size 2x2, 2x3 (top/bottom x left/right)
    # B(5x6) into blocks of size 2x3, 3x3 (top/bottom x left/right) 
    # C(4x6) into blocks of size 2x3 (top/bottom x left/right)
    
    # Strategy: Use the tensor structure with Strassen's 7-multiplication
    # on 2x2x2 sub-tensors.
    
    # Let's use a block approach:
    # A = [[A11, A12], [A21, A22]] where A11,A21 are 2x2, A12,A22 are 2x3
    # B = [[B11, B12], [B21, B22]] where B11,B21 are 2x3, B12,B22 are 3x3
    # C = [[C11, C12], [C21, C22]] where each Ci* is 2x3
    
    # C11 = A11*B11 + A12*B21  (2x2 x 2x3 + 2x3 x 3x3)
    # C12 = A11*B12 + A12*B22  (2x2 x 3x3 + 2x3 x 3x3)
    # C21 = A21*B11 + A22*B21  (2x2 x 2x3 + 2x3 x 3x3)
    # C22 = A21*B12 + A22*B22  (2x2 x 3x3 + 2x3 x 3x3)
    
    # This is complex. Let me use a simpler but effective approach:
    # Directly construct using known low-rank decompositions.
    
    # The best known rank for 4x5x6 matrix multiplication:
    # We can use the fact that MM(4,5,6) can be embedded in larger known decompositions.
    
    # Approach: Use schoolbook but with Strassen optimization on 2x2 sub-blocks.
    # Partition the 4x5x6 tensor into 2x2x2 blocks where possible.
    
    # Actually, let's use a clean Strassen-based construction:
    # Treat the problem as 2x2 blocks of (2x3) matrices.
    
    # A is 4x5. Split into 2x2 blocks: 
    #   A = [[A_tl, A_tr], [A_bl, A_br]]
    #   A_tl: 2x2, A_tr: 2x3, A_bl: 2x2, A_br: 2x3
    # B is 5x6. Split into 2x2 blocks:
    #   B = [[B_tl, B_tr], [B_bl, B_br]]
    #   B_tl: 2x3, B_tr: 2x3, B_bl: 3x3, B_br: 3x3
    # C is 4x6. Split into 2x2 blocks:
    #   C = [[C_tl, C_tr], [C_bl, C_br]]
    #   Each C block: 2x3
    
    # C_tl = A_tl*B_tl + A_tr*B_bl  (2x2*2x3 + 2x3*3x3)
    # C_tr = A_tl*B_tr + A_tr*B_br  (2x2*3x3 + 2x3*3x3)
    # C_bl = A_bl*B_tl + A_br*B_bl  (2x2*2x3 + 2x3*3x3)
    # C_br = A_bl*B_tr + A_br*B_br  (2x2*3x3 + 2x3*3x3)
    
    # Each product of type 2x2 * 2x3 has rank 6 (schoolbook)
    # Each product of type 2x3 * 3x3 has rank 9 (schoolbook)
    # Total schoolbook for this partition: 4*6 + 4*9 = 24 + 36 = 60
    
    # But we can do better by sharing Strassen multiplications.
    
    # Let me just use the direct schoolbook which gives 120, and try to improve.
    # Actually, let's implement a proper Strassen-based approach.
    
    # For now, use the known improvement: the 2x2x2 Strassen trick applied
    # to appropriate sub-blocks.
    
    # Simplest effective approach: use the fact that we can group terms.
    # Let's just do schoolbook first as fallback, then try to optimize.
    
    rank_schoolbook = n * m * p  # 120
    
    # Build schoolbook decomposition as baseline
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Now try to improve using Strassen on 2x2x2 sub-blocks
    # The idea: wherever we have a 2x2x2 sub-tensor (i.e., 2 values of i, 2 of j, 2 of k),
    # we can replace 8 terms with 7 Strassen terms.
    
    # For 4x5x6, we can partition i into {0,1} and {2,3} (two groups of 2)
    # j into {0,1} and {2,3,4} (groups of 2 and 3)
    # k into {0,1} and {2,3,4,5} (groups of 2 and 4)
    
    # Where we have 2x2x2 sub-blocks, apply Strassen.
    # i-group 0 (size 2), j-group 0 (size 2), k-group 0 (size 2): 2x2x2 -> Strassen (7 vs 8)
    # i-group 0 (size 2), j-group 1 (size 3), k-group 0 (size 2): 2x3x2 -> no direct Strassen
    # etc.
    
    # Let me apply Strassen to the 2x2x2 sub-blocks:
    # Sub-block (i in {0,1}, j in {0,1}, k in {0,1}): 8 terms -> 7 Strassen terms
    # Sub-block (i in {2,3}, j in {0,1}, k in {0,1}): 8 terms -> 7 Strassen terms
    # Sub-block (i in {0,1}, j in {0,1}, k in {2,3}): 8 terms -> 7 Strassen terms
    # Sub-block (i in {2,3}, j in {0,1}, k in {2,3}): 8 terms -> 7 Strassen terms
    # Sub-block (i in {0,1}, j in {0,1}, k in {4,5}): 8 terms -> 7 Strassen terms
    # Sub-block (i in {2,3}, j in {0,1}, k in {4,5}): 8 terms -> 7 Strassen terms
    
    # That's 6 sub-blocks of 2x2x2, each saving 1 term: total saving 6
    # Remaining: 120 - 48 = 72 terms from non-2x2x2 blocks
    # Total: 72 + 6*7 = 72 + 42 = 114
    
    # But we can also handle 2x3x2, 2x2x4, etc. blocks more cleverly.
    
    # Let me construct this explicitly.
    
    # Reset and build with Strassen optimization
    U = []
    V = []
    W = []
    
    def add_term(u_val, v_val, w_val):
        """Add a term with given values to U, V, W."""
        U.append(u_val)
        V.append(v_val)
        W.append(w_val)
    
    # Initialize U, V, W as lists of dicts (sparse representation)
    # U[i*m+j] is a list of (rank_idx, value)
    # Actually, let's use dense arrays but build them incrementally.
    
    # Let me use a different approach: build the decomposition as a list of (U_col, V_col, W_col)
    terms = []
    
    def add_strassen_block(i0, j0, k0):
        """Add Strassen decomposition for 2x2x2 block starting at (i0, j0, k0)."""
        # Strassen's 7 multiplications for 2x2 matrix mult:
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # In our tensor: U[i*m+j], V[j*p+k], W[k*n+i]
        # For the 2x2x2 block at (i0,j0,k0):
        # i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
        
        # A is 2x2: A[i0+j, j0+k] for i,j in {0,1} relative
        # B is 2x2: B[j0+k, k0+l] for j,k in {0,1} relative  
        # Wait, I need to be more careful.
        
        # U indexes: i*m + j where i is row of A, j is col of A / row of B
        # V indexes: j*p + k where j is col of A / row of B, k is col of B
        # W indexes: k*n + i where k is col of B, i is row of C / row of A
        
        # For the 2x2x2 block:
        # i_rel in {0,1} -> i = i0 + i_rel
        # j_rel in {0,1} -> j = j0 + j_rel  
        # k_rel in {0,1} -> k = k0 + k_rel
        
        # A sub-block: A[i0+i_rel, j0+j_rel] for i_rel, j_rel in {0,1}
        # B sub-block: B[j0+j_rel, k0+k_rel] for j_rel, k_rel in {0,1}
        # C sub-block: C[i0+i_rel, k0+k_rel] for i_rel, k_rel in {0,1}
        
        # Strassen on this 2x2x2 tensor:
        # The tensor T[i,j,k] = A[i,j] * B[j,k] = C[i,k]
        
        # Strassen multiplications (each is a product of linear forms in A and B):
        # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        # M2 = (A[1,0]+A[1,1]) * B[0,0]
        # M3 = A[0,0] * (B[0,1]-B[1,1])
        # M4 = A[1,1] * (B[1,0]-B[0,0])
        # M5 = (A[0,0]+A[0,1]) * B[1,1]
        # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
        # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
        
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        
        # Each M_r is a product of a linear form in A and a linear form in B.
        # So each contributes one term to the tensor decomposition.
        
        # M1: A form = A[0,0] + A[1,1], B form = B[0,0] + B[1,1]
        #   U contributions: U[i0*m + j0] += 1, U[(i0+1)*m + (j0+1)] += 1
        #   V contributions: V[j0*p + k0] += 1, V[(j0+1)*p + (k0+1)] += 1
        #   W contributions: W[k0*n + i0] += 1, W[(k0+1)*n + (i0+1)] += 1
        #   Wait, that's not right. Each M_r is ONE term in the decomposition.
        #   The U column for this term has the A-form coefficients.
        #   The V column for this term has the B-form coefficients.
        #   The W column for this term has the C-form coefficients.
        
        # For M1: A-form = A[0,0] + A[1,1], B-form = B[0,0] + B[1,1]
        #   U_col: has 1 at position i0*m+j0 and (i0+1)*m+(j0+1)
        #   V_col: has 1 at position j0*p+k0 and (j0+1)*p+(k0+1)
        #   W_col: This is where it gets tricky. The W column encodes how M1 contributes to C.
        #   M1 contributes +1 to C[0,0] and +1 to C[1,1] (from C[0,0]=M1+... and C[1,1]=M1-...)
        #   So W_col: has 1 at position k0*n+i0 and (k0+1)*n+(i0+1)
        
        # Actually, I realize the W matrix in the tensor decomposition is such that:
        # C[i,k] = sum_r U[i*m+j]*V[j*p+k]*W[k*n+i] over all j and r
        # Wait no. The tensor decomposition is:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # So for each rank-r term, we have U_col (length n*m), V_col (length m*p), W_col (length p*n)
        # And T[i,j,k] = sum_r U_col[i*m+j] * V_col[j*p+k] * W_col[k*n+i]
        
        # For a single Strassen multiplication M_r = (sum_a alpha_a * A[a]) * (sum_b beta_b * B[b]):
        # This is a rank-1 term in the tensor where:
        # U_col[i*m+j] = alpha for the (i,j) positions in the A-form
        # V_col[j*p+k] = beta for the (j,k) positions in the B-form
        # W_col[k*n+i] = 1 for all (k,i) in the 2x2 block (since M_r contributes to all C entries with coefficients determined by how M_r appears in C formulas)
        
        # Hmm, actually no. The W column should encode the contribution to C.
        # Let me reconsider.
        
        # T[i,j,k] = A[i,j] * B[j,k]
        # We want T = sum_r U_r ⊗ V_r ⊗ W_r where:
        # T[i,j,k] = sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
        
        # For a rank-1 term corresponding to "A[i,j] * B[j,k]" with specific i,j,k:
        # U_r[i*m+j] = 1, V_r[j*p+k] = 1, W_r[k*n+i] = 1, rest 0.
        
        # For Strassen, each M_r is a product of a linear combination of A entries and a linear combination of B entries.
        # M_r = (sum_{(i,j) in support_A} a_{ij} * A[i,j]) * (sum_{(j,k) in support_B} b_{jk} * B[j,k])
        
        # This expands to: sum_{(i,j)} sum_{(j',k)} a_{ij} * b_{j'k} * A[i,j] * B[j',k]
        # But this only equals the desired contribution if the supports are compatible (j = j' for the product to make sense in the matrix multiplication).
        
        # Actually, Strassen works because the linear forms are chosen so that when you multiply them out, you get the right combinations of A[i,j]*B[j,k] terms.
        
        # For the 2x2 case:
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        # = A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + A[1,1]*B[1,1]
        
        # In the tensor, this corresponds to:
        # U_col[0*m+0] = 1, U_col[1*m+1] = 1 (A-form)
        # V_col[0*p+0] = 1, V_col[1*p+1] = 1 (B-form)  
        # W_col: The contribution to C is:
        #   C[0,0] gets A[0,0]*B[0,0] -> coefficient 1
        #   C[0,1] gets A[0,0]*B[1,1] -> wait, B[1,1] means j=1,k=1, so this contributes to C[0,1]
        #   C[1,0] gets A[1,1]*B[0,0] -> j=0,k=0, contributes to C[1,0]
        #   C[1,1] gets A[1,1]*B[1,1] -> j=1,k=1, contributes to C[1,1]
        
        # So W_col[0*n+0] = 1 (C[0,0]), W_col[1*n+0] = 1 (C[1,0]), W_col[0*n+1] = 1 (C[0,1]), W_col[1*n+1] = 1 (C[1,1])
        # Wait, W index is k*n+i. So:
        # C[0,0]: i=0,k=0 -> W[0*n+0] = W[0]
        # C[0,1]: i=0,k=1 -> W[1*n+0] = W[n]
        # C[1,0]: i=1,k=0 -> W[0*n+1] = W[1]
        # C[1,1]: i=1,k=1 -> W[1*n+1] = W[n+1]
        
        # So W_col has 1 at positions 0, 1, n, n+1 (for the 2x2 block at origin)
        
        # This makes sense! Each Strassen term M_r has:
        # U_col: the A-form coefficients at the appropriate (i,j) positions
        # V_col: the B-form coefficients at the appropriate (j,k) positions
        # W_col: 1 at all (k,i) positions in the 2x2 block (since every A[i,j]*B[j,k] product in the expansion contributes to C[i,k])
        
        # Wait, that's not quite right either. Let me verify with M1:
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        # Expansion: A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + A[1,1]*B[1,1]
        # These contribute to: C[0,0] (from A[0,0]*B[0,0]), C[0,1] (from A[0,0]*B[1,1]), C[1,0] (from A[1,1]*B[0,0]), C[1,1] (from A[1,1]*B[1,1])
        # So yes, W_col has 1 at all four (k,i) positions.
        
        # Now for the actual Strassen C formulas:
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        
        # So each M_r appears in multiple C entries with different coefficients.
        # This means the W_col for each M_r should have the appropriate coefficients!
        
        # Let me redo: For each Strassen term M_r, the W_col encodes the coefficients of M_r in each C[i,k].
        
        # M1: C[0,0]+=1, C[1,1]+=1 -> W_col: W[0*n+0]+=1, W[1*n+1]+=1
        # M2: C[1,0]+=1, C[1,1]+=-1 -> W_col: W[0*n+1]+=1, W[1*n+1]+=-1
        # M3: C[0,1]+=1, C[1,1]+=1 -> W_col: W[1*n+0]+=1, W[1*n+1]+=1
        # M4: C[0,0]+=1, C[1,0]+=1 -> W_col: W[0*n+0]+=1, W[0*n+1]+=1
        # M5: C[0,0]+=-1, C[0,1]+=1 -> W_col: W[0*n+0]+=-1, W[1*n+0]+=1
        # M6: C[1,1]+=1 -> W_col: W[1*n+1]+=1
        # M7: C[0,0]+=1 -> W_col: W[0*n+0]+=1
        
        # And the U and V columns:
        # M1: U: A[0,0]=1, A[1,1]=1; V: B[0,0]=1, B[1,1]=1
        # M2: U: A[1,0]=1, A[1,1]=1; V: B[0,0]=1
        # M3: U: A[0,0]=1; V: B[0,1]=1, B[1,1]=-1
        # M4: U: A[1,1]=1; V: B[1,0]=1, B[0,0]=-1
        # M5: U: A[0,0]=1, A[0,1]=1; V: B[1,1]=1
        # M6: U: A[1,0]=1, A[0,0]=-1; V: B[0,0]=1, B[0,1]=1
        # M7: U: A[0,1]=1, A[1,1]=-1; V: B[1,0]=1, B[1,1]=1
        
        # Now I can build this!
        
        # For the 2x2x2 block at (i0, j0, k0):
        # A positions: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
        # B positions: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
        # C positions: (i0, k0), (i0, k0+1), (i0+1, k0), (i0+1, k0+1)
        
        # U index for A[i,j]: i*m + j
        # V index for B[j,k]: j*p + k
        # W index for C[i,k]: k*n + i
        
        i0, j0, k0 = i0, j0, k0
        
        # Define the 7 Strassen terms
        # Each term: (U_dict, V_dict, W_dict)
        # U_dict: {(i,j): coeff}
        # V_dict: {(j,k): coeff}
        # W_dict: {(i,k): coeff}
        
        strassen_terms = [
            # M1
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            # M2
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            # M3
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            # M4
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            # M5
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            # M6
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            # M7
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]
        
        for u_dict, v_dict, w_dict in strassen_terms:
            U_col = [0] * (n * m)
            V_col = [0] * (m * p)
            W_col = [0] * (p * n)
            
            for (di, dj), coeff in u_dict.items():
                U_col[(i0+di)*m + (j0+dj)] = coeff
            
            for (dj, dk), coeff in v_dict.items():
                V_col[(j0+dj)*p + (k0+dk)] = coeff
            
            for (di, dk), coeff in w_dict.items():
                W_col[(k0+dk)*n + (i0+di)] = coeff
            
            terms.append((U_col, V_col, W_col))
    
    def add_schoolbook_term(i, j, k):
        """Add a single schoolbook term for A[i,j]*B[j,k] -> C[i,k]."""
        U_col = [0] * (n * m)
        V_col = [0] * (m * p)
        W_col = [0] * (p * n)
        U_col[i*m + j] = 1
        V_col[j*p + k] = 1
        W_col[k*n + i] = 1
        terms.append((U_col, V_col, W_col))
    
    # Now, partition the 4x5x6 space into 2x2x2 blocks where possible,
    # and use schoolbook for the rest.
    
    # i: 0,1,2,3
    # j: 0,1,2,3,4
    # k: 0,1,2,3,4,5
    
    # 2x2x2 blocks:
    # (i0,j0,k0) where i0 in {0,2}, j0 in {0,2}, k0 in {0,2,4}
    # But j only goes to 4, so j0 can be 0 or 2 (j0+1 must be <= 4)
    # k0 can be 0, 2, or 4 (k0+1 must be <= 5)
    
    used = set()
    
    # Apply Strassen to all possible 2x2x2 blocks
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                if i0+1 < n and j0+1 < m and k0+1 < p:
                    add_strassen_block(i0, j0, k0)
                    # Mark these (i,j,k) as used
                    for di in range(2):
                        for dj in range(2):
                            for dk in range(2):
                                used.add((i0+di, j0+dj, k0+dk))
    
    # Add schoolbook terms for remaining (i,j,k) not covered
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in used:
                    add_schoolbook_term(i, j, k)
    
    rank = len(terms)
    
    # Convert to the required format
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (U_col, V_col, W_col) in enumerate(terms):
        for idx, val in enumerate(U_col):
            if val != 0:
                U[idx][r] = val
        for idx, val in enumerate(V_col):
            if val != 0:
                V[idx][r] = val
        for idx, val in enumerate(W_col):
            if val != 0:
                W[idx][r] = val
    
    # Verify correctness
    if time.time() < deadline - 10:
        # Build full matrices and verify
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Reconstruct tensor: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        T_reconstructed = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_reconstructed[i,j,k] += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
        
        # True tensor: T[i,j,k] = 1 for all i,j,k (since A[i,j]*B[j,k] = delta-like)
        # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all valid i,j,k
        T_true = np.ones((n, m, p))
        
        if not np.allclose(T_reconstructed, T_true, atol=1e-10):
            # Fallback to schoolbook
            U = [[0] * rank_schoolbook for _ in range(n * m)]
            V = [[0] * rank_schoolbook for _ in range(m * p)]
            W = [[0] * rank_schoolbook for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
            return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
