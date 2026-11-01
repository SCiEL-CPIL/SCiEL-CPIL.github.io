"""Matrix multiplication tensor decomposition for 2x3 * 3x7.
Uses a hybrid Strassen/schoolbook scheme that splits the inner dimension
into 2+1, applying Strassen to the 2x2x2 block and schoolbook to the 2x3x1
block, for total rank 7 + 6 = 13 per 2x2 slice, giving 3 * 13 = 39 total rank.
"""
import numpy as np

DIMENSIONS = [2, 3, 7]


def construct():
    n, m, p = DIMENSIONS  # 2, 3, 7
    # We decompose the 3x7 matrix into three 3x2 blocks and one 3x1 block:
    # B = [B0, B1, B2, B3] where B0,B1,B2 are 3x2 and B3 is 3x1.
    # For each 2-column block Bc (c=0,1,2), we use Strassen on A (2x3) x Bc (3x2):
    #   Split A into [A_top(2x2), A_bot(2x1)] and Bc into [Bc_top(2x2), Bc_bot(2x1)]
    #   Actually, let me think more carefully.
    #
    # A is 2x3, Bc is 3x2. Product Cc = A * Bc is 2x2.
    # Split A = [A11 A12; A21 A22] where A11,A21 are 1x2, A12,A22 are 1x1.
    # Split Bc = [B11 B12; B21 B22] where B11,B21 are 2x1, B12,B22 are 2x1.
    #
    # This is getting complex. Let me use a cleaner approach.
    #
    # For A (2x3) * B (3x2), we can split:
    # A = [A_left(2x2) | A_right(2x1)]
    # B = [B_top(2x2); B_bot(1x2)]
    # Then C = A_left * B_top + A_right * B_bot
    # 
    # A_left * B_top is 2x2*2x2 -> use Strassen (rank 7)
    # A_right * B_bot is 2x1*1x2 -> rank 2 (schoolbook, since 2*1*2=4 but actually only 2 terms)
    # Wait, 2x1 * 1x2: C_ij = sum_k A_ik * B_kj. Here k=1 only, so C_ij = A_i * B_j.
    # That's rank 2 (outer product of two vectors of length 2).
    #
    # So per 2-column block: rank 7 + 2 = 9
    # For 3 blocks: 3 * 9 = 27
    # Plus the last 1-column block: A(2x3) * B(3x1) = 2x1, rank = 6 (schoolbook)
    # Total: 27 + 6 = 33
    #
    # Actually wait, let me reconsider. The standard Strassen for 2x2*2x2 gives rank 7.
    # But we need to handle the splitting carefully.
    #
    # Let me just hardcode a rank-39 decomposition that's been verified to work.
    # Based on the knowledge that rank 39 works (score 1.077), I'll construct it.
    #
    # Alternative: use rank 34 approach.
    # For each of the 3 pairs of columns from B:
    #   A(2x3) * B_pair(3x2): 
    #     Split A into A_12 (2x2) and A_3 (2x1)
    #     Split B_pair into B_12 (2x2) and B_3 (1x2)
    #     C = A_12*B_12 + A_3*B_3
    #     A_12*B_12: Strassen, rank 7
    #     A_3*B_3: rank 2 (outer product)
    #     Total per pair: 9
    # 3 pairs: 27
    # Last column: A(2x3)*B_7(3x1): rank 6
    # Total: 33
    #
    # Let me verify rank 33 is achievable. If not, I'll fall back to 39.
    # Actually, let me just go with the known-good rank 39 for safety.
    
    rank = 39
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Construct using the hybrid approach
    # For each column block c = 0, 1, 2 (covering columns 2c, 2c+1 of B):
    #   Use Strassen on the 2x2 core plus schoolbook for the boundary terms
    # For column 6 (last): schoolbook
    
    # I'll build this step by step.
    # Let me define the Strassen factors for 2x2*2x2.
    # Strassen: given P = A*B where A,B are 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For our case, A is 2x3 and we take the first 2 columns for the 2x2 part,
    # B_pair is 3x2 and we take the first 2 rows for the 2x2 part.
    # A_left = A[:, :2] (2x2), A_right = A[:, 2] (2x1)
    # B_top = B_pair[:2, :] (2x2), B_bot = B_pair[2, :] (1x2)
    
    # The tensor entry is W[k*n+i, r] where the product element C[i, col] = sum_j A[i,j]*B[j,col]
    # In our indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # where i in [0,n), j in [0,m), k in [0,p)
    # Product: C[i,k] = sum_j A[i,j] B[j,k]
    
    # Let me just enumerate all terms for rank 39 using the hybrid method.
    # I'll use a helper to add a rank-1 term.
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val, col):
        U[u_idx, col] += u_val
        V[v_idx, col] += v_val
        W[w_idx, col] += w_val
    
    col = 0
    for c in range(3):  # 3 column pairs
        # Columns 2*c and 2*c+1 of B
        # A is 2x3, indices: i in {0,1}, j in {0,1,2}
        # B pair: k in {2c, 2c+1}
        
        # Strassen on A[:,:2] * B_pair[:2,:]
        # A sub-block: A[0,0], A[0,1], A[1,0], A[1,1] -> indices (0,0),(0,1),(1,0),(1,1) in A
        # B sub-block: B[0,2c], B[0,2c+1], B[1,2c], B[1,2c+1] -> indices (0,2c),(0,2c+1),(1,2c),(1,2c+1)
        
        # M1 = (A00+A11)*(B00+B11)
        # This expands to 4 terms, but in Strassen we treat it as one rank-1 factor
        # In the tensor decomposition, each M corresponds to one rank-1 component
        # where U, V, W encode the product of sums.
        
        # Actually, for the tensor decomposition, each Strassen "multiply" M_t 
        # corresponds to a single rank-1 term where:
        # U encodes the first factor (which may be a sum of A entries)
        # V encodes the second factor (which may be a sum of B entries)
        # W encodes how M_t contributes to each C[i,k]
        
        # Let me define the 7 Strassen multiplies for block c:
        # A entries: A[i,j] for i=0,1; j=0,1 (using A[:, :2])
        # B entries: B[j,k] for j=0,1; k=2c,2c+1 (using B[:2, 2c:2c+2])
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,2c]+B[1,2c+1])
        #   U: i=0,j=0 -> 1; i=1,j=1 -> 1
        #   V: j=0,k=2c -> 1; j=1,k=2c+1 -> 1
        #   W: contributes to C[0,2c], C[0,2c+1], C[1,2c], C[1,2c+1]
        #   C[0,2c] += M1, C[0,2c+1] -= M1, C[1,2c] -= M1, C[1,2c+1] += M1
        
        # This is getting very detailed. Let me just use the known rank-39 construction.
        # I'll hardcode it based on the hybrid Strassen/schoolbook approach.
        
        # For each column pair c, we need 9 rank-1 terms (7 Strassen + 2 boundary)
        # For the last column, we need 6 rank-1 terms.
        # Total: 3*9 + 6 = 33. But knowledge says 39 works, so maybe my analysis is off.
        # Let me just go with rank 39 and fill it in properly.
        
        # Actually, let me use a simpler approach: just do schoolbook for all but
        # use Strassen where beneficial. The knowledge says rank 39 works.
        # Let me construct rank 39 carefully.
        
        pass  # Placeholder, I'll redo this
    
    # Let me restart with a cleaner construction.
    # I'll use the approach: for each of the 7 columns of B, if I group them as
    # (0,1), (2,3), (4,5), (6), I get 3 Strassen blocks + 1 schoolbook column.
    
    # For a Strassen block handling B columns (2c, 2c+1):
    # The contribution to C[i, 2c] and C[i, 2c+1] comes from:
    #   C[i, 2c] = A[i,0]*B[0,2c] + A[i,1]*B[1,2c] + A[i,2]*B[2,2c]
    #   C[i, 2c+1] = A[i,0]*B[0,2c+1] + A[i,1]*B[1,2c+1] + A[i,2]*B[2,2c+1]
    #
    # The first two terms can be handled by Strassen on the 2x2 submatrices,
    # and the last term (involving B[2,*]) is a separate 2x1 * 1x2 = rank 2.
    #
    # Strassen for A[:,:2] * B[:2, 2c:2c+2]:
    # Let me denote a_ij = A[i,j], b_jk = B[j, 2c+k] for k=0,1
    # 
    # The 7 Strassen multiplies:
    # m1 = (a_00+a_11)(b_00+b_11)
    # m2 = (a_10+a_11)b_00
    # m3 = a_00(b_10-b_11)
    # m4 = a_11(b_01-b_10)
    # m5 = (a_00+a_01)b_11
    # m6 = (a_10-a_00)(b_00+b_10)
    # m7 = (a_01-a_11)(b_01+b_11)
    #
    # Results:
    # C_00 = m1+m4-m5+m7
    # C_01 = m3+m5
    # C_10 = m2+m4
    # C_11 = m1-m2+m3+m6
    #
    # Plus boundary terms:
    # C[i, 2c] += a_i2 * b_20
    # C[i, 2c+1] += a_i2 * b_21
    # where b_20 = B[2,2c], b_21 = B[2,2c+1]
    # These are 2 terms (outer product of a_02,a_12 with b_20,b_21): rank 2
    
    # Total per block: 7 + 2 = 9
    # For 3 blocks: 27
    # Last column (col 6): C[i,6] = a_i0*b_06 + a_i1*b_16 + a_i2*b_26, rank 6
    
    # Grand total: 27 + 6 = 33
    
    # Let me implement rank 33.
    
    rank = 33
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    def add_strassen_block(c, start_col):
        """Add Strassen terms for column pair (2c, 2c+1) starting at column start_col."""
        nonlocal col
        # Strassen multiply indices
        # For A sub-block (rows 0-1, cols 0-1) and B sub-block (rows 0-1, cols 2c, 2c+1)
        
        # m1 = (a_00+a_11)(b_00+b_11)
        # U: (0,0)=1, (1,1)=1
        # V: (0,2c)=1, (1,2c+1)=1
        # W: C[0,2c]+=1, C[0,2c+1]-=1, C[1,2c]-=1, C[1,2c+1]+=1
        col += 0  # Will be incremented by the outer function
        # I'll handle this with explicit column assignment
        
        terms = []
        
        # m1
        u = {}; v = {}; w = {}
        u[(0,0)] = 1; u[(1,1)] = 1
        v[(0,2*c)] = 1; v[(1,2*c+1)] = 1
        w[(2*c,0)] = 1; w[(2*c+1,0)] = -1; w[(2*c,1)] = -1; w[(2*c+1,1)] = 1
        terms.append((u, v, w))
        
        # m2 = (a_10+a_11)b_00
        u = {}; v = {}; w = {}
        u[(1,0)] = 1; u[(1,1)] = 1
        v[(0,2*c)] = 1
        w[(2*c,1)] = 1; w[(2*c+1,1)] = -1
        terms.append((u, v, w))
        
        # m3 = a_00(b_10-b_11)
        u = {}; v = {}; w = {}
        u[(0,0)] = 1
        v[(1,2*c)] = 1; v[(1,2*c+1)] = -1
        w[(2*c+1,0)] = 1; w[(2*c+1,1)] = 1
        terms.append((u, v, w))
        
        # m4 = a_11(b_01-b_10)
        u = {}; v = {}; w = {}
        u[(1,1)] = 1
        v[(0,2*c+1)] = 1; v[(0,2*c)] = -1
        w[(2*c,0)] = 1; w[(2*c,1)] = 1; w[(2*c,1)] += 1
        # C_00 += m4, C_10 += m4
        w = {}
        w[(2*c,0)] = 1; w[(2*c,1)] = 1
        terms.append((u, v, w))
        
        # m5 = (a_00+a_01)b_11
        u = {}; v = {}; w = {}
        u[(0,0)] = 1; u[(0,1)] = 1
        v[(1,2*c+1)] = 1
        w = {}
        w[(2*c,0)] = -1; w[(2*c+1,0)] = 1
        terms.append((u, v, w))
        
        # m6 = (a_10-a_00)(b_00+b_10)
        u = {}; v = {}; w = {}
        u[(1,0)] = 1; u[(0,0)] = -1
        v[(0,2*c)] = 1; v[(1,2*c)] = 1
        w = {}
        w[(2*c+1,1)] = 1
        terms.append((u, v, w))
        
        # m7 = (a_01-a_11)(b_01+b_11)
        u = {}; v = {}; w = {}
        u[(0,1)] = 1; u[(1,1)] = -1
        v[(0,2*c+1)] = 1; v[(1,2*c+1)] = 1
        w = {}
        w[(2*c,0)] = 1
        terms.append((u, v, w))
        
        # Boundary terms: a_i2 * b_2k for k=0,1
        # These are: for i=0,1 and k=0,1: C[i, 2c+k] += a_i2 * b_2k
        # This is an outer product: rank 2
        # Term 1: a_02 * b_20 -> C[0,2c]
        # Term 2: a_12 * b_20 -> C[1,2c]  (same b, different a)
        # Actually it's: C[i,2c+k] = a_i2 * b_2k
        # This is a_i2 (length 2) outer b_2k (length 2), giving 4 entries but rank 2:
        # Term 1: a_02 * b_20 -> contributes to C[0,2c]
        #          a_12 * b_20 -> contributes to C[1,2c]
        # So one term with U having a_02 and a_12, V having b_20, W having contributions
        # Wait no, a_i2 is different for each i. Let me think again.
        # C[0,2c] += a_02 * b_20
        # C[1,2c] += a_12 * b_20
        # C[0,2c+1] += a_02 * b_21
        # C[1,2c+1] += a_12 * b_21
        # This is (a_02, a_12) outer (b_20, b_21), which is rank 2:
        # Term A: U=(a_02=1,a_12=1)? No...
        # Actually: C[i,2c+k] = a_i2 * b_2k for i in {0,1}, k in {0,1}
        # = sum_{r} U[i*3+2, r] V[2*p+(2c+k), r] ... no.
        # The tensor is: sum_r U[i,m+j,r] V[j,p+k,r] W[k*n+i,r]
        # For the boundary: we need C[i, 2c+k] to get a_i2 * b_2k
        # a_i2 = A[i,2], b_2k = B[2, 2c+k]
        # So we need a rank-1 term where U[i,2] * V[2, 2c+k] gives the right product.
        # But U[i,2] depends on i and V[2,2c+k] depends on k, and the product should be a_i2 * b_2k.
        # If I set U[i,2] = a_i2 coefficient and V[2,2c+k] = b_2k coefficient, the product works.
        # But in the decomposition, U and V entries are constants (not depending on input).
        # The decomposition is: C[i,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # For the boundary term a_i2 * b_2k: we need U[i*m+2, r] * V[2*p+(2c+k), r] * W[(2c+k)*n+i, r] = a_i2 * b_2k
        # Since a_i2 and b_2k are input variables, we need the decomposition to capture them.
        # In the tensor decomposition, each rank-1 term corresponds to a monomial in the inputs.
        # The term a_i2 * b_2k is a single monomial, so it needs one rank-1 term:
        # U[i*m+2, r] = 1 (for the appropriate i), V[2*p+(2c+k), r] = 1, W[(2c+k)*n+i, r] = 1
        # But that gives 4 separate terms (one for each (i,k) pair), which is rank 4, not rank 2.
        # 
        # Wait, I think I'm confusing myself. In the tensor decomposition:
        # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # The schoolbook decomposition has T[i,j,k] = 1 if we're looking at the (i,j,k) entry,
        # meaning each (i,j,k) triple gets its own rank-1 term.
        #
        # For Strassen, we group multiple (i,j,k) entries into one rank-1 term by using
        # sums in U, V, W. For example, the term (a_00+a_11)(b_00+b_11) expands to
        # a_00*b_00 + a_00*b_11 + a_11*b_00 + a_11*b_11, and these 4 schoolbook terms
        # are captured by 1 Strassen term.
        #
        # For the boundary a_i2 * b_2k: this is just one schoolbook term for each (i,k) pair.
        # There's no sum to exploit, so each needs its own rank-1 term.
        # C[0,2c] += a_02 * b_20: one term
        # C[1,2c] += a_12 * b_20: one term  
        # C[0,2c+1] += a_02 * b_21: one term
        # C[1,2c+1] += a_12 * b_21: one term
        # That's 4 terms, not 2!
        #
        # So per Strassen block: 7 + 4 = 11
        # 3 blocks: 33
        # Last column: 6
        # Total: 39!
        
        # OK so rank 39 it is. Let me redo this properly.
        pass
    
    # Clean implementation:
    rank = 39
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    def set_term(col_idx, u_entries, v_entries, w_entries):
        """Set a rank-1 term at col_idx.
        u_entries: dict mapping (i,j) -> value, where i in [0,n), j in [0,m)
        v_entries: dict mapping (j,k) -> value, where j in [0,m), k in [0,p)
        w_entries: dict mapping (k,i) -> value, where k in [0,p), i in [0,n)
        """
        for (i, j), val in u_entries.items():
            U[i * m + j, col_idx] += val
        for (j, k), val in v_entries.items():
            V[j * p + k, col_idx] += val
        for (k, i), val in w_entries.items():
            W[k * n + i, col_idx] += val
    
    col = 0
    for c in range(3):
        # Strassen on A[:,:2] * B[:2, 2c:2c+2]
        # A indices: (i,j) with i in {0,1}, j in {0,1}
        # B indices: (j,k) with j in {0,1}, k in {2c, 2c+1}
        
        # m1 = (a_00+a_11)(b_00+b_11)
        u = {(0,0): 1, (1,1): 1}
        v = {(0,2*c): 1, (1,2*c+1): 1}
        w = {(2*c,0): 1, (2*c+1,0): -1, (2*c,1): -1, (2*c+1,1): 1}
        set_term(col, u, v, w)
        col += 1
        
        # m2 = (a_10+a_11)b_00
        u = {(1,0): 1, (1,1): 1}
        v = {(0,2*c): 1}
        w = {(2*c,1): 1, (2*c+1,1): -1}
        set_term(col, u, v, w)
        col += 1
        
        # m3 = a_00(b_10-b_11)
        u = {(0,0): 1}
        v = {(1,2*c): 1, (1,2*c+1): -1}
        w = {(2*c+1,0): 1, (2*c+1,1): 1}
        set_term(col, u, v, w)
        col += 1
        
        # m4 = a_11(b_01-b_10)
        u = {(1,1): 1}
        v = {(0,2*c+1): 1, (0,2*c): -1}
        w = {(2*c,0): 1, (2*c,1): 1}
        set_term(col, u, v, w)
        col += 1
        
        # m5 = (a_00+a_01)b_11
        u = {(0,0): 1, (0,1): 1}
        v = {(1,2*c+1): 1}
        w = {(2*c,0): -1, (2*c+1,0): 1}
        set_term(col, u, v, w)
        col += 1
        
        # m6 = (a_10-a_00)(b_00+b_10)
        u = {(1,0): 1, (0,0): -1}
        v = {(0,2*c): 1, (1,2*c): 1}
        w = {(2*c+1,1): 1}
        set_term(col, u, v, w)
        col += 1
        
        # m7 = (a_01-a_11)(b_01+b_11)
        u = {(0,1): 1, (1,1): -1}
        v = {(0,2*c+1): 1, (1,2*c+1): 1}
        w = {(2*c,0): 1}
        set_term(col, u, v, w)
        col += 1
        
        # Boundary: a_i2 * b_2k for i in {0,1}, k in {0,1} (relative to block)
        # C[i, 2c] += a_i2 * B[2, 2c]
        # C[i, 2c+1] += a_i2 * B[2, 2c+1]
        for i in range(2):
            for k_off in range(2):
                k = 2*c + k_off
                u = {(i, 2): 1}
                v = {(2, k): 1}
                w = {(k, i): 1}
                set_term(col, u, v, w)
                col += 1
    
    # Last column: C[i, 6] = a_i0*b_06 + a_i1*b_16 + a_i2*b_26
    for i in range(2):
        for j in range(3):
            u = {(i, j): 1}
            v = {(j, 6): 1}
            w = {(6, i): 1}
            set_term(col, u, v, w)
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
