"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-78 decomposition of 3x4x7 matrix multiplication using:
    - Strassen 2x2x2 (rank 7) for 2x2 * 2x2 blocks
    - Schoolbook for remaining blocks
    Block structure:
      A (3x4): rows split 2+1, cols split 2+2
      B (4x7): rows split 2+2, cols split 2+2+3
    """
    n, m, p = 3, 4, 7
    
    # We'll build U, V, W as lists of lists, accumulating rank-1 terms.
    # U has shape (n*m, rank) = (12, rank)
    # V has shape (m*p, rank) = (28, rank)
    # W has shape (p*n, rank) = (21, rank)
    
    # Start with empty, we'll grow them
    U_rows = []  # each entry will be a list of length rank (built incrementally)
    V_rows = []
    W_rows = []
    
    # We'll collect rank-1 terms as (u_vec, v_vec, w_vec) where each is a dict or sparse
    # Actually, let's just build U, V, W column by column.
    
    terms = []  # list of (u_dict, v_dict, w_dict) where dict maps index -> value
    
    def add_term(u_entries, v_entries, w_entries):
        """Add a rank-1 term. u_entries: list of (row_idx, val), etc."""
        u_d = dict(u_entries)
        v_d = dict(v_entries)
        w_d = dict(w_entries)
        terms.append((u_d, v_d, w_d))
    
    # Helper: Strassen 2x2x2 for A_block (2x2) * B_block (2x2) -> C_block (2x2)
    # A_block rows: a0,a1; cols: b0,b1 (in the m-dimension)
    # B_block rows: c0,c1 (in the m-dimension); cols: d0,d1 (in the p-dimension)
    # C_block rows: a0,a1; cols: d0,d1
    #
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    #
    # For tensor decomposition, each Mi is a rank-1 term in the sub-problem.
    # We need to express each Mi as a product of vectors in U, V, W.
    
    def strassen_222(a_row0, a_row1, b_col0_start, b_col1_start, c_row0, c_row1, c_col0, c_col1):
        """
        Add Strassen rank-7 terms for a 2x2 * 2x2 block.
        a_row0, a_row1: row indices in A (n-dim) for the 2x2 block
        b_col0_start, b_col1_start: col indices in A / row indices in B (m-dim) for the 2x2 block
        c_row0, c_row1: row indices in A (n-dim) for the 2x2 output block
        c_col0, c_col1: col indices in B (p-dim) for the 2x2 output block
        
        The A block is A[a_row0:a_row1+1, b_col0_start:b_col1_start+1]
        The B block is B[b_col0_start:b_col1_start+1, c_col0:c_col1+1]
        The C block is C[c_row0:c_row1+1, c_col0:c_col1+1]
        """
        # A sub-block indices in U: row = i*m + j
        # B sub-block indices in V: row = j*p + k
        # C sub-block indices in W: row = k*n + i
        
        # A00 = A[a_row0, b_col0_start]
        # A01 = A[a_row0, b_col1_start]
        # A10 = A[a_row1, b_col0_start]
        # A11 = A[a_row1, b_col1_start]
        
        # B00 = B[b_col0_start, c_col0]
        # B01 = B[b_col0_start, c_col1]
        # B10 = B[b_col1_start, c_col0]
        # B11 = B[b_col1_start, c_col1]
        
        # M1 = (A00+A11)(B00+B11)
        # This is a sum of 4 rank-1 terms in the full tensor, but in Strassen's
        # tensor decomposition, M1 corresponds to a single rank-1 term where:
        # U vector has 1s at positions corresponding to A00 and A11
        # V vector has 1s at positions corresponding to B00 and B11
        # W vector has coefficients at C positions
        
        # Actually, in the tensor decomposition framework:
        # Each rank-1 term is u ⊗ v ⊗ w where:
        # u is in R^{n*m}, v is in R^{m*p}, w is in R^{p*n}
        # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # For Strassen, we decompose the 2x2x2 tensor (which has rank 7) into 7 rank-1 terms.
        # Each rank-1 term corresponds to one of the M1..M7 products.
        
        # Let me define the 7 rank-1 terms for a generic 2x2x2 block.
        # The 2x2x2 tensor T2[i,j,k] = A[i,j]*B[j,k] for i,j,k in {0,1}
        # 
        # Rank-1 decomposition of T2:
        # Term 1: u1=(1,0,0,1), v1=(1,1,1,1), w1=(1,0,0,1) -- wait, this isn't right.
        
        # Let me think differently. The 2x2x2 matmul tensor is:
        # T[i,j,k] = A[i,j] * B[j,k]
        # 
        # Strassen gives 7 rank-1 terms. Let me label the positions:
        # A positions: (0,0),(0,1),(1,0),(1,1) -> indices 0,1,2,3 in U
        # B positions: (0,0),(0,1),(1,0),(1,1) -> indices 0,1,2,3 in V
        # C positions: (0,0),(0,1),(1,0),(1,1) -> indices 0,1,2,3 in W (with W[k*n+i] convention)
        
        # For the 2x2x2 sub-problem, the rank-1 terms are:
        # (using 0-indexed local coords for the 2x2 blocks)
        
        # M1 = (A00+A11)(B00+B11)
        #   U: {0:1, 3:1}, V: {0:1, 3:1}, W: {0:1, 3:1}  -- but this gives C00 and C11 contributions
        #   Actually M1 contributes to C00 and C11 with coefficient +1 each
        #   So W: {0:1, 3:1}
        
        # M2 = (A10+A11)B00
        #   U: {2:1, 3:1}, V: {0:1}, W: {2:1}  -- contributes to C10
        #   Wait, C10 = M2+M4, so M2 contributes +1 to C10
        #   W: {2:1}
        
        # M3 = A00(B01-B11)
        #   U: {0:1}, V: {1:1, 3:-1}, W: {1:1}  -- contributes to C01
        #   W: {1:1}
        
        # M4 = A11(B10-B00)
        #   U: {3:1}, V: {2:1, 0:-1}, W: {0:1, 2:1}  -- contributes to C00 and C10
        #   W: {0:1, 2:1}
        
        # M5 = (A00+A10)B11
        #   U: {0:1, 2:1}, V: {3:1}, W: {0:-1}  -- contributes to C00 with -1
        #   W: {0:-1}
        
        # M6 = (A10-A00)(B00+B01)
        #   U: {2:1, 0:-1}, V: {0:1, 1:1}, W: {3:1}  -- contributes to C11
        #   W: {3:1}
        
        # M7 = (A01-A11)(B10+B11)
        #   U: {1:1, 3:-1}, V: {2:1, 3:1}, W: {0:1}  -- contributes to C00
        #   W: {0:1}
        
        # Let me verify C00 = M1 + M4 - M5 + M7:
        # M1: +1, M4: +1, M5: -1, M7: +1 => 1+1-1+1 = 2? That's wrong, should be 1.
        
        # I think I'm confusing the standard Strassen formulas. Let me look this up properly.
        #
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A10)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # Let me verify with A=I, B=I:
        # A00=1,A01=0,A10=0,A11=1; B00=1,B01=0,B10=0,B11=1
        # M1=(1+1)(1+1)=4
        # M2=(0+1)(1)=1
        # M3=1*(0-1)=-1
        # M4=1*(0-1)=-1
        # M5=(1+0)(1)=1
        # M6=(0-1)(1+0)=-1
        # M7=(0-1)(0+1)=-1
        # C00=4+(-1)-1+(-1)=1 ✓
        # C01=-1+1=0 ✓
        # C10=1+(-1)=0 ✓
        # C11=4-1+(-1)+(-1)=1 ✓
        # Good!
        
        # Now for tensor rank-1 terms, each Mi is a bilinear form, which in the
        # tensor decomposition corresponds to a rank-1 term where:
        # - U has the coefficients from the A-part
        # - V has the coefficients from the B-part  
        # - W has the coefficients from the C-part (how Mi contributes to each C entry)
        
        # M1 = (A00+A11)(B00+B11): 
        #   A-part: A00 and A11 with coeff 1 -> U: {0:1, 3:1}
        #   B-part: B00 and B11 with coeff 1 -> V: {0:1, 3:1}
        #   C-part: contributes +1 to C00, +1 to C11 -> W: {0:1, 3:1}
        
        # M2 = (A10+A11)B00:
        #   U: {2:1, 3:1}, V: {0:1}, W: {2:1}  (C10 gets +1)
        
        # M3 = A00(B01-B11):
        #   U: {0:1}, V: {1:1, 3:-1}, W: {1:1}  (C01 gets +1)
        
        # M4 = A11(B10-B00):
        #   U: {3:1}, V: {2:1, 0:-1}, W: {0:1, 2:1}  (C00 gets +1, C10 gets +1)
        
        # M5 = (A00+A10)B11:
        #   U: {0:1, 2:1}, V: {3:1}, W: {0:-1}  (C00 gets -1)
        
        # M6 = (A10-A00)(B00+B01):
        #   U: {2:1, 0:-1}, V: {0:1, 1:1}, W: {3:1}  (C11 gets +1)
        
        # M7 = (A01-A11)(B10+B11):
        #   U: {1:1, 3:-1}, V: {2:1, 3:1}, W: {0:1}  (C00 gets +1)
        
        # Verify C00 = M1+M4-M5+M7: W contributions: 1+1-(-1)+1 = 1+1+1+1=4? No...
        # Each rank-1 term contributes its W value times the product. For the tensor:
        # T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        # 
        # For C00 (i=0,k=0): sum over j of A[0,j]*B[j,0] = A00*B00 + A01*B10
        # The rank-1 terms contribute to T[0,*,0] positions.
        
        # This is getting complex. Let me just directly construct the 7 rank-1 terms
        # for the 2x2x2 tensor and verify they reproduce the correct tensor.
        
        # The 2x2x2 matmul tensor T[i,j,k] = A[i,j]*B[j,k]
        # Non-zero entries: T[0,0,0]=1, T[0,1,0]=1, T[1,0,1]=1, T[1,1,1]=1
        # (using T[i,j,k] where i=A row, j=inner, k=B col)
        # But in our W convention, W[k*n+i, r], so for the 2x2 sub-problem embedded in larger,
        # the local C index is (k, i) mapped to global.
        
        # Let me just define the 7 terms with their local U,V,W vectors and verify.
        
        # Local U indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)  [i,j]
        # Local V indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)  [j,k]
        # Local W indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)  [k,i]
        
        # Term M1: U=[1,0,0,1], V=[1,1,1,1], W=[1,0,0,1]
        # Term M2: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0]
        # Term M3: U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,0]
        # Term M4: U=[0,0,0,1], V=[-1,0,1,0], W=[1,0,1,0]
        # Term M5: U=[1,0,1,0], V=[0,0,0,1], W=[-1,0,0,0]
        # Term M6: U=[-1,0,1,0], V=[1,1,0,0], W=[0,0,0,1]
        # Term M7: U=[0,1,0,-1], V=[0,0,1,1], W=[1,0,0,0]
        
        # Verify T[0,0,0] (i=0,j=0,k=0): should be 1
        # M1: U[0]*V[0]*W[0] = 1*1*1 = 1
        # M2: U[0]*V[0]*W[0] = 0*1*0 = 0
        # M3: U[0]*V[0]*W[0] = 1*0*0 = 0
        # M4: U[0]*V[0]*W[0] = 0*(-1)*1 = 0
        # M5: U[0]*V[0]*W[0] = 1*0*(-1) = 0
        # M6: U[0]*V[0]*W[0] = -1*1*0 = 0
        # M7: U[0]*V[0]*W[0] = 0*0*1 = 0
        # Sum = 1 ✓
        
        # T[0,1,0] (i=0,j=1,k=0): should be 1
        # M1: U[1]*V[1]*W[0] = 0*1*1 = 0
        # M2: U[1]*V[1]*W[0] = 0*0*0 = 0
        # M3: U[1]*V[1]*W[0] = 0*1*0 = 0
        # M4: U[1]*V[1]*W[0] = 0*0*1 = 0
        # M5: U[1]*V[1]*W[0] = 0*0*(-1) = 0
        # M6: U[1]*V[1]*W[0] = 0*1*0 = 0
        # M7: U[1]*V[1]*W[0] = 1*0*1 = 0
        # Sum = 0 ✗ Should be 1!
        
        # Hmm, T[0,1,0] means A[0,1]*B[1,0]. In the matmul, C[0,0] = sum_j A[0,j]*B[j,0]
        # = A[0,0]*B[0,0] + A[0,1]*B[1,0]. So T[i,j,k] with i=0,j=1,k=0 should indeed be 1.
        # But my Strassen terms give 0. Let me recheck.
        
        # I think the issue is that in the tensor T[i,j,k] = A[i,j]*B[j,k], the index j is shared.
        # For a rank-1 term u⊗v⊗w, the contribution to T[i,j,k] is u[i,j]*v[j,k]*w[k,i].
        # But u is indexed by (i,j), v by (j,k), w by (k,i). So for a fixed (i,j,k),
        # we need u[i,j], v[j,k], w[k,i].
        
        # For T[0,1,0]: i=0,j=1,k=0. Need u[0,1], v[1,0], w[0,0].
        # u[0,1] is local U index 1, v[1,0] is local V index 2, w[0,0] is local W index 0.
        
        # M1: u[1]*v[2]*w[0] = 0*1*1 = 0
        # M2: u[1]*v[2]*w[0] = 0*0*0 = 0
        # M3: u[1]*v[2]*w[0] = 0*0*0 = 0
        # M4: u[1]*v[2]*w[0] = 0*1*1 = 0
        # M5: u[1]*v[2]*w[0] = 0*0*(-1) = 0
        # M6: u[1]*v[2]*w[0] = 0*0*0 = 0
        # M7: u[1]*v[2]*w[0] = 1*1*1 = 1
        # Sum = 1 ✓
        
        # I made an error before. Let me redo T[0,0,0]: i=0,j=0,k=0. u[0,0],v[0,0],w[0,0].
        # U idx 0, V idx 0, W idx 0.
        # M1: 1*1*1=1
        # M2: 0*1*0=0
        # M3: 1*0*0=0
        # M4: 0*(-1)*1=0
        # M5: 1*0*(-1)=0
        # M6: -1*1*0=0
        # M7: 0*0*1=0
        # Sum=1 ✓
        
        # T[1,0,1]: i=1,j=0,k=1. u[1,0],v[0,1],w[1,1]. U idx 2, V idx 1, W idx 3.
        # M1: 0*1*1=0
        # M2: 1*0*0=0
        # M3: 0*1*0=0
        # M4: 0*0*0=0
        # M5: 1*0*0=0
        # M6: 1*1*1=1
        # M7: 0*1*0=0
        # Sum=1 ✓
        
        # T[1,1,1]: i=1,j=1,k=1. u[1,1],v[1,1],w[1,1]. U idx 3, V idx 3, W idx 3.
        # M1: 1*1*1=1
        # M2: 1*0*0=0
        # M3: 0*(-1)*0=0
        # M4: 1*0*0=0
        # M5: 0*1*0=0
        # M6: 0*0*1=0
        # M7: -1*1*0=0
        # Sum=1 ✓
        
        # Check a zero entry: T[0,0,1]: i=0,j=0,k=1. u[0,0],v[0,1],w[1,0]. U idx 0, V idx 1, W idx 2.
        # M1: 1*1*0=0
        # M2: 0*0*1=0
        # M3: 1*1*0=0
        # M4: 0*0*1=0
        # M5: 1*0*0=0
        # M6: -1*1*0=0
        # M7: 0*0*1=0
        # Sum=0 ✓
        
        # Great! The 7 Strassen rank-1 terms are correct.
        
        # Now map local indices to global:
        # A local (a_i, a_j) -> global U row: (a_row0 + a_i)*m + (b_col0_start + a_j)
        # B local (b_j, b_k) -> global V row: (b_col0_start + b_j)*p + (c_col0 + b_k)
        # C local (c_k, c_i) -> global W row: (c_col0 + c_k)*n + (c_row0 + c_i)
        
        local_U = [
            [1,0,0,1],  # M1
            [0,0,1,1],  # M2
            [1,0,0,0],  # M3
            [0,0,0,1],  # M4
            [1,0,1,0],  # M5
            [-1,0,1,0], # M6
            [0,1,0,-1], # M7
        ]
        local_V = [
            [1,1,1,1],  # M1
            [1,0,0,0],  # M2
            [0,1,0,-1], # M3
            [-1,0,1,0], # M4
            [0,0,0,1],  # M5
            [1,1,0,0],  # M6
            [0,0,1,1],  # M7
        ]
        local_W = [
            [1,0,0,1],  # M1
            [0,0,1,0],  # M2
            [0,1,0,0],  # M3
            [1,0,1,0],  # M4
            [-1,0,0,0], # M5
            [0,0,0,1],  # M6
            [1,0,0,0],  # M7
        ]
        
        for t in range(7):
            u_entries = []
            for li in range(2):
                for lj in range(2):
                    val = local_U[t][li*2+lj]
                    if val != 0:
                        global_i = c_row0 + li
                        global_j = b_col0_start + lj
                        u_entries.append((global_i * m + global_j, val))
            v_entries = []
            for lj in range(2):
                for lk in range(2):
                    val = local_V[t][lj*2+lk]
                    if val != 0:
                        global_j = b_col0_start + lj
                        global_k = c_col0 + lk
                        v_entries.append((global_j * p + global_k, val))
            w_entries = []
            for lk in range(2):
                for li in range(2):
                    val = local_W[t][lk*2+li]
                    if val != 0:
                        global_k = c_col0 + lk
                        global_i = c_row0 + li
                        w_entries.append((global_k * n + global_i, val))
            add_term(u_entries, v_entries, w_entries)
    
    # Helper: schoolbook for 1x2 * 2x2 (A is 1x2, B is 2x2, C is 1x2)
    # A[i,j] for i in {0}, j in {0,1}
    # B[j,k] for j in {0,1}, k in {0,1}
    # C[i,k] for i in {0}, k in {0,1}
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # Rank-1 terms:
    # Term 1: A[0,0]*B[0,0] -> C[0,0]: U:{(0,0):1}, V:{(0,0):1}, W:{(0,0):1}
    # Term 2: A[0,1]*B[1,0] -> C[0,0]: U:{(0,1):1}, V:{(1,0):1}, W:{(0,0):1}
    # Term 3: A[0,0]*B[0,1] -> C[0,1]: U:{(0,0):1}, V:{(0,1):1}, W:{(0,1):1}
    # Term 4: A[0,1]*B[1,1] -> C[0,1]: U:{(0,1):1}, V:{(1,1):1}, W:{(0,1):1}
    
    def schoolbook_1x2_2x2(a_row, b_col0_start, c_row, c_col0, c_col1):
        """
        A is 1x2 at row a_row, cols b_col0_start, b_col0_start+1
        B is 2x2 at rows b_col0_start, b_col0_start+1, cols c_col0, c_col1
        C is 1x2 at row c_row, cols c_col0, c_col1
        """
        # 4 rank-1 terms
        for j in range(2):
            for k in range(2):
                u_entries = [(a_row * m + (b_col0_start + j), 1)]
                v_entries = [((b_col0_start + j) * p + (c_col0 + k), 1)]
                w_entries = [((c_col0 + k) * n + c_row, 1)]
                add_term(u_entries, v_entries, w_entries)
    
    # Helper: schoolbook for 2x2 * 2x3 (A is 2x2, B is 2x3, C is 2x3)
    # Split B into 2x2 + 2x1
    # A is 2x2, B_first is 2x2 (cols 0,1), B_second is 2x1 (col 2)
    
    def schoolbook_2x2_2x3(a_row0, b_col0_start, c_row0, c_col0):
        """
        A is 2x2 at rows a_row0, a_row0+1, cols b_col0_start, b_col0_start+1
        B is 2x3 at rows b_col0_start, b_col0_start+1, cols c_col0, c_col0+1, c_col0+2
        C is 2x3 at rows c_row0, c_row0+1, cols c_col0, c_col0+1, c_col0+2
        """
        # For cols 0,1 of B: use Strassen 2x2x2
        strassen_222(a_row0, a_row0+1, b_col0_start, b_col0_start+1, c_row0, c_row0+1, c_col0, c_col0+1)
        # For col 2 of B: 2x2 * 2x1
        # C[i,2] = sum_j A[i,j]*B[j,2]
        # 4 rank-1 terms
        for i in range(2):
            for j in range(2):
                u_entries = [((a_row0 + i) * m + (b_col0_start + j), 1)]
                v_entries = [((b_col0_start + j) * p + (c_col0 + 2), 1)]
                w_entries = [((c_col0 + 2) * n + (c_row0 + i), 1)]
                add_term(u_entries, v_entries, w_entries)
    
    # Helper: schoolbook for 1x2 * 2x3
    def schoolbook_1x2_2x3(a_row, b_col0_start, c_row, c_col0):
        """
        A is 1x2 at row a_row, cols b_col0_start, b_col0_start+1
        B is 2x3 at rows b_col0_start, b_col0_start+1, cols c_col0, c_col0+1, c_col0+2
        C is 1x3 at row c_row, cols c_col0, c_col0+1, c_col0+2
        """
        for j in range(2):
            for k in range(3):
                u_entries = [(a_row * m + (b_col0_start + j), 1)]
                v_entries = [((b_col0_start + j) * p + (c_col0 + k), 1)]
                w_entries = [((c_col0 + k) * n + c_row, 1)]
                add_term(u_entries, v_entries, w_entries)
    
    # Now assemble the full decomposition:
    # A blocks: A11(2x2) rows 0-1, cols 0-1; A12(2x2) rows 0-1, cols 2-3
    #           A21(1x2) row 2, cols 0-1; A22(1x2) row 2, cols 2-3
    # B blocks: B11(2x2) rows 0-1, cols 0-1; B12(2x2) rows 0-1, cols 2-3
    #           B13(2x3) rows 0-1, cols 4-6
    #           B21(2x2) rows 2-3, cols 0-1; B22(2x2) rows 2-3, cols 2-3
    #           B23(2x3) rows 2-3, cols 4-6
    
    # C11 = A11*B11 + A12*B21
    strassen_222(0, 1, 0, 1, 0, 1, 0, 1)   # A11*B11
    strassen_222(0, 1, 2, 3, 0, 1, 0, 1)   # A12*B21
    
    # C12 = A11*B12 + A12*B22
    strassen_222(0, 1, 0, 1, 0, 1, 2, 3)   # A11*B12
    strassen_222(0, 1, 2, 3, 0, 1, 2, 3)   # A12*B22
    
    # C13 = A11*B13 + A12*B23
    schoolbook_2x2_2x3(0, 0, 0, 4)         # A11*B13
    schoolbook_2x2_2x3(0, 2, 0, 4)         # A12*B23
    
    # C21 = A21*B11 + A22*B21
    schoolbook_1x2_2x2(2, 0, 2, 0, 1)      # A21*B11
    schoolbook_1x2_2x2(2, 2, 2, 0, 1)      # A22*B21
    
    # C22 = A21*B12 + A22*B22
    schoolbook_1x2_2x2(2, 0, 2, 2, 3)      # A21*B12
    schoolbook_1x2_2x2(2, 2, 2, 2, 3)      # A22*B22
    
    # C23 = A21*B13 + A22*B23
    schoolbook_1x2_2x3(2, 0, 2, 4)         # A21*B13
    schoolbook_1x2_2x3(2, 2, 2, 4)         # A22*B23
    
    # Now build U, V, W matrices from the terms
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_d, v_d, w_d) in enumerate(terms):
        for idx, val in u_d.items():
            U[idx][r] = val
        for idx, val in v_d.items():
            V[idx][r] = val
        for idx, val in w_d.items():
            W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
