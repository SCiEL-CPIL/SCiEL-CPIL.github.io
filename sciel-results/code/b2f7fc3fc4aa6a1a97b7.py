<<<<<<< SEARCH
import numpy as np
import time

def construct():
    n, m, p = 4, 4, 6
    deadline = time.time() + 140
    
    # Try to build a rank-26 decomposition
    # Part 1: 4x4 x 4x4 with rank 18
    # Part 2: 4x4 x 4x2 with rank 8 (schoolbook)
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Part 2: Schoolbook for 4x4 x 4x2 (columns 4,5 of B)
    for i in range(n):
        for j in range(m):
            for k in range(2):  # columns 4 and 5
                u_idx = i * m + j
                v_idx = j * p + (k + 4)  # k+4 for columns 4,5
                w_idx = (k + 4) * n + i
                
                U_terms.append({u_idx: 1})
                V_terms.append({v_idx: 1})
                W_terms.append({w_idx: 1})
    
    # Part 1: Rank-18 for 4x4 x 4x4
    # Using a specific tensor decomposition
    # I'll use the following known rank-18 decomposition
    # 
    # The decomposition uses 18 rank-1 tensors.
    # Each rank-1 tensor is defined by:
    #   U vector: nonzero at position (i,j) with value a_ij
    #   V vector: nonzero at position (j,k) with value b_jk  
    #   W vector: nonzero at position (k,i) with value c_ki
    # such that sum_r a_ij * b_jk * c_ki = delta_ik (the multiplication tensor)
    
    # Let me use a construction based on the following:
    # Split indices into even/odd: {0,2} and {1,3}
    # Use 2x2 Strassen on each pair
    
    # Actually, let me just use a verified decomposition from literature
    # For now, let me implement it and verify
    
    # I'll define the 18 terms for 4x4x4
    # Each term: (U_nonzero_positions, V_nonzero_positions, W_nonzero_positions)
    # where each position has a coefficient
    
    # This is getting complex. Let me just return schoolbook for now
    # and add a TODO for optimization.
    
    # Fallback: schoolbook rank 96
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][r_idx] = 1
                V[v_idx][r_idx] = 1
                W[w_idx][r_idx] = 1
                
                r_idx += 1
                
    return {"U": U, "V": V, "W": W}
=======
import numpy as np
import time
from itertools import product

def _strassen_2x2_terms():
    """Return the 7 rank-1 terms for 2x2 x 2x2 -> 2x2 matrix multiplication.
    Each term is (u_dict, v_dict, w_dict) where keys are flat indices.
    u: (i,j) -> i*2+j, v: (j,k) -> j*2+k, w: (k,i) -> k*2+i
    """
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    terms = []
    # M1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
    terms.append(
        ({0: 1, 3: 1},   # A00, A11
         {0: 1, 3: 1},   # B00, B11
         {0: 1, 3: 1})   # C00, C11
    )
    # M2: (A10+A11)B00 -> C10+=1
    terms.append(
        ({2: 1, 3: 1},   # A10, A11
         {0: 1},         # B00
         {2: 1})         # C10
    )
    # M3: A00(B01-B11) -> C01+=1, C11+=1
    terms.append(
        ({0: 1},         # A00
         {1: 1, 3: -1},  # B01-B11
         {1: 1, 3: 1})   # C01, C11
    )
    # M4: A11(B10-B00) -> C00+=1, C10+=1
    terms.append(
        ({3: 1},         # A11
         {2: 1, 0: -1},  # B10-B00
         {0: 1, 2: 1})   # C00, C10
    )
    # M5: (A00+A01)B11 -> C00-=1, C01+=1
    terms.append(
        ({0: 1, 1: 1},   # A00, A01
         {3: 1},         # B11
         {0: -1, 1: 1})  # C00, C01
    )
    # M6: (A10-A00)(B00+B01) -> C11+=1
    terms.append(
        ({2: 1, 0: -1},  # A10-A00
         {0: 1, 1: 1},   # B00, B01
         {3: 1})         # C11
    )
    # M7: (A01-A11)(B10+B11) -> C00+=1
    terms.append(
        ({1: 1, 3: -1},  # A01-A11
         {2: 1, 3: 1},   # B10, B11
         {0: 1})         # C00
    )
    return terms

def _expand_4x4_terms():
    """Expand 2x2 Strassen to 4x4 using block decomposition.
    Split 4x4 into 2x2 blocks of 2x2 submatrices.
    A = [[A00, A01], [A10, A11]] where each block is 2x2.
    This gives 7 * 7 = 49 terms for 4x4 x 4x4 -> 4x4.
    """
    s2_terms = _strassen_2x2_terms()
    terms_4x4 = []
    
    for outer_u, outer_v, outer_w in s2_terms:
        for inner_u, inner_v, inner_w in s2_terms:
            # Combine: outer determines which 2x2 block, inner determines position within block
            u_combined = {}
            v_combined = {}
            w_combined = {}
            
            for (oi, oj), ou_val in outer_u.items():
                for (ii, ij), iu_val in inner_u.items():
                    # Global position: row = oi*2+ii, col = oj*2+ij
                    gi = oi * 2 + ii
                    gj = oj * 2 + ij
                    flat = gi * 4 + gj
                    u_combined[flat] = u_combined.get(flat, 0) + ou_val * iu_val
            
            for (oj, ok), ov_val in outer_v.items():
                for (ij, ik), iv_val in inner_v.items():
                    gi = oj * 2 + ij
                    gj = ok * 2 + ik
                    flat = gi * 4 + gj
                    v_combined[flat] = v_combined.get(flat, 0) + ov_val * iv_val
            
            for (ok, oi), ow_val in outer_w.items():
                for (ik, ii), iw_val in inner_w.items():
                    gi = ok * 2 + ik
                    gj = oi * 2 + ii
                    flat = gi * 4 + gj
                    w_combined[flat] = w_combined.get(flat, 0) + ow_val * iw_val
            
            terms_4x4.append((u_combined, v_combined, w_combined))
    
    return terms_4x4

def _expand_4x4x2_terms():
    """Expand 2x2 Strassen to 4x4 x 4x2 -> 4x2 using block decomposition.
    A is 4x4 (split into 2x2 blocks of 2x2), B is 4x2 (split into 2x2 blocks of 2x1).
    Actually, let's think of B as 4x2 = [[B_top], [B_bottom]] where each is 2x2.
    Wait, 4x2: rows 0-3, cols 0-1. Split into 2x2 blocks: top-left 2x2? No.
    
    Better: B is 4x2. Split rows into 2 groups of 2.
    B = [[B00], [B10]] where B00 is 2x2, B10 is 2x2.
    A = [[A00, A01], [A10, A11]] 4x4.
    
    C = A*B is 4x2.
    C = [[A00*B00 + A01*B10], [A10*B00 + A11*B10]]
    
    This is not a standard Strassen. Let me just use schoolbook for the 4x4x2 part
    but with the 2x2 Strassen applied to the inner dimension.
    
    Actually, for 4x4 x 4x2:
    We can view this as: for each column of B (2 columns), we do 4x4 x 4x1.
    4x4 x 4x1: split A into 2x2 blocks, B column into 2x1 blocks.
    C_col = A00*B0 + A10*B1 (top half) and A01*B0 + A11*B1 (bottom half)
    
    Let me just use a direct approach: schoolbook for 4x4x2 gives rank 32.
    But we can do better by using Strassen on the 2x2 structure.
    
    For 4x4 x 4x2, think of it as 2 independent 4x4 x 4x1 problems.
    For 4x4 x 4x1: A(4x4) * b(4x1) -> c(4x1)
    Split A = [[A00,A01],[A10,A11]] (2x2 blocks of 2x2), b = [b0; b1] (2 blocks of 2x1)
    c = [A00*b0 + A01*b1; A10*b0 + A11*b1]
    
    Using Strassen-like for each 2x2 x 2x1:
    2x2 x 2x1: A(2x2)*b(2x1) -> c(2x1)
    c0 = A00*b0 + A01*b1
    c1 = A10*b0 + A11*b1
    
    This is just schoolbook for 2x2x2x1, rank 4.
    But we can use Strassen trick: 
    M1 = (A00+A11)*b0, M2 = (A10+A11)*b0, M3 = A00*(b0-b1), M4 = A11*(b1-b0)
    Hmm, this doesn't quite work for rectangular.
    
    Let me just use schoolbook for the 4x4x2 part: rank = 4*4*2 = 32.
    Total: 49 + 32 = 81. Still better than 96.
    
    Actually wait - let me reconsider. The inspiration says rank 77 = 49+28.
    So 4x4x2 should be rank 28. Let me think about how.
    
    4x4 x 4x2: We can use the fact that B has only 2 columns.
    Think of it as: A(4x4) * B(4x2) = C(4x2).
    
    Split A into 2x2 blocks: A = [[A00,A01],[A10,A11]]
    Split B into 2x2 blocks: B = [[B00],[B10]] (each 2x2)
    
    C0 = A00*B00 + A01*B10  (2x2)
    C1 = A10*B00 + A11*B10  (2x2)
    
    For C0: use Strassen on (A00,A01) * (B00,B10) -> 2x2
    This is 2x2 x 2x2 -> 2x2 but with A being 2x4 and B being 4x2.
    Actually A00 is 2x2, A01 is 2x2, B00 is 2x2, B10 is 2x2.
    C0 = A00*B00 + A01*B10
    
    We can compute this with 7 Strassen products if we treat it as:
    Let X = A00, Y = A01, Z = B00, W = B10.
    C0 = X*Z + Y*W
    
    This is not a standard matrix multiplication. It's a "bilinear" form.
    
    Alternative: Just use 4 schoolbook terms for each 2x2 block multiplication:
    C0 = A00*B00 + A01*B10: 2*4 = 8 terms
    C1 = A10*B00 + A11*B10: 2*4 = 8 terms
    Total: 16 terms for the 4x4x2 part.
    
    Wait, that's rank 16, not 28. Let me recount.
    
    Actually for 4x4 x 4x2 schoolbook: sum over i,j,k where i in [0,4), j in [0,4), k in [0,2).
    That's 4*4*2 = 32 terms.
    
    With block structure:
    C[i,k] = sum_j A[i,j]*B[j,k]
    
    If we split i into (i0, i1) where i = i0*2+i1, j into (j0,j1), k into (k0) where k=0,1:
    C[(i0,i1),k] = sum_{j0,j1} A[(i0,i1),(j0,j1)] * B[(j0,j1),k]
    
    = sum_{j0,j1} A[i0,i1,j0,j1] * B[j0,j1,k]
    
    This is still 4*4*2 = 32 scalar multiplications in schoolbook.
    
    To get rank 28, we need to use Strassen on the inner 2x2 structure.
    
    Let me try: For fixed i0 (0 or 1) and k (0 or 1):
    C[i0,i1,k] = sum_{j0,j1} A[i0,i1,j0,j1] * B[j0,j1,k]
    
    For fixed i0, k: this is a 2x2 matrix (over i1) times a 2x2 matrix (over j0,j1... no.
    
    Let me think differently. For fixed i0 and k:
    C[i0, :, k] = A[i0, :, :] * B[:, k]
    where A[i0,:,:] is 2x4 and B[:,k] is 4x1.
    
    2x4 * 4x1: split A[i0,:,:] = [A[i0,0,:2], A[i0,0,:2]] (two 2x2 blocks)
    B[:,k] = [B[0:2,k]; B[2:4,k]] (two 2x1 blocks)
    
    C[i0,:,k] = A[i0,0,:2]*B[0:2,k] + A[i0,1,:2]*B[2:4,k]
    
    Each 2x2 * 2x1 is rank 4 in schoolbook, or can be done with fewer.
    2x2 * 2x1: c = A*b where A is 2x2, b is 2x1.
    c0 = A00*b0 + A01*b1
    c1 = A10*b0 + A11*b1
    
    This requires 4 multiplications in schoolbook. Can we do better?
    Not really for 2x2 x 2x1, the rank is 4 (it's a special case).
    
    So total: for each (i0, k) pair: 2 * 4 = 8 terms.
    4 pairs * 8 = 32. Same as schoolbook.
    
    Hmm, let me just go with 49 + 32 = 81 for now. That's still an improvement over 96.
    """
    # Schoolbook for 4x4 x 4x2
    terms = []
    for i in range(4):
        for j in range(4):
            for k in range(2):
                u = {i * 4 + j: 1}
                v = {j * 2 + k: 1}
                w = {k * 4 + i: 1}
                terms.append((u, v, w))
    return terms

def construct():
    n, m, p = 4, 4, 6
    deadline = time.time() + 140
    
    # Part 1: 4x4 x 4x4 with rank 49 (Strassen)
    # Part 2: 4x4 x 4x2 with rank 32 (schoolbook)
    # Total rank: 81
    
    terms_4x4 = _expand_4x4_terms()
    terms_4x4x2 = _expand_4x4x2_terms()
    
    all_terms = terms_4x4 + terms_4x4x2
    rank = len(all_terms)
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x rank
    V = [[0] * rank for _ in range(m * p)]  # 24 x rank
    W = [[0] * rank for _ in range(p * n)]  # 24 x rank
    
    for r, (u_dict, v_dict, w_dict) in enumerate(all_terms):
        for idx, val in u_dict.items():
            U[idx][r] = val
        for idx, val in v_dict.items():
            V[idx][r] = val
        for idx, val in w_dict.items():
            W[idx][r] = val
    
    # Verify correctness
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 if i==k, 0 otherwise (for the 4x4x4 part)
    # For the 4x4x2 part (k=4,5), should equal 1 if appropriate
    
    # Quick verification with random matrices
    A = np.random.randn(4, 4)
    B = np.random.randn(4, 6)
    C_expected = A @ B
    
    U_arr = np.array(U).T  # rank x 16
    V_arr = np.array(V).T  # rank x 24
    W_arr = np.array(W).T  # rank x 24
    
    # Compute C from decomposition
    C_computed = np.zeros((4, 6))
    for r in range(rank):
        # U[:,r] is 16-vector, reshape to 4x4
        u_mat = U_arr[r].reshape(4, 4)
        v_vec = V_arr[r]  # 24-vector
        w_vec = W_arr[r]  # 24-vector
        
        # C += u_mat * (v_vec reshaped) * ... 
        # Actually the decomposition is:
        # C[i,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] for each j
        # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        for j in range(4):
            for k in range(6):
                C_computed[i if False else 0, 0]  # placeholder
    
    # Better verification:
    C_computed = np.zeros((4, 6))
    for i in range(4):
        for k in range(6):
            s = 0.0
            for j in range(4):
                for r in range(rank):
                    s += U[i*4+j][r] * V[j*6+k][r] * W[k*4+i][r]
            C_computed[i,k] = s
    
    # Check
    max_err = np.max(np.abs(C_computed - C_expected))
    if max_err > 1e-10:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    U[u_idx][r_idx] = 1
                    V[v_idx][r_idx] = 1
                    W[w_idx][r_idx] = 1
                    
                    r_idx += 1
    
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE

Wait, I need to double-check the verification logic. The W index for the 4x4x2 part uses `k*n+i` where k is 0 or 1 (local column index in the 4x2 block). But in the full 4x6 problem, the columns are 0-5. Let me fix the indexing.

