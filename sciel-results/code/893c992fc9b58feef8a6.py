"""Matrix multiplication 3,4,7 - rank-minimal tensor decomposition."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s
    
    n, m, p = 3, 4, 7
    
    # We'll build the decomposition using Strassen blocks + schoolbook for remainder
    # Each rank term is a triple (u_vec, v_vec, w_vec) where:
    #   u_vec has length n*m, v_vec has length m*p, w_vec has length p*n
    # The tensor is sum_c u_vec[c] ⊗ v_vec[c] ⊗ w_vec[c]
    # which means T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    rank_terms = []  # list of (U_row_dict, V_row_dict, W_row_dict) or we build arrays
    
    # Let's use a dict-based approach: each term contributes to specific (i,j,k) positions
    # Actually, let's build U, V, W as lists of dicts or sparse representations
    
    # Better: build the full U, V, W matrices incrementally
    # Start with all zeros, add contributions from each rank term
    
    def add_strassen_block(i1, i2, j1, j2, k1, k2, U, V, W, rank):
        """Add a 2x2x2 Strassen block. Returns new rank (7 terms added)."""
        # Strassen multiplications for 2x2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # A is the 2x2 submatrix of the left factor: A[di, dj] corresponds to index (i1+di, j1+dj)
        # B is the 2x2 submatrix of the right factor: B[dj, dk] corresponds to index (j1+dj, k1+dk)
        # C is the 2x2 result: C[di, dk] corresponds to index (i1+di, k1+dk)
        
        # For each multiplication term, we need:
        # U contribution: for the left factor indices
        # V contribution: for the right factor indices
        # W contribution: for the output indices
        
        # M1: (A11+A22)(B11+B22)
        # A11 is at (i1, j1), A22 is at (i2, j2)
        # B11 is at (j1, k1), B22 is at (j2, k2)
        # This contributes to C11, C12, C21, C22 with coefficients
        
        # Let me define the 7 terms more carefully.
        # Term t has:
        #   U: values at positions (i*m+j) for the relevant (i,j)
        #   V: values at positions (j*p+k) for the relevant (j,k)
        #   W: values at positions (k*n+i) for the relevant (k,i)
        
        # M1: left factor gets +1 at (i1,j1) and +1 at (i2,j2)
        #     right factor gets +1 at (j1,k1) and +1 at (j2,k2)
        #     output: C11 += 1, C12 += 1, C21 += 1, C22 += 1
        # Wait, that's not right. Let me re-derive.
        
        # C = A * B where A is 2x2, B is 2x2
        # C11 = A11*B11 + A12*B21
        # C12 = A11*B12 + A12*B22
        # C21 = A21*B11 + A22*B21
        # C22 = A21*B12 + A22*B22
        
        # Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For each multiplication Mx, it's a product of a linear combo of A entries and a linear combo of B entries.
        # In the tensor decomposition, each Mx corresponds to one rank term.
        
        # M1: (A11+A22) * (B11+B22)
        #   U: +1 at (i1*m+j1), +1 at (i2*m+j2)
        #   V: +1 at (j1*p+k1), +1 at (j2*p+k2)
        #   W: +1 at (k1*n+i1) for C11, +1 at (k2*n+i1) for C12, +1 at (k1*n+i2) for C21, +1 at (k2*n+i2) for C22
        #   But wait, the W vector needs to encode the output coefficients.
        
        # Actually, the way it works: the rank term (u, v, w) contributes u[i*m+j] * v[j*p+k] * w[k*n+i] to T[i,j,k].
        # So for a single multiplication term like A_a * B_b, where A_a is the entry at (i_a, j_a) and B_b is at (j_b, k_b):
        #   u[i_a*m+j_a] = 1
        #   v[j_b*p+k_b] = 1
        #   w[k_b*n+i_a] = 1
        # This contributes 1 to T[i_a, j_a, k_b] (since j_a must equal j_b for the product to be nonzero in the tensor sense... wait no).
        
        # Hmm, I need to think about this differently. The matrix multiplication tensor T[i,j,k] represents the coefficient of x_j in (A*B)_{i,k} where A_{i,j} and B_{j,k} are the entries. So T[i,j,k] = 1 for all i,j,k (it's just the identity tensor for multiplication).
        
        # A rank-1 term is u ⊗ v ⊗ w where u has length n*m, v has length m*p, w has length p*n.
        # The contribution to T[i,j,k] is u[i*m+j] * v[j*p+k] * w[k*n+i].
        
        # For a single "multiplication" of A_{i,j} * B_{j,k}, the rank-1 term has:
        #   u[i*m+j] = 1, v[j*p+k] = 1, w[k*n+i] = 1
        # This gives T[i,j,k] += 1, which is correct.
        
        # For Strassen, each Mx is a product of a linear combination of A entries and a linear combination of B entries.
        # E.g., M1 = (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
        # This is NOT a single rank-1 term! It's a sum of 4 rank-1 terms.
        
        # Wait, that can't be right. Strassen reduces 8 multiplications to 7, but each "multiplication" in Strassen is still a product of two linear combinations. In the tensor decomposition, each such product IS a rank-1 term because:
        # (a*A11 + b*A22) * (c*B11 + d*B22) can be represented as a single rank-1 term where:
        #   u[i1*m+j1] = a, u[i2*m+j2] = b
        #   v[j1*p+k1] = c, v[j2*p+k2] = d
        #   w[k1*n+i1] = 1, w[k2*n+i1] = 1, w[k1*n+i2] = 1, w[k2*n+i2] = 1
        # Let's verify: this term contributes to T[i,j,k]:
        #   For i=i1, j=j1, k=k1: u[i1*m+j1]*v[j1*p+k1]*w[k1*n+i1] = a*c*1
        #   For i=i1, j=j1, k=k2: a*d*1
        #   For i=i1, j=j2, k=k1: b*c*1
        #   For i=i1, j=j2, k=k2: b*d*1
        #   For i=i2, j=j1, k=k1: a*c*1
        #   etc.
        # So the total contribution is (a*A11 + b*A22) * (c*B11 + d*B22) distributed across the 4 output positions.
        # Yes! This is a valid rank-1 term.
        
        # So for M1 = (A11+A22)(B11+B22):
        #   a=1, b=1, c=1, d=1
        #   u: +1 at (i1*m+j1), +1 at (i2*m+j2)
        #   v: +1 at (j1*p+k1), +1 at (j2*p+k2)
        #   w: +1 at (k1*n+i1), +1 at (k2*n+i1), +1 at (k1*n+i2), +1 at (k2*n+i2)
        
        # Now for the output reconstruction:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each Mx is a rank-1 term as described. The final C entries are linear combinations of these Mx terms.
        # But in the tensor decomposition, we're decomposing T, not C. The Mx terms ARE the rank terms, and their W vectors encode how they contribute to each output position.
        
        # Let me redefine: for each Mx, the W vector should encode the coefficient of Mx in each C entry.
        # C11 gets M1 with coeff +1, M4 with +1, M5 with -1, M7 with +1
        # C12 gets M3 with +1, M5 with +1
        # C21 gets M2 with +1, M4 with +1
        # C22 gets M1 with +1, M2 with -1, M3 with +1, M6 with +1
        
        # So for M1:
        #   W: +1 at (k1*n+i1) [C11], +1 at (k2*n+i1) [C12], +1 at (k1*n+i2) [C21], +1 at (k2*n+i2) [C22]
        # Wait, C11 is at output (i1, k1), so w[k1*n+i1]. C12 is at (i1, k2), so w[k2*n+i1]. Etc.
        # But the coefficient of M1 in C11 is +1, in C12 is 0 (M1 doesn't appear in C12 formula), etc.
        
        # Let me re-examine:
        # C11 = M1 + M4 - M5 + M7 → M1 coeff in C11 is +1
        # C12 = M3 + M5 → M1 coeff in C12 is 0
        # C21 = M2 + M4 → M1 coeff in C21 is 0
        # C22 = M1 - M2 + M3 + M6 → M1 coeff in C22 is +1
        
        # So M1's W vector: +1 at (k1*n+i1), +1 at (k2*n+i2), 0 elsewhere in the 2x2 block.
        
        # Let me redo all 7:
        # M1 = (A11+A22)(B11+B22): u=(1,0,0,1) at (i1j1,i2j2), v=(1,0,0,1) at (j1k1,j2k2), w: C11:+1, C22:+1
        # M2 = (A21+A22)B11: u=(0,0,1,1) at (i1j1,i2j1,i2j2), v=(1,0) at (j1k1), w: C21:+1, C22:-1
        #   Wait, A21 is at (i2,j1), A22 is at (i2,j2). So u: +1 at (i2*m+j1), +1 at (i2*m+j2)
        #   B11 is at (j1,k1). So v: +1 at (j1*p+k1)
        #   C21 gets M2 with +1, C22 gets M2 with -1
        #   w: +1 at (k1*n+i2) [C21], -1 at (k2*n+i2) [C22]
        
        # Let me be very systematic. I'll define for each Mx:
        #   u_indices: list of (index, coeff) for U
        #   v_indices: list of (index, coeff) for V
        #   w_indices: list of (index, coeff) for W
        
        terms = []
        
        # M1 = (A11+A22)(B11+B22)
        # u: (i1,j1)+1, (i2,j2)+1
        # v: (j1,k1)+1, (j2,k2)+1
        # w: C11=(i1,k1)+1, C22=(i2,k2)+1
        terms.append((
            [(i1*m+j1, 1), (i2*m+j2, 1)],
            [(j1*p+k1, 1), (j2*p+k2, 1)],
            [(k1*n+i1, 1), (k2*n+i2, 1)]
        ))
        
        # M2 = (A21+A22)B11
        # u: (i2,j1)+1, (i2,j2)+1
        # v: (j1,k1)+1
        # w: C21=(i2,k1)+1, C22=(i2,k2)-1
        terms.append((
            [(i2*m+j1, 1), (i2*m+j2, 1)],
            [(j1*p+k1, 1)],
            [(k1*n+i2, 1), (k2*n+i2, -1)]
        ))
        
        # M3 = A11(B21-B22)
        # u: (i1,j1)+1
        # v: (j2,k1)+1, (j2,k2)-1
        # w: C12=(i1,k2)+1, C22=(i2,k2)+1
        # Wait, C12 = M3+M5, so M3 coeff in C12 is +1
        # C22 = M1-M2+M3+M6, so M3 coeff in C22 is +1
        # C12 is at (i1,k2), C22 is at (i2,k2)
        terms.append((
            [(i1*m+j1, 1)],
            [(j2*p+k1, 1), (j2*p+k2, -1)],
            [(k2*n+i1, 1), (k2*n+i2, 1)]
        ))
        
        # M4 = A22(B12-B11)
        # u: (i2,j2)+1
        # v: (j1,k2)+1, (j1,k1)-1
        # w: C11=(i1,k1)+1, C21=(i2,k1)+1
        # C11 = M1+M4-M5+M7, M4 coeff +1
        # C21 = M2+M4, M4 coeff +1
        terms.append((
            [(i2*m+j2, 1)],
            [(j1*p+k2, 1), (j1*p+k1, -1)],
            [(k1*n+i1, 1), (k1*n+i2, 1)]
        ))
        
        # M5 = (A11+A12)B22
        # u: (i1,j1)+1, (i1,j2)+1
        # v: (j2,k2)+1
        # w: C11=(i1,k1)-1, C12=(i1,k2)+1
        # C11 = M1+M4-M5+M7, M5 coeff -1
        # C12 = M3+M5, M5 coeff +1
        terms.append((
            [(i1*m+j1, 1), (i1*m+j2, 1)],
            [(j2*p+k2, 1)],
            [(k1*n+i1, -1), (k2*n+i1, 1)]
        ))
        
        # M6 = (A21-A11)(B11+B12)
        # u: (i2,j1)+1, (i1,j1)-1
        # v: (j1,k1)+1, (j1,k2)+1
        # w: C22=(i2,k2)+1
        # C22 = M1-M2+M3+M6, M6 coeff +1
        terms.append((
            [(i2*m+j1, 1), (i1*m+j1, -1)],
            [(j1*p+k1, 1), (j1*p+k2, 1)],
            [(k2*n+i2, 1)]
        ))
        
        # M7 = (A12-A22)(B21+B22)
        # u: (i1,j2)+1, (i2,j2)-1
        # v: (j2,k1)+1, (j2,k2)+1
        # w: C11=(i1,k1)+1
        # C11 = M1+M4-M5+M7, M7 coeff +1
        terms.append((
            [(i1*m+j2, 1), (i2*m+j2, -1)],
            [(j2*p+k1, 1), (j2*p+k2, 1)],
            [(k1*n+i1, 1)]
        ))
        
        for u_list, v_list, w_list in terms:
            for idx, coeff in u_list:
                U[idx][rank] += coeff
            for idx, coeff in v_list:
                V[idx][rank] += coeff
            for idx, coeff in w_list:
                W[idx][rank] += coeff
            rank += 1
        
        return rank
    
    def add_schoolbook_term(i, j, k, U, V, W, rank):
        """Add a single schoolbook term A_{i,j} * B_{j,k}."""
        U[i*m+j][rank] = 1
        V[j*p+k][rank] = 1
        W[k*n+i][rank] = 1
        return rank + 1
    
    # Initialize U, V, W with a large enough size
    # Max rank: 84 (schoolbook). With 6 Strassen blocks: 84 - 6 = 78
    # Let's allocate for 84 to be safe
    max_rank = n * m * p
    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]
    
    rank = 0
    covered = set()  # set of (i,j,k) triples covered by Strassen blocks
    
    # Place Strassen blocks: i-pair {0,1}, j-pairs {0,1} and {2,3}, k-pairs {0,1}, {2,3}, {4,5}
    i_pairs = [(0, 1)]
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for i1, i2 in i_pairs:
        for j1, j2 in j_pairs:
            for k1, k2 in k_pairs:
                if time.time() - start_time > DEADLINE:
                    break
                # Add all 8 triples to covered
                for di in [0, 1]:
                    for dj in [0, 1]:
                        for dk in [0, 1]:
                            covered.add((i1+di, j1+dj, k1+dk))
                rank = add_strassen_block(i1, i2, j1, j2, k1, k2, U, V, W, rank)
    
    # Add schoolbook terms for uncovered triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    if time.time() - start_time > DEADLINE:
                        break
                    rank = add_schoolbook_term(i, j, k, U, V, W, rank)
    
    # Trim to actual rank
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]
    
    # Verify the decomposition
    # T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For all i,j,k, this should be 1
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        # Fallback to schoolbook
        U = [[0] * max_rank for _ in range(n * m)]
        V = [[0] * max_rank for _ in range(m * p)]
        W = [[0] * max_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        rank = col
        U = [row[:rank] for row in U]
        V = [row[:rank] for row in V]
        W = [row[:rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
