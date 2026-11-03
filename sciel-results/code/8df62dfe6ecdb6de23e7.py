"""Rewritten program for matrix multiplication 4,4,8 rank minimization."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_TIMEOUT = 150

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Rank 112: Strassen (7) x 2 halves x 8 schoolbook inner
    rank = 112
    U = [[0] * rank for _ in range(n * m)]  # 16 x 112
    V = [[0] * rank for _ in range(m * p)]  # 32 x 112
    W = [[0] * rank for _ in range(p * n)]  # 32 x 112
    
    # A index: A is 4x4, row-major: a_idx(i,j) = i*4+j
    # B index: B is 4x8, row-major: b_idx(j,k) = j*8+k
    # W index: W[k*n+i] = W[k*4+i] for C[i,k]
    
    def a_idx(i, j):
        return i * 4 + j
    
    def b_idx(j, k):
        return j * 8 + k
    
    def w_idx(i, k):
        # W[k*n + i]
        return k * 4 + i
    
    col = 0
    
    # For each half of B (4x4), apply Strassen
    for half in range(2):
        # B half offset in column index
        bcol_off = half * 4
        
        # Strassen linear forms for A (4x4 as 2x2 blocks of 2x2)
        # Block (br, bc) entry (ir, ic) -> A index (br*2+ir, bc*2+ic)
        # A block (br, bc):
        #   entry (0,0) = A[(br*2), (bc*2)]
        #   entry (0,1) = A[(br*2), (bc*2+1)]
        #   entry (1,0) = A[(br*2+1), (bc*2)]
        #   entry (1,1) = A[(br*2+1), (bc*2+1)]
        
        # For each Strassen product s (0..6), define:
        #   X[ir][ic] = linear form of A (4 entries, each a sum of A elements)
        #   Y[ir][ic] = linear form of B (4 entries, each a sum of B elements)
        
        # Strassen products (using 1-indexed blocks: 11,12,21,22):
        # s=0: M1 = (A11+A22)(B11+B22)
        # s=1: M2 = (A21+A22)B11
        # s=2: M3 = A11(B12-B22)
        # s=3: M4 = A22(B21-B11)
        # s=4: M5 = (A11+A12)B22
        # s=5: M6 = (A21-A11)(B11+B12)
        # s=6: M7 = (A12-A22)(B21+B22)
        
        # For each s, X is a 2x2 matrix of linear forms of A
        # Y is a 2x2 matrix of linear forms of B
        
        # Let's define X and Y as lists of (coeff, a_index) and (coeff, b_index)
        # X[ir][ic] = list of (coeff, a_idx)
        # Y[ir][ic] = list of (coeff, b_idx)
        
        # Block indices: block 11 = (0,0), 12 = (0,1), 21 = (1,0), 22 = (1,1)
        # Entry (ir, ic) in block (br, bc) -> A row br*2+ir, col bc*2+ic
        
        def a_entry(br, bc, ir, ic):
            """Return (coeff=1, a_index) for A entry in block (br,bc) at position (ir,ic)"""
            return (1, a_idx(br*2+ir, bc*2+ic))
        
        def b_entry(br, bc, ir, ic):
            """Return (coeff=1, b_index) for B entry in block (br,bc) at position (ir,ic) for given half"""
            return (1, b_idx(br*2+ir, bcol_off + bc*2 + ic))
        
        # Define the 7 Strassen products
        # Each product: (X_forms, Y_forms, C_contributions)
        # X_forms[ir][ic] = list of (coeff, a_idx)
        # Y_forms[ir][ic] = list of (coeff, b_idx)
        # C_contributions: dict mapping (c_br, c_bc, cir, cic) -> coeff
        
        products = []
        
        # s=0: M1 = (A11+A22)(B11+B22)
        # X = A11 + A22: X[ir][ic] = A11[ir][ic] + A22[ir][ic]
        X0 = [[a_entry(0,0,ir,ic) + a_entry(1,1,ir,ic) for ic in range(2)] for ir in range(2)]
        # Actually, let's build these properly
        X0 = []
        for ir in range(2):
            row = []
            for ic in range(2):
                row.append([(1, a_idx(0*2+ir, 0*2+ic)), (1, a_idx(1*2+ir, 1*2+ic))])
            X0.append(row)
        Y0 = []
        for ir in range(2):
            row = []
            for ic in range(2):
                row.append([(1, b_idx(0*2+ir, bcol_off+0*2+ic)), (1, b_idx(1*2+ir, 1*2+ic))])
            Y0.append(row)
        # C11 += M1, C12 -= M5 (handled separately), etc.
        # M1 contributes: C11 += M1, C22 -= M1 (with sign from formula)
        # C11 = M1 + M4 - M5 + M7
        # C22 = M1 - M2 + M3 + M6
        # So M1: C11 += M1, C22 += M1
        C0 = {(0,0): 1, (1,1): 1}
        products.append((X0, Y0, C0))
        
        # s=1: M2 = (A21+A22)B11
        X1 = []
        for ir in range(2):
            row = []
            for ic in range(2):
                row.append([(1, a_idx(1*2+ir, 0*2+ic)), (1, a_idx(1*2+ir, 1*2+ic))])
            X1.append(row)
        Y1 = []
        for ir in range(2):
            row = []
            for ic in range(2):
                row.append([(1, b_idx(0*2+ir, bcol_off+0*2+ic))])
            Y1.append(row)
        # C21 += M2, C22 -= M2
        C1 = {(1,0): 1, (1,1): -1}
        products.append((X1, Y1, C1))
        
        # s=2: M3 = A11(B12-B22)
        X2 = []
        for ir in range(2):
            row = []
            for ic in range(2):
                row.append([(1, a_idx(0*2+ir, 0*2+ic))])
            X2.append(row)
        Y2 = []
        for ir in range(2):
            row = []
            for ic in range(2):
                row.append([(1, b_idx(0*2+ir, bcol_off+1*2+ic)), (-1, b_idx(1*2+ir, 1*2+ic))])
            Y2.append(row)
        # C12 += M3, C22 += M3
        C2 = {(0,1): 1, (1,1): 1}
        products.append((X2, Y2, C2))
        
        # s=3: M4 = A22(B21-B11)
        X3 = []
        for ir in range(2):
            row = []
            for ic in range(2):
                row.append([(1, a_idx(1*2+ir, 1*2+ic))])
            X3.append(row)
        Y3 = []
        for ir in range(2):
            row = []
            for ic in range(2):
                row.append([(1, b_idx(1*2+ir, bcol_off+0*2+ic)), (-1, b_idx(0*2+ir, 0*2+ic))])
            Y3.append(row)
        # C11 += M4, C21 += M4
        C3 = {(0,0): 1, (1,0): 1}
        products.append((X3, Y3, C3))
        
        # s=4: M5 = (A11+A12)B22
        X4 = []
        for ir in range(2):
            row = []
            for ic in range(2):
                row.append([(1, a_idx(0*2+ir, 0*2+ic)), (1, a_idx(0*2+ir, 1*2+ic))])
            X4.append(row)
        Y4 = []
        for ir in range(2):
            row = []
            for ic in range(2):
                row.append([(1, b_idx(1*2+ir, bcol_off+1*2+ic))])
            Y4.append(row)
        # C12 += M5, C11 -= M5
        C4 = {(0,1): 1, (0,0): -1}
        products.append((X4, Y4, C4))
        
        # s=5: M6 = (A21-A11)(B11+B12)
        X5 = []
        for ir in range(2):
            row = []
            for ic in range(2):
                row.append([(1, a_idx(1*2+ir, 0*2+ic)), (-1, a_idx(0*2+ir, 0*2+ic))])
            X5.append(row)
        Y5 = []
        for ir in range(2):
            row = []
            for ic in range(2):
                row.append([(1, b_idx(0*2+ir, bcol_off+0*2+ic)), (1, b_idx(0*2+ir, bcol_off+1*2+ic))])
            Y5.append(row)
        # C22 += M6
        C5 = {(1,1): 1}
        products.append((X5, Y5, C5))
        
        # s=6: M7 = (A12-A22)(B21+B22)
        X6 = []
        for ir in range(2):
            row = []
            for ic in range(2):
                row.append([(1, a_idx(0*2+ir, 1*2+ic)), (-1, a_idx(1*2+ir, 1*2+ic))])
            X6.append(row)
        Y6 = []
        for ir in range(2):
            row = []
            for ic in range(2):
                row.append([(1, b_idx(1*2+ir, bcol_off+0*2+ic)), (1, b_idx(1*2+ir, bcol_off+1*2+ic))])
            Y6.append(row)
        # C11 += M7
        C6 = {(0,0): 1}
        products.append((X6, Y6, C6))
        
        # Now for each Strassen product, expand into 8 schoolbook terms
        for s in range(7):
            Xs, Ys, Cs = products[s]
            # 8 schoolbook terms for 2x2x2:
            # Term (tr, tc, ur, uc): X[tr][tc] * Y[ur][uc]
            # Wait, for matrix product Z = X*Y:
            # Z[ir][ic] = sum_k X[ir][k] * Y[k][ic]
            # So the 8 terms are:
            # (0,0,0,0), (0,1,0,1), (0,0,1,0), (0,1,1,1) for Z[0][0], Z[0][1], Z[1][0], Z[1][1]
            # More precisely:
            # Z[0][0] = X[0][0]*Y[0][0] + X[0][1]*Y[1][0]
            # Z[0][1] = X[0][0]*Y[0][1] + X[0][1]*Y[1][1]
            # Z[1][0] = X[1][0]*Y[0][0] + X[1][1]*Y[1][0]
            # Z[1][1] = X[1][0]*Y[0][1] + X[1][1]*Y[1][1]
            
            # Each term: X_entry * Y_entry
            # X_entry is a linear form of A, Y_entry is a linear form of B
            # The product of two linear forms gives a bilinear term
            
            # For each of the 8 terms, we add one rank-1 component
            # Term index: (z_ir, z_ic, x_ic, y_ir) where:
            #   z_ir, z_ic: which entry of Z (the 2x2 result)
            #   x_ic: which column of X (0 or 1)
            #   y_ir: which row of Y (0 or 1)
            
            for z_ir in range(2):
                for z_ic in range(2):
                    for k in range(2):  # sum over k
                        # Term: X[z_ir][k] * Y[k][z_ic]
                        x_entry = Xs[z_ir][k]  # list of (coeff, a_idx)
                        y_entry = Ys[k][z_ic]  # list of (coeff, b_idx)
                        
                        # This is a bilinear form: sum_{a,b} x_coeff_a * y_coeff_b * A[a] * B[b]
                        # For each pair (a_idx_a, b_idx_b), we add to U, V, W
                        # U[a_idx_a] += x_coeff_a
                        # V[b_idx_b] += y_coeff_b  
                        # W[c_idx] += 1 (for the appropriate C position)
                        
                        # But wait - this creates multiple rank-1 terms per schoolbook term
                        # if the linear forms have multiple components!
                        
                        # Hmm, this is the issue. If X[z_ir][k] = A[i1][j1] + A[i2][j2],
                        # then X[z_ir][k] * Y[k][z_ic] is a sum of bilinear terms,
                        # each of which needs its own rank-1 component.
                        
                        # So the rank is NOT simply 7*8*2 = 112!
                        # It depends on the sparsity of the linear forms.
                        
                        # Let me count: For M1 = (A11+A22)(B11+B22):
                        # X[0][0] = A[0][0] + A[2][2] (2 terms)
                        # Y[0][0] = B[0][0] + B[2][2] (2 terms)
                        # Product has 4 bilinear terms -> 4 rank-1 components
                        # 
                        # This makes the total rank much higher than 112!
                        
                        # I think I'm overcomplicating this. Let me reconsider.
                        # 
                        # The standard way to get rank 112 is:
                        # - Use Strassen at the 4x4x4 level (rank 7 for the block structure)
                        # - But each "product" in Strassen is a 2x2x2 matrix multiplication
                        # - A 2x2x2 matrix multiplication has rank 8
                        # - So total = 2 (halves) * 7 (Strassen) * 8 (2x2x2) = 112
                        #
                        # The key is that the linear forms in Strassen are applied to the
                        # ENTIRE 2x2 blocks, not entry-by-entry. The rank-8 decomposition
                        # of 2x2x2 is applied to the block-level matrices.
                        #
                        # So for M1 = (A11+A22)(B11+B22):
                        # Let X = A11+A22 (a 2x2 matrix), Y = B11+B22 (a 2x2 matrix)
                        # Z = X*Y (2x2 matrix)
                        # The 8 rank-1 terms for 2x2x2 are:
                        # Z[0][0] = X[0][0]*Y[0][0] + X[0][1]*Y[1][0]
                        # etc.
                        # But X[0][0] = A[0][0] + A[2][2] is a linear form
                        # So X[0][0]*Y[0][0] = (A[0][0]+A[2][2])*(B[0][0]+B[2][2])
                        # = A[0][0]*B[0][0] + A[0][0]*B[2][2] + A[2][2]*B[0][0] + A[2][2]*B[2][2]
                        # This is 4 separate bilinear terms, each needing its own rank-1 slot!
                        #
                        # So the total rank is NOT 112. It's higher.
                        #
                        # I think the confusion is that "rank 7" for Strassen means 7 scalar
                        # multiplications of 2x2 matrices, but each 2x2 matrix multiplication
                        # still requires 8 scalar multiplications. The "rank" in tensor
                        # decomposition refers to the number of scalar multiplications.
                        #
                        # So for 4x4x4 with Strassen: 7 * 8 = 56 scalar multiplications
                        # For 4x4x8 with two halves: 2 * 56 = 112
                        #
                        # But in the tensor decomposition, each scalar multiplication
                        # corresponds to one rank-1 term. The linear forms can have
                        # multiple terms, but that's fine - each rank-1 term is
                        # u * v * w where u, v, w are vectors.
                        #
                        # So for X[0][0]*Y[0][0] where X[0][0] = A[0][0] + A[2][2]:
                        # This is NOT one rank-1 term. It's the product of two linear forms,
                        # which expands to multiple bilinear terms.
                        #
                        # Wait, no. In tensor decomposition, a rank-1 term is:
                        # sum_{i,j,k} T[i,j,k] = u[i]*v[j]*w[k]
                        # 
                        # For matrix multiplication, T[i,j,k] = A[i,j]*B[j,k] (for the relevant indices)
                        # A rank-1 term corresponds to: u[i]*v[j]*w[k] where
                        # u is a vector over (i,j) indices of A, v over (j,k) of B, w over (i,k) of C.
                        #
                        # So if I have the bilinear term (A[0][0] + A[2][2]) * (B[0][0] + B[2][2]),
                        # this is NOT a single rank-1 term. It's a sum of 4 rank-1 terms:
                        # A[0][0]*B[0][0], A[0][0]*B[2][2], A[2][2]*B[0][0], A[2][2]*B[2][2]
                        #
                        # Each of these contributes to specific entries of C.
                        #
                        # So my original approach of expanding each bilinear term into
                        # its constituent scalar products is correct, but the total rank
                        # will be higher than 112.
                        #
                        # Let me count for M1 = (A11+A22)(B11+B22):
                        # X has 4 entries, each with 2 A-terms. Y has 4 entries, each with 2 B-terms.
                        # 8 schoolbook terms, each is a product of an X-entry (2 terms) and Y-entry (2 terms)
                        # = 4 scalar products per schoolbook term
                        # Total for M1: 8 * 4 = 32 rank-1 terms
                        #
                        # This is way more than 56 total for 4x4x4.
                        #
                        # I think I'm fundamentally misunderstanding something. Let me reconsider.
                        #
                        # Actually, I think the issue is that in the standard Strassen algorithm,
                        # the "rank 7" refers to 7 multiplications of 2x2 MATRICES, not 7
                        # scalar multiplications. Each 2x2 matrix multiplication is itself
                        # a tensor of rank 8. So the total rank for 4x4x4 is 7*8=56.
                        #
                        # But when we write the tensor decomposition explicitly, each of the
                        # 56 rank-1 terms corresponds to ONE scalar multiplication. The linear
                        # forms in u, v, w can have multiple nonzero entries, but that's fine.
                        #
                        # For example, for the first Strassen product M1 = (A11+A22)(B11+B22),
                        # the first schoolbook term is X[0][0]*Y[0][0] where
                        # X[0][0] = A[0][0] + A[2][2] and Y[0][0] = B[0][0] + B[2][2].
                        #
                        # This gives the bilinear form (A[0][0]+A[2][2])(B[0][0]+B[2][2])
                        # which contributes to C[0][0] and C[2][2] (and possibly others).
                        #
                        # But this is NOT a rank-1 tensor! A rank-1 tensor is u[i]*v[j]*w[k].
                        # The product of two linear forms is a sum of rank-1 tensors.
                        #
                        # So I cannot represent (A[0][0]+A[2][2])(B[0][0]+B[2][2]) as a single
                        # rank-1 term in the tensor decomposition.
                        #
                        # This means the total rank is NOT 112. It's higher.
                        #
                        # I think the correct way to get rank 112 is to use a different
                        # approach: instead of Strassen with linear forms, use a direct
                        # rank-56 decomposition of the 4x4x4 tensor. But finding such a
                        # decomposition is non-trivial.
                        #
                        # Alternatively, maybe I should just use schoolbook (rank 128)
                        # and accept that 112 is not easily achievable with this approach.
                        #
                        # Let me just return the schoolbook decomposition.
                        
                        pass
    
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

# EVOLVE-BLOCK-END
