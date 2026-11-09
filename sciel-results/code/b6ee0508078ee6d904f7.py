"""Matrix multiplication 4x4x4 - rank minimization via Strassen-based decomposition."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # We use the Strassen-based rank-18 decomposition for 4x4x4 matrix multiplication.
    # 
    # The idea: split each 4x4 matrix into 2x2 blocks of 2x2 submatrices.
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = A*B = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
    #            [A10*B00 + A11*B10, A10*B01 + A11*B11]]
    #
    # Using Strassen's algorithm for the 2x2 block multiplication:
    # M1 = (A00+A11)*(B00+B11)  -> 2 multiplications of 2x2
    # M2 = (A10+A11)*B00        -> 1 multiplication of 2x2
    # M3 = A00*(B01-B11)        -> 1 multiplication of 2x2
    # M4 = A11*(B10-B00)        -> 1 multiplication of 2x2
    # M5 = (A00+A01)*B11        -> 1 multiplication of 2x2
    # M6 = (A10-A00)*(B00+B01)  -> 1 multiplication of 2x2
    # M7 = (A01-A11)*(B10+B11)  -> 1 multiplication of 2x2
    #
    # Each 2x2 multiplication is done via Strassen: 7 scalar multiplications.
    # Total: 7 * 7 = 49 scalar multiplications.
    #
    # But we can do better. Let's use the rank-18 decomposition directly.
    # Actually, the known minimum rank for 4x4x4 is 49 (from Strassen), 
    # but there might be lower ranks. Let me use a known rank-49 decomposition.
    #
    # Actually, let me just implement the Strassen-based rank-49 decomposition.
    # For each of the 7 Strassen products at the block level, we use 7 scalar 
    # multiplications at the sub-block level, giving 49 total.
    #
    # The factor matrices U, V, W each have shape (16, rank) for n*m=16, m*p=16, p*n=16.
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] where i,j,k in 0..3.
    
    rank = 49
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    # Map: index = i*4 + j for U (i is row of A, j is col of A)
    #       index = j*4 + k for V (j is row of B, k is col of B)
    #       index = k*4 + i for W (k is row of C, i is col of C)
    #
    # Block structure:
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # B00 = B[0:2, 0:2], B01 = B[0:2, 2:4], B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Strassen block products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M5
    #
    # Now for each 2x2 block product, we use Strassen with 7 scalar mults.
    # For a 2x2 product X*Y where X = [[x00,x01],[x10,x11]], Y = [[y00,y01],[y10,y11]]:
    # s1 = (x00+x11)*(y00+y11)
    # s2 = (x10+x11)*y00
    # s3 = x00*(y01-y11)
    # s4 = x11*(y10-y00)
    # s5 = (x00+x01)*y11
    # s6 = (x10-x00)*(y00+y01)
    # s7 = (x01-x11)*(y10+y11)
    # Result: [[s1-s4+s6+s7, s3+s5], [s2+s4, s1-s2+s3+s5]]
    
    def strassen_2x2_indices(X_rows, X_cols, Y_rows, Y_cols):
        """
        Return the 7 rank-1 terms for computing X*Y where X and Y are 2x2.
        X is indexed by (row, col) in the original 4x4, same for Y.
        Each term is (u_idx, v_idx, w_idx) where u_idx = row*4+col for U,
        v_idx = row*4+col for V, w_idx = col*4+row for W.
        """
        x00, x01 = X_rows[0], X_cols[0]
        x10, x11 = X_rows[1], X_cols[0]
        y00, y01 = Y_rows[0], Y_cols[0]
        y10, y11 = Y_rows[1], Y_cols[0]
        
        # s1 = (x00+x11)*(y00+y11)
        # s2 = (x10+x11)*y00
        # s3 = x00*(y01-y11)
        # s4 = x11*(y10-y00)
        # s5 = (x00+x01)*y11
        # s6 = (x10-x00)*(y00+y01)
        # s7 = (x01-x11)*(y10+y11)
        
        terms = []
        # Each Strassen scalar mult becomes a rank-1 term
        # s1: (x00+x11) * (y00+y11) -> two U entries, two V entries
        # We need to expand (x00+x11)*(y00+y11) into rank-1 terms.
        # This requires 4 rank-1 terms: x00*y00, x00*y11, x11*y00, x11*y11
        # But wait, that's not right. The Strassen trick is that we compute
        # the 7 products, each of which is a single scalar multiplication.
        # So each "s" is ONE scalar multiplication, meaning ONE rank-1 term.
        #
        # Actually no. Let me reconsider. In the standard tensor decomposition,
        # each rank-1 term corresponds to one scalar multiplication. The Strassen
        # algorithm uses 7 multiplications for 2x2, so 7 rank-1 terms.
        #
        # For s1 = (x00+x11)*(y00+y11):
        # This is a product of (sum of two A-entries) and (sum of two B-entries).
        # As a rank-1 term: U has 1 in positions x00 and x11, V has 1 in y00 and y11,
        # and W has the result placed in the appropriate C positions.
        #
        # Wait, that's not how it works. A rank-1 tensor term is:
        # U[row, term] * V[col, term] -> contributes to W[out, term]
        # where U[row, term] is the coefficient of A[row] in this term,
        # V[col, term] is the coefficient of B[col] in this term,
        # W[out, term] is the coefficient that multiplies the product to contribute to C[out].
        #
        # For s1 = (x00+x11)*(y00+y11):
        # U[x00, t] = 1, U[x11, t] = 1
        # V[y00, t] = 1, V[y11, t] = 1
        # W[c_out, t] = 1 for the output position c_out.
        # But (x00+x11)*(y00+y11) = x00*y00 + x00*y11 + x11*y00 + x11*y11
        # This contributes to MULTIPLE output positions!
        #
        # I think I'm confusing myself. Let me restart with the correct formulation.
        #
        # In a rank-r decomposition of the matrix multiplication tensor:
        # T(i,j,k) = sum_{c=1}^{r} U[i,c] * V[j,c] * W[k,c]
        # where T is the bilinear map: (A,B) -> C, with C[k,i] = sum_j A[i,j]*B[j,k]
        # Wait, the convention is W[k*n+i, c] where the output is indexed by (k,i) 
        # with k being the row of C and i being the column of C.
        # Actually re-reading: "output factor follows the official trace convention W[k*n+i,r]"
        # And the decomposition is: C = sum_c U_c * outer(V_c, W_c) or similar.
        #
        # Standard form: C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T has indices (i,j,k) with T[i,j,k] = delta(i,k) in some sense...
        # No. The matrix multiplication tensor M has M[i,j,k] = 1 if i==k... no.
        #
        # The bilinear map is: f(A,B) = A*B where (A*B)[i,k] = sum_j A[i,j] B[j,k]
        # As a 3-tensor: T[i,j,k] such that f(A,B) = sum_{i,j,k} A[i,j] B[j,k] T[i,j,k] * E_{i,k}
        # So T[i,j,k] = 1 for all i,j,k? No, that's not right either.
        #
        # Actually: (A*B)[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor entry T[i,j,k] = 1 means that A[i,j]*B[j,k] contributes to C[i,k].
        # The tensor is T[i,j,k] = delta_{i,?} ... 
        #
        # Let me use the standard formulation. The matrix multiplication tensor 
        # for n x n is a tensor in (C^n)^* x (C^n)^* x (C^n) or similar.
        # The rank decomposition: M = sum_{c=1}^r u_c \otimes v_c \otimes w_c
        # where (u_c \otimes v_c \otimes w_c)(A,B) = (u_c^T A)(v_c^T B) w_c
        # and we need sum_c (u_c^T A)(v_c^T B) w_c = A*B.
        #
        # In matrix form: U is (n^2 x r), V is (n^2 x r), W is (n^2 x r)
        # C = U * diag(V^T * A_flattened) * W^T or something.
        #
        # The condition is: for all A, B: 
        # sum_c U[:,c] * (V[:,c]^T * B) * W[:,c]^T ... no.
        #
        # Standard: C[i,k] = sum_c U[i*m+j? ...] 
        #
        # Let me use the interface specification directly:
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # The decomposition satisfies: C[k*n+i? or i*p+k?] = sum_c ...
        #
        # From the schoolbook example:
        # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
        # for c = i*m*p + j*p + k (or similar indexing)
        #
        # So the reconstruction is: C[k,i] = sum_{j,c} A[i,j] * B[j,k] * (contribution from c)
        # With U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c.
        # In schoolbook: for each (i,j,k), there's exactly one c with all three = 1.
        # So C[k,i] = sum_{i,j,k: W[k*n+i,c]=1} A[i,j]*B[j,k] where c corresponds to (i,j,k).
        # This gives C[k,i] = A[i,?]*B[?,k]... 
        #
        # Actually: C[k,i] = sum_c U[i*m+j, c] * A[i,j] ... no.
        # The formula must be: C[k,i] = sum_{j} sum_c A[i,j] * U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * B[j,k]
        # No wait, that doesn't dimensionally work if U,V,W are just coefficients.
        #
        # I think the correct interpretation is:
        # Flatten A to a vector a of length n*m: a[i*m+j] = A[i,j]
        # Flatten B to a vector b of length m*p: b[j*p+k] = B[j,k]
        # Flatten C to a vector c of length p*n: c[k*n+i] = C[k,i]
        # Then c = sum_c (U[:,c]^T a) * (V[:,c]^T b) * W[:,c]
        # i.e., c = W * diag(v) * U^T * a where v = V^T b
        # Or: c[k] = sum_i W[k,c] * U[i,c] * a[i] * V[j,c] * b[j] summed appropriately.
        #
        # More precisely: c = sum_{c=1}^{rank} W[:,c] * (U[:,c]^T @ a) * (V[:,c]^T @ b)
        #
        # So: C[k,i] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
        #
        # For schoolbook: c corresponds to (i0,j0,k0)
        # W[k*n+i, c] = 1 iff k=k0 and i=i0
        # U[i*m+j, c] = 1 iff i=i0 and j=j0  
        # V[j*p+k, c] = 1 iff j=j0 and k=k0
        # So C[k,i] = sum_c W[k*n+i,c] * (U[:,c]^T a) * (V[:,c]^T b)
        # For a given (k,i), the term is non-zero when k=k0, i=i0, and then
        # U gives A[i0,j0], V gives B[j0,k0], so C[k0,i0] = A[i0,j0]*B[j0,k0]
        # Summing over all j0 (each with its own c): C[k,i] = sum_j A[i,j]*B[j,k]. Correct!
        
    # OK so now I understand the formulation. Let me implement the rank-49 Strassen.
    
    # Block indices in the 4x4 matrices:
    # A row blocks: 0 (rows 0-1), 1 (rows 2-3)
    # A col blocks: 0 (cols 0-1), 2 (cols 2-3)
    # Similarly for B.
    
    # A00: rows 0-1, cols 0-1 -> U indices: 0*4+0, 0*4+1, 1*4+0, 1*4+1 = 0,1,4,5
    # A01: rows 0-1, cols 2-3 -> U indices: 0*4+2, 0*4+3, 1*4+2, 1*4+3 = 2,3,6,7
    # A10: rows 2-3, cols 0-1 -> U indices: 2*4+0, 2*4+1, 3*4+0, 3*4+1 = 8,9,12,13
    # A11: rows 2-3, cols 2-3 -> U indices: 2*4+2, 2*4+3, 3*4+2, 3*4+3 = 10,11,14,15
    
    # B00: rows 0-1, cols 0-1 -> V indices: 0*4+0, 0*4+1, 1*4+0, 1*4+1 = 0,1,4,5
    # B01: rows 0-1, cols 2-3 -> V indices: 0*4+2, 0*4+3, 1*4+2, 1*4+3 = 2,3,6,7
    # B10: rows 2-3, cols 0-1 -> V indices: 2*4+0, 2*4+1, 3*4+0, 3*4+1 = 8,9,12,13
    # B11: rows 2-3, cols 2-3 -> V indices: 2*4+2, 2*4+3, 3*4+2, 3*4+3 = 10,11,14,15
    
    # C00: rows 0-1, cols 0-1 -> W indices: 0*4+0, 0*4+1, 1*4+0, 1*4+1 = 0,1,4,5
    # C01: rows 0-1, cols 2-3 -> W indices: 0*4+2, 0*4+3, 1*4+2, 1*4+3 = 2,3,6,7
    # C10: rows 2-3, cols 0-1 -> W indices: 2*4+0, 2*4+1, 3*4+0, 3*4+1 = 8,9,12,13
    # C11: rows 2-3, cols 2-3 -> W indices: 2*4+2, 2*4+3, 3*4+2, 3*4+3 = 10,11,14,15
    
    # For each Strassen block product, we need 7 scalar multiplications.
    # Each scalar multiplication (x*y) where x is a linear combo of A-block entries
    # and y is a linear combo of B-block entries contributes to specific C-block entries.
    #
    # Let's define a helper to add a rank-1 term.
    
    def add_term(col, u_entries, v_entries, w_entries):
        """Add a rank-1 term. u_entries: dict {U_index: coeff}, etc."""
        for idx, coeff in u_entries.items():
            U[idx, col] += coeff
        for idx, coeff in v_entries.items():
            V[idx, col] += coeff
        for idx, coeff in w_entries.items():
            W[idx, col] += coeff
    
    col = 0
    
    # Define block entry lists
    A00_U = [0, 1, 4, 5]   # A[0,0], A[0,1], A[1,0], A[1,1]
    A01_U = [2, 3, 6, 7]   # A[0,2], A[0,3], A[1,2], A[1,3]
    A10_U = [8, 9, 12, 13] # A[2,0], A[2,1], A[3,0], A[3,1]
    A11_U = [10, 11, 14, 15] # A[2,2], A[2,3], A[3,2], A[3,3]
    
    B00_V = [0, 1, 4, 5]   # B[0,0], B[0,1], B[1,0], B[1,1]
    B01_V = [2, 3, 6, 7]   # B[0,2], B[0,3], B[1,2], B[1,3]
    B10_V = [8, 9, 12, 13] # B[2,0], B[2,1], B[3,0], B[3,1]
    B11_V = [10, 11, 14, 15] # B[2,2], B[2,3], B[3,2], B[3,3]
    
    C00_W = [0, 1, 4, 5]   # C[0,0], C[0,1], C[1,0], C[1,1]
    C01_W = [2, 3, 6, 7]   # C[0,2], C[0,3], C[1,2], C[1,3]
    C10_W = [8, 9, 12, 13] # C[2,0], C[2,1], C[3,0], C[3,1]
    C11_W = [10, 11, 14, 15] # C[2,2], C[2,3], C[3,2], C[3,3]
    
    # For a 2x2 block product X*Y -> Z, where X is an A-block (rows x, cols y)
    # and Y is a B-block, the Strassen decomposition gives:
    # Z[0,0] = s1 - s4 + s6 + s7
    # Z[0,1] = s3 + s5
    # Z[1,0] = s2 + s4
    # Z[1,1] = s1 - s2 + s3 + s5
    #
    # where s1 through s7 are the 7 Strassen products.
    # Each s_i is a product of a linear combo of X entries and a linear combo of Y entries.
    # Each s_i contributes to multiple Z entries with coefficients.
    #
    # For each s_i, we add ONE rank-1 term where:
    # U entries: the coefficients of X entries in the first factor of s_i
    # V entries: the coefficients of Y entries in the second factor of s_i
    # W entries: the coefficients in Z for this s_i (from the formulas above)
    
    def add_strassen_block_product(X_U_indices, Y_V_indices, Z_W_indices, z_coeffs):
        """
        X_U_indices: list of 4 U indices for the X block (row0col0, row0col1, row1col0, row1col1)
        Y_V_indices: list of 4 V indices for the Y block
        Z_W_indices: list of 4 W indices for the Z block
        z_coeffs: 7x4 matrix where z_coeffs[s][z] is the coefficient of s_s in Z[z]
        Returns the number of columns used.
        """
        global col
        for s in range(7):
            u_coeff = []
            v_coeff = []
            # Define the 7 Strassen factors
            if s == 0:  # s1 = (x00+x11)*(y00+y11)
                u_vals = [(0, 1), (3, 1)]  # x00 + x11
                v_vals = [(0, 1), (3, 1)]  # y00 + y11
            elif s == 1:  # s2 = (x10+x11)*y00
                u_vals = [(2, 1), (3, 1)]  # x10 + x11
                v_vals = [(0, 1)]          # y00
            elif s == 2:  # s3 = x00*(y01-y11)
                u_vals = [(0, 1)]          # x00
                v_vals = [(1, 1), (3, -1)] # y01 - y11
            elif s == 3:  # s4 = x11*(y10-y00)
                u_vals = [(3, 1)]          # x11
                v_vals = [(2, 1), (0, -1)] # y10 - y00
            elif s == 4:  # s5 = (x00+x01)*y11
                u_vals = [(0, 1), (1, 1)]  # x00 + x01
                v_vals = [(3, 1)]          # y11
            elif s == 5:  # s6 = (x10-x00)*(y00+y01)
                u_vals = [(2, 1), (0, -1)] # x10 - x00
                v_vals = [(0, 1), (1, 1)]  # y00 + y01
            elif s == 6:  # s7 = (x01-x11)*(y10+y11)
                u_vals = [(1, 1), (3, -1)] # x01 - x11
                v_vals = [(2, 1), (3, 1)]  # y10 + y11
            
            u_entries = {}
            for pos, coeff in u_vals:
                u_entries[X_U_indices[pos]] = u_entries.get(X_U_indices[pos], 0) + coeff
            
            v_entries = {}
            for pos, coeff in v_vals:
                v_entries[Y_V_indices[pos]] = v_entries.get(Y_V_indices[pos], 0) + coeff
            
            w_entries = {}
            for z_pos in range(4):
                coeff = z_coeffs[s][z_pos]
                if coeff != 0:
                    w_entries[Z_W_indices[z_pos]] = w_entries.get(Z_W_indices[z_pos], 0) + coeff
            
            add_term(col, u_entries, v_entries, w_entries)
            col += 1
    
    # Z coefficients for Strassen:
    # Z[0,0] = s1 - s4 + s6 + s7 -> coeffs for s0..s6: [1, 0, 0, -1, 0, 1, 1]
    # Z[0,1] = s3 + s5          -> coeffs: [0, 0, 1, 0, 1, 0, 0]
    # Z[1,0] = s2 + s4          -> coeffs: [0, 1, 0, 1, 0, 0, 0]
    # Z[1,1] = s1 - s2 + s3 + s5 -> coeffs: [1, -1, 1, 0, 1, 0, 0]
    Z_strassen = [
        [1, 0, 0, -1, 0, 1, 1],   # Z[0,0]
        [0, 0, 1, 0, 1, 0, 0],    # Z[0,1]
        [0, 1, 0, 1, 0, 0, 0],    # Z[1,0]
        [1, -1, 1, 0, 1, 0, 0],   # Z[1,1]
    ]
    
    # Now compute the 7 Strassen block products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M5
    #
    # Each M_i is a 2x2 block product. When we compute M_i via Strassen,
    # we get 7 rank-1 terms whose W coefficients are Z_strassen.
    # But the final C blocks are LINEAR COMBINATIONS of the M_i blocks.
    # So for each M_i, the contribution to C is: sum over M_i of (coeff * M_i).
    #
    # This means: for each of the 7 Strassen terms within M_i, the W coefficients
    # should be multiplied by the coefficient of M_i in the final C block.
    #
    # Let me restructure: for each M_i, and for each of its 7 Strassen sub-terms,
    # the W coefficient for a given C entry is:
    #   (coefficient of M_i in C_block) * (Z_strassen[s][z_pos])
    # where z_pos maps to the specific C entry.
    
    # Let me define this more carefully.
    # M1 contributes to: C00 (coeff +1), C11 (coeff +1)
    # M2 contributes to: C10 (coeff +1), C11 (coeff -1)
    # M3 contributes to: C01 (coeff +1), C11 (coeff +1)
    # M4 contributes to: C00 (coeff -1), C10 (coeff +1)
    # M5 contributes to: C01 (coeff +1), C11 (coeff +1)
    # M6 contributes to: C00 (coeff +1)
    # M7 contributes to: C00 (coeff +1)
    
    # For each M_i, the 7 Strassen terms produce Z entries (z0,z1,z2,z3) corresponding to
    # the 2x2 block. These map to specific 4x4 entries.
    
    # Let me just do it directly for each M_i.
    
    # Reset
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    col = 0
    
    def add_strassen_with_output_mapping(X_U, Y_V, output_map):
        """
        X_U: 4 U indices for the X 2x2 block
        Y_V: 4 V indices for the Y 2x2 block
        output_map: dict from z_pos (0-3) to list of (W_index, coeff)
        """
        global col
        for s in range(7):
            if s == 0:
                u_vals = [(0, 1), (3, 1)]
                v_vals = [(0, 1), (3, 1)]
            elif s == 1:
                u_vals = [(2, 1), (3, 1)]
                v_vals = [(0, 1)]
            elif s == 2:
                u_vals = [(0, 1)]
                v_vals = [(1, 1), (3, -1)]
            elif s == 3:
                u_vals = [(3, 1)]
                v_vals = [(2, 1), (0, -1)]
            elif s == 4:
                u_vals = [(0, 1), (1, 1)]
                v_vals = [(3, 1)]
            elif s == 5:
                u_vals = [(2, 1), (0, -1)]
                v_vals = [(0, 1), (1, 1)]
            elif s == 6:
                u_vals = [(1, 1), (3, -1)]
                v_vals = [(2, 1), (3, 1)]
            
            u_entries = {}
            for pos, coeff in u_vals:
                u_entries[X_U[pos]] = u_entries.get(X_U[pos], 0) + coeff
            v_entries = {}
            for pos, coeff in v_vals:
                v_entries[Y_V[pos]] = v_entries.get(Y_V[pos], 0) + coeff
            
            w_entries = {}
            for z_pos in range(4):
                z_coeff = Z_strassen[s][z_pos]
                if z_coeff != 0 and z_pos in output_map:
                    for w_idx, out_coeff in output_map[z_pos]:
                        total = z_coeff * out_coeff
                        if total != 0:
                            w_entries[w_idx] = w_entries.get(w_idx, 0) + total
            
            add_term(col, u_entries, v_entries, w_entries)
            col += 1
    
    # M1 = (A00+A11)*(B00+B11)
    # First factor: A00 + A11, second factor: B00 + B11
    # This is NOT a simple 2x2 block product of single blocks. 
    # (A00+A11) is a 2x2 matrix = A00 + A11, (B00+B11) is a 2x2 matrix.
    # The product (A00+A11)*(B00+B11) requires computing the 2x2 matrix product
    # of (A00+A11) and (B00+B11).
    #
    # The entries of (A00+A11):
    # [A00[0,0]+A11[0,0], A00[0,1]+A11[0,1]]   [A[0,0]+A[2,2], A[0,1]+A[2,3]]
    # [A00[1,0]+A11[1,0], A00[1,1]+A11[1,1]] = [A[1,0]+A[3,2], A[1,1]+A[3,3]]
    #
    # Similarly for B00+B11.
    #
    # So for Strassen on this product, X_U would be:
    # X[0,0] = A[0,0]+A[2,2] -> U indices 0 and 10, both coeff 1
    # X[0,1] = A[0,1]+A[2,3] -> U indices 1 and 11, both coeff 1
    # X[1,0] = A[1,0]+A[3,2] -> U indices 4 and 14, both coeff 1
    # X[1,1] = A[1,1]+A[3,3] -> U indices 5 and 15, both coeff 1
    #
    # This means each "X entry" is a sum of two A entries. When we do Strassen,
    # the u_vals reference positions 0-3 in the X block, but each position maps to
    # TWO U indices.
    #
    # Let me generalize: instead of X_U being 4 indices, make it 4 lists of (index, coeff) pairs.
    
    def add_strassen_general(X_entries, Y_entries, output_map):
        """
        X_entries: list of 4 items, each is a list of (U_index, coeff) pairs
        Y_entries: list of 4 items, each is a list of (V_index, coeff) pairs
        output_map: dict from z_pos (0-3) to list of (W_index, coeff)
        """
        global col
        for s in range(7):
            if s == 0:
                u_pos_coeffs = [(0, 1), (3, 1)]
                v_pos_coeffs = [(0, 1), (3, 1)]
            elif s == 1:
                u_pos_coeffs = [(2, 1), (3, 1)]
                v_pos_coeffs = [(0, 1)]
            elif s == 2:
                u_pos_coeffs = [(0, 1)]
                v_pos_coeffs = [(1, 1), (3, -1)]
            elif s == 3:
                u_pos_coeffs = [(3, 1)]
                v_pos_coeffs = [(2, 1), (0, -1)]
            elif s == 4:
                u_pos_coeffs = [(0, 1), (1, 1)]
                v_pos_coeffs = [(3, 1)]
            elif s == 5:
                u_pos_coeffs = [(2, 1), (0, -1)]
                v_pos_coeffs = [(0, 1), (1, 1)]
            elif s == 6:
                u_pos_coeffs = [(1, 1), (3, -1)]
                v_pos_coeffs = [(2, 1), (3, 1)]
            
            u_entries = {}
            for pos, coeff in u_pos_coeffs:
                for idx, sub_coeff in X_entries[pos]:
                    total = coeff * sub_coeff
                    u_entries[idx] = u_entries.get(idx, 0) + total
            
            v_entries = {}
            for pos, coeff in v_pos_coeffs:
                for idx, sub_coeff in Y_entries[pos]:
                    total = coeff * sub_coeff
                    v_entries[idx] = v_entries.get(idx, 0) + total
            
            w_entries = {}
            for z_pos in range(4):
                z_coeff = Z_strassen[s][z_pos]
                if z_coeff != 0 and z_pos in output_map:
                    for w_idx, out_coeff in output_map[z_pos]:
                        total = z_coeff * out_coeff
                        if total != 0:
                            w_entries[w_idx] = w_entries.get(w_idx, 0) + total
            
            add_term(col, u_entries, v_entries, w_entries)
            col += 1
    
    # Define X and Y entries for each M_i
    
    # M1 = (A00+A11)*(B00+B11)
    # X entries (A00+A11):
    M1_X = [
        [(0, 1), (10, 1)],   # [0,0]: A[0,0]+A[2,2]
        [(1, 1), (11, 1)],   # [0,1]: A[0,1]+A[2,3]
        [(4, 1), (14, 1)],   # [1,0]: A[1,0]+A[3,2]
        [(5, 1), (15, 1)],   # [1,1]: A[1,1]+A[3,3]
    ]
    M1_Y = [
        [(0, 1), (10, 1)],   # [0,0]: B[0,0]+B[2,2]
        [(1, 1), (11, 1)],   # [0,1]: B[0,1]+B[2,3]
        [(4, 1), (14, 1)],   # [1,0]: B[1,0]+B[3,2]
        [(5, 1), (15, 1)],   # [1,1]: B[1,1]+B[3,3]
    ]
    # M1 contributes to C00 (coeff +1) and C11 (coeff +1)
    # z_pos 0->C00[0,0]=W[0], 1->C00[0,1]=W[1], 2->C00[1,0]=W[4], 3->C00[1,1]=W[5]
    # AND z_pos 0->C11[0,0]=W[10], 1->C11[0,1]=W[11], 2->C11[1,0]=W[14], 3->C11[1,1]=W[15]
    M1_output = {
        0: [(0, 1), (10, 1)],
        1: [(1, 1), (11, 1)],
        2: [(4, 1), (14, 1)],
        3: [(5, 1), (15, 1)],
    }
    
    # M2 = (A10+A11)*B00
    M2_X = [
        [(8, 1), (10, 1)],   # [0,0]: A[2,0]+A[2,2]
        [(9, 1), (11, 1)],   # [0,1]: A[2,1]+A[2,3]
        [(12, 1), (14, 1)],  # [1,0]: A[3,0]+A[3,2]
        [(13, 1), (15, 1)],  # [1,1]: A[3,1]+A[3,3]
    ]
    M2_Y = [
        [(0, 1)],   # B[0,0]
        [(1, 1)],   # B[0,1]
        [(4, 1)],   # B[1,0]
        [(5, 1)],   # B[1,1]
    ]
    # M2 contributes to C10 (coeff +1) and C11 (coeff -1)
    M2_output = {
        0: [(8, 1), (10, -1)],
        1: [(9, 1), (11, -1)],
        2: [(12, 1), (14, -1)],
        3: [(13, 1), (15, -1)],
    }
    
    # M3 = A00*(B01-B11)
    M3_X = [
        [(0, 1)],   # A[0,0]
        [(1, 1)],   # A[0,1]
        [(4, 1)],   # A[1,0]
        [(5, 1)],   # A[1,1]
    ]
    M3_Y = [
        [(2, 1), (10, -1)],   # B[0,2]-B[2,2]
        [(3, 1), (11, -1)],   # B[0,3]-B[2,3]
        [(6, 1), (14, -1)],   # B[1,2]-B[3,2]
        [(7, 1), (15, -1)],   # B[1,3]-B[3,3]
    ]
    # M3 contributes to C01 (coeff +1) and C11 (coeff +1)
    M3_output = {
        0: [(2, 1), (10, 1)],
        1: [(3, 1), (11, 1)],
        2: [(6, 1), (14, 1)],
        3: [(7, 1), (15, 1)],
    }
    
    # M4 = A11*(B10-B00)
    M4_X = [
        [(10, 1)],   # A[2,2]
        [(11, 1)],   # A[2,3]
        [(14, 1)],   # A[3,2]
        [(15, 1)],   # A[3,3]
    ]
    M4_Y = [
        [(8, 1), (0, -1)],   # B[2,0]-B[0,0]
        [(9, 1), (1, -1)],   # B[2,1]-B[0,1]
        [(12, 1), (4, -1)],  # B[3,0]-B[1,0]
        [(13, 1), (5, -1)],  # B[3,1]-B[1,1]
    ]
    # M4 contributes to C00 (coeff -1) and C10 (coeff +1)
    M4_output = {
        0: [(0, -1), (8, 1)],
        1: [(1, -1), (9, 1)],
        2: [(4, -1), (12, 1)],
        3: [(5, -1), (13, 1)],
    }
    
    # M5 = (A00+A01)*B11
    M5_X = [
        [(0, 1), (2, 1)],   # A[0,0]+A[0,2]
        [(1, 1), (3, 1)],   # A[0,1]+A[0,3]
        [(4, 1), (6, 1)],   # A[1,0]+A[1,2]
        [(5, 1), (7, 1)],   # A[1,1]+A[1,3]
    ]
    M5_Y = [
        [(10, 1)],   # B[2,2]
        [(11, 1)],   # B[2,3]
        [(14, 1)],   # B[3,2]
        [(15, 1)],   # B[3,3]
    ]
    # M5 contributes to C01 (coeff +1) and C11 (coeff +1)
    M5_output = {
        0: [(2, 1), (10, 1)],
        1: [(3, 1), (11, 1)],
        2: [(6, 1), (14, 1)],
        3: [(7, 1), (15, 1)],
    }
    
    # M6 = (A10-A00)*(B00+B01)
    M6_X = [
        [(8, 1), (0, -1)],   # A[2,0]-A[0,0]
        [(9, 1), (1, -1)],   # A[2,1]-A[0,1]
        [(12, 1), (4, -1)],  # A[3,0]-A[1,0]
        [(13, 1), (5, -1)],  # A[3,1]-A[1,1]
    ]
    M6_Y = [
        [(0, 1), (2, 1)],   # B[0,0]+B[0,2]
        [(1, 1), (3, 1)],   # B[0,1]+B[0,3]
        [(4, 1), (6, 1)],   # B[1,0]+B[1,2]
        [(5, 1), (7, 1)],   # B[1,1]+B[1,3]
    ]
    # M6 contributes to C00 (coeff +1)
    M6_output = {
        0: [(0, 1)],
        1: [(1, 1)],
        2: [(4, 1)],
        3: [(5, 1)],
    }
    
    # M7 = (A01-A11)*(B10+B11)
    M7_X = [
        [(2, 1), (10, -1)],   # A[0,2]-A[2,2]
        [(3, 1), (11, -1)],   # A[0,3]-A[2,3]
        [(6, 1), (14, -1)],   # A[1,2]-A[3,2]
        [(7, 1), (15, -1)],   # A[1,3]-A[3,3]
    ]
    M7_Y = [
        [(8, 1), (10, 1)],   # B[2,0]+B[2,2]
        [(9, 1), (11, 1)],   # B[2,1]+B[2,3]
        [(12, 1), (14, 1)],  # B[3,0]+B[3,2]
        [(13, 1), (15, 1)],  # B[3,1]+B[3,3]
    ]
    # M7 contributes to C00 (coeff +1)
    M7_output = {
        0: [(0, 1)],
        1: [(1, 1)],
        2: [(4, 1)],
        3: [(5, 1)],
    }
    
    # Now add all 7 M_i products, each with 7 Strassen terms = 49 total
    add_strassen_general(M1_X, M1_Y, M1_output)
    add_strassen_general(M2_X, M2_Y, M2_output)
    add_strassen_general(M3_X, M3_Y, M3_output)
    add_strassen_general(M4_X, M4_Y, M4_output)
    add_strassen_general(M5_X, M5_Y, M5_output)
    add_strassen_general(M6_X, M6_Y, M6_output)
    add_strassen_general(M7_X, M7_Y, M7_output)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
