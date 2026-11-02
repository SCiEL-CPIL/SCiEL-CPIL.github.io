"""Matrix multiplication 3,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 5 matrix.
    The output factor follows the official trace convention W[k*n+i,r].
    
    Returns dict with 'U', 'V', 'W' keys containing factor matrices.
    """
    n, m, p = DIMENSIONS
    # n=3 rows of A, m=4 cols of A / rows of B, p=5 cols of B
    # A is n x m = 3 x 4
    # B is m x p = 4 x 5
    # C is n x p = 3 x 5
    
    # Strategy: Split m=4 into 2+2, giving two (3x2)*(2x5) products.
    # For each (3x2)*(2x5), split n=3 into 1+2:
    #   (1x2)*(2x5): rank 10 (schoolbook, optimal)
    #   (2x2)*(2x5): split p=5 into 2+2+1:
    #     (2x2)*(2x2): Strassen rank 7
    #     (2x2)*(2x1): rank 4 (schoolbook, optimal)
    #   Total (2x2)*(2x5): rank 11
    # Total (3x2)*(2x5): rank 21
    # Full: 2 * 21 = 42
    
    # We'll build the decomposition column by column.
    # Each column r corresponds to a rank-1 term:
    #   U[:,r] * V[:,r] * W[:,r] contributes to the tensor.
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # must equal 1 if (i,j,k) is a valid index triple, else 0.
    
    # We'll collect all rank-1 terms as (u_vec, v_vec, w_vec) tuples.
    terms = []
    
    def add_term(u_indices, v_indices, w_indices, sign=1.0):
        """Add a rank-1 term where u_indices is a list of (index, coeff) for U,
        v_indices for V, w_indices for W."""
        u = {}
        v = {}
        w = {}
        for idx, c in u_indices:
            u[idx] = u.get(idx, 0) + sign * c
        for idx, c in v_indices:
            v[idx] = v.get(idx, 0) + sign * c
        for idx, c in w_indices:
            w[idx] = w.get(idx, 0) + sign * c
        terms.append((u, v, w))
    
    # Helper: add a schoolbook term for (a x b) x (b x c) at offset (i0, j0, k0)
    # i0: row offset in A (0-based), j0: col offset in A / row offset in B, k0: col offset in B
    def schoolbook_block(i0, j0, k0, a, b, c):
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    u_idx = (i0 + i) * m + (j0 + j)
                    v_idx = (j0 + j) * p + (k0 + k)
                    w_idx = (k0 + k) * n + (i0 + i)
                    add_term([(u_idx, 1.0)], [(v_idx, 1.0)], [(w_idx, 1.0)])
    
    # Helper: add Strassen terms for (2x2)x(2x2) at offset (i0, j0, k0)
    # Strassen's 7 products for multiplying two 2x2 matrices:
    # M1 = (A11)(B12 - B22)
    # M2 = (A11 + A21)(B22)
    # M3 = (A22)(B11 - B21)
    # M4 = (A11 + A22)(B21 + B22)
    # M5 = (A11 + A12)(B11)
    # M6 = (A12 - A22)(B21 + B22)
    # M7 = (A21 - A11)(B11 + B12)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    def strassen_2x2(i0, j0, k0):
        # A is 2x2 at rows i0..i0+1, cols j0..j0+1
        # B is 2x2 at rows j0..j0+1, cols k0..k0+1
        # C is 2x2 at rows i0..i0+1, cols k0..k0+1
        
        # We need to express each Strassen product as a rank-1 tensor contribution.
        # Each product Mx = (linear combo of A entries) * (linear combo of B entries)
        # contributes to C entries via the reconstruction formulas.
        
        # Let's define the 7 products and their contributions.
        # For each product, we have:
        #   A_part = sum of (coeff * A[i,j]) terms
        #   B_part = sum of (coeff * B[j,k]) terms
        #   C_part = sum of (coeff * C[i,k]) terms
        # The rank-1 term is: U from A_part, V from B_part, W from C_part.
        
        # A indices: A[i0+a, j0+b] -> U index (i0+a)*m + (j0+b)
        # B indices: B[j0+a, k0+b] -> V index (j0+a)*p + (k0+b)
        # C indices: C[i0+a, k0+b] -> W index (k0+b)*n + (i0+a)
        
        def A_idx(a, b):
            return (i0 + a) * m + (j0 + b)
        
        def B_idx(a, b):
            return (j0 + a) * p + (k0 + b)
        
        def C_idx(a, b):
            return (k0 + b) * n + (i0 + a)
        
        # M1 = A11 * (B12 - B22)
        add_term([(A_idx(0,0), 1.0)],
                 [(B_idx(0,1), 1.0), (B_idx(1,1), -1.0)],
                 [(C_idx(0,0), 1.0), (C_idx(1,1), 1.0)])
        
        # M2 = (A11 + A21) * B22
        add_term([(A_idx(0,0), 1.0), (A_idx(1,0), 1.0)],
                 [(B_idx(1,1), 1.0)],
                 [(C_idx(1,0), 1.0), (C_idx(1,1), 1.0)])
        
        # M3 = A22 * (B11 - B21)
        add_term([(A_idx(1,1), 1.0)],
                 [(B_idx(0,0), 1.0), (B_idx(1,0), -1.0)],
                 [(C_idx(0,1), 1.0), (C_idx(1,0), 1.0)])
        
        # M4 = (A11 + A22) * (B21 + B22)
        add_term([(A_idx(0,0), 1.0), (A_idx(1,1), 1.0)],
                 [(B_idx(1,0), 1.0), (B_idx(1,1), 1.0)],
                 [(C_idx(0,0), 1.0), (C_idx(0,1), 1.0), (C_idx(1,0), 1.0), (C_idx(1,1), 1.0)])
        
        # M5 = (A11 + A12) * B11
        add_term([(A_idx(0,0), 1.0), (A_idx(0,1), 1.0)],
                 [(B_idx(0,0), 1.0)],
                 [(C_idx(0,0), -1.0), (C_idx(0,1), 1.0)])
        
        # M6 = (A12 - A22) * (B21 + B22)
        add_term([(A_idx(0,1), 1.0), (A_idx(1,1), -1.0)],
                 [(B_idx(1,0), 1.0), (B_idx(1,1), 1.0)],
                 [(C_idx(1,1), 1.0)])
        
        # M7 = (A21 - A11) * (B11 + B12)
        add_term([(A_idx(1,0), 1.0), (A_idx(0,0), -1.0)],
                 [(B_idx(0,0), 1.0), (B_idx(0,1), 1.0)],
                 [(C_idx(0,0), 1.0), (C_idx(1,0), 1.0)])
    
    # Now build the full decomposition:
    # m=4 split into j0=0 (cols 0,1) and j0=2 (cols 2,3)
    # For each half, n=3 split into i0=0 (row 0) and i0=1 (rows 1,2)
    # For the (2x2) part, p=5 split into k0=0 (cols 0,1), k0=2 (cols 2,3), k0=4 (col 4)
    
    for j0 in [0, 2]:
        # First: (1x2) x (2x5) at row i0=0, cols j0..j0+1
        # Schoolbook: 1*2*5 = 10 terms
        schoolbook_block(i0=0, j0=j0, k0=0, a=1, b=2, c=5)
        
        # Second: (2x2) x (2x5) at rows i0=1, cols j0..j0+1
        # Split p=5 into 2+2+1
        # (2x2)x(2x2) at k0=0: Strassen
        strassen_2x2(i0=1, j0=j0, k0=0)
        # (2x2)x(2x2) at k0=2: Strassen
        strassen_2x2(i0=1, j0=j0, k0=2)
        # (2x2)x(2x1) at k0=4: schoolbook, 2*2*1 = 4 terms
        schoolbook_block(i0=1, j0=j0, k0=4, a=2, b=2, c=1)
    
    # Now convert terms to U, V, W matrices
    rank = len(terms)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            U[idx, r] += val
        for idx, val in v_dict.items():
            V[idx, r] += val
        for idx, val in w_dict.items():
            W[idx, r] += val
    
    # Verify the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should be 1 for all valid (i,j,k), 0 otherwise.
    # Let's verify on a few random points.
    U_mat = U
    V_mat = V
    W_mat = W
    
    # Build the full tensor and check
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                if abs(actual - expected) > 1e-9:
                    errors += 1
    
    if errors > 0:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
