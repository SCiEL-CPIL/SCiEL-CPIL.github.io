"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    import time
    from fractions import Fraction
    deadline = time.time() + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank = 49
    
    # Strassen 2x2 decomposition tables
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1
    
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # Build 4x4 via Kronecker product
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            c = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    u1 = U2[i1*2+j1][r1]
                    if u1 == 0:
                        continue
                    for i0 in range(2):
                        for j0 in range(2):
                            u2 = U2[i0*2+j0][r2]
                            if u2 == 0:
                                continue
                            i = i1*2 + i0
                            j = j1*2 + j0
                            U[i*4+j][c] = u1 * u2
            
            for j1 in range(2):
                for k1 in range(2):
                    v1 = V2[j1*2+k1][r1]
                    if v1 == 0:
                        continue
                    for j0 in range(2):
                        for k0 in range(2):
                            v2 = V2[j0*2+k0][r2]
                            if v2 == 0:
                                continue
                            j = j1*2 + j0
                            k = k1*2 + k0
                            V[j*4+k][c] = v1 * v2
            
            for k1 in range(2):
                for i1 in range(2):
                    w1 = W2[k1*2+i1][r1]
                    if w1 == 0:
                        continue
                    for k0 in range(2):
                        for i0 in range(2):
                            w2 = W2[k0*2+i0][r2]
                            if w2 == 0:
                                continue
                            k = k1*2 + k0
                            i = i1*2 + i0
                            W[k*4+i][c] = w1 * w2
    
    # Verify the full tensor identity
    def verify(U_mat, V_mat, W_mat, R):
        """Check that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1 iff (i,j,k) is a valid index"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(R):
                        total += U_mat[i*4+j][r] * V_mat[j*4+k][r] * W_mat[k*4+i][r]
                    # For matrix mult, C[i,k] = sum_j A[i,j]*B[j,k]
                    # Tensor: T[i,j,k] = 1 if this is a valid multiplication term
                    # Actually the tensor is T[i,j,k] where C[i,k] = sum_j A[i,j]*B[j,k]
                    # So T[i,j,k] should be 1 for all i,j,k (it's the multiplication tensor)
                    # Wait, let me reconsider. The tensor for matrix mult is:
                    # C[i,k] = sum_j A[i,j] * B[j,k]
                    # So the 3-index tensor T has T[i,j,k] = 1 for all i,j,k in range
                    if total != 1:
                        return False
        return True
    
    # Attempt rank reduction via Gaussian elimination on the U matrix
    # Convert to Fraction for exact arithmetic
    R = rank
    U_frac = [[Fraction(U[i][r]) for r in range(R)] for i in range(16)]
    V_frac = [[Fraction(V[j][r]) for r in range(R)] for j in range(16)]
    W_frac = [[Fraction(W[k][r]) for r in range(R)] for k in range(16)]
    
    # Try to find linear dependencies among the rank columns
    # A rank term r is redundant if it can be expressed as a linear combination of others
    # We use the fact that if column r of U is a linear combo of other columns,
    # and similarly for V and W, then we can eliminate it
    
    # Strategy: find the rank of U, V, W matrices
    # If any has rank < R, there are dependencies to exploit
    
    def compute_rank(mat):
        """Compute rank of a matrix over Q using Gaussian elimination"""
        rows = len(mat)
        cols = len(mat[0]) if rows > 0 else 0
        M = [row[:] for row in mat]
        r = 0
        for c in range(cols):
            if r >= rows:
                break
            # Find pivot
            pivot = None
            for i in range(r, rows):
                if M[i][c] != 0:
                    pivot = i
                    break
            if pivot is None:
                continue
            M[r], M[pivot] = M[pivot], M[r]
            # Normalize
            pv = M[r][c]
            M[r] = [x / pv for x in M[r]]
            # Eliminate
            for i in range(rows):
                if i != r and M[i][c] != 0:
                    factor = M[i][c]
                    M[i] = [M[i][j] - factor * M[r][j] for j in range(cols)]
            r += 1
        return r
    
    rank_U = compute_rank(U_frac)
    rank_V = compute_rank(V_frac)
    rank_W = compute_rank(W_frac)
    
    # The effective rank is min of the three, but we need to find actual reducible terms
    # Try to identify specific columns that are linearly dependent
    
    # For each column, check if it's in the span of the others
    # This is expensive for 49 columns, so we do it smartly
    
    # Build the set of "active" columns
    active = list(range(R))
    
    # Try removing columns one by one and checking if the decomposition still works
    # We'll use a greedy approach: try to remove each column if possible
    
    # First, let's identify which columns of U are zero
    zero_cols_U = [r for r in range(R) if all(U_frac[i][r] == 0 for i in range(16))]
    zero_cols_V = [r for r in range(R) if all(V_frac[j][r] == 0 for j in range(16))]
    zero_cols_W = [r for r in range(R) if all(W_frac[k][r] == 0 for k in range(16))]
    
    # Remove zero columns
    to_remove = set(zero_cols_U) | set(zero_cols_V) | set(zero_cols_W)
    
    if to_remove:
        keep = [r for r in range(R) if r not in to_remove]
        new_R = len(keep)
        U_new = [[U[i][r] for r in keep] for i in range(16)]
        V_new = [[V[j][r] for r in keep] for j in range(16)]
        W_new = [[W[k][r] for r in keep] for k in range(16)]
        if verify(U_new, V_new, W_new, new_R):
            U, V, W = U_new, V_new, W_new
            R = new_R
    
    # Now try to find linear dependencies
    # For each column, check if removing it breaks the identity
    # We do this by trying to express the column as a combination of others
    
    # Use a more efficient approach: for each candidate column to remove,
    # solve for coefficients that would make the identity hold
    
    # Try removing one column at a time
    removed_any = True
    while removed_any and time.time() < deadline:
        removed_any = False
        for r_try in list(range(R)):
            if time.time() > deadline - 10:
                break
            # Try to express column r_try as linear combo of other columns
            # We need: U[:,r_try] = sum_{s != r_try} alpha_s * U[:,s]
            #           V[:,r_try] = sum_{s != r_try} beta_s * V[:,s]  
            #           W[:,r_try] = sum_{s != r_try} gamma_s * W[:,s]
            # And the identity must still hold
            
            # This is complex. Instead, let's just try removing and verifying.
            # But that's O(R * verification_cost) which is expensive.
            
            # Alternative: check if the column is linearly dependent in U
            others = [s for s in range(R) if s != r_try]
            if not others:
                break
            
            # Check if U[:,r_try] is in span of {U[:,s] for s in others}
            # Build augmented matrix [U_others | U_r_try] and check rank
            # This is still expensive for 49 columns
            
            # Let's just do a simple check: if the column is zero in all of U, V, or W, remove it
            # (Already done above for exact zeros)
            
            # For now, skip the expensive linear dependency check
            pass
    
    # If no reduction found, return the original rank-49 decomposition
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
