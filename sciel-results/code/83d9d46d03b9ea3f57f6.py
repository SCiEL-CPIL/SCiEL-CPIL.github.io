import numpy as np
import time
import itertools
import sys
import random

DIMENSIONS = [4, 5, 5]

def _build_schoolbook(n, m, p):
    """Build the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U,V,W exactly decompose the matrix multiplication tensor."""
    nm, mp, pn = n * m, m * p, p * n
    # Build the multiplication tensor T[i,j,k] = 1
    # Check: for all (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    # Also check that no spurious contributions: the tensor should be exactly delta_{i,j,k}
    # Actually the decomposition only needs to reproduce the tensor, so the above check suffices.
    # But we also need to ensure there's no extra contribution where the tensor entry is 0.
    # Wait - the matrix multiplication tensor T_{i,j,k} = 1 for all (i,j,k) in [n]x[m]x[p].
    # So we just need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all (i,j,k).
    return True

def _verify_full(U, V, W, n, m, p, tol=1e-9):
    """More thorough verification using the tensor contraction."""
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    # For each (i,j,k), compute the contraction
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i*m+j, :]
                v_val = V[j*p+k, :]
                w_val = W[k*n+i, :]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_4x4_4x4():
    """
    Strassen algorithm for 4x4 * 4x4 multiplication.
    Returns U (16, rank), V (16, rank), W (16, rank)
    using rank-8 Strassen for the 4x4 case.
    
    Actually, let's use the standard approach:
    Split 4x4 into 2x2 blocks. Each 2x2*2x2 uses Strassen (rank 7).
    So total rank = 7 for the block computation.
    
    But the indices are tricky. Let me think about this differently.
    
    For multiplying A(4x4) * B(4x4) -> C(4x4):
    Split into 2x2 blocks:
    A = [[A00, A01], [A10, A11]]  (each 2x2)
    B = [[B00, B01], [B10, B11]]  (each 2x2)
    C = [[C00, C01], [C10, C11]]  (each 2x2)
    
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    Using Strassen on the 2x2 block level:
    M1 = (A00+A11)*(B00+B11)  -- 7 multiplications of 2x2 matrices
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 - M4 + M6 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M3 + M2 + M5
    
    Each M is a product of two 2x2 matrices, which itself can be done with Strassen (rank 7) or schoolbook (rank 4).
    
    If we use Strassen for each 2x2*2x2, the total rank is 7 * 7 = 49.
    If we use schoolbook for each 2x2*2x2, the total rank is 7 * 4 = 28.
    
    Wait, that's not right. The rank of the whole thing depends on how we compose.
    
    Let me think about this more carefully. The tensor for 4x4*4x4 has "dimension" 4*4*4 = 64.
    The schoolbook rank is 64. Strassen gives rank 49.
    
    Actually, let me just directly construct the Strassen-based decomposition.
    """
    # For 2x2 * 2x2 Strassen, the rank is 7.
    # The U, V, W for 2x2*2x2 Strassen:
    # U is (4, 7), V is (4, 7), W is (4, 7)
    # where indices are: U[row_in_A, r], V[col_in_B, r], W[out_idx, r]
    
    # Strassen 2x2:
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
    # C11 = M1 - M3 + M2 + M5
    #
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=2, C11=3
    
    U22 = np.zeros((4, 7))
    V22 = np.zeros((4, 7))
    W22 = np.zeros((4, 7))
    
    # r=0: M1 = (A00+A11)*(B00+B11)
    U22[0, 0] = 1; U22[3, 0] = 1
    V22[0, 0] = 1; V22[3, 0] = 1
    W22[0, 0] = 1; W22[3, 0] = 1  # C00 and C11 get +1 from M1
    
    # r=1: M2 = (A10+A11)*B00
    U22[2, 1] = 1; U22[3, 1] = 1
    V22[0, 1] = 1
    W22[2, 1] = 1; W22[3, 1] = 1  # C10 and C11 get +1 from M2
    
    # r=2: M3 = A00*(B01-B11)
    U22[0, 2] = 1
    V22[1, 2] = 1; V22[3, 2] = -1
    W22[1, 2] = 1; W22[3, 2] = -1  # C01 gets +1, C11 gets -1 from M3
    
    # r=3: M4 = A11*(B10-B00)
    U22[3, 3] = 1
    V22[2, 3] = 1; V22[0, 3] = -1
    W22[0, 3] = -1; W22[2, 3] = 1  # C00 gets -1, C10 gets +1 from M4
    
    # r=4: M5 = (A00+A01)*B11
    U22[0, 4] = 1; U22[1, 4] = 1
    V22[3, 4] = 1
    W22[1, 4] = 1; W22[3, 4] = 1  # C01 and C11 get +1 from M5
    
    # r=5: M6 = (A10-A00)*(B00+B01)
    U22[2, 5] = 1; U22[0, 5] = -1
    V22[0, 5] = 1; V22[1, 5] = 1
    W22[0, 5] = 1  # C00 gets +1 from M6
    
    # r=6: M7 = (A01-A11)*(B10+B11)
    U22[1, 6] = 1; U22[3, 6] = -1
    V22[2, 6] = 1; V22[3, 6] = 1
    W22[0, 6] = 1  # C00 gets +1 from M7
    
    return U22, V22, W22

def _strassen_5x5():
    """
    Attempt to construct a Strassen-like decomposition for 5x5 * 5x5.
    
    One approach: pad to 8x8 (next power of 2), use Strassen recursively,
    then project back. But that gives rank 8^3 = 512 for schoolbook, 
    or 7^3 = 343 for Strassen. That's worse than schoolbook 125.
    
    Another approach: Use the fact that 5 = 3 + 2.
    Split A(5x5) into blocks: A = [[A_33, A_32], [A_23, A_22]]
    Similarly for B.
    Then C = A*B involves block multiplications.
    
    This gets complex. Let me try a different approach.
    
    Actually, for this task we need 4x5 * 5x5 -> 4x5.
    Let me think about the structure more carefully.
    
    The multiplication tensor T has dimensions (n,m,p) = (4,5,5).
    T[i,j,k] = 1 for all i in [4], j in [5], k in [5].
    
    The rank of this tensor is what we want to minimize.
    
    Known results:
    - Schoolbook: rank = 4*5*5 = 100
    - The minimum rank is known to be at most... let me think.
    
    For m x n * n x p, the tensor rank is at most mn * np / n = mp * ... no.
    
    Actually, for square n x n * n x n, the Strassen rank is 7^{log2(n)}.
    For n=5, that's not directly applicable.
    
    Let me try a constructive approach:
    1. Start with schoolbook (rank 100)
    2. Try to find rank reductions through algebraic identities
    
    For 4x5 * 5x5:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    One idea: if we can find linear combinations of rows of A and columns of B
    that give the same products with fewer terms, we can reduce rank.
    
    Let me try the following approach:
    - Use the fact that for 4x5*5x5, we can think of it as:
      C = A * B where A is 4x5 and B is 5x5.
    
    - We can write A as a sum of rank-1 matrices (or fewer), but that doesn't help directly.
    
    Let me try a different structural approach:
    Group the 5 columns of A (and 5 rows of B) into groups.
    
    Approach: Use a "bilinear" decomposition with shared factors.
    
    For example, if we define:
    M1 = (A[i,0] + A[i,1]) * (B[0,k] + B[1,k])
    M2 = (A[i,0] - A[i,1]) * (B[0,k] - B[1,k])
    Then A[i,0]*B[0,k] = (M1 + M2)/2 and A[i,1]*B[1,k] = (M1 - M2)/2.
    So 2 products replaced by 2 products - no gain.
    
    But if we can share more:
    M1 = (A[i,0] + A[i,1] + A[i,2]) * (B[0,k] + B[1,k] + B[2,k])
    This gives us 9 products but uses 1 "slot". However, extracting individual products
    requires more work.
    
    Actually, the key insight for Strassen is that you can compute 8 products (2x2*2x2)
    with only 7 multiplications because of algebraic dependencies.
    
    For our case, let me try to find similar dependencies.
    
    Let me try a programmatic search for small rank reductions.
    """
    pass

def _rank_minimization_search(U, V, W, n, m, p, time_budget=60.0):
    """
    Try to reduce the rank of the decomposition by finding linear dependencies.
    """
    nm, mp, pn = n * m, m * p, p * n
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    start = time.time()
    deadline = start + time_budget
    
    # Build the tensor as a set of equations
    # For each (i,j,k), we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # Try to eliminate columns one at a time
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        current_rank = best_U.shape[1]
        if current_rank <= 1:
            break
        
        found = False
        # Try eliminating each column
        for col in range(current_rank):
            if time.time() > deadline:
                break
            
            # Get the column
            u_col = best_U[:, col]
            v_col = best_V[:, col]
            w_col = best_W[:, col]
            
            # Check if column is zero
            if np.allclose(u_col, 0) and np.allclose(v_col, 0) and np.allclose(w_col, 0):
                mask = np.ones(current_rank, dtype=bool)
                mask[col] = False
                new_U = best_U[:, mask]
                new_V = best_V[:, mask]
                new_W = best_W[:, mask]
                if _verify_full(new_U, new_V, new_W, n, m, p):
                    best_U, best_V, best_W = new_U, new_V, new_W
                    best_rank = new_U.shape[1]
                    found = True
                    break
                continue
            
            # Try to express this column as a linear combination of others
            # For each (i,j,k), we need:
            # u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i] = 
            #   sum_{r != col} c_r * U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            
            equations = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        rhs = u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
                        if abs(rhs) > 1e-12:
                            coeffs = []
                            for r in range(current_rank):
                                if r == col:
                                    continue
                                coeffs.append(best_U[i*m+j, r] * best_V[j*p+k, r] * best_W[k*n+i, r])
                            equations.append((np.array(coeffs), rhs))
            
            if not equations:
                continue
            
            A_mat = np.array([eq[0] for eq in equations])
            b_vec = np.array([eq[1] for eq in equations])
            
            try:
                coeffs, residuals, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=1e-12)
                if len(residuals) == 0 or np.max(np.abs(residuals)) < 1e-10:
                    # Verify
                    A_test = A_mat @ coeffs
                    if np.max(np.abs(A_test - b_vec)) < 1e-10:
                        # Create new matrices
                        mask = np.ones(current_rank, dtype=bool)
                        mask[col] = False
                        new_U = best_U[:, mask].copy()
                        new_V = best_V[:, mask].copy()
                        new_W = best_W[:, mask].copy()
                        
                        # Add the combination
                        for idx, c in enumerate(coeffs):
                            if abs(c) > 1e-15:
                                new_U[:, idx] += c * u_col
                                new_V[:, idx] += c * v_col
                                new_W[:, idx] += c * w_col
                        
                        if _verify_full(new_U, new_V, new_W, n, m, p):
                            best_U, best_V, best_W = new_U, new_V, new_W
                            best_rank = new_U.shape[1]
                            found = True
                            break
            except:
                pass
        
        if not found:
            break
    
    return best_U, best_V, best_W

def _try_block_strassen(n, m, p):
    """
    Try to use a block Strassen approach for the 4x5*5x5 case.
    
    Idea: Pad to a size where Strassen applies, then project.
    Or: Find a suitable block decomposition.
    
    For 4x5 * 5x5:
    We can think of A as 4x5 and B as 5x5.
    
    One approach: embed into a larger multiplication.
    For example, pad A to 8x8 and B to 8x8, use Strassen, then extract.
    But the rank would be 7^3 = 343 for the 8x8 case, which is worse than 100.
    
    Another approach: use a smaller Strassen-like construction.
    
    Let me try: split the 5 columns/rows into groups of 2 and 3.
    A = [A_4x2 | A_4x3]
    B = [B_2x5; B_3x5]
    
    C = A_4x2 * B_2x5 + A_4x3 * B_3x5
    
    The first term is 4x2 * 2x5, rank = 4*2*5 = 40 (schoolbook) or potentially less.
    The second term is 4x3 * 3x5, rank = 4*3*5 = 60 (schoolbook) or potentially less.
    Total: 100 (same as schoolbook).
    
    But if we can use Strassen-like tricks on each block:
    4x2 * 2x5: this is like multiplying a 4x2 by 2x5. 
    We can think of it as: C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    This has rank 4*2*5 = 40 at most.
    
    4x3 * 3x5: C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    This has rank 4*3*5 = 60 at most.
    
    Total: 100. No improvement from just splitting.
    
    The key is to find cross-block dependencies.
    
    Let me try a different approach: use random projections to find a low-rank
    approximation, then round to get an exact decomposition.
    
    Actually, let me try a more systematic algebraic approach.
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Try rank minimization
    U2, V2, W2 = _rank_minimization_search(U, V, W, n, m, p, 
                                             time_budget=min(80.0, deadline - (time.time() - start_time)))
    if U2.shape[1] < best_rank:
        best_U, best_V, best_W = U2, V2, W2
        best_rank = U2.shape[1]
    
    # Try to find a better decomposition through algebraic identities
    # Specifically, try to use Strassen-like constructions
    
    # For 4x5*5x5, try the following:
    # Use the fact that we can group indices and find linear dependencies
    
    # Try a "random" rank reduction: perturb the decomposition and see if we can
    # find a lower-rank equivalent
    
    # Try specific algebraic identities
    # Identity 1: (a+b)(c+d) = ac + ad + bc + bd
    # This means 4 products can be computed with 1 "multiplication" if we allow
    # additions. But in tensor rank terms, we still need 4 terms unless we can
    # share across different (i,j,k) tuples.
    
    # Let me try to construct a decomposition with rank < 100 by finding
    # linear dependencies in the tensor.
    
    # The tensor T is 4x5x5 with all entries 1.
    # We want T = sum_r U_r * V_r * W_r (outer products)
    
    # One approach: use the fact that T can be written as:
    # T[i,j,k] = 1 = sum_{j'} delta_{j,j'} * 1
    # But that's just the schoolbook.
    
    # Another approach: find a matrix factorization.
    # T as a 3-tensor can be "matricized" in different ways.
    # T_(1) is a (4*5) x 5 matrix, all ones.
    # T_(2) is a (5*5) x 4 matrix, all ones.
    # T_(3) is a (5*4) x 5 matrix, all ones.
    
    # The rank of T is at least the maximum of the ranks of its matricizations.
    # rank(T_(1)) = 1, rank(T_(2)) = 1, rank(T_(3)) = 1.
    # So the tensor rank is at least 1, which is trivial.
    
    # The actual tensor rank is what we're trying to find.
    
    # Let me try a different constructive approach:
    # Use a "bilinear" form with shared structure.
    
    # Define:
    # For each pair (i, k), C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_j A[i,j] * B[j,k]
    
    # If we define linear combinations of rows of A and columns of B:
    # M1 = (A[i,0] + A[i,1]) * (B[0,k] + B[1,k])
    # M2 = (A[i,0] + A[i,2]) * (B[0,k] + B[2,k])
    # M3 = (A[i,1] + A[i,2]) * (B[1,k] + B[2,k])
    # M4 = A[i,0] * B[0,k]
    # M5 = A[i,1] * B[1,k]
    # M6 = A[i,2] * B[2,k]
    # ...
    # This gets complicated. Let me just try the search approach more aggressively.
    
    # Try to find a decomposition with rank 85 or lower using a more sophisticated search
    
    # Approach: Start from schoolbook and try to merge columns
    # Two columns r1 and r2 can be merged if:
    # U[:,r1] * V[:,r1] * W[:,r1] + U[:,r2] * V[:,r2] * W[:,r2]
    # can be expressed as a sum of fewer rank-1 tensors.
    
    # This is equivalent to checking if the sum of two rank-1 tensors has rank 1.
    # U1*V1*W1 + U2*V2*W2 has rank 1 iff U1 is proportional to U2, 
    # V1 is proportional to V2, and W1 is proportional to W2 (with consistent scaling).
    
    # So we can check: for each pair of columns, check if they can be merged.
    
    # This is O(rank^2 * nm*mp*pn) which is feasible for rank ~ 100.
    
    if time.time() < deadline:
        U3, V3, W3 = best_U.copy(), best_V.copy(), best_W.copy()
        current_rank = U3.shape[1]
        
        for iteration in range(100):
            if time.time() > deadline:
                break
            
            found_merge = False
            for r1 in range(current_rank):
                if time.time() > deadline:
                    break
                for r2 in range(r1 + 1, current_rank):
                    if time.time() > deadline:
                        break
                    
                    # Check if columns r1 and r2 can be merged
                    # U1*V1*W1 + U2*V2*W2 = c * U_new * V_new * W_new
                    # This requires U1/V1/W1 to be proportional to U2/V2/W2
                    # with consistent scaling.
                    
                    # Check U proportionality
                    u1, u2 = U3[:, r1], U3[:, r2]
                    v1, v2 = V3[:, r1], V3[:, r2]
                    w1, w2 = W3[:, r1], W3[:, r2]
                    
                    # Find scaling factors
                    # u1 = a * u2, v1 = b * v2, w1 = c * w2
                    # with a * b * c = 1 (for the sum to be a single rank-1 tensor)
                    
                    # Actually, for the sum to have rank 1:
                    # u1* v1 * w1 + u2 * v2 * w2 = (u1 + s*u2) * (v1 + t*v2) * (w1 + u*w2)
                    # This is not generally true. The sum of two rank-1 tensors
                    # has rank 1 iff they are "parallel" in a specific way.
                    
                    # A simpler check: the sum has rank 1 iff all 2x2 minors
                    # of the combined tensor vanish.
                    
                    # For now, just check if u1 is proportional to u2,
                    # v1 is proportional to v2, and w1 is proportional to w2.
                    
                    def is_proportional(a, b, tol=1e-10):
                        if np.allclose(a, 0) and np.allclose(b, 0):
                            return True, 0.0
                        if np.allclose(a, 0) or np.allclose(b, 0):
                            return False, 0.0
                        # Find the scaling factor
                        mask = np.abs(b) > tol
                        if np.sum(mask) == 0:
                            return False, 0.0
                        factors = a[mask] / b[mask]
                        if np.max(np.abs(factors - factors[0])) < tol:
                            return True, factors[0]
                        return False, 0.0
                    
                    p1, s1 = is_proportional(u1, u2)
                    if p1:
                        p2, s2 = is_proportional(v1, v2)
                        if p2:
                            p3, s3 = is_proportional(w1, w2)
                            if p3 and abs(s1 * s2 * s3 - 1.0) < 1e-8:
                                # They can be merged
                                # New column = u1 + u2 = (1 + s1) * u2 (if u1 = s1 * u2)
                                # Actually, u1* v1 * w1 + u2 * v2 * w2
                                # = s1*u2 * s2*v2 * s3*w2 + u2 * v2 * w2
                                # = (s1*s2*s3 + 1) * u2 * v2 * w2
                                # = (1 + 1) * u2 * v2 * w2 = 2 * u2 * v2 * w2
                                # So the new column is 2 * u2, 2 * v2, 2 * w2? No.
                                # u1*v1*w1 + u2*v2*w2 = (s1*s2*s3 + 1) * u2*v2*w2
                                # = 2 * u2*v2*w2
                                # So we can replace with a single column: u2, v2, w2 with factor 2.
                                # But we need integer/half-integer entries.
                                # So new column = 2*u2, v2, w2? No, that changes the product.
                                # We need: new_u * new_v * new_w = u1*v1*w1 + u2*v2*w2
                                # If u1 = s1*u2, v1 = s2*v2, w1 = s3*w2, and s1*s2*s3 = 1:
                                # u1*v1*w1 = u2*v2*w2
                                # So sum = 2*u2*v2*w2
                                # New column: (2*u2, v2, w2) or (u2, 2*v2, w2) or (u2, v2, 2*w2)
                                # Let's use (u2, v2, 2*w2)
                                
                                # Create new matrices
                                mask = np.ones(current_rank, dtype=bool)
                                mask[r1] = False
                                mask[r2] = False
                                new_U = np.column_stack([U3[:, mask], 2.0 * U3[:, r2]])
                                new_V = np.column_stack([V3[:, mask], V3[:, r2]])
                                new_W = np.column_stack([W3[:, mask], W3[:, r2]])
                                
                                if _verify_full(new_U, new_V, new_W, n, m, p):
                                    U3, V3, W3 = new_U, new_V, new_W
                                    current_rank = new_U.shape[1]
                                    found_merge = True
                                    break
                if found_merge:
                    break
            
            if not found_merge:
                break
        
        if U3.shape[1] < best_rank:
            best_U, best_V, best_W = U3, V3, W3
            best_rank = U3.shape[1]
    
    # Final verification
    if not _verify_full(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook
        best_U, best_V, best_W = _build_schoolbook(n, m, p)
    
    # Convert to lists for the expected output format
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    # Round to remove floating point noise
    U_list = [[round(x, 10) for x in row] for row in U_list]
    V_list = [[round(x, 10) for x in row] for row in V_list]
    W_list = [[round(x, 10) for x in row] for row in W_list]
    
    return {'U': U_list, 'V': V_list, 'W': W_list}
