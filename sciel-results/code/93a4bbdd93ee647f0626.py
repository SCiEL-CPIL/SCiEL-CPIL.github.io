"""Matrix multiplication 2x4x6 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import combinations

DEADLINE = time.time() + 140  # safety margin before 150s kill

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p as numpy arrays."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, max_samples=50):
    """Verify decomposition reproduces matrix multiplication exactly."""
    rank = U.shape[1]
    for _ in range(max_samples):
        A = np.random.randint(-10, 10, (n, m)).astype(float)
        B = np.random.randint(-10, 10, (m, p)).astype(float)
        C_expected = A @ B
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j,r]*A[i,j]) * (sum_j V[j*p+k,r]*B[j,k])
        C_comp = np.zeros((n, p))
        for r in range(rank):
            alpha = U[:, r] @ A.reshape(-1)  # n-vector
            beta = V[:, r] @ B.reshape(-1)   # p-vector
            C_comp += np.outer(alpha, beta) * W[:, r]
        if not np.allclose(C_comp, C_expected, atol=1e-8):
            return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using symbolic-like check on basis matrices."""
    rank = U.shape[1]
    # Check on all basis matrices
    for i in range(n):
        for j in range(m):
            A = np.zeros((n, m))
            A[i, j] = 1
            for k in range(p):
                for l in range(p):
                    B = np.zeros((m, p))
                    B[j, k] = 1  # only the matching j row matters
                    C_expected = A @ B
                    C_comp = np.zeros((n, p))
                    for r in range(rank):
                        alpha = U[:, r] @ A.reshape(-1)
                        beta = V[:, r] @ B.reshape(-1)
                        C_comp += np.outer(alpha, beta) * W[:, r]
                    if not np.allclose(C_comp, C_expected, atol=1e-10):
                        return False
    # Also check that non-matching j rows give zero
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                if j2 == j:
                    continue
                A = np.zeros((n, m))
                A[i, j] = 1
                for k in range(p):
                    B = np.zeros((m, p))
                    B[j2, k] = 1
                    C_expected = A @ B  # should be zero since j != j2
                    C_comp = np.zeros((n, p))
                    for r in range(rank):
                        alpha = U[:, r] @ A.reshape(-1)
                        beta = V[:, r] @ B.reshape(-1)
                        C_comp += np.outer(alpha, beta) * W[:, r]
                    if not np.allclose(C_comp, C_expected, atol=1e-10):
                        return False
    return True

def _try_merge(U, V, W, r1, r2, n, m, p):
    """
    Try to merge terms r1 and r2. If the 2D subspace spanned by 
    (u1,u2) in U-space and (v1,v2) in V-space allows a rank-1 update
    in W-space, we can eliminate one term.
    
    The condition: there exist scalars such that the combined contribution
    of terms r1,r2 equals a single rank-1 term. This happens when
    the 2x2 matrix of W-entries [W[k*n+i, r1], W[k*n+i, r2]] for all (i,k)
    has rank <= 1 in a compatible way with U and V.
    
    More precisely: we need to find u, v, w such that
    u ⊗ v ⊗ w = u1 ⊗ v1 ⊗ w1 + u2 ⊗ v2 ⊗ w2
    This requires the 4 vectors (u1,v1), (u2,v2) to be linearly dependent
    in a specific way.
    
    Practical approach: check if the matrix M where M[i,j,k] represents
    the tensor contribution has a structure allowing merge.
    """
    u1, u2 = U[:, r1], U[:, r2]
    v1, v2 = V[:, r1], V[:, r2]
    w1, w2 = W[:, r1], W[:, r2]
    
    # Check if {u1, u2} are linearly dependent
    if np.linalg.matrix_rank(np.column_stack([u1, u2])) < 2:
        # u1 and u2 are dependent: u2 = c*u1
        c = np.dot(u1, u2) / (np.dot(u1, u1) + 1e-15)
        # Then we need v1 + c*v2 to be proportional to something, and w1 + c*w2...
        # Actually if u2 = c*u1, then term2 = c*(u1 ⊗ v2 ⊗ w2)
        # Combined: u1 ⊗ (v1 + c*v2) ⊗ ... no, that doesn't work directly.
        # The sum is u1⊗v1⊗w1 + c*u1⊗v2⊗w2 = u1 ⊗ (v1⊗w1 + c*v2⊗w2)
        # This is u1 ⊗ M where M is a (m*p x p*n) matrix of rank <= 2.
        # For this to be a single rank-1 term, we need M to have rank 1.
        M = np.outer(v1, w1) + c * np.outer(v2, w2)
        if np.linalg.matrix_rank(M) <= 1:
            # Find the rank-1 decomposition of M
            s, U_s, Vt_s = np.linalg.svd(M)
            if s[0] > 1e-10:
                v_new = s[0] * U_s[:, 0]
                w_new = Vt_s[0, :]
                # Check: u1 ⊗ v_new ⊗ w_new should equal the sum
                # We can replace both terms with one
                return True, u1, v_new, w_new
        return False, None, None, None
    
    # Check if {v1, v2} are linearly dependent
    if np.linalg.matrix_rank(np.column_stack([v1, v2])) < 2:
        c = np.dot(v1, v2) / (np.dot(v1, v1) + 1e-15)
        M = np.outer(u1, w1) + c * np.outer(u2, w2)
        if np.linalg.matrix_rank(M) <= 1:
            s, U_s, Vt_s = np.linalg.svd(M)
            if s[0] > 1e-10:
                u_new = s[0] * U_s[:, 0]
                w_new = Vt_s[0, :]
                return True, u_new, v1, w_new
        return False, None, None, None
    
    # General case: check if the 2x2 tensor sum is rank 1
    # Build the full tensor contribution and check its rank
    # T_sum[i,j,k] = U[i*m+j,r1]*V[j*p+k,r1]*W[k*n+i,r1] + U[i*m+j,r2]*V[j*p+k,r2]*W[k*n+i,r2]
    # This is a 3-tensor. Check if it has tensor rank 1.
    # A 3-tensor has rank 1 iff all its matrix unfoldings have rank 1.
    
    # Mode-1 unfolding: (n) x (m*p) matrix
    T1 = np.zeros((n, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T1[i, j * p + k] = (U[i * m + j, r1] * V[j * p + k, r1] * W[k * n + i, r1] +
                                     U[i * m + j, r2] * V[j * p + k, r2] * W[k * n + i, r2])
    if np.linalg.matrix_rank(T1) > 1:
        return False, None, None, None
    
    # Mode-2 unfolding: (m) x (n*p)
    T2 = np.zeros((m, n * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T2[j, k * n + i] = (U[i * m + j, r1] * V[j * p + k, r1] * W[k * n + i, r1] +
                                     U[i * m + j, r2] * V[j * p + k, r2] * W[k * n + i, r2])
    if np.linalg.matrix_rank(T2) > 1:
        return False, None, None, None
    
    # Mode-3 unfolding: (p) x (n*m)
    T3 = np.zeros((p, n * m))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T3[k, i * m + j] = (U[i * m + j, r1] * V[j * p + k, r1] * W[k * n + i, r1] +
                                     U[i * m + j, r2] * V[j * p + k, r2] * W[k * n + i, r2])
    if np.linalg.matrix_rank(T3) > 1:
        return False, None, None, None
    
    # All unfoldings have rank 1, so tensor rank is 1.
    # Extract the rank-1 factors from the mode-1 unfolding
    s, U_s, Vt_s = np.linalg.svd(T1)
    if s[0] <= 1e-10:
        return False, None, None, None
    u_new = s[0] * U_s[:, 0]  # n-vector, need to reshape to n*m
    # Actually u_new should be n*m vector. Let me redo.
    
    # From T1[i, j*p+k] = u_new[i*m+j] * v_new[j*p+k] * w_new[k*n+i]
    # This is coupled. Let's use a different extraction.
    # Since all unfoldings are rank 1, pick any non-zero entry to anchor.
    # Find first non-zero in T1
    idx = np.unravel_index(np.argmax(np.abs(T1)), T1.shape)
    i0, jpk0 = idx
    val = T1[i0, jpk0]
    if abs(val) < 1e-15:
        return False, None, None, None
    
    # We need to solve for u, v, w such that u[i*m+j]*v[j*p+k]*w[k*n+i] = T1[i, j*p+k]
    # This is underdetermined. Use the fact that we can scale freely.
    # Set w[k*n+i] = 1 for all (k,i) where possible, then solve.
    # Actually, let's just verify the merge works by construction.
    
    # For now, return that merge is possible but don't extract exact factors.
    # We'll handle extraction separately.
    return True, None, None, None

def _reduce_rank(U, V, W, n, m, p, max_rounds=50):
    """
    Iteratively try to reduce rank by merging pairs of terms.
    Returns the reduced decomposition.
    """
    U, V, W = U.copy(), V.copy(), W.copy()
    rank = U.shape[1]
    
    for round_num in range(max_rounds):
        if time.time() > DEADLINE - 10:
            break
        
        merged = False
        for r1 in range(rank):
            if time.time() > DEADLINE - 10:
                break
            for r2 in range(r1 + 1, rank):
                # Try to merge r1 and r2
                can_merge, u_new, v_new, w_new = _try_merge(U, V, W, r1, r2, n, m, p)
                if can_merge:
                    if u_new is not None:
                        # We have explicit new factors
                        U[:, r1] = u_new
                        V[:, r1] = v_new
                        W[:, r1] = w_new
                        # Remove r2
                        U = np.delete(U, r2, axis=1)
                        V = np.delete(V, r2, axis=1)
                        W = np.delete(W, r2, axis=1)
                        rank -= 1
                        merged = True
                        break
                    else:
                        # Merge possible but no explicit factors - skip for now
                        pass
            if merged:
                break
        
        if not merged:
            break
    
    return U, V, W

def _block_strassen(n, m, p):
    """
    Construct block Strassen decomposition for 2x4x6.
    Split into 2x2 blocks and use Strassen for each.
    """
    strassen_U = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,1,1), (1,1,-1)]
    ]
    strassen_V = [
        [(0,0,1), (1,1,1)],
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (0,0,-1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (1,1,1)]
    ]
    strassen_W = [
        [(0,0,1), (1,1,1)],
        [(1,0,1)],
        [(0,1,1)],
        [(0,0,-1), (1,0,1)],
        [(0,0,1), (0,1,1)],
        [(0,0,1)],
        [(1,1,1)]
    ]
    
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for a_block in range(2):
        for b_block in range(3):
            for t_idx in range(7):
                r = a_block * 21 + b_block * 7 + t_idx
                for (i, j_local, coeff) in strassen_U[t_idx]:
                    j_global = a_block * 2 + j_local
                    U[i * m + j_global, r] += coeff
                for (j_local, k_local, coeff) in strassen_V[t_idx]:
                    j_global = a_block * 2 + j_local
                    k_global = b_block * 2 + k_local
                    V[j_global * p + k_global, r] += coeff
                for (i, k_local, coeff) in strassen_W[t_idx]:
                    k_global = b_block * 2 + k_local
                    W[k_global * n + i, r] += coeff
    
    return U, V, W

def _to_fraction_matrix(mat, tol=1e-9):
    """Convert float matrix to Fraction matrix, rounding near-integers."""
    result = []
    for row in mat:
        new_row = []
        for val in row:
            if abs(val) < tol:
                new_row.append(Fraction(0))
            elif abs(val - round(val)) < tol:
                new_row.append(Fraction(int(round(val))))
            elif abs(2*val - round(2*val)) < tol:
                new_row.append(Fraction(int(round(2*val)), 2))
            else:
                new_row.append(Fraction(val).limit_denominator(1000000))
        result.append(new_row)
    return result

def _to_list(mat):
    """Convert numpy array to nested list of fractions."""
    return _to_fraction_matrix(mat)

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x6 matrix.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook (rank 48)
    U, V, W = _schoolbook(n, m, p)
    
    # Try rank reduction
    U, V, W = _reduce_rank(U, V, W, n, m, p, max_rounds=100)
    rank_schoolbook = U.shape[1]
    
    # Also try block Strassen (rank 42)
    if time.time() < DEADLINE - 20:
        U_s, V_s, W_s = _block_strassen(n, m, p)
        if _verify(U_s, V_s, W_s, n, m, p, max_samples=30):
            # Try to reduce the Strassen decomposition too
            U_s, V_s, W_s = _reduce_rank(U_s, V_s, W_s, n, m, p, max_rounds=100)
            rank_strassen = U_s.shape[1]
            
            # Use whichever is better
            if rank_strassen < rank_schoolbook:
                U, V, W = U_s, V_s, W_s
                rank_schoolbook = rank_strassen
    
    # Final verification
    if not _verify(U, V, W, n, m, p, max_samples=50):
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    # Convert to list format with fractions
    U_list = _to_list(U)
    V_list = _to_list(V)
    W_list = _to_list(W)
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
