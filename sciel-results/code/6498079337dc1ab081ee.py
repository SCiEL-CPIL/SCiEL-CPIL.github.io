"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy import linalg

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=2, m=4, p=4):
    """Verify the decomposition. Returns max error."""
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err


def _verify_fast(U, V, W, n=2, m=4, p=4):
    """Vectorized verification. Returns max error."""
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for k in range(p):
            # For fixed i,k, we need: sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # = W[k*n+i, :] dot (sum over j of outer(U[i*m+j,:], V[j*p+k,:]))
            # Actually: sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # = sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] * V[j*p+k,r])
            # Let S[j,r] = U[i*m+j,r] * V[j*p+k,r] for each j
            # Then sum_j S[j,r] for each r, dot with W[k*n+i,:]
            for j in range(m):
                pass  # Just doing it directly for clarity
            # Direct but vectorized over r:
            prod_sum = np.zeros(rank)
            for j in range(m):
                prod_sum += U[i*m+j, :] * V[j*p+k, :]
            val = np.dot(prod_sum, W[k*n+i, :])
            err = abs(val - 1.0)
            if err > max_err:
                max_err = err
    return max_err


def _build_strassen_28():
    """Build rank-28 block Strassen decomposition for 2x4 * 4x4.
    
    Splits into four 2x2 * 2x2 sub-problems, each using 7 Strassen terms.
    """
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen terms: each is (u_coeffs, v_coeffs, w_coeffs)
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Verified Strassen formulation:
    # M1 = (A00+A11)(B00+B11): C00+=1, C11+=1
    # M2 = (A10+A11)B00:       C10+=1, C11-=1
    # M3 = A00(B10-B11):        C01+=1, C11+=1
    # M4 = A11(B01-B10):        C00+=1, C10+=1
    # M5 = (A00+A10)B11:        C00-=1, C01+=1
    # M6 = (A01-A10)(B00+B10):  C00+=1, C01-=1, C10+=1, C11-=1
    # M7 = (A01-A11)(B01+B11):  C00+=1, C01+=1, C10-=1, C11-=1
    
    strassen_terms = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
        ([(0,1)], [(2,1),(3,-1)], [(1,1),(3,1)]),
        ([(3,1)], [(1,1),(2,-1)], [(0,1),(2,1)]),
        ([(0,1),(2,1)], [(3,1)], [(0,-1),(1,1)]),
        ([(1,1),(2,-1)], [(0,1),(2,1)], [(0,1),(1,-1),(2,1),(3,-1)]),
        ([(1,1),(3,-1)], [(1,1),(3,1)], [(0,1),(1,1),(2,-1),(3,-1)]),
    ]
    
    def map_u(a_row, a_col_local):
        """Map A local (row, col within 2x2 block) to global U index.
        a_col_local: 0 or 1 for left block, 2 or 3 for right block."""
        return a_row * 4 + a_col_local
    
    def map_v(b_row, b_col):
        """Map B (row, col) to global V index."""
        return b_row * 4 + b_col
    
    def map_w(c_col, c_row):
        """Map C (col, row) to global W index using trace convention W[k*n+i]."""
        return c_col * 2 + c_row
    
    def apply_block(a_col_offset, b_row_offset, b_col_offset, c_col_offset, start):
        for t in strassen_terms:
            u_c, v_c, w_c = t
            for (li, coeff) in u_c:
                lr, lc = li // 2, li % 2
                U[map_u(lr, a_col_offset + lc), start] += coeff
            for (li, coeff) in v_c:
                lr, lc = li // 2, li % 2
                V[map_v(b_row_offset + lr, b_col_offset + lc), start] += coeff
            for (li, coeff) in w_c:
                lr, lc = li // 2, li % 2
                W[map_w(c_col_offset + lc, lr), start] += coeff
            start += 1
        return start
    
    # Four 2x2 blocks:
    # A_L * B_TL -> C_L (a_col=0, b_row=0, b_col=0, c_col=0)
    # A_R * B_BL -> C_L (a_col=2, b_row=2, b_col=0, c_col=0)
    # A_L * B_TR -> C_R (a_col=0, b_row=0, b_col=2, c_col=2)
    # A_R * B_BR -> C_R (a_col=2, b_row=2, b_col=2, c_col=2)
    
    col = apply_block(0, 0, 0, 0, 0)
    col = apply_block(2, 2, 0, 0, col)
    col = apply_block(0, 0, 2, 2, col)
    col = apply_block(2, 2, 2, 2, col)
    
    return U, V, W


def _build_schoolbook():
    """Build the rank-32 schoolbook decomposition."""
    n, m, p = 2, 4, 4
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W


def _find_rank_dependency(U, V, W, n=2, m=4, p=4):
    """Check if the rank-1 tensor components are linearly dependent.
    
    Returns (alpha, r_elim) if a dependency is found, else (None, None).
    alpha[r_elim] = 1, and sum_{r} alpha[r] * T_r = 0.
    """
    rank = U.shape[1]
    N = n * m * p  # 32
    
    # Build tensor vector for each rank-1 component
    T = np.zeros((rank, N))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[r, i*m*p + j*p + k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # SVD to find null space
    try:
        U_svd, S_svd, Vt_svd = np.linalg.svd(T, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, None
    
    tol = 1e-10
    # Count significant singular values
    num_rank = np.sum(S_svd > tol)
    
    if num_rank >= rank:
        return None, None
    
    # The null space is spanned by the last (rank - num_rank) rows of Vt
    # Take the last row (corresponding to smallest singular value)
    null_vec = Vt_svd[-1, :]
    
    # Find the component with largest magnitude to eliminate
    r_elim = int(np.argmax(np.abs(null_vec)))
    
    if abs(null_vec[r_elim]) < 1e-15:
        return None, None
    
    alpha = null_vec / null_vec[r_elim]
    alpha[r_elim] = 1.0
    
    # Verify the dependency
    recon = np.zeros(N)
    for r in range(rank):
        if abs(alpha[r]) > 1e-15:
            recon += alpha[r] * T[r, :]
    
    if np.max(np.abs(recon)) > 1e-8:
        return None, None
    
    return alpha, r_elim


def _reduce_rank_one(U, V, W, alpha, r_elim, n=2, m=4, p=4):
    """Given a dependency sum_r alpha[r] * T_r = 0 with alpha[r_elim]=1,
    try to construct a valid rank-(rank-1) decomposition.
    
    This is non-trivial: we need to find new U', V', W' of rank-1 columns
    such that the sum still equals the original tensor.
    
    Strategy: We can't simply remove one term. Instead, we use the dependency
    to perform a change of basis in the coefficient space.
    
    Actually, the correct approach: if T_{r_elim} = sum_{r != r_elim} alpha[r] * T_r,
    then the tensor is the same whether we include term r_elim or the combination.
    But we can't just drop it - we need to absorb it into the other terms.
    
    The proper way: we need to find U', V', W' with rank-1 columns such that
    sum_{r' < rank-1} U'_{:,r'} * V'_{:,r'} * W'_{:,r'} = original tensor.
    
    This is equivalent to finding a rank-(rank-1) decomposition, which is generally
    not possible by simple elimination. The dependency only tells us the 28 rank-1
    tensors are linearly dependent as vectors, which is always true since they live
    in a 32-dim space and 28 < 32... wait, 28 < 32 so they could be independent.
    
    Let me reconsider. If they ARE linearly dependent, it means there exists a
    non-trivial linear combination that gives zero tensor. This doesn't directly
    give a lower-rank decomposition.
    
    However, we can use this to perturb the decomposition and try to find one
    where one term becomes zero (can be removed).
    """
    rank = U.shape[1]
    new_rank = rank - 1
    
    # The dependency means: T_{r_elim} = sum_{r != r_elim} alpha[r] * T_r
    # where T_r[i,j,k] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # This doesn't directly help construct a lower-rank decomposition.
    # We return the original as fallback.
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Build rank-28 Strassen decomposition
    U, V, W = _build_strassen_28()
    
    # Verify
    err = _verify(U, V, W)
    
    if err < 1e-9:
        # Valid rank-28 decomposition - this is our best known
        # The structural question: is rank < 28 possible?
        # For 2x4 * 4x4, the tensor lives in R^{8} x R^{16} x R^{8}
        # Lower bound: max(8, 16, 8) = 16, but that's weak.
        # The 2x4*4x4 multiplication can be seen as 2 independent 1x4*4x4
        # plus coupling... Actually it's one 2x4 * 4x4 product.
        # Known: rank of m x n * n x p multiplication is at least max(m*n, n*p, m*p)
        # = max(8, 16, 8) = 16. And at most m*n*p = 32.
        # Strassen gives 28. Can we do better?
        # 
        # Key structural insight: the 2x4*4x4 tensor has a special structure
        # because n=2 is small. We can potentially use the fact that the
        # first matrix has only 2 rows to find a more efficient decomposition.
        #
        # For now, return rank-28 as the safe answer.
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        U, V, W = _build_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
