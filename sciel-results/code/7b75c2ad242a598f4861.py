"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START


def _get_strassen_2x2x2():
    """
    Return U, V, W factors for 2x2x2 matrix multiplication (Strassen, rank 7).
    U: (4, 7) indexed as U[i*2+j, c] for A[i,j]
    V: (4, 7) indexed as V[j*2+k, c] for B[j,k]
    W: (4, 7) indexed as W[k*2+i, c] for C[k,i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # U: A factors
    U[0, 0] = 1   # a in m0
    U[3, 0] = 1   # d in m0
    U[2, 1] = 1   # c in m1
    U[3, 1] = 1   # d in m1
    U[0, 2] = 1   # a in m2
    U[3, 3] = 1   # d in m3
    U[0, 4] = 1   # a in m4
    U[1, 4] = 1   # b in m4
    U[2, 5] = 1   # c in m5
    U[0, 5] = -1  # -a in m5
    U[1, 6] = 1   # b in m6
    U[3, 6] = -1  # -d in m6

    # V: B factors
    V[0, 0] = 1   # e in m0
    V[3, 0] = 1   # h in m0
    V[0, 1] = 1   # e in m1
    V[1, 2] = 1   # f in m2
    V[3, 2] = -1  # -h in m2
    V[2, 3] = 1   # g in m3
    V[0, 3] = -1  # -e in m3
    V[3, 4] = 1   # h in m4
    V[0, 5] = 1   # e in m5
    V[1, 5] = 1   # f in m5
    V[2, 6] = 1   # g in m6
    V[3, 6] = 1   # h in m6

    # W: C factors (W[k*2+i, c])
    W[0, 0] = 1   # m0 in C[0,0]
    W[0, 3] = 1   # m3 in C[0,0]
    W[0, 4] = -1  # -m4 in C[0,0]
    W[0, 6] = 1   # m6 in C[0,0]
    W[2, 2] = 1   # m2 in C[0,1]
    W[2, 4] = 1   # m4 in C[0,1]
    W[1, 1] = 1   # m1 in C[1,0]
    W[1, 3] = 1   # m3 in C[1,0]
    W[3, 0] = 1   # m0 in C[1,1]
    W[3, 1] = -1  # -m1 in C[1,1]
    W[3, 2] = 1   # m2 in C[1,1]
    W[3, 5] = 1   # m5 in C[1,1]

    return U, V, W


def _build_strassen_3x5x6(n, m, p):
    """
    Build tensor decomposition for 3x5 @ 5x6 using block-Strassen structure.
    
    Split: top 2 rows use Strassen on j-pairs, bottom 1 row uses schoolbook.
    Top: 3 k-pairs * (7 + 7 + 4) = 54
    Bottom: 5 * 6 = 30
    Total: 84
    """
    sU, sV, sW = _get_strassen_2x2x2()

    rank = 84
    U = np.zeros((n * m, rank))  # 15 x 84
    V = np.zeros((m * p, rank))  # 30 x 84
    W = np.zeros((p * n, rank))  # 18 x 84

    col = 0

    for k_block in range(3):
        k0 = k_block * 2
        k1 = k0 + 1

        # Top 2 rows (i=0,1), j in {0,1}: Strassen 2x2x2
        for c in range(7):
            U[0, col] = sU[0, c]
            U[1, col] = sU[1, c]
            U[5, col] = sU[2, c]
            U[6, col] = sU[3, c]
            V[k0, col] = sV[0, c]
            V[k0 + 1, col] = sV[1, c]
            V[6 + k0, col] = sV[2, c]
            V[6 + k0 + 1, col] = sV[3, c]
            W[k0 * 3, col] = sW[0, c]
            W[k0 * 3 + 1, col] = sW[1, c]
            W[(k0 + 1) * 3, col] = sW[2, c]
            W[(k0 + 1) * 3 + 1, col] = sW[3, c]
            col += 1

        # Top 2 rows (i=0,1), j in {2,3}: Strassen 2x2x2
        for c in range(7):
            U[2, col] = sU[0, c]
            U[3, col] = sU[1, c]
            U[7, col] = sU[2, c]
            U[8, col] = sU[3, c]
            V[12 + k0, col] = sV[0, c]
            V[12 + k0 + 1, col] = sV[1, c]
            V[18 + k0, col] = sV[2, c]
            V[18 + k0 + 1, col] = sV[3, c]
            W[k0 * 3, col] = sW[0, c]
            W[k0 * 3 + 1, col] = sW[1, c]
            W[(k0 + 1) * 3, col] = sW[2, c]
            W[(k0 + 1) * 3 + 1, col] = sW[3, c]
            col += 1

        # Top 2 rows (i=0,1), j=4: schoolbook 2x1x2
        for i in range(2):
            for k in range(2):
                U[i, col] = 1
                V[24 + k0 + k, col] = 1
                W[(k0 + k) * 3 + i, col] = 1
                col += 1

        # Bottom row (i=2): schoolbook 1x5x2
        for j in range(5):
            for k in range(2):
                U[10 + j, col] = 1
                V[j * 6 + k0 + k, col] = 1
                W[(k0 + k) * 3 + 2, col] = 1
                col += 1

    if col != rank:
        if col > rank:
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
        else:
            rank = col
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]

    return U, V, W, rank


def _verify_fast(U, V, W, n, m, p):
    """Fast verification using einsum."""
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)


def _verify(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)

    r = U.shape[1]

    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)

    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)

    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True


def _try_randomized_rank_reduction(U, V, W, n, m, p, deadline, max_iter=200):
    """
    Try to reduce rank below 84 using randomized search.
    
    Strategy: Start with the rank-84 solution, then try to find a rank-83
    decomposition by adding a random perturbation and checking if the
    perturbed system has a valid solution.
    
    We test if there exists a non-trivial combination of columns that
    sums to zero in the tensor sense.
    """
    r = U.shape[1]
    start_rank = r
    
    # Try to find linear dependencies among the rank terms
    # A rank term c contributes: sum over (i,j,k) of U[i,j,c]*V[j,k,c]*W[k,i,c] * delta
    
    # For each column, compute its "signature" in tensor space
    # The tensor T has dimensions n*m*p = 90
    # Each column c contributes a rank-1 tensor: U_col * V_col * W_col (outer product structure)
    
    # Build the 90-dimensional representation of each column
    # T_c[i,j,k] = U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # We want to find coefficients alpha_c such that sum_c alpha_c * T_c = 0
    # and not all alpha_c are zero.
    
    # Build matrix M where M[(i,j,k), c] = U[i,j,c] * V[j,k,c] * W[k,i,c]
    # This is a 90 x r matrix
    # If rank(M) < r, there's a linear dependency.
    
    M = np.zeros((n * m * p, r))
    for c in range(r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M[(i * m + j) * p + k, c] = U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
    
    # Check rank of M
    try:
        rank_M = np.linalg.matrix_rank(M, tol=1e-10)
        if rank_M < r:
            # Found a dependency! Find the null vector
            _, s_vals, Vt = np.linalg.svd(M, full_matrices=False)
            # The last row(s) of Vt give null space vectors
            null_dim = r - rank_M
            null_vectors = Vt[rank_M:]  # shape (null_dim, r)
            
            # Try removing one column at a time using the null vectors
            # For each null vector, we can add it to any column and subtract from others
            # to create a new valid decomposition with potentially fewer non-zero columns.
            
            # Simple approach: for each column, check if it can be expressed as
            # a linear combination of others using the null space
            removed = set()
            
            for c in range(r):
                if c in removed:
                    continue
                if time.time() > deadline - 10:
                    break
                
                # Check if column c is in the span of other columns in M-space
                # Build M without column c
                mask = [c2 for c2 in range(r) if c2 != c and c2 not in removed]
                if len(mask) == 0:
                    continue
                    
                M_reduced = M[:, mask]
                target = M[:, c]
                
                # Check if target is in span of M_reduced
                # Use least squares
                try:
                    coeffs, residuals, rank_reduced, _ = np.linalg.lstsq(M_reduced, target, rcond=None)
                    # Check reconstruction error
                    recon = M_reduced @ coeffs
                    err = np.linalg.norm(target - recon) / (np.linalg.norm(target) + 1e-15)
                    
                    if err < 1e-8:
                        # Column c is redundant in tensor space
                        removed.add(c)
                except:
                    pass
            
            if removed:
                keep = [c for c in range(r) if c not in removed]
                new_U = U[:, keep]
                new_V = V[:, keep]
                new_W = W[:, keep]
                new_rank = len(keep)
                
                if new_rank < r and _verify_fast(new_U, new_V, new_W, n, m, p):
                    return new_U, new_V, new_W, new_rank
                    
    except Exception as e:
        pass
    
    return None


def _try_perturbation_search(U, V, W, n, m, p, deadline, n_trials=50):
    """
    Try to find a rank-83 decomposition by perturbing the rank-84 solution
    and solving the resulting underdetermined system.
    
    Idea: If we have rank r = 84, we can try to set one column to be a
    linear combination of others and check if the tensor equation still holds.
    """
    r = U.shape[1]
    start_rank = r
    
    # For each pair of columns (a, b), try to eliminate one by expressing
    # it as a scalar multiple of the other (after scaling)
    # This is a simple check for proportional columns
    
    removed = set()
    for a in range(r):
        if a in removed:
            continue
        if time.time() > deadline - 15:
            break
        for b in range(a + 1, r):
            if b in removed:
                continue
            
            # Check if column b is proportional to column a in all three matrices
            ua, ub = U[:, a], U[:, b]
            va, vb = V[:, a], V[:, b]
            wa, wb = W[:, a], W[:, b]
            
            # Find scale from first non-zero element
            scale = None
            for idx in range(len(ua)):
                if abs(ua[idx]) > 1e-12:
                    scale = ub[idx] / ua[idx]
                    break
            
            if scale is None:
                continue
            
            if (np.allclose(ua * scale, ub, atol=1e-9) and
                np.allclose(va * scale, vb, atol=1e-9) and
                np.allclose(wa * scale, wb, atol=1e-9)):
                removed.add(b)
    
    if removed:
        keep = [c for c in range(r) if c not in removed]
        new_U = U[:, keep]
        new_V = V[:, keep]
        new_W = W[:, keep]
        new_rank = len(keep)
        
        if new_rank < r and _verify_fast(new_U, new_V, new_W, n, m, p):
            return new_U, new_V, new_W, new_rank
    
    return None


def _schoolbook_3x5x6(n, m, p):
    """Fallback: schoolbook matrix multiplication, rank n*m*p = 90."""
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
    
    return U, V, W, rank


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s wall-clock

    # Primary: block-Strassen rank-84 solution
    U, V, W, rank = _build_strassen_3x5x6(n, m, p)

    # Verify the primary solution
    if _verify_fast(U, V, W, n, m, p):
        # Try to reduce rank further
        if time.time() < deadline - 30:
            result = _try_randomized_rank_reduction(U, V, W, n, m, p, deadline)
            if result is not None:
                U, V, W, rank = result
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        if time.time() < deadline - 15:
            result = _try_perturbation_search(U, V, W, n, m, p, deadline)
            if result is not None:
                U, V, W, rank = result
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    U, V, W, rank = _schoolbook_3x5x6(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
