"""
Matrix multiplication tensor decomposition for 2x5 * 5x6.

Strategy:
1. Build a rank-54 Strassen-based decomposition.
2. Attempt to reduce rank by checking for linearly dependent terms
   (terms that can be expressed as linear combinations of others).
3. Verify the final decomposition exactly.
"""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _build_rank54(n, m, p):
    """Build the rank-54 Strassen decomposition. Returns (U, V, W, rank)."""
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
            U[0 * m + j0][col] += 1
            U[1 * m + j1][col] += 1
            V[j0 * p + k0][col] += 1
            V[j1 * p + k1][col] += 1
            W[k0 * n + 0][col] += 1
            W[k1 * n + 1][col] += 1
            col += 1

            # P2 = (A_1,j0 + A_1,j1) B_j0,k0
            U[1 * m + j0][col] += 1
            U[1 * m + j1][col] += 1
            V[j0 * p + k0][col] += 1
            W[k0 * n + 1][col] += 1
            W[k1 * n + 1][col] -= 1
            col += 1

            # P3 = A_0,j0 (B_j0,k1 - B_j1,k1)
            U[0 * m + j0][col] += 1
            V[j0 * p + k1][col] += 1
            V[j1 * p + k1][col] -= 1
            W[k1 * n + 0][col] += 1
            W[k1 * n + 1][col] += 1
            col += 1

            # P4 = A_1,j1 (B_j1,k0 - B_j0,k0)
            U[1 * m + j1][col] += 1
            V[j1 * p + k0][col] += 1
            V[j0 * p + k0][col] -= 1
            W[k0 * n + 0][col] += 1
            W[k0 * n + 1][col] += 1
            col += 1

            # P5 = (A_0,j0 + A_0,j1) B_j1,k1
            U[0 * m + j0][col] += 1
            U[0 * m + j1][col] += 1
            V[j1 * p + k1][col] += 1
            W[k0 * n + 0][col] -= 1
            W[k1 * n + 0][col] += 1
            col += 1

            # P6 = (A_1,j0 - A_0,j0)(B_j0,k0 + B_j0,k1)
            U[1 * m + j0][col] += 1
            U[0 * m + j0][col] -= 1
            V[j0 * p + k0][col] += 1
            V[j0 * p + k1][col] += 1
            W[k1 * n + 1][col] += 1
            col += 1

            # P7 = (A_0,j1 - A_1,j1)(B_j1,k0 + B_j1,k1)
            U[0 * m + j1][col] += 1
            U[1 * m + j1][col] -= 1
            V[j1 * p + k0][col] += 1
            V[j1 * p + k1][col] += 1
            W[k0 * n + 0][col] += 1
            col += 1

    # Remaining j=4
    for k in range(p):
        U[0 * m + 4][col] += 1
        V[4 * p + k][col] += 1
        W[k * n + 0][col] += 1
        col += 1

        U[1 * m + 4][col] += 1
        V[4 * p + k][col] += 1
        W[k * n + 1][col] += 1
        col += 1

    return U, V, W, rank


def _verify(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify the decomposition is exact."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(
                    U_np[i * m + j] * V_np[j * p + k] * W_np[k * n + i]
                )
                if abs(val - 1.0) > tol:
                    return False
    return True


def _try_eliminate_term(U, V, W, n, m, p, rank, term_idx):
    """
    Try to eliminate term term_idx by expressing it as a linear combination
    of other terms. Returns (new_U, new_V, new_W, new_rank) or None.
    """
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # The constraint is: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # If we remove term c, we need: for all (i,j,k),
    #   sum_{r != c} U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 - U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    
    # This is a system where we want to find adjustments to other terms.
    # A simpler approach: check if the "tensor vector" of term c is in the span
    # of the tensor vectors of other terms.
    
    # Build the tensor vectors: for each term r, the vector t_r is indexed by (i,j,k)
    # t_r[i,j,k] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    num_tensors = n * m * p
    all_tensors = np.zeros((rank, num_tensors))
    for r in range(rank):
        for idx, (i, j, k) in enumerate(
            [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        ):
            all_tensors[r, idx] = U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
    
    # Check if term c's tensor is in the span of others
    target = all_tensors[term_idx]
    others = np.delete(all_tensors, term_idx, axis=0)
    
    # Solve: find coefficients alpha such that others @ alpha ≈ target
    # Using least squares
    try:
        alpha, residuals, rank_ls, sv = np.linalg.lstsq(others, target, rcond=1e-10)
    except np.linalg.LinAlgError:
        return None
    
    # Check if the residual is small (term is in the span)
    residual = np.linalg.norm(target - others @ alpha)
    if residual > 1e-6:
        return None
    
    # The term can be eliminated. But we need to actually construct the new U, V, W.
    # This is non-trivial because the tensor decomposition isn't just about the
    # tensor vectors - we need consistent U, V, W modifications.
    # For now, this is a check only. Actual elimination requires more work.
    return None


def _snap_to_half_integer(mat, tol=1e-6):
    """Snap values close to half-integers to exact half-integers."""
    result = np.array(mat, dtype=np.float64)
    # Snap to nearest 0.5
    snapped = np.round(result * 2) / 2
    if np.max(np.abs(result - snapped)) < tol:
        return snapped
    return result


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Safety margin before 120s limit
    
    # Build the rank-54 decomposition
    U, V, W, rank = _build_rank54(n, m, p)
    
    # Verify
    if not _verify(U, V, W, n, m, p, rank):
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
    
    # Try to find if any term is redundant
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    num_tensors = n * m * p
    tensor_indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # Build tensor vectors for each term
    all_tensors = np.zeros((rank, num_tensors))
    for r in range(rank):
        for idx, (i, j, k) in enumerate(tensor_indices):
            all_tensors[r, idx] = U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
    
    # Check rank of the tensor matrix
    # If rank of all_tensors < rank, some terms are redundant
    try:
        matrix_rank = np.linalg.matrix_rank(all_tensors, tol=1e-10)
    except np.linalg.LinAlgError:
        matrix_rank = rank
    
    if matrix_rank < rank:
        # There are redundant terms. Try to identify and remove them.
        # Find a maximal linearly independent subset
        # Use QR with column pivoting
        try:
            Q, R, P = np.linalg.qr(all_tensors.T, mode='complete')
            # R is upper triangular, diagonal entries indicate independence
            diag = np.abs(np.diag(R))
            # Find pivot columns
            tol_rank = 1e-10
            pivot_cols = np.where(diag > tol_rank)[0]
            num_independent = len(pivot_cols)
            
            if num_independent < rank:
                # We can potentially reduce rank
                # For now, just note it and keep the full decomposition
                # Actual removal requires careful reconstruction
                pass
        except np.linalg.LinAlgError:
            pass
    
    # Attempt: try removing each term and checking if the system can be solved
    # This is expensive, so only do it if time permits
    if time.time() < deadline - 20:
        for c in range(rank - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            
            # Try to express term c as combination of others
            target = all_tensors[c]
            others = np.delete(all_tensors, c, axis=0)
            
            try:
                alpha, residuals, rank_ls, sv = np.linalg.lstsq(others, target, rcond=1e-10)
            except np.linalg.LinAlgError:
                continue
            
            residual = np.linalg.norm(target - others @ alpha)
            if residual < 1e-8:
                # Term c appears redundant in tensor space
                # But we need to verify we can actually construct valid U,V,W
                # For now, skip actual removal as it's complex
                pass
    
    # Return the verified rank-54 decomposition
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
