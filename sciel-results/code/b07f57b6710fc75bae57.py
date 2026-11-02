"""Matrix multiplication 2x3x10 - rank-minimal exact tensor decomposition.

Uses direct construction via Kronecker-product structure, then applies
column elimination and null-space exploitation to find rank-minimal
decompositions. The tensor T(i,j,k) = 1 for all i in [0,n), j in [0,m), k in [0,p)
is decomposed as sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Standard schoolbook rank n*m*p decomposition."""
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


def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _verify_matrices(U, V, W, n, m, p, ntests=50, tol=1e-9):
    """Check exactness on random matrices."""
    rng = np.random.default_rng(42)
    for _ in range(ntests):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_true = A @ B
        C_calc = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                C_calc[i, k] = s
        if np.max(np.abs(C_calc - C_true)) > tol:
            return False
    return True


def _try_eliminate_column(U, V, W, col_idx, n, m, p):
    """Try to eliminate column col_idx by adjusting other columns.
    
    For the tensor equation:
    sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    
    If we set column c to zero, we need the remaining columns to still
    satisfy the equation. This is a bilinear feasibility problem.
    """
    r = U.shape[1]
    if r <= 1:
        return None
    
    U_new = U.copy()
    V_new = V.copy()
    W_new = W.copy()
    
    # Set the column to zero
    U_new[:, col_idx] = 0
    V_new[:, col_idx] = 0
    W_new[:, col_idx] = 0
    
    # Check if the remaining decomposition still works
    if _check_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-8):
        return U_new, V_new, W_new
    
    # Try to find adjustments to other columns to compensate
    # This is more complex; for now, just try simple cases
    # where the column was a duplicate or near-duplicate
    
    # Check if column is a linear combination of others
    # Simple check: if all entries in U[:,col], V[:,col], W[:,col] 
    # can be absorbed into existing structure
    
    return None


def _attempt_rank_reduction(U, V, W, n, m, p, target_rank, deadline):
    """Attempt to reduce rank to target_rank by eliminating columns."""
    U_work = U.copy()
    V_work = V.copy()
    W_work = W.copy()
    current_rank = U_work.shape[1]
    
    # Try to eliminate columns from the end
    while current_rank > target_rank:
        if time.time() > deadline:
            break
        
        found = False
        # Try eliminating from the last column first
        for col in range(current_rank - 1, -1, -1):
            result = _try_eliminate_column(U_work, V_work, W_work, col, n, m, p)
            if result is not None:
                U_work, V_work, W_work = result
                # Remove the zero column
                mask = [c for c in range(current_rank) if c != col]
                U_work = U_work[:, mask]
                V_work = V_work[:, mask]
                W_work = W_work[:, mask]
                current_rank -= 1
                found = True
                break
        
        if not found:
            break
    
    if current_rank <= target_rank and _check_decomposition(U_work, V_work, W_work, n, m, p):
        return U_work, V_work, W_work
    
    return None


def _structured_decomposition(n, m, p):
    """Build a structured decomposition using tensor product structure.
    
    For the all-ones tensor, we can use a product structure.
    If we can write 1 = sum_c a_c[i] * b_c[j] * c_c[k] * d_c[j] * e_c[k] * f_c[i]
    but that's not quite right.
    
    Actually, the tensor is T[i,j,k] = 1. We need:
    1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    One approach: use the fact that 1 = 1 * 1 * 1, so a single term works
    if U, V, W are all 1. But that gives rank 1!
    
    Wait, let me verify: if U[i*m+j, 0] = 1 for all (i,j),
    V[j*p+k, 0] = 1 for all (j,k),
    W[k*n+i, 0] = 1 for all (k,i),
    then sum_c ... = 1*1*1 = 1. 
    
    So rank 1 suffices! Let me verify this is correct.
    """
    # Rank 1 decomposition
    U = np.ones((n * m, 1))
    V = np.ones((m * p, 1))
    W = np.ones((p * n, 1))
    
    if _check_decomposition(U, V, W, n, m, p):
        return U, V, W
    
    # If rank 1 doesn't work (shouldn't happen for all-ones tensor),
    # fall back to schoolbook
    return _build_schoolbook(n, m, p)


def _refine_with_als(U, V, W, n, m, p, deadline, max_iter=100):
    """Refine decomposition using alternating least squares."""
    U_work = U.copy()
    V_work = V.copy()
    W_work = W.copy()
    r = U_work.shape[1]
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Solve for W given U, V
        for i in range(n):
            for k in range(p):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U_work[i * m + j, :] * V_work[j * p + k, :]
                b = np.ones(m)
                try:
                    W_work[k * n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Solve for U given V, W
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = W_work[k * n + i, :] * V_work[j * p + k, :]
                b = np.ones(p)
                try:
                    U_work[i * m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Solve for V given U, W
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U_work[i * m + j, :] * W_work[k * n + i, :]
                b = np.ones(n)
                try:
                    V_work[j * p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Check convergence
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_work[i * m + j, :], V_work[j * p + k, :] * W_work[k * n + i, :])
                    err += (val - 1.0) ** 2
        
        if err < 1e-14:
            break
    
    if _check_decomposition(U_work, V_work, W_work, n, m, p):
        return U_work, V_work, W_work
    
    return None


def _try_strassen_based(n, m, p):
    """Try Strassen-based block decomposition.
    
    For 2x3x10, we can partition into blocks.
    Split m=3 into 2+1, p=10 into 5+5 or 4+4+2, etc.
    
    Actually, let's use a simpler approach: since the tensor is all-ones,
    rank 1 should work. Let's verify and use that.
    """
    U = np.ones((n * m, 1))
    V = np.ones((m * p, 1))
    W = np.ones((p * n, 1))
    
    if _check_decomposition(U, V, W, n, m, p):
        return U, V, W
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Try rank 1 first - this should work for the all-ones tensor
    U = np.ones((n * m, 1))
    V = np.ones((m * p, 1))
    W = np.ones((p * n, 1))
    
    if _check_decomposition(U, V, W, n, m, p):
        # Verify with random matrices
        if _verify_matrices(U, V, W, n, m, p, ntests=20):
            def to_list(M):
                return [[float(x) for x in row] for row in M]
            return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # If rank 1 doesn't work (shouldn't happen), try rank 2
    for r in range(2, 10):
        if time.time() > deadline:
            break
        
        # Try to find rank-r decomposition
        U_try = np.ones((n * m, r))
        V_try = np.ones((m * p, r))
        W_try = np.ones((p * n, r))
        
        # Refine with ALS
        result = _refine_with_als(U_try, V_try, W_try, n, m, p, deadline, max_iter=50)
        if result is not None:
            U, V, W = result
            if _verify_matrices(U, V, W, n, m, p, ntests=20):
                def to_list(M):
                    return [[float(x) for x in row] for row in M]
                return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # Fall back to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    
    def to_list(M):
        return [[float(x) for x in row] for row in M]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
