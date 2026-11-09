"""Matrix multiplication 2,4,6 – rank-minimal decomposition via block decomposition."""
DIMENSIONS = [2, 4, 6]

import time
import numpy as np
from scipy.optimize import lsq_linear

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Standard O(nmp) decomposition for multiplying (n x m) by (m x p).
    Returns U (n*m x nmp), V (m*p x nmp), W (p*n x nmp)."""
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
    return U, V, W


def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen)."""
    n_s, m_s, p_s = 2, 2, 2
    rank_s = 7
    U = np.zeros((n_s * m_s, rank_s), dtype=float)
    V = np.zeros((m_s * p_s, rank_s), dtype=float)
    W = np.zeros((p_s * n_s, rank_s), dtype=float)

    # M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # c00, c11

    # M2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1  # c10, c11

    # M3 = a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # c01, c11

    # M4 = a11(b10-b00)
    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[2, 3] = 1  # c00, c10

    # M5 = (a00+a10)b11
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # c00, c01

    # M6 = (a11-a10)(b00+b10)
    U[2, 5] = -1; U[3, 5] = 1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1  # c11

    # M7 = (a01-a11)(b01+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # c00

    return U, V, W


def _verify_decomp(U, V, W, n, m, p, rank, tol=1e-10):
    """Verify decomposition."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _build_2x4x2_from_strassen(U_s, V_s, W_s):
    """Build 2x4x2 decomposition (rank 14) from two 2x2x2 Strassen blocks."""
    n, m, p = 2, 4, 2
    rank = 14
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Block 1: j in {0,1}
    for r in range(7):
        for i in range(2):
            for jl in range(2):
                U[i * 4 + jl, r] = U_s[i * 2 + jl, r]
                for k in range(2):
                    V[jl * 2 + k, r] = V_s[jl * 2 + k, r]
                    W[k * 2 + i, r] = W_s[k * 2 + i, r]

    # Block 2: j in {2,3}
    for r in range(7):
        for i in range(2):
            for jl in range(2):
                U[i * 4 + (jl + 2), r + 7] = U_s[i * 2 + jl, r]
                for k in range(2):
                    V[(jl + 2) * 2 + k, r + 7] = V_s[jl * 2 + k, r]
                    W[k * 2 + i, r + 7] = W_s[k * 2 + i, r]

    return U, V, W


def _build_2x4x6_from_blocks(U1, V1, W1):
    """
    Build 2x4x6 decomposition from three 2x4x2 blocks.
    k blocks: {0,1}, {2,3}, {4,5}
    Total rank = 3 * 14 = 42.
    """
    n, m, p = 2, 4, 6
    rank1 = U1.shape[1]  # 14
    total_rank = rank1 * 3  # 42
    
    U_total = np.zeros((n * m, total_rank), dtype=float)
    V_total = np.zeros((m * p, total_rank), dtype=float)
    W_total = np.zeros((p * n, total_rank), dtype=float)

    for b in range(3):
        offset = b * rank1
        U_total[:, offset:offset + rank1] = U1
        # V: j in 0..3, k_local in 0..1 -> k_global = k_local + b*2
        for j in range(4):
            for kl in range(2):
                V_total[j * 6 + (kl + b * 2), offset:offset + rank1] = V1[j * 2 + kl, :]
        # W: k_local in 0..1, i in 0..1 -> k_global = k_local + b*2
        for kl in range(2):
            for i in range(2):
                W_total[(kl + b * 2) * 2 + i, offset:offset + rank1] = W1[kl * 2 + i, :]

    return U_total, V_total, W_total


def _try_lower_rank_decomposition(n, m, p, target_rank, deadline):
    """
    Attempt to find a decomposition with rank <= target_rank using lsq_linear.
    Formulate the tensor equation as a linear system and solve for one factor
    given random initializations of the other two, then iterate.
    """
    if time.time() > deadline:
        return None
    
    # The matrix multiplication tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want T[i,j,k] = 1 for all (i,j,k).
    # This is a 3-way tensor of size n x m x p.
    # Flatten: for each (i,j,k), we have one equation.
    # Total equations: n*m*p = 48 for 2x4x6.
    
    n, m, p = 2, 4, 6
    num_eqs = n * m * p
    
    # Strategy: fix U and W randomly (or with structure), solve for V, then fix V and W, solve for U, etc.
    # But lsq_linear solves min ||A x - b||. 
    # The equation is bilinear in (U, V, W). We can't directly solve for all three.
    # 
    # Alternative: use the known structure. 
    # Let's try to find a decomposition with rank < 42.
    # 
    # Actually, a simpler approach for small sizes:
    # We know the schoolbook rank is 48.
    # Strassen-based gives 42.
    # Let's see if we can do better with a general search.
    
    # For now, this is a placeholder that returns None to avoid wasting time
    # on a complex search that may not succeed within the deadline.
    # The block decomposition with rank 42 is likely near-optimal for this small size.
    
    # Let's try one quick thing: check if rank 41 or 40 is possible with a random restart
    # using lsq_linear in an alternating least squares fashion.
    
    def _als_decompose(n, m, p, rank, iters=10, seed=42):
        """Alternating least squares for tensor decomposition."""
        rng = np.random.RandomState(seed)
        
        # Initialize factors
        U = rng.randn(n * m, rank) * 0.1
        V = rng.randn(m * p, rank) * 0.1
        W = rng.randn(p * n, rank) * 0.1
        
        for it in range(iters):
            if time.time() > deadline:
                return None
            
            # Update U: for each (i,j), solve for U[i*m+j, :]
            # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
            # This is linear in U[i*m+j, :]
            # Let a_k = V[j*p+k, :] * W[k*n+i, :] (element-wise product, shape (rank,))
            # Then sum_r U[i*m+j, r] * a_k[r] = 1 for all k
            # So we need U[i*m+j, :] * A^T = ones(p) where A[k, r] = a_k[r]
            # i.e., A @ U[i*m+j, :]^T = ones(p)
            
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve A @ u = ones(p)
                    try:
                        u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i * m + j, :] = u
                    except:
                        pass
            
            # Update V: for each (j,k), solve for V[j*p+k, :]
            # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
            # Let b_i = U[i*m+j, :] * W[k*n+i, :]
            # B @ v = ones(n) where B[i, r] = b_i[r]
            
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, rank))
                    for i in range(n):
                        B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        v, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        V[j * p + k, :] = v
                    except:
                        pass
            
            # Update W: for each (k,i), solve for W[k*n+i, :]
            # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
            # Let c_j = U[i*m+j, :] * V[j*p+k, :]
            # C @ w = ones(m) where C[j, r] = c_j[r]
            
            for k in range(p):
                for i in range(n):
                    C = np.zeros((m, rank))
                    for j in range(m):
                        C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        w, _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                        W[k * n + i, :] = w
                    except:
                        pass
        
        # Verify
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > 1e-8:
                        return None
        
        return U, V, W
    
    # Try ranks from 41 down to 30
    for rank in range(min(target_rank, 41), max(target_rank - 15, 20) - 1, -1):
        if time.time() > deadline:
            break
        
        # Try a few random seeds
        for seed in range(5):
            if time.time() > deadline:
                break
            
            result = _als_decompose(n, m, p, rank, iters=20, seed=seed)
            if result is not None:
                U, V, W = result
                # Snap to half-integers
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                
                # Verify snapped version
                if _verify_decomp(U_snapped, V_snapped, W_snapped, n, m, p, rank, tol=1e-10):
                    return U_snapped, V_snapped, W_snapped
                
                # If snapping fails, try integer snapping
                U_snapped_int = np.round(U)
                V_snapped_int = np.round(V)
                W_snapped_int = np.round(W)
                
                if _verify_decomp(U_snapped_int, V_snapped_int, W_snapped_int, n, m, p, rank, tol=1e-10):
                    return U_snapped_int, V_snapped_int, W_snapped_int
    
    return None


def construct():
    """Construct rank-minimal exact tensor decomposition for 2x4 * 4x6 multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 2, 4, 6

    U_s, V_s, W_s = _strassen_2x2x2()

    if not _verify_decomp(U_s, V_s, W_s, 2, 2, 2, 7):
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U1, V1, W1 = _build_2x4x2_from_strassen(U_s, V_s, W_s)

    if not _verify_decomp(U1, V1, W1, 2, 4, 2, 14):
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Build the full 2x4x6 decomposition
    U_total, V_total, W_total = _build_2x4x6_from_blocks(U1, V1, W1)
    total_rank = U_total.shape[1]

    if not _verify_decomp(U_total, V_total, W_total, n, m, p, total_rank):
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find a lower rank decomposition
    best_U, best_V, best_W = U_total, V_total, W_total
    best_rank = total_rank

    if time.time() < deadline - 30:  # Leave 30s safety margin
        lower_result = _try_lower_rank_decomposition(n, m, p, best_rank - 1, deadline)
        if lower_result is not None:
            U_lower, V_lower, W_lower = lower_result
            lower_rank = U_lower.shape[1]
            if lower_rank < best_rank:
                best_U, best_V, best_W = U_lower, V_lower, W_lower
                best_rank = lower_rank

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
