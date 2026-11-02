"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition.

Uses a structured decomposition that exploits the n=2 structure to share
intermediate products across column blocks of B, aiming for rank < 44.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]


def _get_target_indices():
    n, m, p = DIMENSIONS
    target = np.zeros((n * m, m * p, p * n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = True
    return target


def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    T = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])

    target = _get_target_indices()
    target_err = np.max(np.abs(T[target] - 1.0))
    non_target_err = np.max(np.abs(T[~target])) if (~target).any() else 0.0
    return max(target_err, non_target_err) <= tol


def _build_rank44():
    """Explicit rank-44 decomposition: 4 blocks of (7 Strassen + 4 outer)."""
    n, m, p = 2, 3, 8
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    for pair in range(4):
        k0 = 2 * pair
        k1 = 2 * pair + 1

        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0, col] = 1; U[4, col] = 1
        V[k0, col] = 1; V[8 + k1, col] = 1
        W[2 * k0, col] = 1; W[2 * k1 + 1, col] = 1
        col += 1

        # M2 = (A10 + A11) * B0k0
        U[3, col] = 1; U[4, col] = 1
        V[k0, col] = 1
        W[2 * k0 + 1, col] = 1; W[2 * k1 + 1, col] = -1
        col += 1

        # M3 = A00 * (B0k1 - B1k1)
        U[0, col] = 1
        V[k1, col] = 1; V[8 + k1, col] = -1
        W[2 * k1, col] = 1; W[2 * k1 + 1, col] = 1
        col += 1

        # M4 = A11 * (B1k0 - B0k0)
        U[4, col] = 1
        V[8 + k0, col] = 1; V[k0, col] = -1
        W[2 * k0, col] = 1; W[2 * k0 + 1, col] = 1
        col += 1

        # M5 = (A00 + A01) * B1k1
        U[0, col] = 1; U[1, col] = 1
        V[8 + k1, col] = 1
        W[2 * k0, col] = -1; W[2 * k1, col] = 1
        col += 1

        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[3, col] = 1; U[0, col] = -1
        V[k0, col] = 1; V[k1, col] = 1
        W[2 * k1 + 1, col] = 1
        col += 1

        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[1, col] = 1; U[4, col] = -1
        V[8 + k0, col] = 1; V[8 + k1, col] = 1
        W[2 * k0, col] = 1
        col += 1

        # P1 = A02 * B2k0
        U[2, col] = 1
        V[16 + k0, col] = 1
        W[2 * k0, col] = 1
        col += 1

        # P2 = A02 * B2k1
        U[2, col] = 1
        V[16 + k1, col] = 1
        W[2 * k1, col] = 1
        col += 1

        # P3 = A12 * B2k0
        U[5, col] = 1
        V[16 + k0, col] = 1
        W[2 * k0 + 1, col] = 1
        col += 1

        # P4 = A12 * B2k1
        U[5, col] = 1
        V[16 + k1, col] = 1
        W[2 * k1 + 1, col] = 1
        col += 1

    assert col == 44
    return U.tolist(), V.tolist(), W.tolist()


def _build_rank40_attempt():
    """
    Attempt rank-40 decomposition by sharing 2 terms across blocks.
    
    Idea: The 4 outer-product terms per block (P1-P4) use U indices 2,5 and 
    V indices 16+k. These are the same across blocks except for which k.
    Instead of 4 separate terms per block, we can use a Strassen-like 
    approach for the 2x1 * 1x2 part using only 3 terms per block.
    
    For A_right (2x1) * B_bottom (1x2):
      C[i,k] = A[i,2] * B[2,k]
    
    This is an outer product of a 2-vector and a 2-vector. The rank-1 tensor
    decomposition of an outer product v ⊗ w (where v is 2-dim, w is 2-dim)
    as a contribution to the multiplication tensor requires... actually each
    (i,k) pair is a separate tensor entry, so we need 4 terms unless we can
    share.
    
    But wait - we can use:
      T1: U=[0,0,1,0,0,0], V=[0,...,1(at 16+k0),...], W=[1,0,0,0,...] for C[0,k0]
      T2: U=[0,0,1,0,0,0], V=[0,...,1(at 16+k1),...], W=[0,0,1,0,...] for C[0,k1]
      T3: U=[0,0,0,0,0,1], V=[0,...,1(at 16+k0),...], W=[0,1,0,0,...] for C[1,k0]
      T4: U=[0,0,0,0,0,1], V=[0,...,1(at 16+k1),...], W=[0,0,0,1,...] for C[1,k1]
    
    These 4 terms have distinct W patterns, so they can't be merged directly.
    
    Alternative: use a 2x2 Strassen-like trick on the outer product:
      Let x = A[0,2], y = A[1,2], u = B[2,k0], v = B[2,k1]
      C[0,k0] = x*u, C[0,k1] = x*v, C[1,k0] = y*u, C[1,k1] = y*v
      
      We can write:
        m1 = (x+y)*(u+v) = x*u + x*v + y*u + y*v
        m2 = (x+y)*u = x*u + y*u
        m3 = x*(v-u) = x*v - x*u
        
      Then:
        C[0,k0] = m2 - m3 = (x+y)*u - x*(v-u) = xu + yu - xv + xu = 2xu + yu - xv  -- NO
      
      Let me redo:
        C[0,k0] = xu
        C[0,k1] = xv
        C[1,k0] = yu
        C[1,k1] = yv
        
      Using:
        m1 = (x+y)(u+v) = xu + xv + yu + yv
        m2 = (x+y)u = xu + yu
        m3 = x(v-u) = xv - xu
        
      C[0,k0] = m2 - m3 + ... hmm this is getting complicated.
      
      Actually the standard result is that the rank of the 2x1 * 1x2 multiplication
      tensor (which is just the outer product tensor) is 4, not reducible.
      
    So let me try a different approach: reduce the Strassen part from 7 to 6.
    """
    # This attempt is unlikely to work cleanly; return None to signal failure
    return None


def _try_rank_reduction(U, V, W, target_rank, deadline):
    """Try to reduce rank by removing terms."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    current_rank = U_arr.shape[1]

    if current_rank <= target_rank:
        return None

    target = _get_target_indices()

    for r in range(current_rank):
        if time.time() > deadline:
            break
        mask = [i for i in range(current_rank) if i != r]
        U_sub = U_arr[:, mask]
        V_sub = V_arr[:, mask]
        W_sub = W_arr[:, mask]
        T_sub = np.zeros((n * m, m * p, p * n))
        for rr in range(len(mask)):
            T_sub += np.einsum('a,b,c->abc', U_sub[:, rr], V_sub[:, rr], W_sub[:, rr])
        target_err = np.max(np.abs(T_sub[target] - 1.0))
        non_target_err = np.max(np.abs(T_sub[~target])) if (~target).any() else 0.0
        if max(target_err, non_target_err) < 1e-9:
            return U_sub.tolist(), V_sub.tolist(), W_sub.tolist()

    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    # Build the explicit rank-44 decomposition
    U44, V44, W44 = _build_rank44()

    if not _verify(U44, V44, W44, tol=1e-9):
        rank = n * m * p
        U_fb = np.zeros((n * m, rank))
        V_fb = np.zeros((m * p, rank))
        W_fb = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    best_U, best_V, best_W = U44, V44, W44
    best_rank = 44

    # Try to reduce rank below 44
    if time.time() < deadline - 5:
        reduced = _try_rank_reduction(U44, V44, W44, 43, deadline)
        if reduced is not None:
            U_r, V_r, W_r = reduced
            if _verify(U_r, V_r, W_r, tol=1e-9):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = len(U_r[0])

    return {"U": best_U, "V": best_V, "W": best_W}
