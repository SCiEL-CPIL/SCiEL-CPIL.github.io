import numpy as np
import time
import random

DIMENSIONS = [2, 2, 9]
_START_TIME = time.time()
_TIMEOUT = 140  # safety margin before 150s kill


def _build_rank_32(n, m, p):
    """
    Build a rank-32 decomposition using Strassen for 4 blocks of 2x2x2
    and schoolbook for 1 block of 2x2x1.
    """
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        # M1: (A00+A11)*(B00+B11)
        U[0][col] = 1
        U[3][col] = 1
        V[0*p + k0][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1

        # M2: (A10+A11)*B00
        U[2][col] = 1
        U[3][col] = 1
        V[0*p + k0][col] = 1
        W[k0*n + 1][col] = 1
        W[k1*n + 1][col] = -1
        col += 1

        # M3: A00*(B10-B00)
        U[0][col] = 1
        V[0*p + k1][col] = 1
        V[1*p + k1][col] = -1
        W[k1*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1

        # M4: A11*(B01-B11)
        U[3][col] = 1
        V[1*p + k0][col] = 1
        V[0*p + k0][col] = -1
        W[k0*n + 0][col] = 1
        W[k0*n + 1][col] = 1
        col += 1

        # M5: (A00+A01)*B11
        U[0][col] = 1
        U[1][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = -1
        W[k1*n + 0][col] = 1
        col += 1

        # M6: (A10-A00)*(B00+B01)
        U[2][col] = 1
        U[0][col] = -1
        V[0*p + k0][col] = 1
        V[0*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1

        # M7: (A01-A11)*(B10+B11)
        U[1][col] = 1
        U[3][col] = -1
        V[1*p + k0][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1

        U[1][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1

        U[2][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1

        U[3][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)

    return U, V, W


def _validate(U, V, W, n, m, p):
    """Check if the decomposition exactly decomposes the multiplication tensor."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    T_recon = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
    return np.allclose(T, T_recon, atol=1e-10)


def _try_remove_terms(U, V, W, n, m, p):
    """Try to remove terms from the decomposition if they are linearly dependent."""
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT - 10:
        return None

    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    T_full = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)

    if not np.allclose(T, T_full, atol=1e-10):
        return None

    # Flatten each term tensor to a vector for fast linear independence checks
    flat_dim = n * m * m * p * p * n
    T_flat = np.zeros((rank, flat_dim), dtype=float)
    for r in range(rank):
        T_flat[r] = np.einsum('i,j,k->ijk', Ua[:, r], Va[:, r], Wa[:, r]).flatten()

    # Find null space of T_flat (rows)
    try:
        if time.time() - _START_TIME > _TIMEOUT - 5:
            return None
        U_svd, s, Vt = np.linalg.svd(T_flat, full_matrices=False)
        tol = 1e-10
        num_zero = sum(1 for si in s if si < tol)
    except Exception:
        return None

    if num_zero == 0:
        # All terms are linearly independent - no single or pair removal possible
        # without changing coefficients
        return None

    # For each null space vector, try to eliminate the term with largest coefficient
    null_vecs = Vt[-num_zero:].T  # shape (rank, num_zero)
    for nv in null_vecs:
        if time.time() - _START_TIME > _TIMEOUT - 5:
            return None
        max_idx = np.argmax(np.abs(nv))
        if abs(nv[max_idx]) < 1e-10:
            continue
        # T[max_idx] = -sum_{r != max_idx} (nv[r]/nv[max_idx]) * T_r
        # T = sum_{r != max_idx} (1 - nv[r]/nv[max_idx]) * T_r
        new_mask = []
        new_coeffs = []
        for r in range(rank):
            if r == max_idx:
                continue
            c = 1.0 - nv[r] / nv[max_idx]
            if abs(c) > 1e-10:
                new_mask.append(r)
                new_coeffs.append(c)

        if len(new_mask) < rank:
            U_new = Ua[:, new_mask] * np.array(new_coeffs)[np.newaxis, :]
            V_new = Va[:, new_mask]
            W_new = Wa[:, new_mask]
            T_check = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
            if np.allclose(T, T_check, atol=1e-10):
                return (
                    U_new.tolist(),
                    V_new.tolist(),
                    W_new.tolist()
                )

    return None


def _search_rank_31(n, m, p, U0, V0, W0, max_iterations=20000):
    """
    Search for a rank-31 decomposition using tensor flattening and matrix rank
    bound analysis. The 2x2x9 multiplication tensor has structure that may allow
    a rank-31 decomposition if the 32 Strassen+schoolbook terms are linearly
    dependent in the tensor space.
    
    Uses the matrix rank bound: for a rank-r decomposition of T (shape 4x18x18),
    the flattening T_{i|(jk)} is a 4x(18*18) matrix with rank <= 4.
    More critically, we check if the 32 term-tensors are linearly independent.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT - 20:
        return None

    Ua = np.array(U0, dtype=float)
    Va = np.array(V0, dtype=float)
    Wa = np.array(W0, dtype=float)
    rank = Ua.shape[1]  # 32

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    T_full = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
    if not np.allclose(T, T_full, atol=1e-10):
        return None

    # Flatten each term tensor to a vector
    flat_dim = n * m * m * p * p * n  # 4 * 18 * 18 = 1296
    T_flat = np.zeros((rank, flat_dim), dtype=float)
    for r in range(rank):
        T_flat[r] = np.einsum('i,j,k->ijk', Ua[:, r], Va[:, r], Wa[:, r]).flatten()

    # Find the null space of T_flat (rows) - this tells us about linear dependencies
    try:
        if time.time() - _START_TIME > _TIMEOUT - 10:
            return None
        U_svd, s, Vt = np.linalg.svd(T_flat, full_matrices=False)
        tol = 1e-10
        num_zero = sum(1 for si in s if si < tol)
    except Exception:
        return None

    if num_zero == 0:
        # All 32 terms are linearly independent in the tensor space.
        # A rank-31 decomposition would require a fundamentally different structure.
        # Try a targeted perturbation search: randomly perturb one term and see
        # if the remaining 31 can be adjusted to compensate.
        if time.time() - _START_TIME > _TIMEOUT - 5:
            return None

        # For each term to eliminate, try to find coefficients for the other 31
        # such that sum_{r != target} c_r * T_r = T
        # This is a linear system: T_flat_without_target @ c = T_flat_target + T_flat_target
        # Actually: sum_{r != target} c_r * T_r = T - 0 (since T = sum_all T_r)
        # So: T_flat_without_target @ c = T - T_target = sum_{r != target} T_r
        # Which means c = ones if the terms are independent.
        # So this only works if T_target is in the span of the others, which
        # contradicts num_zero == 0.
        
        # Alternative: try to merge two terms. If T_a + T_b can be written as
        # a single rank-1 tensor, we can merge them.
        # T_a + T_b = U_a⊗V_a⊗W_a + U_b⊗V_b⊗W_b
        # This is rank-1 only if one of them is zero or they are proportional.
        # Check proportionality for all pairs:
        for a in range(rank):
            if time.time() - _START_TIME > _TIMEOUT - 5:
                return None
            for b in range(a + 1, rank):
                # Check if T_a and T_b are proportional
                # T_a = u_a ⊗ v_a ⊗ w_a, T_b = u_b ⊗ v_b ⊗ w_b
                # Proportional iff u_a ∝ u_b, v_a ∝ v_b, w_a ∝ w_b
                ua, va, wa = Ua[:, a], Va[:, a], Wa[:, a]
                ub, vb, wb = Ua[:, b], Va[:, b], Wa[:, b]
                
                # Check if ua is proportional to ub
                if np.allclose(ua, 0, atol=1e-10) or np.allclose(ub, 0, atol=1e-10):
                    continue
                # Find scale factor
                nonzero_a = np.where(np.abs(ua) > 1e-10)[0]
                nonzero_b = np.where(np.abs(ub) > 1e-10)[0]
                if len(nonzero_a) == 0 or len(nonzero_b) == 0:
                    continue
                # Use first nonzero element
                ia = nonzero_a[0]
                ib = nonzero_b[0]
                scale = ua[ia] / ub[ib]
                if np.allclose(ua, scale * ub, atol=1e-10) and \
                   np.allclose(va, scale * vb, atol=1e-10) and \
                   np.allclose(wa, scale * wb, atol=1e-10):
                    # Terms are proportional - merge them
                    # T_a + T_b = (1+scale) * T_b (if ua = scale*ub)
                    # Actually: T_a = scale * T_b, so T_a + T_b = (1+scale)*T_b
                    # We can just add them into one term
                    if abs(1 + scale) > 1e-10:
                        mask = [r for r in range(rank) if r != a]
                        U_new = Ua[:, mask].copy()
                        V_new = Va[:, mask].copy()
                        W_new = Wa[:, mask].copy()
                        # Add T_a into T_b: find b's index in mask
                        b_idx = mask.index(b)
                        U_new[:, b_idx] += scale * Ua[:, a]
                        V_new[:, b_idx] += scale * Va[:, a]
                        W_new[:, b_idx] += scale * Wa[:, a]
                        # Wait, this is wrong. T_a + T_b is not (1+scale)*T_b in general
                        # unless they're exactly proportional. Let me just verify.
                        T_check = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
                        if np.allclose(T, T_check, atol=1e-10):
                            return (U_new.tolist(), V_new.tolist(), W_new.tolist())

        return None

    # We have linear dependencies. For each null space vector, try to eliminate
    # the term with the largest coefficient.
    null_vecs = Vt[-num_zero:].T  # shape (rank, num_zero)
    for nv in null_vecs:
        if time.time() - _START_TIME > _TIMEOUT - 5:
            return None
        max_idx = np.argmax(np.abs(nv))
        if abs(nv[max_idx]) < 1e-10:
            continue
        # T[max_idx] = -sum_{r != max_idx} (nv[r]/nv[max_idx]) * T_r
        # T = sum_{r != max_idx} (1 - nv[r]/nv[max_idx]) * T_r
        new_mask = []
        new_coeffs = []
        for r in range(rank):
            if r == max_idx:
                continue
            c = 1.0 - nv[r] / nv[max_idx]
            if abs(c) > 1e-10:
                new_mask.append(r)
                new_coeffs.append(c)

        if len(new_mask) < rank:
            U_new = Ua[:, new_mask] * np.array(new_coeffs)[np.newaxis, :]
            V_new = Va[:, new_mask]
            W_new = Wa[:, new_mask]
            T_check = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
            if np.allclose(T, T_check, atol=1e-10):
                return (
                    U_new.tolist(),
                    V_new.tolist(),
                    W_new.tolist()
                )

    return None


def construct():
    n, m, p = DIMENSIONS

    # Build rank-32 decomposition
    U, V, W = _build_rank_32(n, m, p)

    # Validate
    if not _validate(U, V, W, n, m, p):
        # Fallback: schoolbook rank 36
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try to reduce rank via linear dependency analysis (fast SVD-based)
    reduced = _try_remove_terms(U, V, W, n, m, p)
    if reduced is not None:
        U2, V2, W2 = reduced
        if _validate(U2, V2, W2, n, m, p):
            return {"U": U2, "V": V2, "W": W2}

    # Try to find rank-31 via tensor flattening and targeted search
    if time.time() - _START_TIME < _TIMEOUT - 15:
        r31 = _search_rank_31(n, m, p, U, V, W)
        if r31 is not None:
            U3, V3, W3 = r31
            if _validate(U3, V3, W3, n, m, p):
                return {"U": U3, "V": V3, "W": W3}

    return {"U": U, "V": V, "W": W}
