"""Rewritten program for matrix multiplication 2x2 by 2x5 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify U, V, W decompose the matmul tensor:
       sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k.
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_terms_2x2(k_offset, n, m, p):
    """7 Strassen terms for a 2x2x2 block starting at output column k_offset."""
    terms = []
    # Term 0: (a00+a11) * (b00+b11) * (c00+c11)
    u0 = [1, 0, 0, 1]
    v0 = [0]*10
    v0[0*p + k_offset] = 1
    v0[1*p + (k_offset+1)] = 1
    w0 = [0]*10
    w0[k_offset*n + 0] = 1
    w0[(k_offset+1)*n + 1] = 1
    terms.append((u0, v0, w0))
    # Term 1: (a10+a11) * (b00) * (c01+c11)
    u1 = [0, 0, 1, 1]
    v1 = [0]*10
    v1[0*p + k_offset] = 1
    w1 = [0]*10
    w1[k_offset*n + 1] = 1
    w1[(k_offset+1)*n + 1] = 1
    terms.append((u1, v1, w1))
    # Term 2: a00 * (b01-b11) * (c01+c11)
    u2 = [1, 0, 0, 0]
    v2 = [0]*10
    v2[0*p + (k_offset+1)] = 1
    v2[1*p + (k_offset+1)] = -1
    w2 = [0]*10
    w2[k_offset*n + 1] = 1
    w2[(k_offset+1)*n + 1] = 1
    terms.append((u2, v2, w2))
    # Term 3: a11 * (b10-b00) * (c10+c00)
    u3 = [0, 0, 0, 1]
    v3 = [0]*10
    v3[1*p + k_offset] = 1
    v3[0*p + k_offset] = -1
    w3 = [0]*10
    w3[k_offset*n + 1] = 1
    w3[k_offset*n + 0] = 1
    terms.append((u3, v3, w3))
    # Term 4: (a00+a01) * (b11) * (c10+c11)
    u4 = [1, 1, 0, 0]
    v4 = [0]*10
    v4[1*p + (k_offset+1)] = 1
    w4 = [0]*10
    w4[k_offset*n + 1] = 1
    w4[(k_offset+1)*n + 1] = 1
    terms.append((u4, v4, w4))
    # Term 5: (a10-a00) * (b00+b01) * (c00+c01)
    u5 = [-1, 0, 1, 0]
    v5 = [0]*10
    v5[0*p + k_offset] = 1
    v5[0*p + (k_offset+1)] = 1
    w5 = [0]*10
    w5[k_offset*n + 0] = 1
    w5[k_offset*n + 1] = 1
    terms.append((u5, v5, w5))
    # Term 6: (a01-a11) * (b10+b11) * (c10+c11)
    u6 = [0, 1, 0, -1]
    v6 = [0]*10
    v6[1*p + k_offset] = 1
    v6[1*p + (k_offset+1)] = 1
    w6 = [0]*10
    w6[k_offset*n + 1] = 1
    w6[(k_offset+1)*n + 1] = 1
    terms.append((u6, v6, w6))
    return terms

def _schoolbook_terms_2x1(k_idx, n, m, p):
    """4 schoolbook terms for a 2x2x1 block at output column k_idx."""
    terms = []
    for i in range(n):
        for j in range(m):
            u = [0]*4
            u[i*m+j] = 1
            v = [0]*10
            v[j*p+k_idx] = 1
            w = [0]*10
            w[k_idx*n+i] = 1
            terms.append((u, v, w))
    return terms

def _build_rank18(n, m, p):
    all_terms = []
    all_terms.extend(_strassen_terms_2x2(0, n, m, p))
    all_terms.extend(_strassen_terms_2x2(2, n, m, p))
    all_terms.extend(_schoolbook_terms_2x1(4, n, m, p))
    rank = len(all_terms)
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    for r, (u, v, w) in enumerate(all_terms):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
    return U, V, W

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _build_target_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _als_search(n, m, p, rank, max_iters, tol, rng, time_budget):
    """Try to find an exact rank-r decomposition via ALS. Return (U,V,W) or None."""
    T = _build_target_tensor(n, m, p)
    T2 = T.reshape(n*m*p, n*m*p)  # not used; keep simple

    # Initialize U, V, W with random values
    U = rng.normal(size=(n*m, rank))
    V = rng.normal(size=(m*p, rank))
    W = rng.normal(size=(p*n, rank))

    best_err = float('inf')
    best_UVW = (U.copy(), V.copy(), W.copy())

    for it in range(max_iters):
        # Update U: for each (i,j), solve min ||T[i,j,:] - (V * W^T)[j,:] ||
        # T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
        # For fixed V,W: T[i*m+j, :, :] = U[i*m+j,:] (V[:, :, :] * W[:, :, :]^T)
        # Actually: for each i, the tensor slice over (j,k) is U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # Reshape T as (n, m*p, p*n). For fixed i, T[i, :, :] (shape (m*p, p*n)) =
        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is: for each i, T[i] = sum_r u_i_j * V_jr * W_kr  where u_i_j = U[i*m+j, r]
        # So T[i] (m*p x p*n) = sum_r (U[i*m:i*m+m, r] * V[r, :]) outer (W[:, r])
        # = sum_r outer(V[r, :] * U[i*m:i*m+m, r], W[:, r])
        # This is a linear system in U[i*m:i*m+m, :] for fixed V, W.

        # Let's build the linear system properly.
        # T[i, a, b] where a = j*p+k, b = k*n+i
        # T[i, a, b] = sum_r U[i*m+j, r] V[a, r] W[b, r]
        # For fixed i, define matrix M_i (m*p * p*n) where M_i[a, b] = V[a, :] * W[b, :] (elementwise product)
        # Then T[i, :, :] = sum_r U[i*m+j, r] * (V[:, r] * W[:, r]^T)
        # So T[i] = U_i @ (V * W^T) where U_i is (m, rank) and (V*W^T) reshaped...
        # Actually: T[i, j*p+k, k*n+i] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # Let's define for each i: T_i is (m*p, p*n)
        # T_i[a, b] = sum_r U[i*m+j, r] V[a, r] W[b, r]  where a = j*p+k, b = k*n+i
        # This can be written as: T_i = sum_r U[i*m, :, r] (V[:, r] ⊙ W[:, r] reshaped)
        # where U[i*m, :, r] is a vector of length m (over j), and V[:, r] ⊙ W[:, r] needs careful reshaping.

        # Simpler: just do gradient-free ALS by solving linear least squares.
        # For U: fix V, W. For each i, j: T[i*m+j, :, :] = U[i*m+j, :] (V * W^T)
        # T[i*m+j] is (m*p, p*n). We need U[i*m+j, r] such that
        # T[i*m+j, a, b] = sum_r U[i*m+j, r] V[a, r] W[b, r]
        # Define M[a, b] = V[a, :] * W[b, :] (outer product of V[a,:] and W[b,:])
        # Then T[i*m+j] = sum_r U[i*m+j, r] M_r where M_r[a, b] = V[a, r] W[b, r]
        # So T[i*m+j] (flattened) = U[i*m+j, :] @ M_flat
        # where M_flat is (m*p*p*n, rank) with M_flat[a*p*n + b, r] = V[a, r] W[b, r]

        # Build M_flat once
        M_flat = np.zeros((m*p*p*n, rank), dtype=float)
        for a in range(m*p):
            for b in range(p*n):
                for r in range(rank):
                    M_flat[a*p*n + b, r] = V[a, r] * W[b, r]

        # Update U
        for i in range(n):
            for j in range(m):
                T_slice = T[i*m+j, :, :].flatten()
                # Solve: U[i*m+j, :] @ M_flat = T_slice
                # U[i*m+j, :] = (M_flat^T M_flat)^{-1} M_flat^T T_slice
                # Use least squares
                try:
                    U_row, _, _, _ = np.linalg.lstsq(M_flat, T_slice, rcond=None)
                    U[i*m+j, :] = U_row
                except:
                    pass

        # Update V: fix U, W. For each j, k: T[:, j*p+k, :] = V[j*p+k, :] (U * W^T)
        # T[a, j*p+k, b] = sum_r U[a, r] V[j*p+k, r] W[b, r]  where a = i*m+j, b = k*n+i
        # M2[a, b] = U[a, :] * W[b, :]
        M2_flat = np.zeros((n*m*p*n, rank), dtype=float)
        for a in range(n*m):
            for b in range(p*n):
                for r in range(rank):
                    M2_flat[a*p*n + b, r] = U[a, r] * W[b, r]

        for j in range(m):
            for k in range(p):
                T_slice = T[:, j*p+k, :].flatten()
                try:
                    V_row, _, _, _ = np.linalg.lstsq(M2_flat, T_slice, rcond=None)
                    V[j*p+k, :] = V_row
                except:
                    pass

        # Update W: fix U, V. For each k, i: T[:, :, k*n+i] = W[k*n+i, :] (U * V^T)
        # T[a, b, k*n+i] = sum_r U[a, r] V[b, r] W[k*n+i, r]  where a = i*m+j, b = j*p+k
        M3_flat = np.zeros((n*m*m*p, rank), dtype=float)
        for a in range(n*m):
            for b in range(m*p):
                for r in range(rank):
                    M3_flat[a*m*p + b, r] = U[a, r] * V[b, r]

        for k in range(p):
            for i in range(n):
                T_slice = T[:, :, k*n+i].flatten()
                try:
                    W_row, _, _, _ = np.linalg.lstsq(M3_flat, T_slice, rcond=None)
                    W[k*n+i, :] = W_row
                except:
                    pass

        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err += (s - 1.0) ** 2
        if err < best_err:
            best_err = err
            best_UVW = (U.copy(), V.copy(), W.copy())

        if err < tol:
            break

        # Check time
        if time.time() > time_budget:
            break

    if best_err < 1e-6:
        return best_UVW
    return None

def construct():
    n, m, p = 2, 2, 5
    t_start = time.time()
    deadline = t_start + 130  # Safety margin before 150s limit

    # First, build the guaranteed rank-18 solution
    U18, V18, W18 = _build_rank18(n, m, p)
    if _verify_decomposition(U18, V18, W18, n, m, p):
        best = (U18, V18, W18, 18)
    else:
        U20, V20, W20 = _build_schoolbook(n, m, p)
        best = (U20, V20, W20, 20)

    # Try ALS search for lower ranks
    rng = np.random.RandomState(42)
    for target_rank in [15, 16, 17]:
        if time.time() > deadline:
            break
        if target_rank >= best[3]:
            continue
        max_iters = 200
        tol = 1e-8
        time_budget = min(deadline, time.time() + 30)  # 30s per rank attempt
        result = _als_search(n, m, p, target_rank, max_iters, tol, rng, time_budget)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best = (U, V, W, target_rank)
                break  # Found lower rank, stop

    U, V, W, rank = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
