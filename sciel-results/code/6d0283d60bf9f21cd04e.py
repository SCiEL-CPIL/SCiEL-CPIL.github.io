"""Rank-98 tensor decomposition for 4x4 * 4x8 using Kronecker Strassen.

Uses the rank-49 Kronecker-product Strassen decomposition for 4x4x4,
applied independently to the two 4-column halves of B to get rank 98.

The Kronecker construction: T_4 = T_2 ⊗ T_2 (tensor Kronecker product)
gives a rank-49 decomposition of the 4x4x4 matrix multiplication tensor.

Interface: construct() -> {'U', 'V', 'W'}
  U: shape (n*m, rank) = (16, 98), rows indexed i*m+j
  V: shape (m*p, rank) = (32, 98), rows indexed j*p+k
  W: shape (p*n, rank) = (32, 98), rows indexed k*n+i  (official trace convention)
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]


def _strassen_2x2_tensors():
    """
    Return the 7 rank-1 tensors (u, v, w) for 2x2x2 Strassen.
    Each is a flat array of length 4 (row-major 2x2).
    Convention: sum_t u_t[i,j]*v_t[j,k]*w_t[k,i] = 1 for all i,j,k in {0,1}
    """
    terms = []

    # Term 0: M1 = (A00+A11)(B00+B11)
    u = np.zeros(4); u[0] = 1; u[3] = 1
    v = np.zeros(4); v[0] = 1; v[3] = 1
    w = np.zeros(4); w[0] = 1; w[3] = 1
    terms.append((u, v, w))

    # Term 1: M2 = (A10+A11)B00
    u = np.zeros(4); u[2] = 1; u[3] = 1
    v = np.zeros(4); v[0] = 1
    w = np.zeros(4); w[2] = 1; w[3] = -1
    terms.append((u, v, w))

    # Term 2: M3 = A00(B10-B11)
    u = np.zeros(4); u[0] = 1
    v = np.zeros(4); v[2] = 1; v[3] = -1
    w = np.zeros(4); w[1] = 1; w[3] = 1
    terms.append((u, v, w))

    # Term 3: M4 = A11(B01-B00)
    u = np.zeros(4); u[3] = 1
    v = np.zeros(4); v[1] = 1; v[0] = -1
    w = np.zeros(4); w[0] = -1; w[2] = 1
    terms.append((u, v, w))

    # Term 4: M5 = (A00+A01)B11
    u = np.zeros(4); u[0] = 1; u[1] = 1
    v = np.zeros(4); v[3] = 1
    w = np.zeros(4); w[0] = 1
    terms.append((u, v, w))

    # Term 5: M6 = (A10-A00)(B00+B01)
    u = np.zeros(4); u[2] = 1; u[0] = -1
    v = np.zeros(4); v[0] = 1; v[1] = 1
    w = np.zeros(4); w[3] = 1
    terms.append((u, v, w))

    # Term 6: M7 = (A01-A11)(B10+B11)
    u = np.zeros(4); u[1] = 1; u[3] = -1
    v = np.zeros(4); v[2] = 1; v[3] = 1
    w = np.zeros(4); w[0] = -1; w[1] = 1
    terms.append((u, v, w))

    return terms


def _kron_strassen_4x4x4():
    """
    Build rank-49 decomposition for 4x4x4 using Kronecker product of Strassen.
    Returns U (16x49), V (16x49), W (16x49) as numpy arrays.
    Row convention: U[i*4+j], V[j*4+k], W[k*4+i]
    """
    s_terms = _strassen_2x2_tensors()
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))

    for a in range(7):
        ua, va, wa = s_terms[a]
        for b in range(7):
            ub, vb, wb = s_terms[b]
            t = a * 7 + b
            for i in range(4):
                for j in range(4):
                    u_val = ua[(i // 2) * 2 + (j // 2)] * ub[(i % 2) * 2 + (j % 2)]
                    if u_val != 0:
                        U[i * 4 + j, t] = u_val

            for j in range(4):
                for k in range(4):
                    v_val = va[(j // 2) * 2 + (k // 2)] * vb[(j % 2) * 2 + (k % 2)]
                    if v_val != 0:
                        V[j * 4 + k, t] = v_val

            for k in range(4):
                for i in range(4):
                    w_val = wa[(k // 2) * 2 + (i // 2)] * wb[(k % 2) * 2 + (i % 2)]
                    if w_val != 0:
                        W[k * 4 + i, t] = w_val

    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1 for all i,j,k."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for t in range(rank):
                    val += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _schoolbook(n, m, p):
    """Rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U[ii * m + jj, col] = 1.0
                V[jj * p + kk, col] = 1.0
                W[kk * n + ii, col] = 1.0
                col += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit

    try:
        # Build rank-49 4x4x4 decomposition via Kronecker Strassen
        U4, V4, W4 = _kron_strassen_4x4x4()

        # Verify the 4x4x4 decomposition
        if not _verify_decomposition(U4, V4, W4, 4, 4, 4):
            # Fall back to schoolbook for 4x4x4
            U4, V4, W4 = _schoolbook(4, 4, 4)
            sub_rank = 64
        else:
            sub_rank = 49

        # Build 4x4x8 by splitting B into left (cols 0-3) and right (cols 4-7)
        total_rank = 2 * sub_rank
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))

        # Left half: B[:, 0:4], C[:, 0:4]
        for t in range(sub_rank):
            U[:, t] = U4[:, t]
            for j in range(m):
                for k in range(4):
                    V[j * p + k, t] = V4[j * 4 + k, t]
            for k in range(4):
                for i in range(n):
                    W[k * n + i, t] = W4[k * 4 + i, t]

        # Right half: B[:, 4:8], C[:, 4:8]
        for t in range(sub_rank):
            col = sub_rank + t
            U[:, col] = U4[:, t]
            for j in range(m):
                for k in range(4):
                    V[j * p + (k + 4), col] = V4[j * 4 + k, t]
            for k in range(4):
                for i in range(n):
                    W[(k + 4) * n + i, col] = W4[k * 4 + i, t]

        # Verify the full 4x4x8 decomposition
        if not _verify_decomposition(U, V, W, n, m, p):
            # Fall back to schoolbook
            U, V, W = _schoolbook(n, m, p)

        # Convert to lists of floats (the interface expects lists)
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]

        return {"U": U_list, "V": V_list, "W": W_list}

    except Exception:
        # Ultimate fallback: schoolbook
        U, V, W = _schoolbook(n, m, p)
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
