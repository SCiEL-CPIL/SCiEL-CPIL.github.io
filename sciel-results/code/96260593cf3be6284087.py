import numpy as np
import time

DIMENSIONS = [5, 5, 5]

_T0 = time.time()
_LIMIT = 150.0
_MARGIN = 5.0

def _deadline_exceeded():
    return (time.time() - _T0) > (_LIMIT - _MARGIN)

# EVOLVE-BLOCK-START

def _strassen_2x2_terms():
    """
    Return the 7 rank-1 terms for 2x2x2 matrix multiplication.
    Each term is (u_dict, v_dict, w_dict) where keys are local (i,j) or (j,k) or (k,i) indices.
    """
    return [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]


def _build_5x5_decomposition(n, m, p):
    """
    Construct the 5x5x5 matrix multiplication tensor decomposition using:
    - 4x4x4 block with Strassen (rank 49)
    - Remaining 61 terms with schoolbook
    Total rank: 110
    """
    cols_u = []
    cols_v = []
    cols_w = []

    def add_term(u_dict, v_dict, w_dict):
        """Add a rank-1 term, filtering zeros."""
        u_dict = {k: v for k, v in u_dict.items() if v != 0}
        v_dict = {k: v for k, v in v_dict.items() if v != 0}
        w_dict = {k: v for k, v in w_dict.items() if v != 0}

        if not u_dict or not v_dict or not w_dict:
            return

        cols_u.append(u_dict)
        cols_v.append(v_dict)
        cols_w.append(w_dict)

    # --- Strassen 4x4x4 Decomposition ---
    s2x2 = _strassen_2x2_terms()

    def add_strassen_4x4(offset_i, offset_j, offset_k):
        """
        Add the 49 rank-1 terms for a 4x4x4 block using recursive Strassen.
        """
        # Macro definitions for 4x4 Strassen (same structure as 2x2)
        macro_defs = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]

        for u_m, v_m, w_m in macro_defs:
            for u_s, v_s, w_s in s2x2:
                u_dict = {}
                v_dict = {}
                w_dict = {}

                # Expand U: (i, j) indices
                for (li, lj), val_u_s in u_s.items():
                    for (mi, mj), val_u_m in u_m.items():
                        val = val_u_s * val_u_m
                        if val != 0:
                            gi = offset_i + mi * 2 + li
                            gj = offset_j + mj * 2 + lj
                            idx = gi * m + gj
                            u_dict[idx] = u_dict.get(idx, 0) + val

                # Expand V: (j, k) indices
                for (lj, lk), val_v_s in v_s.items():
                    for (mj, mk), val_v_m in v_m.items():
                        val = val_v_s * val_v_m
                        if val != 0:
                            gj = offset_j + mj * 2 + lj
                            gk = offset_k + mk * 2 + lk
                            idx = gj * p + gk
                            v_dict[idx] = v_dict.get(idx, 0) + val

                # Expand W: (k, i) indices
                for (li, lk), val_w_s in w_s.items():
                    for (mi, mk), val_w_m in w_m.items():
                        val = val_w_s * val_w_m
                        if val != 0:
                            gi = offset_i + mi * 2 + li
                            gk = offset_k + mk * 2 + lk
                            idx = gk * n + gi
                            w_dict[idx] = w_dict.get(idx, 0) + val

                add_term(u_dict, v_dict, w_dict)

    def add_schoolbook(i_set, j_set, k_set):
        """Add schoolbook terms for given index sets."""
        for i in i_set:
            for j in j_set:
                for k in k_set:
                    add_term(
                        {i * m + j: 1.0},
                        {j * p + k: 1.0},
                        {k * n + i: 1.0}
                    )

    # Build decomposition
    # 4x4x4 block at indices 0-3
    add_strassen_4x4(0, 0, 0)

    # Off-diagonal blocks with schoolbook
    G4 = [0, 1, 2, 3]
    G1 = [4]

    add_schoolbook(G4, G4, G1)  # i in 0-3, j in 0-3, k=4 -> 16 terms
    add_schoolbook(G4, G1, G4)  # i in 0-3, j=4, k in 0-3 -> 16 terms
    add_schoolbook(G1, G4, G4)  # i=4, j in 0-3, k in 0-3 -> 16 terms
    add_schoolbook(G4, G1, G1)  # i in 0-3, j=4, k=4 -> 4 terms
    add_schoolbook(G1, G4, G1)  # i=4, j in 0-3, k=4 -> 4 terms
    add_schoolbook(G1, G1, G4)  # i=4, j=4, k in 0-3 -> 4 terms
    add_schoolbook(G1, G1, G1)  # i=4, j=4, k=4 -> 1 term

    total_rank = len(cols_u)

    # Construct numpy arrays
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    for c in range(total_rank):
        for idx, val in cols_u[c].items():
            U[idx, c] = val
        for idx, val in cols_v[c].items():
            V[idx, c] = val
        for idx, val in cols_w[c].items():
            W[idx, c] = val

    return U, V, W


def _verify_decomposition(U, V, W, n=5, m=5, p=5, tol=1e-9):
    """
    Verify that the decomposition exactly decomposes the matrix multiplication tensor.
    Check all (a, b, c) triples where the tensor entry should be 1 or 0.
    """
    # Check all 125 entries that should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
                if abs(val - 1.0) > tol:
                    return False

    # Sample some entries that should be 0 to catch errors
    rng = np.random.RandomState(42)
    for _ in range(2000):
        a = rng.randint(0, n * m)
        b = rng.randint(0, m * p)
        c = rng.randint(0, p * n)

        # Extract indices
        i1 = a // m
        j1 = a % m
        j2 = b // p
        k1 = b % p
        k2 = c // n
        i2 = c % n

        # This entry should be 1 iff i1==i2, j1==j2, k1==k2
        if i1 == i2 and j1 == j2 and k1 == k2:
            continue

        val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
        if abs(val) > tol:
            return False

    return True


def _construct_schoolbook(n, m, p):
    """Fallback schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
    return U, V, W


def construct():
    """
    Construct the factor matrices U, V, W for the 5x5x5 matrix multiplication tensor.
    Uses a 4+1 block partition: 4x4x4 block with Strassen (rank 49) and schoolbook
    for the remaining 61 terms, giving total rank 110.
    """
    start_time = time.time()
    deadline = 120.0
    n, m, p = DIMENSIONS

    # Fallback: schoolbook
    U_school, V_school, W_school = _construct_schoolbook(n, m, p)

    # Try the 4+1 Strassen decomposition
    try:
        if time.time() - start_time < deadline - 30:
            U, V, W = _build_5x5_decomposition(n, m, p)
            rank = U.shape[1]

            if time.time() - start_time < deadline - 15:
                if _verify_decomposition(U, V, W, n, m, p):
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
    except Exception:
        pass

    # Return schoolbook as fallback
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }

# EVOLVE-BLOCK-END
