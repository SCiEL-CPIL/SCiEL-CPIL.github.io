import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 120.0  # safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    """
    Return a low-rank exact decomposition of the 5x5x5 matrix multiplication tensor.

    Strategy (block partition):
      G0 = {0,1}, G1 = {2,3}, G2 = {4}
    For blocks where all three groups have size 2, use Strassen's 7-term algorithm.
    For all other blocks, use schoolbook multiplication.

    The decomposition is verified against the full multiplication tensor before
    returning. If verification fails (should not happen), fall back to pure
    schoolbook (rank 125).
    """
    n, m, p = 5, 5, 5

    G0 = [0, 1]
    G1 = [2, 3]
    G2 = [4]
    groups = [G0, G1, G2]

    terms = []  # each term: (u_dict, v_dict, w_dict)

    def add_term(u_entries, v_entries, w_entries):
        u = {}
        v = {}
        w = {}
        for idx, val in u_entries:
            if val != 0:
                u[idx] = u.get(idx, 0) + val
        for idx, val in v_entries:
            if val != 0:
                v[idx] = v.get(idx, 0) + val
        for idx, val in w_entries:
            if val != 0:
                w[idx] = w.get(idx, 0) + val
        # Remove zero entries
        u = {k: v for k, v in u.items() if v != 0}
        v = {k: v for k, v in v.items() if v != 0}
        w = {k: v for k, v in w.items() if v != 0}
        terms.append((u, v, w))

    def add_strassen_block(i_group, j_group, k_group):
        """Add Strassen's 7-term decomposition for a 2x2x2 block."""
        i0, i1 = i_group[0], i_group[1]
        j0, j1 = j_group[0], j_group[1]
        k0, k1 = k_group[0], k_group[1]

        # M1 = (A00 + A11) * (B00 + B11)
        add_term(
            [(i0 * m + j0, 1), (i1 * m + j1, 1)],
            [(j0 * p + k0, 1), (j1 * p + k1, 1)],
            [(k0 * n + i0, 1), (k1 * n + i1, 1)]
        )

        # M2 = (A10 + A11) * B00
        add_term(
            [(i1 * m + j0, 1), (i1 * m + j1, 1)],
            [(j0 * p + k0, 1)],
            [(k0 * n + i1, 1)]
        )

        # M3 = A00 * (B10 - B11)
        add_term(
            [(i0 * m + j0, 1)],
            [(j1 * p + k0, 1), (j1 * p + k1, -1)],
            [(k0 * n + i0, 1)]
        )

        # M4 = A11 * (B01 - B00)
        add_term(
            [(i1 * m + j1, 1)],
            [(j0 * p + k1, 1), (j0 * p + k0, -1)],
            [(k1 * n + i1, 1)]
        )

        # M5 = (A00 + A01) * B11
        add_term(
            [(i0 * m + j0, 1), (i0 * m + j1, 1)],
            [(j1 * p + k1, 1)],
            [(k1 * n + i0, 1)]
        )

        # M6 = (A10 - A00) * (B00 + B01)
        add_term(
            [(i1 * m + j0, 1), (i0 * m + j0, -1)],
            [(j0 * p + k0, 1), (j0 * p + k1, 1)],
            [(k0 * n + i1, 1)]
        )

        # M7 = (A01 - A11) * (B10 + B11)
        add_term(
            [(i0 * m + j1, 1), (i1 * m + j1, -1)],
            [(j1 * p + k0, 1), (j1 * p + k1, 1)],
            [(k0 * n + i0, 1)]
        )

    # Build the decomposition
    for i_group in groups:
        for j_group in groups:
            for k_group in groups:
                if _time_left() < 5.0:
                    # Time running out, use schoolbook for remaining
                    for i in i_group:
                        for j in j_group:
                            for k in k_group:
                                add_term(
                                    [(i * m + j, 1)],
                                    [(j * p + k, 1)],
                                    [(k * n + i, 1)]
                                )
                    continue

                if len(i_group) == 2 and len(j_group) == 2 and len(k_group) == 2:
                    add_strassen_block(i_group, j_group, k_group)
                else:
                    # Schoolbook multiplication
                    for i in i_group:
                        for j in j_group:
                            for k in k_group:
                                add_term(
                                    [(i * m + j, 1)],
                                    [(j * p + k, 1)],
                                    [(k * n + i, 1)]
                                )

    rank = len(terms)

    # Build factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r, (u, v, w) in enumerate(terms):
        for idx, val in u.items():
            U[idx][r] = val
        for idx, val in v.items():
            V[idx][r] = val
        for idx, val in w.items():
            W[idx][r] = val

    # Verify the decomposition
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback: pure schoolbook
    print("Verification failed, using schoolbook fallback")
    return _schoolbook_decomposition(n, m, p)


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    
    The tensor T[i,j,k] = 1 if (i,j,k) is a valid multiplication index, else 0.
    Actually, the matrix multiplication tensor is:
      T_{(i,j),(j,k),(k,i)} = 1  for all i,j,k
    where the three modes are (i,j), (j,k), (k,i) with the trace convention.
    
    The decomposition says:
      T = sum_r U_r ⊗ V_r ⊗ W_r
    
    So for each (a, b, c) where a = i*m+j, b = j*p+k, c = k*n+i:
      T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    We need T[a,b,c] = 1 when the indices are consistent (j from a matches j from b,
    k from b matches k from c, i from c matches i from a) and 0 otherwise.
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    rank = U_arr.shape[1]
    
    # Check all (a, b, c) combinations
    for a in range(n * m):
        i_a = a // m
        j_a = a % m
        for b in range(m * p):
            j_b = b // p
            k_b = b % p
            for c in range(p * n):
                k_c = c // n
                i_c = c % n
                
                # Expected value: 1 if j_a == j_b and k_b == k_c and i_c == i_a, else 0
                expected = 1.0 if (j_a == j_b and k_b == k_c and i_c == i_a) else 0.0
                
                # Computed value
                if expected != 0.0:
                    # Only need to check where expected is nonzero for efficiency
                    # sum_r U[a,r] * V[b,r] * W[c,r]
                    val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                    if abs(val - expected) > 1e-9:
                        return False
                else:
                    # Check that it's zero
                    val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                    if abs(val) > 1e-9:
                        return False
    
    return True


def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}
