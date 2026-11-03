"""Combined program: structured decomposition for 3x4 * 4x6 matrix multiplication."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    _t0 = time.time()
    _DEADLINE = 115.0  # safety margin before 120s limit

    n, m, p = 3, 4, 6
    terms = []  # Each term: (u_dict, v_dict, w_dict)

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        """Add Strassen decomposition for A[i1:i2+1, j1:j2+1] * B[j1:j2+1, k1:k2+1] -> C[i1:i2+1, k1:k2+1]"""
        a = (i1, j1)
        b = (i1, j2)
        c = (i2, j1)
        d = (i2, j2)
        e = (j1, k1)
        f = (j1, k2)
        g = (j2, k1)
        h = (j2, k2)

        C11 = (i1, k1)
        C12 = (i1, k2)
        C21 = (i2, k1)
        C22 = (i2, k2)

        # M1 = (a+d)(e+h), C: C11:+1, C22:+1
        terms.append((
            {a: 1, d: 1},
            {e: 1, h: 1},
            {C11: 1, C22: 1}
        ))
        # M2 = (c+d)*e, C: C21:+1, C22:-1
        terms.append((
            {c: 1, d: 1},
            {e: 1},
            {C21: 1, C22: -1}
        ))
        # M3 = a*(f-h), C: C12:+1, C22:+1
        terms.append((
            {a: 1},
            {f: 1, h: -1},
            {C12: 1, C22: 1}
        ))
        # M4 = d*(g-e), C: C11:+1, C21:+1
        terms.append((
            {d: 1},
            {g: 1, e: -1},
            {C11: 1, C21: 1}
        ))
        # M5 = (a+b)*h, C: C11:-1, C12:+1
        terms.append((
            {a: 1, b: 1},
            {h: 1},
            {C11: -1, C12: 1}
        ))
        # M6 = (c-a)*(e+f), C: C22:+1
        terms.append((
            {c: 1, a: -1},
            {e: 1, f: 1},
            {C22: 1}
        ))
        # M7 = (b-d)*(g+h), C: C11:+1
        terms.append((
            {b: 1, d: -1},
            {g: 1, h: 1},
            {C11: 1}
        ))

    def add_schoolbook_term(i, j, k):
        """Add term for A[i,j] * B[j,k] -> C[i,k]"""
        terms.append((
            {(i, j): 1},
            {(j, k): 1},
            {(i, k): 1}
        ))

    # Part 1: Row 0 of A, schoolbook (24 terms)
    for j in range(4):
        for k in range(6):
            add_schoolbook_term(0, j, k)

    # Part 2: Rows 1-2 of A, using Strassen for 2x2 blocks
    # A[1:3, 0:2] * B[0:2, k1:k2] for each column pair
    # A[1:3, 2:4] * B[2:4, k1:k2] for each column pair
    for (k1, k2) in [(0, 1), (2, 3), (4, 5)]:
        # Left A block: rows 1,2; cols 0,1
        add_strassen_block(1, 2, 0, 1, k1, k2)
        # Right A block: rows 1,2; cols 2,3
        add_strassen_block(1, 2, 2, 3, k1, k2)

    # Now convert terms to U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][col] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][col] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][col] = coeff

    # Deadline guard
    if time.time() - _t0 > _DEADLINE:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
