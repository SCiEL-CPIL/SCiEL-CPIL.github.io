import time

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 10.0

    n, m, p = DIMENSIONS
    R = 44

    # Pre-build the 44-term tensor as a list of (u, v, w) tuples
    # where u is a list of (row_idx, val), v is list of (col_idx, val), w is list of (out_idx, val)
    terms = []

    for b in range(4):
        if time.time() > deadline:
            break
        k0 = 2 * b
        k1 = 2 * b + 1

        # Strassen 2x2 * 2x2 for top part
        # M1: (A00+A11)(B0k0+B1k1)
        terms.append(([ (0,1),(4,1) ], [ (0*p+k0,1),(1*p+k1,1) ], [ (k0*n+0,1),(k1*n+1,1) ]))
        # M2: (A10+A11)B0k0
        terms.append(([ (3,1),(4,1) ], [ (0*p+k0,1) ], [ (k0*n+1,1),(k1*n+1,-1) ]))
        # M3: A00(B0k1-B1k1)
        terms.append(([ (0,1) ], [ (0*p+k1,1),(1*p+k1,-1) ], [ (k1*n+0,1),(k1*n+1,1) ]))
        # M4: A11(B1k0-B0k0)
        terms.append(([ (4,1) ], [ (1*p+k0,1),(0*p+k0,-1) ], [ (k0*n+0,1),(k0*n+1,1) ]))
        # M5: (A00+A01)B1k1
        terms.append(([ (0,1),(1,1) ], [ (1*p+k1,1) ], [ (k0*n+0,-1),(k1*n+0,1) ]))
        # M6: (A10-A00)(B0k0+B0k1)
        terms.append(([ (3,1),(0,-1) ], [ (0*p+k0,1),(0*p+k1,1) ], [ (k1*n+1,1) ]))
        # M7: (A01-A11)(B1k0+B1k1)
        terms.append(([ (1,1),(4,-1) ], [ (1*p+k0,1),(1*p+k1,1) ], [ (k0*n+0,1) ]))

        # Schoolbook for bottom: A(2x1) * B(1x2)
        # P1: A02 * B2k0
        terms.append(([ (2,1) ], [ (2*p+k0,1) ], [ (k0*n+0,1) ]))
        # P2: A02 * B2k1
        terms.append(([ (2,1) ], [ (2*p+k1,1) ], [ (k1*n+0,1) ]))
        # P3: A12 * B2k0
        terms.append(([ (5,1) ], [ (2*p+k0,1) ], [ (k0*n+1,1) ]))
        # P4: A12 * B2k1
        terms.append(([ (5,1) ], [ (2*p+k1,1) ], [ (k1*n+1,1) ]))

    # Now build U, V, W from terms
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]

    for t, (u_list, v_list, w_list) in enumerate(terms):
        for idx, val in u_list:
            U[idx][t] = val
        for idx, val in v_list:
            V[idx][t] = val
        for idx, val in w_list:
            W[idx][t] = val

    return {"U": U, "V": V, "W": W}
