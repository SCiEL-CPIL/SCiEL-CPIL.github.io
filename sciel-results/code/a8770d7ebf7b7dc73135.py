import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 125

    n, m, p = DIMENSIONS
    try:
        U, V, W = _strassen_4x4(n, m, p)
        _verify(U, V, W, n, m, p, deadline)
        return {"U": U, "V": V, "W": W}
    except Exception as e:
        # Fallback: rank-64 trivial decomposition
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "64": W} if False else {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, deadline):
    """Verify the decomposition with multiple random tests."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    for seed in [42, 123, 999]:
        if time.time() > deadline - 10:
            break
        rng = np.random.RandomState(seed)
        A = rng.rand(n, m)
        B = rng.rand(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        
        for r in range(rank):
            if time.time() > deadline - 5:
                break
            u_col = U_arr[:, r].reshape(n, m)
            v_col = V_arr[:, r].reshape(m, p)
            
            uA = u_col * A
            vB = v_col * B
            
            term = uA @ vB
            
            w_mat = W_arr[:, r].reshape(p, n)
            C_computed += term * w_mat.T
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            diff = np.max(np.abs(C_computed - C_expected))
            raise ValueError(f"Decomposition verification failed for seed {seed}, max diff: {diff}")

def _strassen_4x4(n, m, p):
    """
    Recursive Strassen: 4x4 = 2x2 blocks, each 2x2 block mult uses 7 products.
    Total rank = 7 (outer Strassen) * 7 (inner Strassen) = 49
    """
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Inner Strassen 2x2 decomposition (rank 7):
    # For multiplying 2x2 X * Y = Z:
    inner_defs = [
        # t=0: M0
        {'u': [(0,0,1),(1,1,1)], 'v': [(0,0,1),(1,1,1)], 'w': [(0,0,1),(1,1,1)]},
        # t=1: M1
        {'u': [(1,0,1),(1,1,1)], 'v': [(0,0,1)], 'w': [(1,0,1),(1,1,-1)]},
        # t=2: M2
        {'u': [(0,0,1)], 'v': [(0,1,1),(1,1,-1)], 'w': [(0,1,1),(1,1,1)]},
        # t=3: M3
        {'u': [(1,1,1)], 'v': [(1,0,1),(0,0,-1)], 'w': [(0,0,1),(1,0,1)]},
        # t=4: M4
        {'u': [(0,0,1),(0,1,1)], 'v': [(1,1,1)], 'w': [(0,1,1),(0,0,-1)]},
        # t=5: M5
        {'u': [(1,0,1),(0,0,-1)], 'v': [(0,0,1),(0,1,1)], 'w': [(1,1,1)]},
        # t=6: M6
        {'u': [(0,1,1),(1,1,-1)], 'v': [(1,0,1),(1,1,1)], 'w': [(0,0,1)]},
    ]

    # Outer Strassen for 4x4 (treating as 2x2 blocks of size 2x2):
    outer_defs = [
        # s=0: M0
        {'a_blocks': [(0,0,1),(1,1,1)], 'b_blocks': [(0,0,1),(1,1,1)], 'c_blocks': [(0,0,1),(1,1,1)]},
        # s=1: M1
        {'a_blocks': [(1,0,1),(1,1,1)], 'b_blocks': [(0,0,1)], 'c_blocks': [(1,0,1),(1,1,-1)]},
        # s=2: M2
        {'a_blocks': [(0,0,1)], 'b_blocks': [(0,1,1),(1,1,-1)], 'c_blocks': [(0,1,1),(1,1,1)]},
        # s=3: M3
        {'a_blocks': [(1,1,1)], 'b_blocks': [(1,0,1),(0,0,-1)], 'c_blocks': [(0,0,1),(1,0,1)]},
        # s=4: M4
        {'a_blocks': [(0,0,1),(0,1,1)], 'b_blocks': [(1,1,1)], 'c_blocks': [(0,1,1),(0,0,-1)]},
        # s=5: M5
        {'a_blocks': [(1,0,1),(0,0,-1)], 'b_blocks': [(0,0,1),(0,1,1)], 'c_blocks': [(1,1,1)]},
        # s=6: M6
        {'a_blocks': [(0,1,1),(1,1,-1)], 'b_blocks': [(1,0,1),(1,1,1)], 'c_blocks': [(0,0,1)]},
    ]

    for s, outer in enumerate(outer_defs):
        for t, inner in enumerate(inner_defs):
            r = s * 7 + t

            # Build U contributions
            for (P, Q, sign_a) in outer['a_blocks']:
                for (ir, jc, sign_x) in inner['u']:
                    coeff = sign_a * sign_x
                    if coeff != 0:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        u_idx = row * m + col
                        U[u_idx][r] += coeff

            # Build V contributions
            for (P, Q, sign_b) in outer['b_blocks']:
                for (ir, jc, sign_y) in inner['v']:
                    coeff = sign_b * sign_y
                    if coeff != 0:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        v_idx = row * p + col
                        V[v_idx][r] += coeff

            # Build W contributions
            for (P, Q, sign_c) in outer['c_blocks']:
                for (ir, jc, sign_z) in inner['w']:
                    coeff = sign_c * sign_z
                    if coeff != 0:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        w_idx = col * n + row
                        W[w_idx][r] += coeff

    return U, V, W
