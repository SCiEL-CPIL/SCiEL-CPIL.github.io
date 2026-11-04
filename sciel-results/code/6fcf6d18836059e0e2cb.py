import numpy as np
import time

_START = time.time()
DEADLINE = 115.0

def _time_left():
    return DEADLINE - (time.time() - _START)

def construct():
    n, m, p = 4, 4, 6
    U_list = []
    V_list = []
    W_list = []

    def add_term(u_dict, v_dict, w_dict):
        u_col = [0.0] * (n * m)
        v_col = [0.0] * (m * p)
        w_col = [0.0] * (p * n)
        for (i, j), c in u_dict.items():
            u_col[i * m + j] += c
        for (j, k), c in v_dict.items():
            v_col[j * p + k] += c
        for (i, k), c in w_dict.items():
            w_col[k * n + i] += c
        U_list.append(u_col)
        V_list.append(v_col)
        W_list.append(w_col)

    # Strassen 2x2*2x2 steps: (U, V, W) each dict {(r,c): coeff}
    S2 = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    # Outer Strassen steps for 4x4*4x6:
    # Each: (A_blocks, B_blocks, C_blocks) where blocks are '00','01','10','11'
    # A is 4x4, B is 4x6, C is 4x6
    # A blocks: 2x2, B blocks: 2x3, C blocks: 2x3
    OS = [
        ({'00':1, '11':1}, {'00':1, '11':1}, {'00':1, '11':1}),
        ({'10':1, '11':1}, {'00':1}, {'10':1, '11':-1}),
        ({'00':1}, {'01':1, '11':-1}, {'01':1, '11':1}),
        ({'11':1}, {'10':1, '00':-1}, {'00':1, '10':1}),
        ({'00':1, '01':1}, {'11':1}, {'00':-1, '01':1}),
        ({'10':1, '00':-1}, {'00':1, '01':1}, {'11':1}),
        ({'01':1, '11':-1}, {'10':1, '11':1}, {'00':1}),
    ]

    def blk_off(block, d1, d2):
        if block == '00': return (0, 0)
        elif block == '01': return (0, d2)
        elif block == '10': return (d1, 0)
        elif block == '11': return (d1, d2)
        raise ValueError(block)

    for s in range(7):
        a_b, b_b, c_b = OS[s]
        
        # Part 1: Strassen for 2x2*2x2 (first 2 cols of each B block)
        for iu, iv, iw in S2:
            u_dict, v_dict, w_dict = {}, {}, {}
            for ab, ac in a_b.items():
                ai, aj = blk_off(ab, 2, 2)
                for (lr, lc), ic in iu.items():
                    u_dict[(ai+lr, aj+lc)] = u_dict.get((ai+lr, aj+lc), 0) + ac*ic
            for bb, bc in b_b.items():
                bj, bk = blk_off(bb, 2, 3)
                for (lr, lc), ic in iv.items():
                    v_dict[(bj+lr, bk+lc)] = v_dict.get((bj+lr, bk+lc), 0) + bc*ic
            for cb, cc in c_b.items():
                ci, ck = blk_off(cb, 2, 3)
                for (lr, lc), ic in iw.items():
                    w_dict[(ci+lr, ck+lc)] = w_dict.get((ci+lr, ck+lc), 0) + cc*ic
            add_term(u_dict, v_dict, w_dict)
        
        # Part 2: Schoolbook for 2x2*2x1 (3rd col of each B block)
        # For each (a_blk, b_blk, c_blk) combo in the outer step:
        # A[a_blk] * B[b_blk, col2] -> C[c_blk, col2]
        # This is 2x2 * 2x1 = 4 scalar mults
        for ab, ac in a_b.items():
            ai, aj = blk_off(ab, 2, 2)
            for bb, bc in b_b.items():
                bj, bk = blk_off(bb, 2, 3)
                for cb, cc in c_b.items():
                    ci, ck = blk_off(cb, 2, 3)
                    for r in range(2):
                        for c in range(2):
                            add_term(
                                {(ai+r, aj+c): ac},
                                {(bj+r, bk+2): bc},
                                {(ci+r, ck+2): cc}
                            )

    rank = len(U_list)
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    # Verify
    try:
        rng = np.random.default_rng(42)
        for _ in range(5):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            Af = A.reshape(n*m)
            Bf = B.reshape(m*p)
            Ct = (A @ B).T.reshape(p*n)
            Cs = W.T @ ((U.T @ Af) * (V.T @ Bf))
            if not np.allclose(Cs, Ct, atol=1e-8):
                U = np.zeros((n*m, n*m*p))
                V = np.zeros((m*p, n*m*p))
                W = np.zeros((p*n, n*m*p))
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U[i*m+j, idx] = 1
                            V[j*p+k, idx] = 1
                            W[k*n+i, idx] = 1
                            idx += 1
                break
    except:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
