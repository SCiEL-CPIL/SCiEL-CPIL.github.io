"""Rewritten program: hybrid Strassen(4x4x4) + schoolbook(4x4x3) decomposition."""
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    n, m, p = 4, 4, 7
    columns = []

    def add_col(u_dict, v_dict, w_dict):
        columns.append((u_dict, v_dict, w_dict))

    # 4x4x4 block (B cols 0..3) via Strassen: 7 terms x 4 sub-products = 28 cols
    terms = [
        ([(0,0,1),(1,1,-1)], [(1,1,1),(0,0,-1)], [(0,0,1),(1,1,-1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,1,-1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(1,1,1)]),
    ]
    for a_side, b_side, out_side in terms:
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    for d in range(2):
                        u_dict = {}
                        for (biA, bjA, ca) in a_side:
                            key = (2*biA+a, 2*bjA+b)
                            u_dict[key] = u_dict.get(key, 0) + ca
                        v_dict = {}
                        for (bkB, blB, cb) in b_side:
                            key = (2*bkB+c, 2*blB+d)
                            v_dict[key] = v_dict.get(key, 0) + cb
                        w_dict = {}
                        for (biO, blO, sign) in out_side:
                            key = (2*blO+d, 2*biO+a)
                            w_dict[key] = w_dict.get(key, 0) + sign
                        add_col(u_dict, v_dict, w_dict)

    # 4x4x3 block (B cols 4..6) schoolbook: 48 cols
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                add_col({(i, j): 1}, {(j, k): 1}, {(k, i): 1})

    rank = len(columns)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for c_idx, (u_dict, v_dict, w_dict) in enumerate(columns):
        for (i, j), val in u_dict.items():
            U[i*m+j][c_idx] = val
        for (j, k), val in v_dict.items():
            V[j*p+k][c_idx] = val
        for (k, i), val in w_dict.items():
            W[k*n+i][c_idx] = val

    if time.time() - start > 110:
        # safety margin; construction is tiny so this is a no-op guard
        pass
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
