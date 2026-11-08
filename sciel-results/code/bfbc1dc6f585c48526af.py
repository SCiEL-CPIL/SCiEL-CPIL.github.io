import time
import numpy as np

def construct():
    """
    Rank-49 tensor decomposition for 4x4 matrix multiplication
    using 2-level Strassen's algorithm.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    n, m, p = 4, 4, 4
    rank = 49

    U = np.zeros((n * m, rank), dtype=float)  # (16, 49)
    V = np.zeros((m * p, rank), dtype=float)  # (16, 49)
    W = np.zeros((p * n, rank), dtype=float)  # (16, 49)

    def block_entries(br, bc):
        return [[(2*br+r, 2*bc+c) for c in range(2)] for r in range(2)]

    def single_block_form(br, bc):
        res = []
        for r in range(2):
            row = []
            for c in range(2):
                row.append({(2*br+r, 2*bc+c): 1})
            res.append(row)
        return res

    def sum_block_form(br1, bc1, br2, bc2):
        res = []
        for r in range(2):
            row = []
            for c in range(2):
                row.append({(2*br1+r, 2*bc1+c): 1, (2*br2+r, 2*bc2+c): 1})
            res.append(row)
        return res

    def diff_block_form(br1, bc1, br2, bc2):
        res = []
        for r in range(2):
            row = []
            for c in range(2):
                row.append({(2*br1+r, 2*bc1+c): 1, (2*br2+r, 2*bc2+c): -1})
            res.append(row)
        return res

    top_products = [
        ({'x': 'sum', 'p1': (0,0), 'p2': (1,1)}, 'A',
         {'y': 'sum', 'p1': (0,0), 'p2': (1,1)}, 'B',
         {(0,0): 1, (1,1): 1}),
        ({'x': 'sum', 'p1': (1,0), 'p2': (1,1)}, 'A',
         {'y': 'single', 'p': (0,0)}, 'B',
         {(1,0): 1, (1,1): -1}),
        ({'x': 'single', 'p': (0,0)}, 'A',
         {'y': 'diff', 'p1': (0,1), 'p2': (1,1)}, 'B',
         {(0,1): 1, (1,1): 1}),
        ({'x': 'single', 'p': (1,1)}, 'A',
         {'y': 'diff', 'p1': (1,0), 'p2': (0,0)}, 'B',
         {(0,0): 1, (1,0): 1}),
        ({'x': 'sum', 'p1': (0,0), 'p2': (0,1)}, 'A',
         {'y': 'single', 'p': (1,1)}, 'B',
         {(0,0): -1, (0,1): 1}),
        ({'x': 'diff', 'p1': (1,0), 'p2': (0,0)}, 'A',
         {'y': 'sum', 'p1': (0,0), 'p2': (0,1)}, 'B',
         {(1,1): 1}),
        ({'x': 'diff', 'p1': (0,1), 'p2': (1,1)}, 'A',
         {'y': 'sum', 'p1': (1,0), 'p2': (1,1)}, 'B',
         {(0,0): 1}),
    ]

    D = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, -1]],
        [[0, 1], [0, 1]],
        [[1, 0], [1, 0]],
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]],
    ]

    u_entries = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,1,1), (1,1,-1)],
    ]
    v_entries = [
        [(0,0,1), (1,1,1)],
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (0,0,-1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (1,1,1)],
    ]

    def make_form(params, is_A):
        t = params['x'] if is_A else params['y']
        if t == 'single':
            br, bc = params['p']
            return single_block_form(br, bc)
        elif t == 'sum':
            br1, bc1 = params['p1']
            br2, bc2 = params['p2']
            return sum_block_form(br1, bc1, br2, bc2)
        elif t == 'diff':
            br1, bc1 = params['p1']
            br2, bc2 = params['p2']
            return diff_block_form(br1, bc1, br2, bc2)
        else:
            raise ValueError(f"Unknown form type: {t}")

    for k in range(7):
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
            
        x_params, x_side, y_params, y_side, c_contrib = top_products[k]
        X = make_form(x_params, True)
        Y = make_form(y_params, False)

        for s in range(7):
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}

            r = 7 * k + s

            u_form = {}
            for (lr, lc, coeff) in u_entries[s]:
                for (gi, gj), c in X[lr][lc].items():
                    u_form[(gi, gj)] = u_form.get((gi, gj), 0) + coeff * c

            v_form = {}
            for (lr, lc, coeff) in v_entries[s]:
                for (gi, gj), c in Y[lr][lc].items():
                    v_form[(gi, gj)] = v_form.get((gi, gj), 0) + coeff * c

            for (gi, gj), coeff in u_form.items():
                if coeff != 0:
                    U[gi * m + gj, r] += coeff

            for (gi, gj), coeff in v_form.items():
                if coeff != 0:
                    V[gi * m + gj, r] += coeff

            for (br, bc), c_coeff in c_contrib.items():
                for lr in range(2):
                    for lc in range(2):
                        d_coeff = D[s][lr][lc]
                        if d_coeff == 0:
                            continue
                        total_coeff = c_coeff * d_coeff
                        if total_coeff == 0:
                            continue
                        i_global = 2 * br + lr
                        j_global = 2 * bc + lc
                        W[j_global * n + i_global, r] += total_coeff

    # Convert to integers if they are close to integers
    U = np.rint(U).astype(int)
    V = np.rint(V).astype(int)
    W = np.rint(W).astype(int)

    return {"U": U, "V": V, "W": W}
