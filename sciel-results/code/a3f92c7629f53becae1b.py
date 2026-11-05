import numpy as np
import time
import itertools
import random

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    deadline = start_time + 130.0

    # Rank-56 Strassen decomposition
    rank_56 = 56
    U = [[0] * rank_56 for _ in range(n * m)]
    V = [[0] * rank_56 for _ in range(m * p)]
    W = [[0] * rank_56 for _ in range(p * n)]

    strassen_U = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1},
    ]
    strassen_V = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1},
    ]
    strassen_W = [
        {(0,0): 1, (1,1): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(0,0): -1, (1,0): 1},
        {(1,1): 1},
        {(0,0): 1},
    ]

    def add_block(base_rank, i_base, j_base, k_base):
        for r in range(7):
            cr = base_rank + r
            for (li, lj), val in strassen_U[r].items():
                gi = i_base + li
                gj = j_base + lj
                U[gi * m + gj][cr] += val
            for (lj, lk), val in strassen_V[r].items():
                gj = j_base + lj
                gk = k_base + lk
                V[gj * p + gk][cr] += val
            for (lk, li), val in strassen_W[r].items():
                gk = k_base + lk
                gi = i_base + li
                W[gk * n + gi][cr] += val

    # 8 blocks
    add_block(0, 0, 0, 0)
    add_block(7, 0, 0, 2)
    add_block(14, 0, 0, 4)
    add_block(21, 0, 0, 6)
    add_block(28, 0, 2, 0)
    add_block(35, 0, 2, 2)
    add_block(42, 0, 2, 4)
    add_block(49, 0, 2, 6)

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    _triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    _u_idx = np.array([i * m + j for i, j, k in _triples])
    _v_idx = np.array([j * p + k for i, j, k in _triples])
    _w_idx = np.array([k * n + i for i, j, k in _triples])

    def verify(U_arr, V_arr, W_arr, rank_val):
        u_vals = U_arr[_u_idx, :rank_val]
        v_vals = V_arr[_v_idx, :rank_val]
        w_vals = W_arr[_w_idx, :rank_val]
        products = u_vals * v_vals * w_vals
        sums = products.sum(axis=1)
        return np.all(np.abs(sums - 1.0) < 1e-9)

    if not verify(U_np, V_np, W_np, rank_56):
        return {"U": U, "V": V, "W": W}

    best_rank = rank_56
    best_U = U_np.copy()
    best_V = V_np.copy()
    best_W = W_np.copy()

    # Greedy removal with random restarts
    # We try to remove as many columns as possible.
    # Strategy:
    # 1. Randomly shuffle column indices.
    # 2. Iterate through indices. If removing current column keeps validity, remove it.
    # 3. Repeat this process with different shuffles.
    
    current_U = U_np.copy()
    current_V = V_np.copy()
    current_W = W_np.copy()
    current_rank = rank_56
    active_indices = list(range(rank_56))

    def try_greedy_remove(U_curr, V_curr, W_curr, rank_curr, indices_order, deadline):
        U_work = U_curr.copy()
        V_work = V_curr.copy()
        W_work = W_curr.copy()
        rank_work = rank_curr
        active = [True] * rank_curr
        
        for idx in indices_order:
            if time.time() > deadline - 5:
                break
            if not active[idx]:
                continue
            
            # Try removing idx
            # Create mask
            mask = np.ones(rank_work, dtype=bool)
            mask[idx] = False
            
            # Verify
            U_test = U_work[:, mask]
            V_test = V_work[:, mask]
            W_test = W_work[:, mask]
            
            if verify(U_test, V_test, W_test, rank_work - 1):
                # Accept removal
                active[idx] = False
                U_work = U_test
                V_work = V_test
                W_work = W_test
                rank_work -= 1
        
        return rank_work, U_work, V_work, W_work, active

    # Run multiple restarts
    rng = np.random.default_rng(42)
    num_restarts = 20
    if time.time() < deadline - 30:
        for i in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            order = np.arange(rank_56)
            rng.shuffle(order)
            
            r, U_new, V_new, W_new, _ = try_greedy_remove(U_np, V_np, W_np, rank_56, order, deadline)
            
            if r < best_rank:
                best_rank = r
                best_U = U_new
                best_V = V_new
                best_W = W_new
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
