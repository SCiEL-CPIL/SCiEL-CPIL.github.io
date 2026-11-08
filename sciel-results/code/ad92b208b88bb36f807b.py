import numpy as np

def construct():
    n, m, p = 3, 4, 6
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    curr_rank = 0
    
    def add_mults(u_specs, v_specs, w_specs):
        nonlocal curr_rank
        if curr_rank >= rank:
            return
        for idx, val in u_specs:
            if val != 0:
                U[idx, curr_rank] += val
        for idx, val in v_specs:
            if val != 0:
                V[idx, curr_rank] += val
        for idx, val in w_specs:
            if val != 0:
                W[idx, curr_rank] += val
        curr_rank += 1

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # Part 1: 1x4x6 (Rank 4)
    for j in range(4):
        u_spec = [(u_idx(0, j), 1)]
        v_spec = [(v_idx(j, k), 1) for k in range(6)]
        w_spec = [(w_idx(k, 0), 1) for k in range(6)]
        add_mults(u_spec, v_spec, w_spec)

    # Part 2: 2x4x6 (Rank 24)
    # Using 3 * R(2,4,2) = 3 * 8 = 24
    def add_2x4x2(j_offset, k_offset):
        # j_offset: 0 or 2
        # k_offset: 0, 2, 4
        for local_k in range(2):
            global_k = k_offset + local_k
            for local_j in range(2):
                global_j = j_offset + local_j
                for i in range(2):
                    global_i = 1 + i
                    u_spec = [(u_idx(global_i, global_j), 1)]
                    v_spec = [(v_idx(global_j, global_k), 1)]
                    w_spec = [(w_idx(global_k, global_i), 1)]
                    add_mults(u_spec, v_spec, w_spec)

    # k=0,1
    add_2x4x2(0, 0)
    add_2x4x2(2, 0)
    # k=2,3
    add_2x4x2(0, 2)
    add_2x4x2(2, 2)
    # k=4,5
    add_2x4x2(0, 4)
    add_2x4x2(2, 4)

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
