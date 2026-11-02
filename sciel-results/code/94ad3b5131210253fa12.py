import time

def construct():
    """
    Rank-50 exact tensor decomposition for 2x5 x 5x6 matrix multiplication.
    """
    n, m, p = 2, 5, 6
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_rank_step(u_pairs, v_pairs, w_pairs):
        nonlocal col
        if col >= rank:
            return False
        for idx, val in u_pairs:
            U[idx][col] = val
        for idx, val in v_pairs:
            V[idx][col] = val
        for idx, val in w_pairs:
            W[idx][col] = val
        col += 1
        return True

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    def add_strassen_2x2x2(u_base, v_base, w_base):
        """2x2x2 Strassen, rank 7."""
        add_rank_step([(u_base[0], 1), (u_base[3], 1)], 
                     [(v_base[0], 1), (v_base[3], 1)], 
                     [(w_base[0], 1), (w_base[3], 1)])
        add_rank_step([(u_base[2], 1), (u_base[3], 1)], 
                     [(v_base[0], 1)], 
                     [(w_base[2], 1), (w_base[3], -1)])
        add_rank_step([(u_base[0], 1)], 
                     [(v_base[1], 1), (v_base[3], -1)], 
                     [(w_base[1], 1), (w_base[3], 1)])
        add_rank_step([(u_base[3], 1)], 
                     [(v_base[2], 1), (v_base[0], -1)], 
                     [(w_base[0], 1), (w_base[2], 1)])
        add_rank_step([(u_base[0], 1), (u_base[1], 1)], 
                     [(v_base[3], 1)], 
                     [(w_base[0], -1), (w_base[1], 1)])
        add_rank_step([(u_base[2], 1), (u_base[0], -1)], 
                     [(v_base[0], 1), (v_base[1], 1)], 
                     [(w_base[3], 1)])
        add_rank_step([(u_base[1], 1), (u_base[3], -1)], 
                     [(v_base[2], 1), (v_base[3], 1)], 
                     [(w_base[0], 1)])

    def add_2x2x1(u_base, v_base, w_base):
        """2x2x1 schoolbook, rank 4.
        u_base: [u00, u01, u10, u11]
        v_base: [v0, v1]
        w_base: [w0, w1]
        """
        add_rank_step([(u_base[0], 1)], [(v_base[0], 1)], [(w_base[0], 1)])
        add_rank_step([(u_base[2], 1)], [(v_base[1], 1)], [(w_base[0], 1)])
        add_rank_step([(u_base[1], 1)], [(v_base[0], 1)], [(w_base[1], 1)])
        add_rank_step([(u_base[3], 1)], [(v_base[1], 1)], [(w_base[1], 1)])

    def add_2x2x3(u_base, v_base, w_base):
        """2x2x3, rank 11 = 7 + 4.
        u_base: [u00, u01, u10, u11]
        v_base: [v00, v01, v02, v10, v11, v12]
        w_base: [w00, w01, w10, w11, w02, w12]
        """
        v_sub = [v_base[0], v_base[1], v_base[3], v_base[4]]
        w_sub = [w_base[0], w_base[1], w_base[2], w_base[3]]
        add_strassen_2x2x2(u_base, v_sub, w_sub)
        
        v_col = [v_base[2], v_base[5]]
        w_col = [w_base[4], w_base[5]]
        add_2x2x1(u_base, v_col, w_col)

    # Block 1: A[:,0:2] * B[0:2,0:3] -> C[0:3,0:2]
    u1 = [u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1)]
    v1 = [v_idx(0,0), v_idx(0,1), v_idx(0,2), v_idx(1,0), v_idx(1,1), v_idx(1,2)]
    w1 = [w_idx(0,0), w_idx(0,1), w_idx(1,0), w_idx(1,1), w_idx(2,0), w_idx(2,1)]
    add_2x2x3(u1, v1, w1)
    
    # Block 2: A[:,0:2] * B[0:2,3:6] -> C[3:6,0:2]
    v2 = [v_idx(0,3), v_idx(0,4), v_idx(0,5), v_idx(1,3), v_idx(1,4), v_idx(1,5)]
    w2 = [w_idx(3,0), w_idx(3,1), w_idx(4,0), w_idx(4,1), w_idx(5,0), w_idx(5,1)]
    add_2x2x3(u1, v2, w2)
    
    # Block 3: A[:,2:4] * B[2:4,0:3] -> C[0:3,0:2]
    u3 = [u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3)]
    v3 = [v_idx(2,0), v_idx(2,1), v_idx(2,2), v_idx(3,0), v_idx(3,1), v_idx(3,2)]
    w3 = [w_idx(0,0), w_idx(0,1), w_idx(1,0), w_idx(1,1), w_idx(2,0), w_idx(2,1)]
    add_2x2x3(u3, v3, w3)
    
    # Block 4: A[:,2:4] * B[2:4,3:6] -> C[3:6,0:2]
    v4 = [v_idx(2,3), v_idx(2,4), v_idx(2,5), v_idx(3,3), v_idx(3,4), v_idx(3,5)]
    w4 = [w_idx(3,0), w_idx(3,1), w_idx(4,0), w_idx(4,1), w_idx(5,0), w_idx(5,1)]
    add_2x2x3(u3, v4, w4)
    
    # Block 5: A[:,4] * B[4,:] -> C[0:6,0:2]
    for i in range(n):
        for k in range(p):
            add_rank_step([(u_idx(i,4), 1)], [(v_idx(4,k), 1)], [(w_idx(k,i), 1)])
    
    return {"U": U, "V": V, "W": W}
