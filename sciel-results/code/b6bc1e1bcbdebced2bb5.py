import time

def construct():
    n, m, p = 2, 5, 6
    # Split B into two 3-column blocks
    # Total rank = 2 * (9 + 9 + 1) = 38
    rank = 38
    
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
        # u_base: [u00, u01, u10, u11]
        # v_base: [v00, v01, v10, v11]
        # w_base: [w00, w01, w10, w11]
        
        # m1 = (u00+u11)(v00+v11)
        add_rank_step([(u_base[0], 1), (u_base[3], 1)], 
                     [(v_base[0], 1), (v_base[3], 1)], 
                     [(w_base[0], 1), (w_base[3], 1)])
                     
        # m2 = (u10+u11)*v00
        add_rank_step([(u_base[2], 1), (u_base[3], 1)], 
                     [(v_base[0], 1)], 
                     [(w_base[2], 1), (w_base[3], -1)])
                     
        # m3 = u00*(v01-v11)
        add_rank_step([(u_base[0], 1)], 
                     [(v_base[1], 1), (v_base[3], -1)], 
                     [(w_base[1], 1), (w_base[3], 1)])
                     
        # m4 = u11*(v10-v00)
        add_rank_step([(u_base[3], 1)], 
                     [(v_base[2], 1), (v_base[0], -1)], 
                     [(w_base[0], 1), (w_base[2], 1)])
                     
        # m5 = (u00+u01)*v11
        add_rank_step([(u_base[0], 1), (u_base[1], 1)], 
                     [(v_base[3], 1)], 
                     [(w_base[0], -1), (w_base[1], 1)])
                     
        # m6 = (u10-u00)*(v00+v01)
        add_rank_step([(u_base[2], 1), (u_base[0], -1)], 
                     [(v_base[0], 1), (v_base[1], 1)], 
                     [(w_base[3], 1)])
                     
        # m7 = (u01-u11)*(v10+v11)
        add_rank_step([(u_base[1], 1), (u_base[3], -1)], 
                     [(v_base[2], 1), (v_base[3], 1)], 
                     [(w_base[0], 1)])

    def add_2x2x1(u_base, v_base, w_base):
        # u_base: [u00, u01, u10, u11] (only 2 used? No, 2x2x1 means B is 2x1)
        # A is 2x2, B is 2x1. C is 2x1.
        # Schoolbook: 4 terms.
        # u00 v00 -> w00
        # u01 v01 -> w01? No.
        # B is 2x1. Indices j=0,1. k=0.
        # C is 2x1. Indices i=0,1. k=0.
        # C[i, 0] = A[i, 0]B[0, 0] + A[i, 1]B[1, 0].
        # w_base should be [w00, w10] (k=0, i=0 and k=0, i=1).
        # Actually w_idx(k, i).
        # w00 = w_idx(0, 0)
        # w10 = w_idx(0, 1)
        
        # Term 1: A[0,0] B[0,0] -> C[0,0]
        add_rank_step([(u_base[0], 1)], [(v_base[0], 1)], [(w_base[0], 1)])
        # Term 2: A[1,0] B[0,0] -> C[1,0]
        add_rank_step([(u_base[2], 1)], [(v_base[0], 1)], [(w_base[1], 1)])
        # Term 3: A[0,1] B[1,0] -> C[0,0]
        add_rank_step([(u_base[1], 1)], [(v_base[1], 1)], [(w_base[0], 1)])
        # Term 4: A[1,1] B[1,0] -> C[1,0]
        add_rank_step([(u_base[3], 1)], [(v_base[1], 1)], [(w_base[1], 1)])

    def add_2x1x3(u_base, v_base, w_base):
        # A is 2x1, B is 1x3. C is 2x3.
        # Rank 1.
        # U: u00, u10.
        # V: v00, v01, v02.
        # W: w00, w10, w01, w11, w02, w12.
        # w_idx(k, i).
        # k=0,1,2. i=0,1.
        
        # U[0,0] = A[0,0]
        # U[1,0] = A[1,0]
        # V[0,0] = B[0,0]
        # V[0,1] = B[0,1]
        # V[0,2] = B[0,2]
        # W[k, i] = 1
        
        add_rank_step([(u_base[0], 1), (u_base[1], 1)], 
                     [(v_base[0], 1), (v_base[1], 1), (v_base[2], 1)], 
                     [(w_base[0], 1), (w_base[1], 1), (w_base[2], 1), (w_base[3], 1), (w_base[4], 1), (w_base[5], 1)])

    def add_2x2x3(u_base, v_base, w_base):
        # A is 2x2, B is 2x3. C is 2x3.
        # Split B into 2x2 and 2x1.
        # B_L: v00, v01, v10, v11 (cols 0,1)
        # B_R: v02, v12 (col 2)
        # C_L: w00, w10, w01, w11 (cols 0,1)
        # C_R: w02, w12 (col 2)
        
        # Strassen for B_L
        u_s = u_base
        v_s = [v_base[0], v_base[1], v_base[2], v_base[3]]
        w_s = [w_base[0], w_base[1], w_base[2], w_base[3]]
        add_strassen_2x2x2(u_s, v_s, w_s)
        
        # Schoolbook for B_R
        u_s2 = u_base
        v_s2 = [v_base[4], v_base[5]]
        w_s2 = [w_base[4], w_base[5]]
        add_2x2x1(u_s2, v_s2, w_s2)

    # Process B col block 0 (cols 0,1,2)
    # A cols 0,1 vs B rows 0,1
    u_00 = [u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1)]
    v_00 = [v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1), v_idx(0,2), v_idx(1,2)]
    w_00 = [w_idx(0,0), w_idx(1,0), w_idx(0,1), w_idx(1,1), w_idx(2,0), w_idx(2,1)]
    add_2x2x3(u_00, v_00, w_00)
    
    # A cols 2,3 vs B rows 2,3
    u_11 = [u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3)]
    v_11 = [v_idx(2,0), v_idx(2,1), v_idx(3,0), v_idx(3,1), v_idx(2,2), v_idx(3,2)]
    w_11 = [w_idx(0,0), w_idx(1,0), w_idx(0,1), w_idx(1,1), w_idx(2,0), w_idx(2,1)]
    add_2x2x3(u_11, v_11, w_11)
    
    # A col 4 vs B row 4
    u_22 = [u_idx(0,4), u_idx(1,4)]
    v_22 = [v_idx(4,0), v_idx(4,1), v_idx(4,2)]
    w_22 = [w_idx(0,0), w_idx(1,0), w_idx(0,1), w_idx(1,1), w_idx(0,2), w_idx(1,2)]
    add_2x1x3(u_22, v_22, w_22)

    # Process B col block 1 (cols 3,4,5)
    # A cols 0,1 vs B rows 0,1
    u_33 = [u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1)]
    v_33 = [v_idx(0,3), v_idx(0,4), v_idx(1,3), v_idx(1,4), v_idx(0,5), v_idx(1,5)]
    w_33 = [w_idx(3,0), w_idx(4,0), w_idx(3,1), w_idx(4,1), w_idx(5,0), w_idx(5,1)]
    add_2x2x3(u_33, v_33, w_33)
    
    # A cols 2,3 vs B rows 2,3
    u_44 = [u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3)]
    v_44 = [v_idx(2,3), v_idx(2,4), v_idx(3,3), v_idx(3,4), v_idx(2,5), v_idx(3,5)]
    w_44 = [w_idx(3,0), w_idx(4,0), w_idx(3,1), w_idx(4,1), w_idx(5,0), w_idx(5,1)]
    add_2x2x3(u_44, v_44, w_44)
    
    # A col 4 vs B row 4
    u_55 = [u_idx(0,4), u_idx(1,4)]
    v_55 = [v_idx(4,3), v_idx(4,4), v_idx(4,5)]
    w_55 = [w_idx(3,0), w_idx(4,0), w_idx(3,1), w_idx(4,1), w_idx(5,0), w_idx(5,1)]
    add_2x1x3(u_55, v_55, w_55)

    return {"U": U, "V": V, "W": W}
