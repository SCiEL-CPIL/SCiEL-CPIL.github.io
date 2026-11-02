import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0

    n, m, p = 2, 5, 6
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0

    def add_rank_step(u_pairs, v_pairs, w_pairs):
        nonlocal col
        if col >= rank:
            return False
        if time.time() - start_time > deadline:
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

    def add_strassen(j1, j2, k1, k2):
        """
        Adds Strassen rank-7 terms for A[:, j1:j2+1] * B[j1:j2+1, k1:k2+1]
        assuming i is 0,1.
        """
        # m1 = (a00+a11)(b00+b11)
        # a00=U[0,j1], a11=U[1,j2], b00=V[j1,k1], b11=V[j2,k2]
        # C00=W[k1,0], C11=W[k2,1]
        add_rank_step([(u_idx(0,j1), 1), (u_idx(1,j2), 1)], 
                     [(v_idx(j1,k1), 1), (v_idx(j2,k2), 1)], 
                     [(w_idx(k1,0), 1), (w_idx(k2,1), 1)])
                     
        # m2 = (a10+a11)*b00
        # a10=U[1,j1], a11=U[1,j2], b00=V[j1,k1]
        # C10=W[k1,1], C11=W[k2,1] (-1)
        add_rank_step([(u_idx(1,j1), 1), (u_idx(1,j2), 1)], 
                     [(v_idx(j1,k1), 1)], 
                     [(w_idx(k1,1), 1), (w_idx(k2,1), -1)])
                     
        # m3 = a00*(b01-b11)
        # a00=U[0,j1], b01=V[j1,k2], b11=V[j2,k2]
        # C01=W[k2,0], C11=W[k2,1]
        add_rank_step([(u_idx(0,j1), 1)], 
                     [(v_idx(j1,k2), 1), (v_idx(j2,k2), -1)], 
                     [(w_idx(k2,0), 1), (w_idx(k2,1), 1)])
                     
        # m4 = a11*(b10-b00)
        # a11=U[1,j2], b10=V[j2,k1], b00=V[j1,k1]
        # C00=W[k1,0], C10=W[k1,1]
        add_rank_step([(u_idx(1,j2), 1)], 
                     [(v_idx(j2,k1), 1), (v_idx(j1,k1), -1)], 
                     [(w_idx(k1,0), 1), (w_idx(k1,1), 1)])
                     
        # m5 = (a00+a01)*b11
        # a00=U[0,j1], a01=U[0,j2], b11=V[j2,k2]
        # C00=W[k1,0] (-1), C01=W[k2,0]
        add_rank_step([(u_idx(0,j1), 1), (u_idx(0,j2), 1)], 
                     [(v_idx(j2,k2), 1)], 
                     [(w_idx(k1,0), -1), (w_idx(k2,0), 1)])
                     
        # m6 = (a10-a00)*(b00+b01)
        # a10=U[1,j1], a00=U[0,j1], b00=V[j1,k1], b01=V[j1,k2]
        # C11=W[k2,1]
        add_rank_step([(u_idx(1,j1), 1), (u_idx(0,j1), -1)], 
                     [(v_idx(j1,k1), 1), (v_idx(j1,k2), 1)], 
                     [(w_idx(k2,1), 1)])
                     
        # m7 = (a01-a11)*(b10+b11)
        # a01=U[0,j2], a11=U[1,j2], b10=V[j2,k1], b11=V[j2,k2]
        # C00=W[k1,0]
        add_rank_step([(u_idx(0,j2), 1), (u_idx(1,j2), -1)], 
                     [(v_idx(j2,k1), 1), (v_idx(j2,k2), 1)], 
                     [(w_idx(k1,0), 1)])

    def add_schoolbook_2x1x2(j, k1, k2):
        """
        Adds schoolbook rank-4 terms for A[:, j] * B[j, k1:k2+1]
        """
        # i=0, j, k1
        add_rank_step([(u_idx(0,j), 1)], [(v_idx(j,k1), 1)], [(w_idx(k1,0), 1)])
        # i=0, j, k2
        add_rank_step([(u_idx(0,j), 1)], [(v_idx(j,k2), 1)], [(w_idx(k2,0), 1)])
        # i=1, j, k1
        add_rank_step([(u_idx(1,j), 1)], [(v_idx(j,k1), 1)], [(w_idx(k1,1), 1)])
        # i=1, j, k2
        add_rank_step([(u_idx(1,j), 1)], [(v_idx(j,k2), 1)], [(w_idx(k2,1), 1)])

    # Part 1: 2x2x2 (j=0,1; k=0,1)
    add_strassen(0, 1, 0, 1)
    
    # Part 2: 2x2x4 (j=0,1; k=2,3,4,5)
    # Split into k=2,3 and k=4,5
    add_strassen(0, 1, 2, 3)
    add_strassen(0, 1, 4, 5)
    
    # Part 3: 2x3x2 (j=2,3,4; k=0,1)
    # Split j=2,3 (Strassen) and j=4 (Schoolbook)
    add_strassen(2, 3, 0, 1)
    add_schoolbook_2x1x2(4, 0, 1)
    
    # Part 4: 2x3x4 (j=2,3,4; k=2,3,4,5)
    # Split k=2,3 and k=4,5
    # Each is 2x3x2
    # k=2,3: j=2,3 (Strassen) + j=4 (Schoolbook)
    add_strassen(2, 3, 2, 3)
    add_schoolbook_2x1x2(4, 2, 3)
    
    # k=4,5: j=2,3 (Strassen) + j=4 (Schoolbook)
    add_strassen(2, 3, 4, 5)
    add_schoolbook_2x1x2(4, 4, 5)

    return {"U": U, "V": V, "W": W}
