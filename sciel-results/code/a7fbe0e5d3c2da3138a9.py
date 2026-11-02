import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0

    n, m, p = 2, 5, 6
    # Rank calculation:
    # A cols: 2, 2, 1
    # B rows: 2, 2, 1
    # B cols: 2, 2, 2
    # Blocks:
    # (2,2)x(2,2) -> 3 * 7 = 21
    # (2,1)x(2,2) -> 3 * 4 = 12
    # (1,2)x(2,2) -> 3 * 4 = 12
    # (1,1)x(2,2) -> 3 * 2 = 6
    # Total = 51
    
    rank = 51
    
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

    def add_strassen_2x2x2(j1, j2, k1, k2):
        """
        Adds Strassen rank-7 terms for A[:, j1:j2+1] * B[j1:j2+1, k1:k2+1]
        A is 2x2, B is 2x2, C is 2x2.
        j1, j2 are the row indices of B (col indices of A).
        k1, k2 are the col indices of B.
        """
        # m1 = (a00+a11)(b00+b11)
        add_rank_step([(u_idx(0,j1), 1), (u_idx(1,j2), 1)], 
                     [(v_idx(j1,k1), 1), (v_idx(j2,k2), 1)], 
                     [(w_idx(k1,0), 1), (w_idx(k2,1), 1)])
                     
        # m2 = (a10+a11)*b00
        add_rank_step([(u_idx(1,j1), 1), (u_idx(1,j2), 1)], 
                     [(v_idx(j1,k1), 1)], 
                     [(w_idx(k1,1), 1), (w_idx(k2,1), -1)])
                     
        # m3 = a00*(b01-b11)
        add_rank_step([(u_idx(0,j1), 1)], 
                     [(v_idx(j1,k2), 1), (v_idx(j2,k2), -1)], 
                     [(w_idx(k2,0), 1), (w_idx(k2,1), 1)])
                     
        # m4 = a11*(b10-b00)
        add_rank_step([(u_idx(1,j2), 1)], 
                     [(v_idx(j2,k1), 1), (v_idx(j1,k1), -1)], 
                     [(w_idx(k1,0), 1), (w_idx(k1,1), 1)])
                     
        # m5 = (a00+a01)*b11
        add_rank_step([(u_idx(0,j1), 1), (u_idx(0,j2), 1)], 
                     [(v_idx(j2,k2), 1)], 
                     [(w_idx(k1,0), -1), (w_idx(k2,0), 1)])
                     
        # m6 = (a10-a00)*(b00+b01)
        add_rank_step([(u_idx(1,j1), 1), (u_idx(0,j1), -1)], 
                     [(v_idx(j1,k1), 1), (v_idx(j1,k2), 1)], 
                     [(w_idx(k2,1), 1)])
                     
        # m7 = (a01-a11)*(b10+b11)
        add_rank_step([(u_idx(0,j2), 1), (u_idx(1,j2), -1)], 
                     [(v_idx(j2,k1), 1), (v_idx(j2,k2), 1)], 
                     [(w_idx(k1,0), 1)])

    def add_2x1x2(j, k1, k2):
        """
        Adds rank-4 terms for A[:, j] * B[j, k1:k2+1]
        A is 2x1, B is 1x2, C is 2x2.
        """
        # C00 = A0j * Bj,k1
        add_rank_step([(u_idx(0,j), 1)], [(v_idx(j,k1), 1)], [(w_idx(k1,0), 1)])
        # C01 = A0j * Bj,k2
        add_rank_step([(u_idx(0,j), 1)], [(v_idx(j,k2), 1)], [(w_idx(k2,0), 1)])
        # C10 = A1j * Bj,k1
        add_rank_step([(u_idx(1,j), 1)], [(v_idx(j,k1), 1)], [(w_idx(k1,1), 1)])
        # C11 = A1j * Bj,k2
        add_rank_step([(u_idx(1,j), 1)], [(v_idx(j,k2), 1)], [(w_idx(k2,1), 1)])

    def add_1x2x2(i, j1, j2, k):
        """
        Adds rank-4 terms for A[i, j1:j2+1] * B[j1:j2+1, k]
        A is 1x2, B is 2x1, C is 1x1.
        Wait, C is 2x2? No, i is fixed.
        A is 1x2 (row i). B is 2x1 (col k).
        C is 1x1 (element i,k).
        Rank 4?
        C_ik = A_i,j1 B_j1,k + A_i,j2 B_j2,k.
        This is 2 terms.
        So rank 2.
        """
        # C_ki = A_i,j1 * B_j1,k
        add_rank_step([(u_idx(i,j1), 1)], [(v_idx(j1,k), 1)], [(w_idx(k,i), 1)])
        # C_ki = A_i,j2 * B_j2,k
        add_rank_step([(u_idx(i,j2), 1)], [(v_idx(j2,k), 1)], [(w_idx(k,i), 1)])

    def add_1x1x2(i, j, k1, k2):
        """
        Adds rank-2 terms for A[i, j] * B[j, k1:k2+1]
        A is 1x1, B is 1x2, C is 1x2.
        """
        # C_k1,i = A_i,j * B_j,k1
        add_rank_step([(u_idx(i,j), 1)], [(v_idx(j,k1), 1)], [(w_idx(k1,i), 1)])
        # C_k2,i = A_i,j * B_j,k2
        add_rank_step([(u_idx(i,j), 1)], [(v_idx(j,k2), 1)], [(w_idx(k2,i), 1)])

    # Partitions:
    # A cols (B rows): 0,1 (size 2), 2,3 (size 2), 4 (size 1)
    # B cols: 0,1 (size 2), 2,3 (size 2), 4,5 (size 2)
    
    # Block (2,2) for A cols 0,1 and B cols 0,1
    add_strassen_2x2x2(0, 1, 0, 1)
    # Block (2,2) for A cols 0,1 and B cols 2,3
    add_strassen_2x2x2(0, 1, 2, 3)
    # Block (2,2) for A cols 0,1 and B cols 4,5
    add_strassen_2x2x2(0, 1, 4, 5)
    
    # Block (2,1) for A cols 0,1 (size 2) and B col 4 (size 1)
    # Wait, the partition is A cols 0,1 (2), 2,3 (2), 4 (1).
    # B rows 0,1 (2), 2,3 (2), 4 (1).
    
    # A cols 0,1 (2) with B rows 2,3 (2) -> 2x2x2
    add_strassen_2x2x2(2, 3, 0, 1)
    add_strassen_2x2x2(2, 3, 2, 3)
    add_strassen_2x2x2(2, 3, 4, 5)
    
    # A cols 0,1 (2) with B row 4 (1) -> 2x1x2
    add_2x1x2(4, 0, 1)
    add_2x1x2(4, 2, 3)
    add_2x1x2(4, 4, 5)
    
    # A col 4 (1) with B rows 0,1 (2) -> 1x2x2
    # Note: add_1x2x2 takes i, j1, j2, k
    # We need to cover all i (0,1) and all k (0..5)
    for i in range(2):
        for k1, k2 in [(0,1), (2,3), (4,5)]:
            # k is the column index of B.
            # We need to handle k1 and k2 separately?
            # No, add_1x2x2 handles a single k.
            # So we call it for k1 and k2.
            add_1x2x2(i, 0, 1, k1)
            add_1x2x2(i, 0, 1, k2)
            
    # A col 4 (1) with B rows 2,3 (2) -> 1x2x2
    for i in range(2):
        for k1, k2 in [(0,1), (2,3), (4,5)]:
            add_1x2x2(i, 2, 3, k1)
            add_1x2x2(i, 2, 3, k2)
            
    # A col 4 (1) with B row 4 (1) -> 1x1x2
    for i in range(2):
        for k1, k2 in [(0,1), (2,3), (4,5)]:
            add_1x1x2(i, 4, k1, k2)

    return {"U": U, "V": V, "W": W}
