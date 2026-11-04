import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = 2, 4, 5
    nm = n * m # 8
    mp = m * p # 20
    np_ = n * p # 10
    
    rank = 36
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((np_, rank))
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        if col >= rank:
            return
        for idx, val in zip(u_idx, u_val):
            U[idx, col] += val
        for idx, val in zip(v_idx, v_val):
            V[idx, col] += val
        for idx, val in zip(w_idx, w_val):
            W[idx, col] += val
        col += 1
        
    # Block 1: A1 (2x2) * B1 (2x5)
    # A1 indices: 0,1,4,5
    # B1 indices: 0..4, 5..9
    # Z1 indices: 0..9 (c = k*2 + i)
    
    # Strassen 2x2x2 part 1 (cols 0,1 of B1)
    # Y1 is B1[:, 0:2] -> indices 0,1,5,6
    # X is A1 -> indices 0,1,4,5
    
    # a1 = (x11+x22)(y11+y22)
    add_term([0, 5], [1, 1], [0, 6], [1, 1], [0, 3], [1, 1])
    # a2 = (x21+x22)y11
    add_term([4, 5], [1, 1], [0], [1], [2], [1])
    # a3 = x11(y12-y22)
    add_term([0], [1], [1, 6], [1, -1], [1, 3], [1, 1])
    # a4 = x22(y21-y11)
    add_term([5], [1], [5, 0], [1, -1], [0, 2], [1, 1])
    # a5 = (x11+x12)y22
    add_term([0, 1], [1, 1], [6], [1], [1, 0], [1, -1])
    # a6 = (x21-x11)(y11+y12)
    add_term([4, 0], [1, -1], [0, 1], [1, 1], [3], [1])
    # a7 = (x12-x22)(y21+y22)
    add_term([1, 5], [1, -1], [5, 6], [1, 1], [0], [1])
    
    # Strassen 2x2x2 part 2 (cols 2,3 of B1)
    # Y2 is B1[:, 2:4] -> indices 2,3,7,8
    # X is A1
    
    # b1 = (x11+x22)(y13+y24)
    add_term([0, 5], [1, 1], [2, 8], [1, 1], [4, 7], [1, 1])
    # b2 = (x21+x22)y13
    add_term([4, 5], [1, 1], [2], [1], [6], [1])
    # b3 = x11(y14-y24)
    add_term([0], [1], [3, 8], [1, -1], [5, 7], [1, 1])
    # b4 = x22(y23-y13)
    add_term([5], [1], [7, 2], [1, -1], [4, 6], [1, 1])
    # b5 = (x11+x12)y24
    add_term([0, 1], [1, 1], [8], [1], [5, 4], [1, -1])
    # b6 = (x21-x11)(y13+y14)
    add_term([4, 0], [1, -1], [2, 3], [1, 1], [7], [1])
    # b7 = (x12-x22)(y23+y24)
    add_term([1, 5], [1, -1], [7, 8], [1, 1], [4], [1])
    
    # Schoolbook 2x2x1 part (col 4 of B1)
    # Y3 is B1[:, 4:5] -> indices 4, 9
    # X is A1
    
    # c1 = x11*y15
    add_term([0], [1], [4], [1], [8], [1])
    # c2 = x12*y25
    add_term([1], [1], [9], [1], [8], [1])
    # c3 = x21*y15
    add_term([4], [1], [4], [1], [9], [1])
    # c4 = x22*y25
    add_term([5], [1], [9], [1], [9], [1])
    
    # Block 2: A2 (2x2) * B2 (2x5)
    # A2 indices: 2,3,6,7
    # B2 indices: 10..14, 15..19
    # Z2 indices: 0..9 (same c indices as Z1, added to)
    
    # Strassen 2x2x2 part 1 (cols 0,1 of B2)
    # Y1 is B2[:, 0:2] -> indices 10,11,15,16
    # X is A2 -> indices 2,3,6,7
    
    # a1 = (x11+x22)(y11+y22)
    add_term([2, 7], [1, 1], [10, 16], [1, 1], [0, 3], [1, 1])
    # a2 = (x21+x22)y11
    add_term([6, 7], [1, 1], [10], [1], [2], [1])
    # a3 = x11(y12-y22)
    add_term([2], [1], [11, 16], [1, -1], [1, 3], [1, 1])
    # a4 = x22(y21-y11)
    add_term([7], [1], [15, 10], [1, -1], [0, 2], [1, 1])
    # a5 = (x11+x12)y22
    add_term([2, 3], [1, 1], [16], [1], [1, 0], [1, -1])
    # a6 = (x21-x11)(y11+y12)
    add_term([6, 2], [1, -1], [10, 11], [1, 1], [3], [1])
    # a7 = (x12-x22)(y21+y22)
    add_term([3, 7], [1, -1], [15, 16], [1, 1], [0], [1])
    
    # Strassen 2x2x2 part 2 (cols 2,3 of B2)
    # Y2 is B2[:, 2:4] -> indices 12,13,17,18
    # X is A2
    
    # b1 = (x11+x22)(y13+y24)
    add_term([2, 7], [1, 1], [12, 18], [1, 1], [4, 7], [1, 1])
    # b2 = (x21+x22)y13
    add_term([6, 7], [1, 1], [12], [1], [6], [1])
    # b3 = x11(y14-y24)
    add_term([2], [1], [13, 18], [1, -1], [5, 7], [1, 1])
    # b4 = x22(y23-y13)
    add_term([7], [1], [17, 12], [1, -1], [4, 6], [1, 1])
    # b5 = (x11+x12)y24
    add_term([2, 3], [1, 1], [18], [1], [5, 4], [1, -1])
    # b6 = (x21-x11)(y13+y14)
    add_term([6, 2], [1, -1], [12, 13], [1, 1], [7], [1])
    # b7 = (x12-x22)(y23+y24)
    add_term([3, 7], [1, -1], [17, 18], [1, 1], [4], [1])
    
    # Schoolbook 2x2x1 part (col 4 of B2)
    # Y3 is B2[:, 4:5] -> indices 14, 19
    # X is A2
    
    # c1 = x11*y15
    add_term([2], [1], [14], [1], [8], [1])
    # c2 = x12*y25
    add_term([3], [1], [19], [1], [8], [1])
    # c3 = x21*y15
    add_term([6], [1], [14], [1], [9], [1])
    # c4 = x22*y25
    add_term([7], [1], [19], [1], [9], [1])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
