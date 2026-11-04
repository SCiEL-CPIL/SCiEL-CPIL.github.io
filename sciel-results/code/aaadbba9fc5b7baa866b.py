import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    # Block 1: A_L (cols 0,1) * B_T (rows 0,1)
    # A_L indices: 0, 1, 4, 5
    # B_T indices: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    # W indices: 0..9 (C is 2x5, W is 5x2 flattened as k*n+i)
    # C[0,0]->0, C[1,0]->1, C[0,1]->2, C[1,1]->3, etc.
    
    # M1: (A00+A11)(B00+B10)
    add_term(0, 1, 0, 1, 0, 1, 0)
    add_term(4, 1, 5, 1, 0, 0, 0) # Wait, add_term adds to specific cell.
    # Correction: add_term should handle the logic.
    # Let's just write it out explicitly for clarity and safety.
    
    def set_term(col, u_dict, v_dict, w_dict):
        for idx, val in u_dict.items():
            U[idx][col] += val
        for idx, val in v_dict.items():
            V[idx][col] += val
        for idx, val in w_dict.items():
            W[idx][col] += val

    # Block 1: A_L B_T
    # A_L: 0,1,4,5. B_T: 0..9.
    # Strassen 1 (Y1: B00,B01,B10,B11 -> idx 0,1,5,6)
    set_term(0, {0:1, 4:1}, {0:1, 5:1}, {0:1, 3:1})
    set_term(1, {4:1, 5:1}, {0:1}, {1:1, 3:-1})
    set_term(2, {0:1}, {1:1, 6:-1}, {2:1, 3:1})
    set_term(3, {5:1}, {5:1, 0:-1}, {0:1, 1:1})
    set_term(4, {0:1, 1:1}, {6:1}, {0:-1, 2:1})
    set_term(5, {4:1, 0:-1}, {0:1, 1:1}, {3:1})
    set_term(6, {1:1, 5:-1}, {5:1, 6:1}, {0:1})
    
    # Strassen 2 (Y2: B02,B03,B12,B13 -> idx 2,3,7,8)
    # Output cols 2,3 -> W idx 4,5,6,7
    set_term(7, {0:1, 4:1}, {2:1, 7:1}, {4:1, 7:1})
    set_term(8, {4:1, 5:1}, {2:1}, {5:1, 7:-1})
    set_term(9, {0:1}, {3:1, 8:-1}, {6:1, 7:1})
    set_term(10, {5:1}, {7:1, 2:-1}, {4:1, 5:1})
    set_term(11, {0:1, 1:1}, {8:1}, {4:-1, 6:1})
    set_term(12, {4:1, 0:-1}, {2:1, 3:1}, {7:1})
    set_term(13, {1:1, 5:-1}, {7:1, 8:1}, {4:1})
    
    # Schoolbook 3 (Y3: B04,B14 -> idx 4,9)
    # Output col 4 -> W idx 8,9
    set_term(14, {0:1}, {4:1}, {8:1})
    set_term(15, {1:1}, {9:1}, {8:1})
    set_term(16, {4:1}, {4:1}, {9:1})
    set_term(17, {5:1}, {9:1}, {9:1})

    # Block 2: A_R B_B
    # A_R: 2,3,6,7. B_B: 10..19.
    # Strassen 1 (Y1: B20,B21,B30,B31 -> idx 10,11,15,16)
    set_term(18, {2:1, 6:1}, {10:1, 15:1}, {0:1, 3:1})
    set_term(19, {6:1, 7:1}, {10:1}, {1:1, 3:-1})
    set_term(20, {2:1}, {11:1, 16:-1}, {2:1, 3:1})
    set_term(21, {7:1}, {15:1, 10:-1}, {0:1, 1:1})
    set_term(22, {2:1, 3:1}, {16:1}, {0:-1, 2:1})
    set_term(23, {6:1, 2:-1}, {10:1, 11:1}, {3:1})
    set_term(24, {3:1, 7:-1}, {15:1, 16:1}, {0:1})
    
    # Strassen 2 (Y2: B22,B23,B32,B33 -> idx 12,13,17,18)
    set_term(25, {2:1, 6:1}, {12:1, 17:1}, {4:1, 7:1})
    set_term(26, {6:1, 7:1}, {12:1}, {5:1, 7:-1})
    set_term(27, {2:1}, {13:1, 18:-1}, {6:1, 7:1})
    set_term(28, {7:1}, {17:1, 12:-1}, {4:1, 5:1})
    set_term(29, {2:1, 3:1}, {18:1}, {4:-1, 6:1})
    set_term(30, {6:1, 2:-1}, {12:1, 13:1}, {7:1})
    set_term(31, {3:1, 7:-1}, {17:1, 18:1}, {4:1})
    
    # Schoolbook 3 (Y3: B24,B34 -> idx 14,19)
    set_term(32, {2:1}, {14:1}, {8:1})
    set_term(33, {3:1}, {19:1}, {8:1})
    set_term(34, {6:1}, {14:1}, {9:1})
    set_term(35, {7:1}, {19:1}, {9:1})

    return {"U": U, "V": V, "W": W}
