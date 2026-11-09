import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 20
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a column
    def add_col(col_idx, u_indices, v_indices, w_indices):
        for idx in u_indices:
            U[idx][col_idx] += 1
        for idx in v_indices:
            V[idx][col_idx] += 1
        for idx, val in w_indices:
            W[idx][col_idx] += val

    # Strassen for A11 * B11 (2x2 * 2x2)
    # Local A: (0,0), (0,1), (1,0), (1,1) -> Global U: 0,1,3,4
    # Local B: (0,0), (0,1), (1,0), (1,1) -> Global V: 0,1,4,5
    # Local C: (0,0), (0,1), (1,0), (1,1) -> Global W: 0,1,3,4 (k*n+i)
    # W index: k*3 + i.
    # C(0,0) -> W[0]
    # C(0,1) -> W[1]
    # C(1,0) -> W[3]
    # C(1,1) -> W[4]
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # U: 0, 4
    # V: 0, 5
    # W: 0, 4
    add_col(0, [0, 4], [0, 5], [(0, 1), (4, 1)])
    
    # M2: (A10+A11)B00 -> C10, C11 (with signs)
    # C10 is W[3], C11 is W[4]?
    # Local C row 1 col 0 -> Global C row 1 col 0 -> W[0*3+1] = W[3]
    # Local C row 1 col 1 -> Global C row 1 col 1 -> W[1*3+1] = W[4]
    # M2 contributes to C10 (+) and C11 (-) ?
    # C20 = M2 + M4. C21 = M1 - M2 + M3 + M6.
    # So M2: W[3]=1, W[4]=-1.
    # U: 3, 4 (A10, A11)
    # V: 0 (B00)
    add_col(1, [3, 4], [0], [(3, 1), (4, -1)])
    
    # M3: A00(B10-B11) -> C01, C11
    # C01 is W[1], C11 is W[4]
    # M3 contributes to C01 (+) and C11 (+)
    # C01 = M3 + M5. C11 = M3 + M5.
    # So M3: W[1]=1, W[4]=1.
    # U: 0 (A00)
    # V: 1, 5 (B10, B11) with signs? B10 - B11.
    # V[1] = 1, V[5] = -1.
    add_col(2, [0], [1, 5], [(1, 1), (4, 1)])
    # Wait, V signs are in V.
    # So V[1]=1, V[5]=-1.
    # Let's re-verify add_col. It adds 1.
    # I should modify add_col to take values or handle signs in V/W.
    # Let's just set values directly.

    # Reset U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def set_col(col_idx, u_vals, v_vals, w_vals):
        for idx, val in u_vals.items():
            U[idx][col_idx] = val
        for idx, val in v_vals.items():
            V[idx][col_idx] = val
        for idx, val in w_vals.items():
            W[idx][col_idx] = val

    # Strassen 1: A11 * B11
    # M1
    set_col(0, {0:1, 4:1}, {0:1, 5:1}, {0:1, 4:1})
    # M2
    set_col(1, {3:1, 4:1}, {0:1}, {3:1, 4:-1})
    # M3
    set_col(2, {0:1}, {1:1, 5:-1}, {1:1, 4:1})
    # M4: A11(B00-B01) -> C10, C11?
    # C10 = M1+M4-M5+M7. C11 = M3+M5.
    # M4 contributes to C10 (+) and C20 (+).
    # C10 is W[3]. C20 is W[3]? No.
    # Local C row 1 col 0 -> W[3].
    # Local C row 2 col 0 -> W[6]?
    # Global C row 2 col 0 -> W[0*3+2] = W[2].
    # Wait. W index is k*n + i.
    # C(i,k).
    # Local C(1,0) -> Global C(1,0) -> W[0*3+1] = W[3].
    # Local C(2,0) -> Global C(2,0) -> W[0*3+2] = W[2].
    # M4 contributes to C10 (+) and C20 (+).
    # So W[3]=1, W[2]=1.
    # U: 4 (A11)
    # V: 0, 1 (B00, B01) with signs 1, -1.
    set_col(3, {4:1}, {0:1, 1:-1}, {3:1, 2:1})
    # M5: (A00+A01)B11 -> C01, C11
    # C01 = M3+M5. C11 = M3+M5.
    # M5 contributes to C01 (-) and C11 (+)?
    # C01 = M3+M5. C11 = M3+M5.
    # Wait, standard Strassen:
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    # So M5 contributes to C01 (+) and C11 (-)? No.
    # C01 = M3+M5.
    # C11 = M1-M2+M3+M6. M5 is not in C11.
    # M5 is in C01 and C00?
    # C00 = M1+M4-M5+M7.
    # So M5 contributes to C00 (-) and C01 (+).
    # C00 is W[0]. C01 is W[1].
    # U: 0, 1 (A00, A01)
    # V: 5 (B11)
    set_col(4, {0:1, 1:1}, {5:1}, {0:-1, 1:1})
    # M6: (A10-A00)(B00+B01) -> C11
    # C11 = M1-M2+M3+M6.
    # M6 contributes to C11 (+).
    # C11 is W[4].
    # U: 3, 0 (A10, A00) with signs 1, -1.
    # V: 0, 1 (B00, B01)
    set_col(5, {3:1, 0:-1}, {0:1, 1:1}, {4:1})
    # M7: (A01-A11)(B10+B11) -> C00
    # C00 = M1+M4-M5+M7.
    # M7 contributes to C00 (+).
    # C00 is W[0].
    # U: 1, 4 (A01, A11) with signs 1, -1.
    # V: 1, 5 (B10, B11)
    set_col(6, {1:1, 4:-1}, {1:1, 5:1}, {0:1})

    # Strassen 2: A11 * B12 (2x2 * 2x2)
    # A11: U 0,1,3,4
    # B12: V indices for rows 0,1 and cols 2,3.
    # B(0,2)=V[2], B(0,3)=V[3], B(1,2)=V[6], B(1,3)=V[7].
    # C12: W indices for rows 0,1 and cols 2,3.
    # C(0,2)=W[2], C(0,3)=W[3], C(1,2)=W[5], C(1,3)=W[6].
    
    # M1
    set_col(7, {0:1, 4:1}, {2:1, 7:1}, {2:1, 6:1})
    # M2
    set_col(8, {3:1, 4:1}, {2:1}, {5:1, 6:-1})
    # M3
    set_col(9, {0:1}, {3:1, 7:-1}, {3:1, 6:1})
    # M4
    set_col(10, {4:1}, {2:1, 3:-1}, {5:1, 2:1})
    # M5
    set_col(11, {0:1, 1:1}, {7:1}, {2:-1, 3:1})
    # M6
    set_col(12, {3:1, 0:-1}, {2:1, 3:1}, {6:1})
    # M7
    set_col(13, {1:1, 4:-1}, {3:1, 7:1}, {2:1})

    # Rank 1 terms
    # A12 (2x1) * B21 (1x2) -> C11 (2x2)
    # A12: U 0, 3 (rows 0,1 col 2)
    # B21: V 8, 9 (row 2 cols 0,1)
    # C11: W 0, 1, 3, 4
    # Tensor is all ones.
    # U: 0,3 -> 1
    # V: 8,9 -> 1
    # W: 0,1,3,4 -> 1
    set_col(14, {0:1, 3:1}, {8:1, 9:1}, {0:1, 1:1, 3:1, 4:1})
    
    # A12 (2x1) * B22 (1x2) -> C12 (2x2)
    # A12: U 0,3
    # B22: V 10,11 (row 2 cols 2,3)
    # C12: W 2,3,5,6
    set_col(15, {0:1, 3:1}, {10:1, 11:1}, {2:1, 3:1, 5:1, 6:1})
    
    # A21 (1x2) * B11 (2x2) -> C21 (1x2)
    # A21: U 6,7 (row 2 cols 0,1)
    # B11: V 0,1,4,5
    # C21: W 2,3 (row 2 cols 0,1)
    set_col(16, {6:1, 7:1}, {0:1, 1:1, 4:1, 5:1}, {2:1, 3:1})
    
    # A21 (1x2) * B12 (2x2) -> C22 (1x2)
    # A21: U 6,7
    # B12: V 2,3,6,7
    # C22: W 5,6
    set_col(17, {6:1, 7:1}, {2:1, 3:1, 6:1, 7:1}, {5:1, 6:1})
    
    # A22 (1x1) * B21 (1x2) -> C21 (1x2)
    # A22: U 8
    # B21: V 8,9
    # C21: W 2,3
    set_col(18, {8:1}, {8:1, 9:1}, {2:1, 3:1})
    
    # A22 (1x1) * B22 (1x2) -> C22 (1x2)
    # A22: U 8
    # B22: V 10,11
    # C22: W 5,6
    set_col(19, {8:1}, {10:1, 11:1}, {5:1, 6:1})

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
