import numpy as np

DIMENSIONS = [2, 2, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 18
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to add a rank-1 term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        U[u_idx, col] += u_val
        V[v_idx, col] += v_val
        W[w_idx, col] += w_val

    # Block 1: Cols 0, 1 (Rank 7)
    # Strassen indices for 2x2x2
    # Local k=0,1 -> Global k=0,1
    # V index = j * p + k
    # W index = k * n + i
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    add_term([0, 3], [1, 1], [0, 6], [1, 1], [0, 3], [1, 1], 0)
    # M2: (A10+A11)B00 -> C10
    add_term([2, 3], [1, 1], [0], [1], [1], [1], 1)
    # M3: A00(B01-B11) -> C01
    add_term([0], [1], [1, 6], [1, -1], [2], [1], 2)
    # M4: A11(B10-B00) -> C00, C10
    add_term([3], [1], [5, 0], [1, -1], [0, 1], [1, 1], 3)
    # M5: (A00+A01)B11 -> C00(-1), C01
    add_term([0, 1], [1, 1], [6], [1], [0, 2], [-1, 1], 4)
    # M6: (A10-A00)(B00+B01) -> C11
    add_term([2, 0], [1, -1], [0, 1], [1, 1], [3], [1], 5)
    # M7: (A01-A11)(B10+B11) -> C00, C11
    add_term([1, 3], [1, -1], [5, 6], [1, 1], [0, 3], [1, 1], 6)

    # Block 2: Cols 2, 3 (Rank 7)
    # Local k=0,1 -> Global k=2,3
    # V index = j * p + k
    # W index = k * n + i
    
    # M1: (A00+A11)(B02+B13) -> C02, C13
    add_term([0, 3], [1, 1], [2, 8], [1, 1], [4, 7], [1, 1], 7)
    # M2: (A10+A11)B02 -> C12
    add_term([2, 3], [1, 1], [2], [1], [5], [1], 8)
    # M3: A00(B03-B13) -> C03
    add_term([0], [1], [3, 8], [1, -1], [6], [1], 9)
    # M4: A11(B12-B02) -> C02, C12
    add_term([3], [1], [7, 2], [1, -1], [4, 5], [1, 1], 10)
    # M5: (A00+A01)B13 -> C02(-1), C03
    add_term([0, 1], [1, 1], [8], [1], [4, 6], [-1, 1], 11)
    # M6: (A10-A00)(B02+B03) -> C13
    add_term([2, 0], [1, -1], [2, 3], [1, 1], [7], [1], 12)
    # M7: (A01-A11)(B12+B13) -> C02, C13
    add_term([1, 3], [1, -1], [7, 8], [1, 1], [4, 7], [1, 1], 13)

    # Block 3: Col 4 (Rank 4)
    # C04 = A00*B04 + A01*B14
    # C14 = A10*B04 + A11*B14
    # We can use 4 rank-1 terms directly.
    
    # Term 1: A00 * B04 -> C04
    add_term([0], [1], [4], [1], [8], [1], 14)
    # Term 2: A01 * B14 -> C04
    add_term([1], [1], [9], [1], [8], [1], 15)
    # Term 3: A10 * B04 -> C14
    add_term([2], [1], [4], [1], [9], [1], 16)
    # Term 4: A11 * B14 -> C14
    add_term([3], [1], [9], [1], [9], [1], 17)

    # Convert to lists if required, but dict of lists is fine.
    # The task says "factor matrices of shapes ... with integer/half-integer entries"
    # Returning numpy arrays is usually accepted if they are converted or if the evaluator handles them.
    # The seed program returns lists of lists. I will return lists of lists to be safe.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
