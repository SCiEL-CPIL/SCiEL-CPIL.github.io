import numpy as np
import time

def construct():
    n, m, p = 3, 4, 6
    deadline = time.time() + 110
    
    # Strassen 2x2x2 decomposition factors
    # U_s[i,j], V_s[j,k], W_s[k,i] for s=0..6
    U_strassen = [
        [[1, 0], [0, 1]],  # s=0: m1
        [[0, 0], [1, 1]],  # s=1: m2
        [[1, 0], [0, 0]],  # s=2: m3
        [[0, 0], [0, 1]],  # s=3: m4
        [[1, 1], [0, 0]],  # s=4: m5
        [[-1, 0], [1, 0]], # s=5: m6
        [[0, 1], [-1, 0]], # s=6: m7
    ]
    V_strassen = [
        [[1, 0], [0, 1]],  # s=0
        [[1, 0], [0, 0]],  # s=1
        [[0, 1], [-1, 0]], # s=2
        [[-1, 0], [1, 0]], # s=3
        [[0, 0], [0, 1]],  # s=4
        [[1, 1], [0, 0]],  # s=5
        [[0, 1], [1, 1]],  # s=6
    ]
    W_strassen = [
        [[1, 0], [0, 1]],  # s=0: W[k,i], k=0,1; i=0,1
        [[0, 1], [-1, 0]], # s=1
        [[0, 1], [0, 1]],  # s=2
        [[1, 0], [1, 0]],  # s=3
        [[-1, 0], [1, 0]], # s=4
        [[0, 0], [0, 1]],  # s=5
        [[1, 0], [0, 0]],  # s=6
    ]
    
    # Build the full decomposition
    U = [[0] * 54 for _ in range(n * m)]  # 12 x 54
    V = [[0] * 54 for _ in range(m * p)]  # 24 x 54
    W = [[0] * 54 for _ in range(p * n)]  # 18 x 54
    
    col = 0
    
    # Middle groups: {0,1} and {2,3}
    middle_groups = [(0, 1), (2, 3)]
    
    for j1, j2 in middle_groups:
        # Row groups: {0,1} and {2}
        # Col groups: {0,1}, {2,3}, {4,5}
        
        # For rows {0,1} and each col pair, use Strassen 2x2x2
        row_pairs = [(0, 1)]
        single_rows = [2]
        col_pairs = [(0, 1), (2, 3), (4, 5)]
        
        for i1, i2 in row_pairs:
            for k1, k2 in col_pairs:
                # 2x2x2 sub-block: rows {i1,i2}, middle {j1,j2}, cols {k1,k2}
                for s in range(7):
                    # U[i,j,s] for i in {i1,i2}, j in {j1,j2}
                    for di in range(2):
                        for dj in range(2):
                            i = i1 + di
                            j = j1 + dj
                            U[i * m + j][col] = U_strassen[s][di][dj]
                    # V[j,k,s] for j in {j1,j2}, k in {k1,k2}
                    for dj in range(2):
                        for dk in range(2):
                            j = j1 + dj
                            k = k1 + dk
                            V[j * p + k][col] = V_strassen[s][dj][dk]
                    # W[k,i,s] for k in {k1,k2}, i in {i1,i2}
                    for dk in range(2):
                        for di in range(2):
                            k = k1 + dk
                            i = i1 + di
                            W[k * n + i][col] = W_strassen[s][dk][di]
                    col += 1
        
        # For single row 2 and each col pair, use schoolbook (rank 2)
        for i in single_rows:
            for k1, k2 in col_pairs:
                for j in [j1, j2]:
                    for k in [k1, k2]:
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
    
    assert col == 54, f"Expected 54 columns, got {col}"
    
    # Verify the decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(54):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1) > 1e-9:
                    raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {val}")
    
    return {"U": U, "V": V, "W": W}
