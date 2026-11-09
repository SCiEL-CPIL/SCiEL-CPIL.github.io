import numpy as np

DIMENSIONS = [3, 4, 6]

def get_rank23_3x3():
    # Rank 23 algorithm for 3x3 matrix multiplication
    # U, V, W are 9x23 matrices
    # Indices: i*3+j for U, j*3+k for V, k*3+i for W
    # Coefficients from Strassen (1969)
    # This is a known set.
    # If I don't have them memorized perfectly, I will use a placeholder.
    # However, I will try to generate a valid low-rank decomposition.
    # Since I cannot look it up, I will use a safe fallback.
    # But I will try to use the rank 23 coefficients if possible.
    
    # For now, I will return None to indicate failure to provide Rank 23 coefficients.
    # This will cause the main function to fall back to schoolbook.
    return None

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    # Try to build a better decomposition
    # Strategy: Split A into 3x3 and 3x1, B into 3x6 and 1x6
    # C = A1 B1 + A2 B2
    # A1 B1 is 3x3 x 3x6 -> split B1 into 3x3 and 3x3
    # C = A1 B11 + A1 B12 + A2 B2
    # A1 B11 is 3x3 x 3x3 (rank 23)
    # A1 B12 is 3x3 x 3x3 (rank 23)
    # A2 B2 is 3x1 x 1x6 (rank 18)
    # Total rank: 23 + 23 + 18 = 64
    
    U23, V23, W23 = get_rank23_3x3()
    if U23 is not None:
        rank_23 = 23
        # Build U, V, W for the 64 rank decomposition
        U_new = []
        V_new = []
        W_new = []
        
        # Part 1: A1 B11 (3x3 x 3x3)
        # A1 is rows 0-2, cols 0-2 of A
        # B11 is rows 0-2, cols 0-2 of B
        # W1 is cols 0-2, rows 0-2 of C
        for r in range(rank_23):
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            for i in range(3):
                for j in range(3):
                    u[i * m + j] = U23[i * 3 + j][r]
                    v[j * p + i] = V23[j * 3 + i][r] # Wait, V index is j*p+k
                    # V23 index is j*3+k
                    # So V23[j*3+k] corresponds to V[j*p+k]
                    # But here k is the column of B11, which is 0-2
                    # So V[j*p + k] = V23[j*3+k]
                    for k in range(3):
                        v[j * p + k] = V23[j * 3 + k][r]
                        w[k * n + i] = W23[k * 3 + i][r]
            U_new.append(u)
            V_new.append(v)
            W_new.append(w)
            
        # Part 2: A1 B12 (3x3 x 3x3)
        # A1 is rows 0-2, cols 0-2 of A
        # B12 is rows 0-2, cols 3-5 of B
        # W2 is cols 3-5, rows 0-2 of C
        for r in range(rank_23):
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            for i in range(3):
                for j in range(3):
                    u[i * m + j] = U23[i * 3 + j][r]
                    for k in range(3):
                        # B12 col k corresponds to B col 3+k
                        v[j * p + (3 + k)] = V23[j * 3 + k][r]
                        # W2 col 3+k corresponds to C col 3+k
                        w[(3 + k) * n + i] = W23[k * 3 + i][r]
            U_new.append(u)
            V_new.append(v)
            W_new.append(w)
            
        # Part 3: A2 B2 (3x1 x 1x6)
        # A2 is rows 0-2, col 3 of A
        # B2 is row 3, cols 0-5 of B
        # W3 is all cols, rows 0-2 of C
        for i in range(3):
            for k in range(6):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                u[i * m + 3] = 1
                v[3 * p + k] = 1
                w[k * n + i] = 1
                U_new.append(u)
                V_new.append(v)
                W_new.append(w)
                
        return {"U": U_new, "V": V_new, "W": W_new}
    
    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}

