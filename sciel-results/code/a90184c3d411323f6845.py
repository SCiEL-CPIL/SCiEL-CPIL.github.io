import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=4
    # U shape (n*m, rank) = (6, rank)
    # V shape (m*p, rank) = (12, rank)
    # W shape (p*n, rank) = (8, rank)
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    # A is 2x2, B is 2x2
    # A indices in U: a00, a01, a10, a11
    # B indices in V: b00, b01, b10, b11
    # C indices in W: c00, c01, c10, c11
    def add_strassen_block(a_idx, b_idx, c_idx):
        nonlocal col
        # M1 = (a00+a11)(b00+b11)
        U[a_idx[0]][col] += 1
        U[a_idx[1]][col] += 1
        V[b_idx[0]][col] += 1
        V[b_idx[1]][col] += 1
        # C00 += M1, C11 += M1
        W[c_idx[0]][col] += 1
        W[c_idx[1]][col] += 1
        col += 1
        
        # M2 = (a10+a11)b00
        U[a_idx[2]][col] += 1
        U[a_idx[1]][col] += 1
        V[b_idx[0]][col] += 1
        # C10 += M2, C11 -= M2
        W[c_idx[2]][col] += 1
        W[c_idx[1]][col] -= 1
        col += 1
        
        # M3 = a00(b10-b11)
        U[a_idx[0]][col] += 1
        V[b_idx[2]][col] += 1
        V[b_idx[3]][col] -= 1
        # C01 += M3, C11 += M3
        W[c_idx[3]][col] += 1
        W[c_idx[1]][col] += 1
        col += 1
        
        # M4 = a11(b00-b01)
        U[a_idx[1]][col] += 1
        V[b_idx[0]][col] += 1
        V[b_idx[1]][col] -= 1
        # C00 += M4, C10 += M4
        W[c_idx[0]][col] += 1
        W[c_idx[2]][col] += 1
        col += 1
        
        # M5 = (a00+a01)b11
        U[a_idx[0]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[3]][col] += 1
        # C00 -= M5, C01 += M5
        W[c_idx[0]][col] -= 1
        W[c_idx[3]][col] += 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[a_idx[2]][col] += 1
        U[a_idx[0]][col] -= 1
        V[b_idx[0]][col] += 1
        V[b_idx[1]][col] += 1
        # C11 += M6
        W[c_idx[1]][col] += 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[a_idx[3]][col] += 1
        U[a_idx[1]][col] -= 1
        V[b_idx[2]][col] += 1
        V[b_idx[3]][col] += 1
        # C00 += M7
        W[c_idx[0]][col] += 1
        col += 1

    # Part 1: A1 (2x2) * B1 (2x4)
    # A1 indices: a00(0), a01(1), a10(3), a11(4)
    a1_idx = [0, 1, 3, 4]
    
    # B1a (cols 0,1): b00(0), b01(1), b10(4), b11(5)
    b1a_idx = [0, 1, 4, 5]
    # C indices for cols 0,1: c00(0), c01(1), c10(2), c11(3)
    c1a_idx = [0, 1, 2, 3]
    
    add_strassen_block(a1_idx, b1a_idx, c1a_idx)
    
    # B1b (cols 2,3): b02(2), b03(3), b12(6), b13(7)
    b1b_idx = [2, 3, 6, 7]
    # C indices for cols 2,3: c02(4), c03(5), c12(6), c13(7)
    c1b_idx = [4, 5, 6, 7]
    
    add_strassen_block(a1_idx, b1b_idx, c1b_idx)
    
    # Part 2: A2 (2x1) * B2 (1x4)
    # A2 indices: a02(2), a12(5)
    # B2 indices: b20(8), b21(9), b22(10), b23(11)
    # C indices: all
    
    # Term 1: a02 * b20 -> c00
    U[2][col] = 1
    V[8][col] = 1
    W[0][col] = 1
    col += 1
    
    # Term 2: a12 * b20 -> c10
    U[5][col] = 1
    V[8][col] = 1
    W[2][col] = 1
    col += 1
    
    # Term 3: a02 * b21 -> c01
    U[2][col] = 1
    V[9][col] = 1
    W[1][col] = 1
    col += 1
    
    # Term 4: a12 * b21 -> c11
    U[5][col] = 1
    V[9][col] = 1
    W[3][col] = 1
    col += 1
    
    # Term 5: a02 * b22 -> c02
    U[2][col] = 1
    V[10][col] = 1
    W[4][col] = 1
    col += 1
    
    # Term 6: a12 * b22 -> c12
    U[5][col] = 1
    V[10][col] = 1
    W[6][col] = 1
    col += 1
    
    # Term 7: a02 * b23 -> c03
    U[2][col] = 1
    V[11][col] = 1
    W[5][col] = 1
    col += 1
    
    # Term 8: a12 * b23 -> c13
    U[5][col] = 1
    V[11][col] = 1
    W[7][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Simple verification
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    n, m, p = DIMENSIONS
    # Random A, B
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    
    # Flatten
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    # Compute C_flat using decomposition
    C_flat = np.zeros(p * n)
    for r in range(U.shape[1]):
        prod = np.dot(U[:, r], A_flat) * np.dot(V[:, r], B_flat)
        C_flat += prod * W[:, r]
    
    # Reshape C_flat to (p, n) and transpose to (n, p)
    C_decomp = C_flat.reshape(p, n).T
    
    print("Max diff:", np.max(np.abs(C - C_decomp)))
    print("Rank:", U.shape[1])
