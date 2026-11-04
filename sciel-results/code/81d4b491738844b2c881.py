import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# Strassen tensor for 2x2x2
# U, V, W are 4x7 matrices
# Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
# T = sum_c U[i,j,c] V[j,k,c] W[k,i,c]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Strassen coefficients for 2x2x2
    # Standard Strassen decomposition
    # U (A factors), V (B factors), W (C factors)
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # Strassen products:
    # p1 = (a00 + a11) * (b00 + b11)
    # p2 = (a10 + a11) * b00
    # p3 = a00 * (b01 - b11)
    # p4 = a11 * (b10 - b00)
    # p5 = (a00 + a01) * b11
    # p6 = (a10 - a00) * (b00 + b10)
    # p7 = (a01 - a11) * (b01 + b11)
    
    # U matrix (A coefficients)
    # Rows: a00, a01, a10, a11
    # Cols: p1..p7
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0], # a00
        [0, 0, 0, 0, 1, 0, 1],  # a01
        [0, 1, 0, 0, 0, 1, 0],  # a10
        [1, 1, 0, 1, 0, 0, -1]  # a11
    ]
    
    # V matrix (B coefficients)
    # Rows: b00, b01, b10, b11
    # Cols: p1..p7
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0], # b00
        [0, 0, 1, 0, 0, 0, 1],  # b01
        [0, 0, 0, 1, 0, 1, 0],  # b10
        [1, 0, -1, 0, 1, 0, 1]  # b11
    ]
    
    # W matrix (C coefficients)
    # Rows: c00, c01, c10, c11
    # Cols: p1..p7
    # c00 = p1 - p4 + p5 + p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p6
    W_strassen = [
        [1, 0, 0, -1, 1, 0, 1], # c00
        [0, 0, 1, 0, 1, 0, 0],  # c01
        [0, 1, 0, 1, 0, 0, 0],  # c10
        [1, -1, 1, 0, 0, 1, 0]  # c11
    ]
    
    # Convert to numpy arrays for Kronecker product
    U_s = np.array(U_strassen)
    V_s = np.array(V_strassen)
    W_s = np.array(W_strassen)
    
    # Kronecker product for 4x4x4
    # U_444 = U_s (x) U_s
    # Size: (4*4) x (7*7) = 16 x 49
    U_444 = np.kron(U_s, U_s)
    V_444 = np.kron(V_s, V_s)
    W_444 = np.kron(W_s, W_s)
    
    # We need to construct U, V, W for 4x4x7
    # Rank = 49 (for first 4 cols) + 48 (schoolbook for last 3 cols) = 97
    rank = 97
    U = np.zeros((16, rank))
    V = np.zeros((28, rank))
    W = np.zeros((28, rank))
    
    # Part 1: First 4 columns (k=0..3)
    # U_444 has shape (16, 49)
    # U[i*4+j, c] = U_444[i*4+j, c]
    U[:, 0:49] = U_444
    
    # V_444 has shape (16, 49)
    # V_444[j*4+k', c] where k' in 0..3
    # In full V, index is j*7 + k
    # So V[j*7 + k', c] = V_444[j*4 + k', c] for k' in 0..3
    for j in range(4):
        for k_prime in range(4):
            V[j*7 + k_prime, 0:49] = V_444[j*4 + k_prime, 0:49]
            
    # W_444 has shape (16, 49)
    # W_444[k'*4 + i, c] where k' in 0..3
    # In full W, index is k*4 + i
    # So W[k'*4 + i, c] = W_444[k'*4 + i, c] for k' in 0..3
    W[:, 0:49] = W_444
    
    # Part 2: Last 3 columns (k=4..6)
    # Schoolbook for 4x4x3
    # Rank 48
    col_offset = 49
    for i in range(4):
        for j in range(4):
            for k_local in range(3):
                k = 4 + k_local
                c = col_offset + (i*4 + j)*3 + k_local
                # U[i*4+j, c] = 1
                U[i*4 + j, c] = 1
                # V[j*7+k, c] = 1
                V[j*7 + k, c] = 1
                # W[k*4+i, c] = 1
                W[k*4 + i, c] = 1
                
    # Convert to lists of lists of integers (or fractions if needed, but here they are ints)
    U_list = [[int(x) for x in row] for row in U]
    V_list = [[int(x) for x in row] for row in V]
    W_list = [[int(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
