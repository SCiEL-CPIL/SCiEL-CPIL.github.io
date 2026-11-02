import numpy as np
import time

def construct():
    n, m, p = 3, 3, 8
    # Dimensions:
    # U: (n*m, rank) = (9, rank)
    # V: (m*p, rank) = (24, rank)
    # W: (p*n, rank) = (24, rank)
    
    # We aim for rank 68.
    # 28 Strassen terms + 40 Schoolbook terms.
    
    rank = 68
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Helper to add a term
    # u_indices: list of (index, coeff) for U
    # v_indices: list of (index, coeff) for V
    # w_indices: list of (index, coeff) for W
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx, coeff in u_indices:
            U[idx, col] += coeff
        for idx, coeff in v_indices:
            V[idx, col] += coeff
        for idx, coeff in w_indices:
            W[idx, col] += coeff
        col += 1
        
    # 1. Strassen Terms for A[0:2, 0:2] * B[0:2, :]
    # A[0:2, 0:2] indices in U (row = i*m+j):
    # A[0,0] -> 0*3+0 = 0
    # A[0,1] -> 0*3+1 = 1
    # A[1,0] -> 1*3+0 = 3
    # A[1,1] -> 1*3+1 = 4
    
    # B indices in V (row = j*p+k):
    # B[0, k] -> 0*8+k = k
    # B[1, k] -> 1*8+k = 8+k
    
    # C indices in W (row = k*n+i):
    # C[0, k] -> k*3+0 = 3k
    # C[1, k] -> k*3+1 = 3k+1
    
    a00_idx = 0
    a01_idx = 1
    a10_idx = 3
    a11_idx = 4
    
    for c_b in range(4):
        # B block columns: 2*c_b, 2*c_b+1
        b00_idx = 0 * p + 2 * c_b       # B[0, 2c]
        b01_idx = 0 * p + (2 * c_b + 1) # B[0, 2c+1]
        b10_idx = 1 * p + 2 * c_b       # B[1, 2c]
        b11_idx = 1 * p + (2 * c_b + 1) # B[1, 2c+1]
        
        # C block columns: 2*c_b, 2*c_b+1
        c00_idx = (2 * c_b) * n + 0     # C[0, 2c]
        c01_idx = (2 * c_b + 1) * n + 0 # C[0, 2c+1]
        c10_idx = (2 * c_b) * n + 1     # C[1, 2c]
        c11_idx = (2 * c_b + 1) * n + 1 # C[1, 2c+1]
        
        # m1 = (a00 + a11) * (b00 + b11)
        add_term(
            [(a00_idx, 1), (a11_idx, 1)],
            [(b00_idx, 1), (b11_idx, 1)],
            [(c00_idx, 1), (c11_idx, 1)]
        )
        
        # m2 = (a10 + a11) * b00
        add_term(
            [(a10_idx, 1), (a11_idx, 1)],
            [(b00_idx, 1)],
            [(c10_idx, 1), (c11_idx, -1)] # C[1,2c] += m2, C[1,2c+1] -= m2
        )
        
        # m3 = a00 * (b01 - b11)
        add_term(
            [(a00_idx, 1)],
            [(b01_idx, 1), (b11_idx, -1)],
            [(c01_idx, 1), (c11_idx, 1)] # C[0,2c+1] += m3, C[1,2c+1] += m3
        )
        
        # m4 = a11 * (b10 - b00)
        add_term(
            [(a11_idx, 1)],
            [(b10_idx, 1), (b00_idx, -1)],
            [(c00_idx, -1), (c10_idx, 1)] # C[0,2c] -= m4, C[1,2c] += m4
        )
        
        # m5 = (a00 + a01) * b11
        add_term(
            [(a00_idx, 1), (a01_idx, 1)],
            [(b11_idx, 1)],
            [(c00_idx, -1), (c01_idx, 1)] # C[0,2c] -= m5, C[0,2c+1] += m5
        )
        
        # m6 = (a10 - a00) * (b00 + b01)
        add_term(
            [(a10_idx, 1), (a00_idx, -1)],
            [(b00_idx, 1), (b01_idx, 1)],
            [(c11_idx, 1)] # C[1,2c+1] += m6
        )
        
        # m7 = (a01 - a11) * (b10 + b11)
        add_term(
            [(a01_idx, 1), (a11_idx, -1)],
            [(b10_idx, 1), (b11_idx, 1)],
            [(c00_idx, 1), (c01_idx, 1)] # C[0,2c] += m7, C[0,2c+1] += m7
        )
        
    # 2. Schoolbook Terms for the rest
    # The full multiplication is sum_{j=0}^{2} A[i,j] B[j,k]
    # We have already handled j=0,1 for i=0,1 using Strassen.
    # So we need to add:
    # - i=0,1, j=2
    # - i=2, j=0,1
    # - i=2, j=2
    
    # i=0,1, j=2
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
            
    # i=2, j=0,1
    for j in range(2):
        for k in range(8):
            u_idx = 2 * m + j
            v_idx = j * p + k
            w_idx = k * n + 2
            add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
            
    # i=2, j=2
    for k in range(8):
        u_idx = 2 * m + 2
        v_idx = 2 * p + k
        w_idx = k * n + 2
        add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
        
    # Verify rank
    # col should be 68
    assert col == 68, f"Expected 68, got {col}"
    
    # Verification (optional, but good for debugging)
    # Check if the decomposition is exact
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # We expect T[i,j,k] = 1 for all i,j,k
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
