"""Tensor decomposition for 2x2 * 2x13 matrix multiplication."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s limit
    
    n, m, p = 2, 2, 13
    
    # Strategy: 6 blocks of 2 columns (12 cols) + 1 block of 1 column
    # For each 2-column block, use rank-7 (Strassen) decomposition
    # For the 1-column block, use rank-4 (schoolbook)
    # Total: 6*7 + 4 = 46
    
    # Build the full tensor T of shape (n*m, m*p, p*n) = (4, 26, 26)
    # T[a,b,c] = 1 if a=i*m+k, b=k'*p+j, c=j'*n+i and k=k', j=j', i=i'
    # i.e., T[i*m+k, k*p+j, j*n+i] = 1
    
    # Strassen decomposition for a 2x2x2 sub-block:
    # Local indices:
    #   A: idx = i*2+k, i in {0,1}, k in {0,1} -> 4 entries
    #   B: idx = k*2+j, k in {0,1}, j in {0,1} -> 4 entries  
    #   C: idx = j*2+i, j in {0,1}, i in {0,1} -> 4 entries
    #
    # The 7 Strassen products (standard formulation):
    # P1 = (a00+a11)*(b00+b11)
    # P2 = (a10+a11)*b00
    # P3 = a00*(b10-b11)
    # P4 = a11*(b01-b00)
    # P5 = (a00+a01)*b11
    # P6 = (a10-a00)*(b00+b10)
    # P7 = (a01-a11)*(b01+b11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P1 - P2 + P6
    # C10 = P2 + P4 - P3 + P5
    # C11 = P1 - P4 + P2 - P6
    #
    # For the tensor decomposition, each product Pt is ONE rank-1 term:
    #   U[:,t] = coefficients for A
    #   V[:,t] = coefficients for B
    #   W[:,t] = coefficients for C
    #
    # Let me define the 7 terms:
    # A local indices: 0=a00, 1=a01, 2=a10, 3=a11
    # B local indices: 0=b00, 1=b01, 2=b10, 3=b11
    # C local indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # P1: A=[1,0,0,1], B=[1,0,0,1], C=[1,0,0,1]
    # P2: A=[0,0,1,1], B=[1,0,0,0], C=[0,-1,1,1]
    # P3: A=[1,0,0,0], B=[0,0,1,-1], C=[0,1,-1,0]
    # P4: A=[0,0,0,1], B=[0,1,-1,0], C=[1,0,1,-1]
    # P5: A=[1,1,0,0], B=[0,0,0,1], C=[0,0,-1,0]
    # P6: A=[-1,0,1,0], B=[1,0,1,0], C=[0,1,0,-1]
    # P7: A=[0,1,0,-1], B=[0,0,0,1], C=[1,0,0,0]
    
    # Let me verify by computing the tensor from these 7 terms:
    # For each term t, contribution to T[a,b,c] = U[a,t]*V[b,t]*W[c,t]
    # Sum over all t should give the matmul tensor.
    
    U_local = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # a00
        [0, 0, 0, 0, 1,  0, 1],  # a01
        [0, 1, 0, 0, 0,  1, 0],  # a10
        [1, 1, 0, 1, 0,  0,-1],  # a11
    ], dtype=float)
    
    V_local = np.array([
        [1, 1, 0, 0, 0, 1, 0],  # b00
        [0, 0, 0, 1, 0, 0, 0],  # b01
        [0, 0, 1,-1, 0, 1, 0],  # b10
        [1, 0,-1, 0, 1, 0, 1],  # b11
    ], dtype=float)
    
    W_local = np.array([
        [1, 0, 0, 1, 0, 0, 1],  # C00
        [0,-1, 1, 0, 0, 1, 0],  # C01
        [0, 1,-1, 1,-1, 0, 0],  # C10
        [1, 1, 0,-1, 0,-1, 0],  # C11
    ], dtype=float)
    
    # Verify: compute the tensor from these 7 terms
    T_computed = np.zeros((4, 4, 4), dtype=float)
    for t in range(7):
        T_computed += np.einsum('a,b,c->abc', U_local[:,t], V_local[:,t], W_local[:,t])
    
    # Expected tensor: T[i*2+k, k*2+j, j*2+i] = 1
    T_expected = np.zeros((4, 4, 4), dtype=float)
    for i in range(2):
        for k in range(2):
            for j in range(2):
                T_expected[i*2+k, k*2+j, j*2+i] = 1
    
    # Check if they match
    if not np.allclose(T_computed, T_expected, atol=1e-10):
        # Fallback: use schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k][col] = 1
                    V[k * p + j][col] = 1
                    W[j * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Build the full decomposition using Strassen blocks
    # 6 blocks of 2 columns each (columns 0-11), then 1 block of 1 column (column 12)
    
    total_rank = 6 * 7 + 4  # 46
    U_full = np.zeros((n * m, total_rank), dtype=float)
    V_full = np.zeros((m * p, total_rank), dtype=float)
    W_full = np.zeros((p * n, total_rank), dtype=float)
    
    col_offset = 0
    
    # 6 Strassen blocks for columns 0-11
    for block in range(6):
        j0 = block * 2  # first column of this block
        j1 = block * 2 + 1  # second column of this block
        
        for t in range(7):
            # U: local index i*2+k maps to global i*m+k = i*2+k (same)
            for i in range(2):
                for k in range(2):
                    U_full[i*m+k, col_offset + t] = U_local[i*2+k, t]
            
            # V: local index k*2+j_local maps to global k*p + (j0 + j_local)
            for k in range(2):
                for jl in range(2):
                    j_global = j0 + jl
                    V_full[k*p + j_global, col_offset + t] = V_local[k*2+jl, t]
            
            # W: local index j_local*2+i maps to global (j0+j_local)*n + i
            for jl in range(2):
                for i in range(2):
                    j_global = j0 + jl
                    W_full[j_global*n + i, col_offset + t] = W_local[jl*2+i, t]
        
        col_offset += 7
    
    # 1 schoolbook block for column 12
    j = 12
    for i in range(n):
        for k in range(m):
            U_full[i*m+k, col_offset] = 1
            V_full[k*p+j, col_offset] = 1
            W_full[j*n+i, col_offset] = 1
            col_offset += 1
    
    # Verify the full decomposition
    T_full_computed = np.zeros((n*m, m*p, p*n), dtype=float)
    for t in range(total_rank):
        T_full_computed += np.einsum('a,b,c->abc', U_full[:,t], V_full[:,t], W_full[:,t])
    
    T_full_expected = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T_full_expected[i*m+k, k*p+j, j*n+i] = 1
    
    if not np.allclose(T_full_computed, T_full_expected, atol=1e-10):
        # Fallback: use schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k][col] = 1
                    V[k * p + j][col] = 1
                    W[j * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to lists of lists with integer/half-integer entries
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    # Round to nearest 0.5 to handle floating point issues
    def round_to_half(x):
        return round(x * 2) / 2
    
    U_list = [[round_to_half(v) for v in row] for row in U_list]
    V_list = [[round_to_half(v) for v in row] for row in V_list]
    W_list = [[round_to_half(v) for v in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
