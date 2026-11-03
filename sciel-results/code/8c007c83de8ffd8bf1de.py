import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Part 1: 4x4x4 using Strassen (rank 49)
    # Part 2: 4x4x5 using schoolbook (rank 80)
    # Total rank: 129
    
    rank = 49 + 80
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2 patterns
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # T1: A=(a+d), B=(e+h)
    # T2: A=(c+d), B=e
    # T3: A=a, B=(f-h)
    # T4: A=d, B=(g-e)
    # T5: A=(a+b), B=h
    # T6: A=(c-a), B=(e+f)
    # T7: A=(b-d), B=(g+h)
    
    # C[0,0] = T1 + T4 - T5 + T7
    # C[0,1] = T3 + T5
    # C[1,0] = T2 + T4
    # C[1,1] = T1 - T2 + T3 + T6
    
    # A patterns for each T (2x2 matrix of coefficients for a,b,c,d)
    A_pat = [
        [[1,0],[0,1]],  # T1
        [[0,0],[1,1]],  # T2
        [[1,0],[0,0]],  # T3
        [[0,0],[0,1]],  # T4
        [[1,1],[0,0]],  # T5
        [[-1,0],[1,0]], # T6
        [[0,1],[0,-1]], # T7
    ]
    
    # B patterns for each T (2x2 matrix of coefficients for e,f,g,h)
    B_pat = [
        [[1,0],[0,1]],  # T1
        [[1,0],[0,0]],  # T2
        [[0,1],[0,-1]], # T3
        [[0,-1],[1,0]], # T4
        [[0,0],[0,1]],  # T5
        [[1,1],[0,0]],  # T6
        [[0,0],[1,1]],  # T7
    ]
    
    # C patterns for each T (2x2 matrix of coefficients for C[0,0],C[0,1],C[1,0],C[1,1])
    C_pat = [
        [[1,0],[0,1]],  # T1
        [[0,0],[1,0]],  # T2
        [[0,1],[0,1]],  # T3
        [[1,0],[1,0]],  # T4
        [[-1,1],[0,0]], # T5
        [[0,0],[0,1]],  # T6
        [[1,0],[0,0]],  # T7
    ]
    
    # Block-level Strassen for 4x4x4:
    # S1: A=A11+A22, B=B11+B22, C: C11+=S1, C22+=S1
    # S2: A=A21+A22, B=B11, C: C21+=S2
    # S3: A=A11, B=B12-B22, C: C12+=S3, C22+=S3
    # S4: A=A22, B=B21-B11, C: C11+=S4, C21+=S4
    # S5: A=A11+A12, B=B22, C: C11-=S5, C12+=S5
    # S6: A=A21-A11, B=B11+B12, C: C22+=S6
    # S7: A=A12-A22, B=B21+B22, C: C11+=S7
    
    # Block indices: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
    # B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1)
    # C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1)
    
    # For each block-level factor, define:
    # A_blocks: list of (block_row, block_col, coeff)
    # B_blocks: list of (block_row, block_col, coeff)
    # C_blocks: list of (block_row, block_col, coeff)
    
    block_factors = [
        # S1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # S2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1)]),
        # S3
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # S4
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # S5
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # S6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # S7
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    col = 0
    for A_blocks, B_blocks, C_blocks in block_factors:
        for t in range(7):
            # For this Strassen term t, compute U, V, W contributions
            # U: for each A element (i,j) in the 2x2, U[i,j] = sum over A_blocks of coeff * A_pat[t][i][j]
            # But we need to map to global indices.
            
            # A block (br, bc) means the 2x2 block at rows 2*br to 2*br+1, cols 2*bc to 2*bc+1
            # Within that block, element (i,j) is at global row 2*br+i, col 2*bc+j
            
            U_vec = {}
            for (br, bc, coeff) in A_blocks:
                for i in range(2):
                    for j in range(2):
                        a_coeff = A_pat[t][i][j] * coeff
                        if a_coeff != 0:
                            global_i = 2 * br + i
                            global_j = 2 * bc + j
                            key = (global_i, global_j)
                            U_vec[key] = U_vec.get(key, 0) + a_coeff
            
            V_vec = {}
            for (br, bc, coeff) in B_blocks:
                for j in range(2):
                    for k in range(2):
                        b_coeff = B_pat[t][j][k] * coeff
                        if b_coeff != 0:
                            global_j = 2 * br + j
                            global_k = 2 * bc + k
                            key = (global_j, global_k)
                            V_vec[key] = V_vec.get(key, 0) + b_coeff
            
            W_vec = {}
            for (br, bc, coeff) in C_blocks:
                for k in range(2):
                    for i in range(2):
                        c_coeff = C_pat[t][k][i] * coeff
                        if c_coeff != 0:
                            global_k = 2 * bc + k  # C col = B col block
                            global_i = 2 * br + i  # C row = A row block
                            key = (global_k, global_i)
                            W_vec[key] = W_vec.get(key, 0) + c_coeff
            
            # Add to U, V, W
            for (gi, gj), coeff in U_vec.items():
                U[gi * m + gj, col] += coeff
            for (gj, gk), coeff in V_vec.items():
                V[gj * p + gk, col] += coeff
            for (gk, gi), coeff in W_vec.items():
                W[gk * n + gi, col] += coeff
            
            col += 1
    
    # Part 2: 4x4x5 schoolbook for columns 4-8
    for i in range(n):
        for j in range(m):
            for k in range(4, p):  # columns 4 to 8
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Verify col == rank
    assert col == rank, f"Expected {rank}, got {col}"
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
