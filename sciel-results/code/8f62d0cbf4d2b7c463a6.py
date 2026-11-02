"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 120.0
    
    nm = n * m  # 8
    mp = m * p  # 28
    pn = p * n  # 14
    
    # Use a structured decomposition based on Winograd 2x2x2
    # Winograd 2x2x2: rank 7
    # U_w (4x7), V_w (4x7), W_w (4x7)
    
    # Standard Winograd 2x2x2 decomposition:
    # C = A * B where A, B are 2x2
    # Let a11, a12, a21, a22 and b11, b12, b21, b22
    # Winograd: 
    #   x1 = a11*b11
    #   x2 = a12*b22
    #   x3 = (a11+a12)*(b11+b21)
    #   x4 = (a11+a21)*(b11+b12)
    #   x5 = (a12+a22)*(b21+b22)
    #   x6 = (a11-a12)*b12
    #   x7 = (a21-a22)*b21
    #   c11 = x1 + x3 + x4 - x6 - x7
    #   c12 = x2 + x3 + x5 + x6
    #   c21 = x2 + x4 + x5 - x6
    #   c22 = x1 + x3 + x4 + x5 + x7
    
    # Actually, let me use a cleaner formulation. The standard Winograd 2x2x2:
    # U_w rows indexed by (i,j) for A[i,j], V_w rows by (j,k) for B[j,k], W_w rows by (i,k) for C[i,k]
    
    # For 2x2x2, the tensor has 8 ones (one for each i,j,k combination)
    # Winograd reduces this to rank 7.
    
    # Let me define the Winograd 2x2x2 factors:
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    U_w = np.array([
        [1, 0, 1, 1, 0, 1, 0],  # a00
        [0, 1, 1, 0, 1, -1, 0], # a01
        [0, 0, 0, 1, 0, 0, 1],  # a10
        [0, 1, 0, 0, 1, 0, -1], # a11
    ], dtype=float)
    
    V_w = np.array([
        [1, 0, 1, 1, 0, 0, 0],  # b00
        [0, 0, 1, 1, 0, 1, 0],  # b01
        [0, 0, 1, 0, 1, 0, 1],  # b10
        [0, 1, 0, 0, 1, 0, 0],  # b11
    ], dtype=float)
    
    W_w = np.array([
        [1, 0, 1, 1, 0, 0, 0],  # c00
        [0, 1, 1, 0, 1, 1, 0],  # c01
        [0, 0, 0, 1, 1, -1, 1], # c10
        [1, 0, 1, 0, 0, 0, 1],  # c11
    ], dtype=float)
    
    # Verify Winograd 2x2x2
    def check_222(U, V, W):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    a = i * 2 + j
                    b = j * 2 + k
                    c = k * 2 + i
                    s = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(s - 1.0) > 1e-6:
                        return False
        # Check zeros
        for a in range(4):
            i_a, j_a = divmod(a, 2)
            for b in range(4):
                j_b, k_b = divmod(b, 2)
                for c in range(4):
                    k_c, i_c = divmod(c, 2)
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        continue
                    s = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(s) > 1e-6:
                        return False
        return True
    
    # Now for 2x4x7, use a block decomposition
    # Split m=4 into two blocks of size 2
    # Block 1: j=0,1 -> 2x2x7
    # Block 2: j=2,3 -> 2x2x7
    
    # For each 2x2x7 block, we need a decomposition
    # 2x2x7: A is 2x2, B is 2x7, C is 2x7
    # Schoolbook rank: 2*2*7 = 28
    
    # Can we reduce 2x2x7? 
    # One approach: use Winograd 2x2x2 for each column of B
    # But that would give 7 * 7 = 49, worse than 28
    
    # Better: use the structure of 2x2x7 directly
    # Let me try to find a rank-20 or so decomposition for 2x2x7
    
    # Actually, for the full 2x4x7, let me try a different approach:
    # Use a tensor network / matrix product state approach
    
    # For now, let's use schoolbook and try to optimize
    rank_school = n * m * p  # 56
    U = np.zeros((nm, rank_school))
    V = np.zeros((mp, rank_school))
    W = np.zeros((pn, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    def check_valid(U, V, W, rank):
        if time.time() > deadline - 2:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(s - 1.0) > 1e-6:
                        return False
        for a in range(nm):
            i_a = a // m
            j_a = a % m
            for b in range(mp):
                j_b = b // p
                k_b = b % p
                for c in range(pn):
                    k_c = c // n
                    i_c = c % n
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        continue
                    s = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(s) > 1e-6:
                        return False
        return True
    
    if not check_valid(U, V, W, rank_school):
        # Shouldn't happen, but fallback
        pass
    
    # Try to reduce rank by identifying linear dependencies
    # This is complex, so for now return schoolbook
    
    U_list = [[float(U[i, j]) for j in range(rank_school)] for i in range(nm)]
    V_list = [[float(V[i, j]) for j in range(rank_school)] for i in range(mp)]
    W_list = [[float(W[i, j]) for j in range(rank_school)] for i in range(pn)]
    
    def clean_val(x):
        if abs(x - round(x)) < 1e-10:
            return int(round(x))
        elif abs(2*x - round(2*x)) < 1e-10:
            return round(2*x) / 2
        else:
            return float(x)
    
    U_list = [[clean_val(x) for x in row] for row in U_list]
    V_list = [[clean_val(x) for x in row] for row in V_list]
    W_list = [[clean_val(x) for x in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
