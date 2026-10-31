import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We assume n=m=p=4 for this specific task
    if n != 4 or m != 4 or p != 4:
        # Fallback to schoolbook if not 4x4x4
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Build Strassen 4x4 decomposition
    # Recursive helper
    def build_strassen(size):
        if size == 1:
            # 1x1 matmul: rank 1
            U = [[1]]
            V = [[1]]
            W = [[1]]
            return U, V, W
        
        h = size // 2
        # Get sub-decomposition
        U_sub, V_sub, W_sub = build_strassen(h)
        R_sub = len(U_sub[0]) # Rank of sub-problem
        
        # Parent dimensions
        N = size * size
        R_parent = 7 * R_sub
        
        U = [[0] * R_parent for _ in range(N)]
        V = [[0] * R_parent for _ in range(N)]
        W = [[0] * R_parent for _ in range(N)]
        
        # Define block mappings for A, B, C
        # Blocks are (0,0), (0,1), (1,0), (1,1) corresponding to 11, 12, 21, 22
        # Sub-index (i', j') in [0, h-1] x [0, h-1]
        
        # For each of the 7 Strassen products, define:
        # A_map: list of (parent_block, sign)
        # B_map: list of (parent_block, sign)
        # C_map: list of (parent_block, sign)
        
        # r=1: M1 = (A11+A22)(B11+B22)
        # r=2: M2 = (A21+A22)B11
        # r=3: M3 = A11(B12-B22)
        # r=4: M4 = A22(B21-B11)
        # r=5: M5 = (A11+A12)B22
        # r=6: M6 = (A21-A11)(B11+B12)
        # r=7: M7 = (A12-A22)(B21+B22)
        
        # C sums:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        A_maps = [
            [(0, 1), (3, 1)],      # r=1: A11 + A22
            [(2, 1), (3, 1)],      # r=2: A21 + A22
            [(0, 1)],              # r=3: A11
            [(3, 1)],              # r=4: A22
            [(0, 1), (1, 1)],      # r=5: A11 + A12
            [(2, 1), (0, -1)],     # r=6: A21 - A11
            [(1, 1), (3, -1)],     # r=7: A12 - A22
        ]
        
        B_maps = [
            [(0, 1), (3, 1)],      # r=1: B11 + B22
            [(0, 1)],              # r=2: B11
            [(1, 1), (3, -1)],     # r=3: B12 - B22
            [(2, 1), (0, -1)],     # r=4: B21 - B11
            [(3, 1)],              # r=5: B22
            [(0, 1), (1, 1)],      # r=6: B11 + B12
            [(2, 1), (3, 1)],      # r=7: B21 + B22
        ]
        
        C_maps = [
            [(0, 1), (3, 1)],      # r=1: C11 (+1), C22 (+1)
            [(2, 1), (3, -1)],     # r=2: C21 (+1), C22 (-1)
            [(1, 1), (3, 1)],      # r=3: C12 (+1), C22 (+1)
            [(0, 1), (2, 1)],      # r=4: C11 (+1), C21 (+1)
            [(0, -1), (1, 1)],     # r=5: C11 (-1), C12 (+1)
            [(3, 1)],              # r=6: C22 (+1)
            [(0, 1)],              # r=7: C11 (+1)
        ]
        
        # Helper to get parent index from block and sub-index
        def get_parent_index(block, i_prime, j_prime, size):
            # block: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            bi, bj = divmod(block, 2)
            h = size // 2
            i = bi * h + i_prime
            j = bj * h + j_prime
            return i * size + j
        
        # Fill U, V, W
        for r in range(7):
            base_col = r * R_sub
            for c in range(R_sub):
                col = base_col + c
                for i_prime in range(h):
                    for j_prime in range(h):
                        sub_idx = i_prime * h + j_prime
                        val_u = U_sub[sub_idx][c]
                        val_v = V_sub[sub_idx][c]
                        val_w = W_sub[sub_idx][c]
                        
                        if val_u != 0:
                            for block, sign in A_maps[r]:
                                p_idx = get_parent_index(block, i_prime, j_prime, size)
                                U[p_idx][col] += sign * val_u
                                
                        if val_v != 0:
                            for block, sign in B_maps[r]:
                                p_idx = get_parent_index(block, i_prime, j_prime, size)
                                V[p_idx][col] += sign * val_v
                                
                        if val_w != 0:
                            for block, sign in C_maps[r]:
                                p_idx = get_parent_index(block, i_prime, j_prime, size)
                                W[p_idx][col] += sign * val_w
                                
        return U, V, W

    U, V, W = build_strassen(4)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
