"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
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
    
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-10:
                        return False
        return True
    
    # Strassen 2x2x2 decomposition (rank 7)
    strassen_U = [
        [1, 0, 1, 0, 1, -1, 0],   # a00
        [0, 0, 0, 0, 1, 0, 1],    # a01
        [0, 1, 0, 0, 0, 1, 0],    # a10
        [1, 1, 0, 1, 0, 0, -1]    # a11
    ]
    
    strassen_V = [
        [1, 1, 0, -1, 0, 1, 0],   # b00
        [0, 0, 0, 1, 0, 0, 1],    # b01
        [0, 0, 1, 0, 0, 1, 0],    # b10
        [1, 0, -1, 0, 1, 0, 1]    # b11
    ]
    
    strassen_W = [
        [1, 0, 0, -1, 1, 0, 1],   # C00 (k=0, i=0)
        [0, 1, 0, 0, 0, 1, 0],    # C10 (k=0, i=1)
        [0, 0, 1, 1, 0, 0, 0],    # C01 (k=1, i=0)
        [1, 0, -1, 0, 0, 1, 1]    # C11 (k=1, i=1)
    ]
    
    # Try rank-28 decomposition:
    # Use 4 Strassen blocks (4*7=28) for columns 0-7, and handle column 8
    # by absorbing it into the Strassen structure.
    # 
    # Alternative: use a modified approach where we share some terms.
    # 
    # Actually, let's try a cleaner structural approach:
    # For 2x2x9, use the fact that we can write:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Apply Strassen to the 2x2 part, giving 7 "products" M_c.
    # Each M_c is a scalar that multiplies a linear combination of B's entries.
    # For each of the 9 columns k, we need to express B[j,k] in terms of 
    # the Strassen factors.
    #
    # The key insight: for each column k, the contribution to C[i,k] is:
    # C[i,k] = sum_c W[k,i,c] * (sum_j U[i,j,c] * A[i,j]) * (sum_j V[j,k,c] * B[j,k])
    # 
    # But wait, that's not quite right. The correct form is:
    # C[i,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] summed over j
    # No, the tensor decomposition is:
    # 1 = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] for each (i,j,k)
    #
    # This means for each fixed (i,j,k), the sum over c must equal 1.
    #
    # Let's try a rank-27 approach by finding a clever combination.
    # 
    # For now, let's try rank 28: 4 Strassen blocks for 8 columns + 1 schoolbook
    # But that's 4*7+1 = 29. Hmm.
    #
    # Actually, the current approach gives 32. Let me try to do better.
    #
    # New idea: use a 2x2x3 decomposition of lower rank, then combine.
    # For 2x2x3, schoolbook is 12. Can we do better?
    #
    # Let's try to construct a rank-11 decomposition for 2x2x3.
    
    def try_2x2x3_decomposition():
        """Try to find a rank-11 or lower decomposition for 2x2x3."""
        # For 2x2x3, we need:
        # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i in {0,1}, j in {0,1}, k in {0,1,2}
        # 
        # That's 12 equations. With rank r, we have 4r + 6r + 6r = 16r unknowns.
        # For r=11, we have 176 unknowns and 12 equations, so it should be feasible.
        #
        # Let's use a constructive approach based on Strassen.
        # 
        # Strassen for 2x2x2 gives rank 7. For the third column, we need 4 more terms
        # (one for each (i,j) pair), giving rank 11.
        #
        # But maybe we can do better by sharing structure.
        
        # Use Strassen for columns 0,1 (rank 7) + schoolbook for column 2 (rank 4) = rank 11
        # This is the same as the current approach scaled down.
        
        # Let's try to find if rank 10 is possible.
        # For now, return rank 11.
        rank = 11
        U = [[0] * rank for _ in range(4)]  # 2x2 = 4 entries
        V = [[0] * rank for _ in range(6)]  # 2x3 = 6 entries
        W = [[0] * rank for _ in range(6)]  # 3x2 = 6 entries
        
        col = 0
        # Strassen for columns 0,1
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * 2 + j][col] = strassen_U[i * 2 + j][c]
            for j in range(2):
                V[j * 3 + 0][col] = strassen_V[j * 2 + 0][c]
                V[j * 3 + 1][col] = strassen_V[j * 2 + 1][c]
            for i in range(2):
                W[0 * 2 + i][col] = strassen_W[0 * 2 + i][c]
                W[1 * 2 + i][col] = strassen_W[1 * 2 + i][c]
            col += 1
        
        # Schoolbook for column 2
        for i in range(2):
            for j in range(2):
                U[i * 2 + j][col] = 1
                V[j * 3 + 2][col] = 1
                W[2 * 2 + i][col] = 1
                col += 1
        
        # Verify
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    total = 0
                    for c in range(rank):
                        total += U[i * 2 + j][c] * V[j * 3 + k][c] * W[k * 2 + i][c]
                    if abs(total - 1) > 1e-10:
                        return None
        
        return {"U": U, "V": V, "W": W}
    
    # Try the 2x2x3 decomposition
    result_2x2x3 = try_2x2x3_decomposition()
    
    if result_2x2x3 is not None:
        # Combine three 2x2x3 blocks for 2x2x9
        # Each block handles 3 columns
        total_rank = 3 * 11  # 33
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        for block in range(3):
            k0 = block * 3
            for c in range(11):
                col = block * 11 + c
                for i in range(n):
                    for j in range(m):
                        U[i * m + j][col] = result_2x2x3["U"][i * m + j][c]
                for j in range(m):
                    for kk in range(3):
                        V[j * p + k0 + kk][col] = result_2x2x3["V"][j * 3 + kk][c]
                for i in range(n):
                    for kk in range(3):
                        W[(k0 + kk) * n + i][col] = result_2x2x3["W"][(kk) * 2 + i][c]
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # Fall back to the rank-32 approach (4 Strassen blocks + schoolbook)
    total_rank = 32
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1
        for c in range(7):
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = strassen_U[i * m + j][c]
            for j in range(m):
                V[j * p + k0][col] = strassen_V[j * 2 + 0][c]
                V[j * p + k1][col] = strassen_V[j * 2 + 1][c]
            for i in range(n):
                W[k0 * n + i][col] = strassen_W[0 * 2 + i][c]
                W[k1 * n + i][col] = strassen_W[1 * 2 + i][c]
            col += 1
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 8][col] = 1
            W[8 * n + i][col] = 1
            col += 1
    
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Ultimate fallback: schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
