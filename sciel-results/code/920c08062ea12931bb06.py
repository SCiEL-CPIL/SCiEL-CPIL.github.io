"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS  # 2, 2, 9
    
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
    
    # Strategy: Use the cyclic group Z_9 structure
    # For the p=9 dimension, we can use a basis that exploits periodicity
    # 
    # Key idea: Instead of treating each k independently, use a set of
    # "frequency" basis vectors over Z_9 to create shared structure.
    #
    # The tensor condition is:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    #
    # For fixed (i,j), this must hold for all k in {0,...,8}.
    # If we can make V[j*p+k, c] and W[k*n+i, c] be related by a
    # group character, we can reduce the number of independent columns.
    #
    # Construction using Z_9 characters:
    # Let omega be a primitive 9th root of unity (complex).
    # Character chi_a(k) = omega^(a*k) for a in {0,...,8}
    #
    # But we need rational/integer coefficients. Instead, use the
    # fact that the sum of all 9 characters at any point gives 9*delta.
    #
    # Alternative structural approach:
    # Split the 9 columns into 3 groups of 3, and for each group use
    # a 2x2x3 decomposition, then combine.
    #
    # Actually, let's try a more direct approach:
    # Use the fact that for the k-dimension, we can use a "cyclic" 
    # decomposition where V and W are related by shifts.
    #
    # Let's try: rank = 3 * (2x2x3 rank)
    # For 2x2x3, schoolbook is 12, but we can do better.
    #
    # Actually, the simplest structural improvement:
    # Use 3 blocks of 2x2x3, each with rank 8 (if we can find it)
    # Total rank = 24
    #
    # For 2x2x3: Can we do rank 8?
    # Schoolbook is 12. Strassen doesn't directly apply.
    # Let's try rank 10 for 2x2x3, total 30. Still better than 32.
    #
    # Let me try a different approach: use the linearity in k.
    # For fixed (i,j), the equation is:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is a linear equation in the "products" V[j*p+k, c]*W[k*n+i, c]
    # for each k.
    #
    # If we fix U[i*m+j, c] values, we need:
    #   sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1 for all k
    #
    # Let X_j,k,c = V[j*p+k, c] * W[k*n+i, c] (depends on i through W)
    # This is complicated because W depends on i.
    #
    # Let me try a concrete construction with rank 27:
    # 3 groups of 3 k-values, each group using a 2x2x3 rank-9 decomposition
    # 3 * 9 = 27 < 32
    
    # For 2x2x3 with rank 9 (schoolbook is 12, so 9 is a reduction):
    # Actually 2x2x3 schoolbook is 2*2*3=12. Rank 9 would be an improvement.
    # Let's see if rank 9 is achievable for 2x2x3.
    
    # For now, let's try a simpler structural change:
    # Use 3 blocks, each handling 3 k-values, with rank 10 per block = 30 total
    # Or try to find a rank-9 decomposition for 2x2x3.
    
    # Let's attempt rank 27 construction:
    # 3 blocks of 3 k-values each, rank 9 per block
    
    # For a 2x2x3 block (k in {k0, k0+1, k0+2}):
    # U is 4x9, V is 6x9, W is 6x9
    # We need sum_c U[i*2+j,c] * V[j*3+k,c] * W[k*2+i,c] = 1
    
    # Schoolbook for 2x2x3 has rank 12. Let's try rank 9.
    # This is non-trivial. Let's use a different approach.
    
    # Alternative: Use the current rank-32 approach but try to optimize
    # the last schoolbook column (rank 4) into the Strassen blocks.
    
    # Actually, let me try a completely different structural approach:
    # Use the fact that 9 = 3*3 and do a two-level decomposition.
    # Level 1: 3 groups of 3
    # Level 2: Within each group, use a 2x2x3 decomposition
    
    # For 2x2x3, let's use rank 10 (slightly better than schoolbook 12)
    # Total: 3 * 10 = 30 < 32
    
    # Let me construct a rank-10 decomposition for 2x2x3:
    # This requires finding 10 terms M_c such that:
    # (AB)[i,k] = sum_j A[i,j] B[j,k] = sum_c (linear in A) * (linear in B) * (linear in C)
    
    # For 2x2x3, the output C is 2x3, so 6 output entries.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # Let's try to find a rank-10 decomposition by combining terms.
    # This is a research-level problem. For now, let's use rank 11 per block.
    # Total: 3 * 11 = 33 > 32. Not helpful.
    
    # Let me try rank 10 per block: 3*10 = 30.
    # I'll attempt to construct it explicitly.
    
    # For 2x2x3, let's use the following approach:
    # Define 10 products:
    # M1 = (a00+a10) * (b00+b10)  # covers C[0,0] and C[1,0] partially
    # M2 = (a00+a10) * (b01+b11)  # covers C[0,1] and C[1,1] partially
    # M3 = (a00+a10) * (b02+b12)  # covers C[0,2] and C[1,2] partially
    # M4 = a00 * b00
    # M5 = a00 * b01
    # M6 = a00 * b02
    # M7 = a10 * b00
    # M8 = a10 * b01
    # M9 = a10 * b02
    # M10 = (a00-a10) * (b10-b00)  # correction term
    
    # This is getting complex. Let me just return the current rank-32 solution
    # but with a cleaner implementation, and add a deadline guard.
    
    # Actually, let me try one more structural idea:
    # Use 9 columns where each column handles one k-value, but share
    # the U factors across k-values using the cyclic structure.
    
    # For each i,j, U[i*2+j, c] can be the same for all c in a group.
    # If we group k-values into 3 groups of 3, and use the same U
    # for all 3 k-values in a group, we save 2 U-factors per group.
    
    # But this doesn't directly reduce rank unless V and W also share.
    
    # Let me just implement the rank-32 solution cleanly with deadline guard.
    
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
    
    # Build rank-32 decomposition
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
    
    # Schoolbook for column 8 (rank 4)
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 8][col] = 1
            W[8 * n + i][col] = 1
            col += 1
    
    result = {"U": U, "V": V, "W": W}
    
    if verify(result["U"], result["V"], result["W"]):
        return result
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return schoolbook()
    
    return schoolbook()
# EVOLVE-BLOCK-END
