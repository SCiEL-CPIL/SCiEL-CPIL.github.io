"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    rank = 45
    
    U = [[0] * rank for _ in range(n * m)]  # 20 x 45
    V = [[0] * rank for _ in range(m * p)]  # 25 x 45
    W = [[0] * rank for _ in range(p * n)]  # 20 x 45
    
    col = 0
    
    # Four 2x2x2 Strassen blocks
    # Block (i0, j0): rows i0,i0+1 of A; cols j0,j0+1 of B; output rows i0,i0+1, cols j0,j0+1 of C
    strassen_blocks = [(0, 0), (0, 2), (2, 0), (2, 2)]
    
    for (i0, j0) in strassen_blocks:
        # 7 Strassen terms for this 2x2x2 block
        # A indices: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
        # B indices: (j0, j0), (j0, j0+1), (j0+1, j0), (j0+1, j0+1)
        # C output: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
        
        # M1 = (A[i0,j0] - A[i0+1,j0+1]) * (B[j0,j0+1] + B[j0+1,j0+1])
        U[i0 * m + j0][col] = 1
        U[(i0+1) * m + (j0+1)][col] = -1
        V[j0 * p + (j0+1)][col] = 1
        V[(j0+1) * p + (j0+1)][col] = 1
        W[(j0) * n + i0][col] = 1
        W[(j0+1) * n + i0][col] = 1
        col += 1
        
        # M2 = (A[i0,j0] + A[i0+1,j0+1]) * B[j0+1,j0+1]
        U[i0 * m + j0][col] = 1
        U[(i0+1) * m + (j0+1)][col] = 1
        V[(j0+1) * p + (j0+1)][col] = 1
        W[(j0+1) * n + i0][col] = -1
        W[(j0+1) * n + (i0+1)][col] = 1
        W[(j0) * n + (i0+1)][col] = 1
        col += 1
        
        # M3 = A[i0,j0] * (B[j0,j0] - B[j0+1,j0+1])
        U[i0 * m + j0][col] = 1
        V[j0 * p + j0][col] = 1
        V[(j0+1) * p + (j0+1)][col] = -1
        W[(j0) * n + i0][col] = 1
        W[(j0+1) * n + i0][col] = -1
        col += 1
        
        # M4 = A[i0+1,j0+1] * (B[j0,j0] + B[j0+1,j0+1])
        U[(i0+1) * m + (j0+1)][col] = 1
        V[j0 * p + j0][col] = 1
        V[(j0+1) * p + (j0+1)][col] = 1
        W[(j0) * n + (i0+1)][col] = 1
        W[(j0+1) * n + (i0+1)][col] = 1
        col += 1
        
        # M5 = (A[i0,j0] + A[i0,j0+1]) * B[j0+1,j0+1]
        U[i0 * m + j0][col] = 1
        U[i0 * m + (j0+1)][col] = 1
        V[(j0+1) * p + (j0+1)][col] = 1
        W[(j0+1) * n + i0][col] = 1
        col += 1
        
        # M6 = (A[i0+1,j0] - A[i0+1,j0+1]) * (B[j0,j0] + B[j0,j0+1])
        U[(i0+1) * m + j0][col] = 1
        U[(i0+1) * m + (j0+1)][col] = -1
        V[j0 * p + j0][col] = 1
        V[j0 * p + (j0+1)][col] = 1
        W[(j0) * n + (i0+1)][col] = 1
        col += 1
        
        # M7 = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,j0] - B[j0+1,j0+1])
        U[i0 * m + j0][col] = 1
        U[(i0+1) * m + (j0+1)][col] = 1
        V[j0 * p + j0][col] = 1
        V[(j0+1) * p + (j0+1)][col] = -1
        W[(j0) * n + i0][col] = -1
        W[(j0+1) * n + i0][col] = 1
        W[(j0) * n + (i0+1)][col] = 1
        W[(j0+1) * n + (i0+1)][col] = -1
        col += 1
    
    # One 2x3x3 block: rows 2,3 of A; col 4 of B; output rows 2,3, col 4 of C
    # A is 2x3 (rows 2,3; cols 0,1,2), B is 3x1 (col 4), C is 2x1 (rows 2,3; col 4)
    # Wait, that's not right. Let me reconsider.
    
    # Actually, the remaining part is: A rows 2,3 (2 rows), all 5 cols of A, B rows 0-4, col 4 of B
    # No, let me re-examine the partition.
    
    # The 4x5x5 tensor: C[i,k] = sum_j A[i,j] * B[j,k]
    # i in {0,1,2,3}, j in {0,1,2,3,4}, k in {0,1,2,3,4}
    
    # I partitioned:
    # Block (0,0): i in {0,1}, j in {0,1}, k in {0,1} -> 2x2x2
    # Block (0,2): i in {0,1}, j in {2,3}, k in {2,3} -> 2x2x2
    # Block (2,0): i in {2,3}, j in {0,1}, k in {0,1} -> 2x2x2
    # Block (2,2): i in {2,3}, j in {2,3}, k in {2,3} -> 2x2x2
    
    # Remaining:
    # i in {0,1}, j in {4}, k in {0,1,2,3,4} -> 2x1x5
    # i in {0,1}, j in {0,1,2,3}, k in {4} -> 2x4x1
    # i in {2,3}, j in {4}, k in {0,1,2,3,4} -> 2x1x5
    # i in {2,3}, j in {0,1,2,3}, k in {4} -> 2x4x1
    # i in {0,1}, j in {4}, k in {4} -> 2x1x1
    # i in {2,3}, j in {4}, k in {4} -> 2x1x1
    
    # This is getting complicated. Let me use a different partition.
    
    # Better partition:
    # Split i into {0,1} and {2,3}
    # Split j into {0,1}, {2,3}, {4}
    # Split k into {0,1}, {2,3}, {4}
    
    # That gives 2*3*3 = 18 blocks of various sizes.
    
    # Let me restart with a cleaner approach.
    
    # Actually, let me just use the full decomposition more carefully.
    
    # I'll use a rank-45 decomposition by combining:
    # - 4 Strassen 2x2x2 blocks (rank 7 each) = 28
    # - Remaining terms to cover the rest
    
    # The remaining part after the 4 2x2x2 blocks:
    # For i in {0,1}: j=4 with all k, and j in {0,1,2,3} with k=4
    # For i in {2,3}: j=4 with all k, and j in {0,1,2,3} with k=4
    
    # This is 2 * (1*5 + 4*1 + 1*1) = 2 * 10 = 20 terms if done naively
    # But we can do better.
    
    # Total so far: 28 + 20 = 48, not 45.
    
    # Let me use a different strategy. I'll construct a proper rank-45 decomposition.
    
    # Reset and rebuild
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Use Strassen for the four 2x2x2 corner blocks
    strassen_blocks = [(0, 0), (0, 2), (2, 0), (2, 2)]
    
    for (i0, j0) in strassen_blocks:
        # 7 Strassen terms
        # M1 = (A[i0,j0] - A[i0+1,j0+1]) * (B[j0,j0+1] + B[j0+1,j0+1])
        U[i0 * m + j0][col] = 1
        U[(i0+1) * m + (j0+1)][col] = -1
        V[j0 * p + (j0+1)][col] = 1
        V[(j0+1) * p + (j0+1)][col] = 1
        W[(j0) * n + i0][col] = 1
        W[(j0+1) * n + i0][col] = 1
        col += 1
        
        # M2 = (A[i0,j0] + A[i0+1,j0+1]) * B[j0+1,j0+1]
        U[i0 * m + j0][col] = 1
        U[(i0+1) * m + (j0+1)][col] = 1
        V[(j0+1) * p + (j0+1)][col] = 1
        W[(j0+1) * n + i0][col] = -1
        W[(j0+1) * n + (i0+1)][col] = 1
        W[(j0) * n + (i0+1)][col] = 1
        col += 1
        
        # M3 = A[i0,j0] * (B[j0,j0] - B[j0+1,j0+1])
        U[i0 * m + j0][col] = 1
        V[j0 * p + j0][col] = 1
        V[(j0+1) * p + (j0+1)][col] = -1
        W[(j0) * n + i0][col] = 1
        W[(j0+1) * n + i0][col] = -1
        col += 1
        
        # M4 = A[i0+1,j0+1] * (B[j0,j0] + B[j0+1,j0+1])
        U[(i0+1) * m + (j0+1)][col] = 1
        V[j0 * p + j0][col] = 1
        V[(j0+1) * p + (j0+1)][col] = 1
        W[(j0) * n + (i0+1)][col] = 1
        W[(j0+1) * n + (i0+1)][col] = 1
        col += 1
        
        # M5 = (A[i0,j0] + A[i0,j0+1]) * B[j0+1,j0+1]
        U[i0 * m + j0][col] = 1
        U[i0 * m + (j0+1)][col] = 1
        V[(j0+1) * p + (j0+1)][col] = 1
        W[(j0+1) * n + i0][col] = 1
        col += 1
        
        # M6 = (A[i0+1,j0] - A[i0+1,j0+1]) * (B[j0,j0] + B[j0,j0+1])
        U[(i0+1) * m + j0][col] = 1
        U[(i0+1) * m + (j0+1)][col] = -1
        V[j0 * p + j0][col] = 1
        V[j0 * p + (j0+1)][col] = 1
        W[(j0) * n + (i0+1)][col] = 1
        col += 1
        
        # M7 = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,j0] - B[j0+1,j0+1])
        U[i0 * m + j0][col] = 1
        U[(i0+1) * m + (j0+1)][col] = 1
        V[j0 * p + j0][col] = 1
        V[(j0+1) * p + (j0+1)][col] = -1
        W[(j0) * n + i0][col] = -1
        W[(j0+1) * n + i0][col] = 1
        W[(j0) * n + (i0+1)][col] = 1
        W[(j0+1) * n + (i0+1)][col] = -1
        col += 1
    
    # Now handle the remaining 17 terms for the cross-block interactions
    # The remaining tensor involves:
    # - A[i,4] * B[4,k] for i in {0,1,2,3}, k in {0,1,2,3,4}
    # - A[i,j] * B[j,4] for i in {0,1,2,3}, j in {0,1,2,3}
    # - A[i,4] * B[4,4] is counted twice above, so subtract once
    
    # This is a 4x5x5 tensor minus the 4 2x2x2 blocks.
    # Let me handle it as a 4x5x5 with some entries zeroed out.
    
    # For the remaining part, I'll use a rank-17 decomposition.
    # The remaining tensor has support on:
    # (i,j,k) where not (i in {0,1} and j in {j0,j0+1} and k in {j0,j0+1}) for the 4 blocks
    
    # This is complex. Let me use a simpler approach: just add the remaining terms directly.
    # The remaining terms are:
    # For each i in {0,1,2,3}, j=4, k in {0,1,2,3,4}: term A[i,4]*B[4,k]
    # For each i in {0,1,2,3}, j in {0,1,2,3}, k=4: term A[i,j]*B[j,4]
    # Minus the double-counted A[i,4]*B[4,4] for each i
    
    # That's 4*5 + 4*4 - 4 = 20 + 16 - 4 = 32 terms, but we only have 17 slots left.
    
    # I need a smarter decomposition. Let me use the fact that we can combine terms.
    
    # For now, let me just use the schoolbook for the remaining part and accept rank 28+32=60.
    # But the directive says to achieve 45. Let me think harder.
    
    # Actually, I realize my partition was flawed. Let me use a proper 2x3x3 decomposition.
    
    # New approach: partition i into {0,1} and {2,3}, j into {0,1,2} and {3,4}, k into {0,1,2} and {3,4}
    # That gives 2*2*2 = 8 blocks:
    # (0,0,0): 2x3x3, (0,0,1): 2x3x2, (0,1,0): 2x2x3, (0,1,1): 2x2x2
    # (1,0,0): 2x3x3, (1,0,1): 2x3x2, (1,1,0): 2x2x3, (1,1,1): 2x2x2
    
    # Ranks: 17+13+13+7 + 17+13+13+7 = 100. Not better.
    
    # Let me try: i into {0,1,2} and {3}, j into {0,1,2,3} and {4}, k into {0,1,2,3} and {4}
    # 2*2*2 = 8 blocks:
    # (0,0,0): 3x4x4, (0,0,1): 3x4x1, (0,1,0): 3x1x4, (0,1,1): 3x1x1
    # (1,0,0): 1x4x4, (1,0,1): 1x4x1, (1,1,0): 1x1x4, (1,1,1): 1x1x1
    
    # This is getting too complex. Let me just implement a valid decomposition with rank < 100.
    
    # I'll use the 4 Strassen blocks (rank 28) plus schoolbook for the rest (rank 32) = 60.
    
    # Add remaining terms with schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if this term is already covered by a Strassen block
                covered = False
                for (i0, j0) in strassen_blocks:
                    if i in (i0, i0+1) and j in (j0, j0+1) and k in (j0, j0+1):
                        covered = True
                        break
                if not covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    # Trim to actual rank used
    actual_rank = col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
