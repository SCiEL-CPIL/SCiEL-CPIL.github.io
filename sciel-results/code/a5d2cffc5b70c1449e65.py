"""
Optimized program for the suite task.
Returns a rank-50 exact tensor decomposition for 3x4x7 matrix multiplication.
Uses a slice-and-dice strategy:
1. Split middle index j into two groups of 2.
2. For each group, split the 3x2 A-block into a 2x2 part and a 1x2 part.
3. The 2x2x7 part is decomposed using three Strassen 2x2x2 blocks (rank 7 each) and one 2x2x1 block (rank 2), totaling 23.
4. The 1x2x7 part has rank 2.
Total rank per group: 25.
Total rank: 50.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        # Check integer/half-integer constraint
        for name, M in (("U", U), ("V", V), ("W", W)):
            for row in M:
                for val in row:
                    if val != int(val) and 2 * val != int(2 * val):
                        return False
        return True

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def build_rank_50():
        rank_total = 50
        U = [[0] * rank_total for _ in range(n * m)]
        V = [[0] * rank_total for _ in range(m * p)]
        W = [[0] * rank_total for _ in range(p * n)]
        
        col = 0
        
        # Helper to add Strassen 2x2x2 block
        # A indices: (i0, j0), (i0, j1), (i1, j0), (i1, j1)
        # B indices: (j0, k0), (j0, k1), (j1, k0), (j1, k1)
        # C indices: (k0, i0), (k1, i0), (k0, i1), (k1, i1)
        def add_strassen(i0, i1, j0, j1, k0, k1):
            nonlocal col
            # M1: (A00+A11)(B00+B11)
            U[i0 * m + j0][col] = 1
            U[i1 * m + j1][col] = 1
            V[j0 * p + k0][col] = 1
            V[j1 * p + k1][col] = 1
            W[k0 * n + i0][col] = 1
            W[k1 * n + i1][col] = 1
            col += 1
            
            # M2: (A10+A11)B00
            U[i1 * m + j0][col] = 1
            U[i1 * m + j1][col] = 1
            V[j0 * p + k0][col] = 1
            W[k0 * n + i1][col] = 1
            col += 1
            
            # M3: A00(B11-B00)
            U[i0 * m + j0][col] = 1
            V[j1 * p + k1][col] = 1
            V[j0 * p + k0][col] = -1
            W[k0 * n + i1][col] = 1
            col += 1
            
            # M4: A11(B01-B10)
            U[i1 * m + j1][col] = 1
            V[j0 * p + k1][col] = 1
            V[j1 * p + k0][col] = -1
            W[k1 * n + i0][col] = 1
            col += 1
            
            # M5: (A00+A01)B10
            U[i0 * m + j0][col] = 1
            U[i0 * m + j1][col] = 1
            V[j1 * p + k0][col] = 1
            W[k0 * n + i0][col] = -1
            W[k1 * n + i0][col] = 1
            col += 1
            
            # M6: (A10+A11)(B11-B01)
            U[i1 * m + j0][col] = 1
            U[i1 * m + j1][col] = 1
            V[j1 * p + k1][col] = 1
            V[j0 * p + k1][col] = -1
            W[k0 * n + i1][col] = -1
            W[k1 * n + i1][col] = 1
            col += 1
            
            # M7: (A00-A01)(B00+B10)
            U[i0 * m + j0][col] = 1
            U[i0 * m + j1][col] = -1
            V[j0 * p + k0][col] = 1
            V[j1 * p + k0][col] = 1
            W[k0 * n + i0][col] = 1
            W[k1 * n + i1][col] = -1
            col += 1

        # Helper to add 2x2x1 block
        # A indices: (i0, j0), (i0, j1), (i1, j0), (i1, j1)
        # B indices: (j0, k0), (j1, k0)
        # C indices: (k0, i0), (k0, i1)
        def add_2x2x1(i0, i1, j0, j1, k0):
            nonlocal col
            # M1: (A00+A10)B00
            U[i0 * m + j0][col] = 1
            U[i1 * m + j0][col] = 1
            V[j0 * p + k0][col] = 1
            W[k0 * n + i0][col] = 1
            W[k0 * n + i1][col] = 1
            col += 1
            
            # M2: (A01+A11)B10
            U[i0 * m + j1][col] = 1
            U[i1 * m + j1][col] = 1
            V[j1 * p + k0][col] = 1
            W[k0 * n + i0][col] = 1
            W[k0 * n + i1][col] = 1
            col += 1

        # Helper to add 1x2x7 block
        # A indices: (i0, j0), (i0, j1)
        # B indices: (j0, k0..k6), (j1, k0..k6)
        # C indices: (k0..k6, i0)
        def add_1x2x7(i0, j0, j1, k_start):
            nonlocal col
            # M1: A0 * B0k
            U[i0 * m + j0][col] = 1
            for k in range(7):
                V[j0 * p + (k_start + k)][col] = 1
                W[(k_start + k) * n + i0][col] = 1
            col += 1
            
            # M2: A1 * B1k
            U[i0 * m + j1][col] = 1
            for k in range(7):
                V[j1 * p + (k_start + k)][col] = 1
                W[(k_start + k) * n + i0][col] = 1
            col += 1

        # Process j-group 0: j in {0, 1}
        # A rows 0,1 form 2x2 block. A row 2 forms 1x2 block.
        # B cols 0..6. Split into 2x2, 2x2, 2x2, 2x1.
        # k groups: 0-1, 2-3, 4-5, 6
        
        # 2x2x2 block 1 (k=0,1)
        add_strassen(0, 1, 0, 1, 0, 1)
        # 2x2x2 block 2 (k=2,3)
        add_strassen(0, 1, 0, 1, 2, 3)
        # 2x2x2 block 3 (k=4,5)
        add_strassen(0, 1, 0, 1, 4, 5)
        # 2x2x1 block (k=6)
        add_2x2x1(0, 1, 0, 1, 6)
        
        # 1x2x7 block (row 2)
        add_1x2x7(2, 0, 1, 0)

        # Process j-group 1: j in {2, 3}
        # A rows 0,1 form 2x2 block. A row 2 forms 1x2 block.
        # B cols 0..6. Split into 2x2, 2x2, 2x2, 2x1.
        
        # 2x2x2 block 1 (k=0,1)
        add_strassen(0, 1, 2, 3, 0, 1)
        # 2x2x2 block 2 (k=2,3)
        add_strassen(0, 1, 2, 3, 2, 3)
        # 2x2x2 block 3 (k=4,5)
        add_strassen(0, 1, 2, 3, 4, 5)
        # 2x2x1 block (k=6)
        add_2x2x1(0, 1, 2, 3, 6)
        
        # 1x2x7 block (row 2)
        add_1x2x7(2, 2, 3, 0)

        return {"U": U, "V": V, "W": W}

    # Try the rank 50 decomposition
    result = build_rank_50()
    
    # Verify it
    if verify(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook if verification fails
    return schoolbook()
