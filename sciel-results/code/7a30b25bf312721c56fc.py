import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s margin
    
    # Winograd 3x3x3 decomposition (rank 7)
    # U_win333: shape (3*3, 7) = (9, 7)
    # V_win333: shape (3*3, 7) = (9, 7)
    # W_win333: shape (3*3, 7) = (9, 7)
    
    # Standard Winograd 3x3x3 factors (using 1/2 where needed)
    # U rows indexed by (i,j) where i in {0,1,2}, j in {0,1,2}
    # V rows indexed by (j,k) where j in {0,1,2}, k in {0,1,2}
    # W rows indexed by (k,i) where k in {0,1,2}, i in {0,1,2}
    
    U_win333 = [
        # i=0, j=0: [1, 0, 0, 0, 0, 0, 0]
        [1, 0, 0, 0, 0, 0, 0],
        # i=0, j=1: [0, 1, 1, 0, 0, 0, 0]
        [0, 1, 1, 0, 0, 0, 0],
        # i=0, j=2: [0, 1, -1, 0, 0, 0, 0]
        [0, 1, -1, 0, 0, 0, 0],
        # i=1, j=0: [1, 1, 0, 0, 0, 0, 0]
        [1, 1, 0, 0, 0, 0, 0],
        # i=1, j=1: [0, 0, 0, 1, 1, 0, 0]
        [0, 0, 0, 1, 1, 0, 0],
        # i=1, j=2: [0, 0, 0, 1, -1, 0, 0]
        [0, 0, 0, 1, -1, 0, 0],
        # i=2, j=0: [1, -1, 0, 0, 0, 0, 0]
        [1, -1, 0, 0, 0, 0, 0],
        # i=2, j=1: [0, 0, 0, 1, 1, 1, 0]
        [0, 0, 0, 1, 1, 1, 0],
        # i=2, j=2: [0, 0, 0, 1, -1, 1, 0]
        [0, 0, 0, 1, -1, 1, 0],
    ]
    
    V_win333 = [
        # j=0, k=0: [1, 0, 0, 0, 0, 0, 0]
        [1, 0, 0, 0, 0, 0, 0],
        # j=0, k=1: [1, 1, 0, 0, 0, 0, 0]
        [1, 1, 0, 0, 0, 0, 0],
        # j=0, k=2: [1, -1, 0, 0, 0, 0, 0]
        [1, -1, 0, 0, 0, 0, 0],
        # j=1, k=0: [0, 1, 1, 0, 0, 0, 0]
        [0, 1, 1, 0, 0, 0, 0],
        # j=1, k=1: [0, 0, 0, 1, 1, 0, 0]
        [0, 0, 0, 1, 1, 0, 0],
        # j=1, k=2: [0, 0, 0, 1, -1, 0, 0]
        [0, 0, 0, 1, -1, 0, 0],
        # j=2, k=0: [0, 1, -1, 0, 0, 0, 0]
        [0, 1, -1, 0, 0, 0, 0],
        # j=2, k=1: [0, 0, 0, 1, 1, 1, 0]
        [0, 0, 0, 1, 1, 1, 0],
        # j=2, k=2: [0, 0, 0, 1, -1, 1, 0]
        [0, 0, 0, 1, -1, 1, 0],
    ]
    
    W_win333 = [
        # k=0, i=0: [1, 0, 0, 0, 0, 0, 0]
        [1, 0, 0, 0, 0, 0, 0],
        # k=0, i=1: [0, 1, 0, 0, 0, 0, 0]
        [0, 1, 0, 0, 0, 0, 0],
        # k=0, i=2: [0, 0, 0, 0, 0, 0, 1]
        [0, 0, 0, 0, 0, 0, 1],
        # k=1, i=0: [0, 1, 1, 0, 0, 0, 0]
        [0, 1, 1, 0, 0, 0, 0],
        # k=1, i=1: [0, 0, 0, 1, 0, 0, 0]
        [0, 0, 0, 1, 0, 0, 0],
        # k=1, i=2: [0, 0, 0, 0, 0, 1, 1]
        [0, 0, 0, 0, 0, 1, 1],
        # k=2, i=0: [0, 1, -1, 0, 0, 0, 0]
        [0, 1, -1, 0, 0, 0, 0],
        # k=2, i=1: [0, 0, 0, 1, 0, 0, 0]
        [0, 0, 0, 1, 0, 0, 0],
        # k=2, i=2: [0, 0, 0, 0, 0, 1, -1]
        [0, 0, 0, 0, 0, 1, -1],
    ]
    
    # Winograd 3x2x2 decomposition (rank 4) for 3x4x2 block
    # This multiplies a 3x4 matrix by a 4x2 matrix to get 3x2
    # We can decompose this as two 3x2x2 blocks (columns 0-1 and 2-3 of A)
    # Actually, let's think differently: 3x4x2 can be split into two 3x2x2 blocks
    
    # Winograd 3x2x2 (rank 4)
    # U_win322: shape (3*2, 4) = (6, 4) - for A which is 3x2
    # V_win322: shape (2*2, 4) = (4, 4) - for B which is 2x2
    # W_win322: shape (2*3, 4) = (6, 4) - for C which is 2x3 (transposed)
    
    # Standard Winograd 3x2x2 factors
    U_win322 = [
        # i=0, j=0: [1, 0, 0, 0]
        [1, 0, 0, 0],
        # i=0, j=1: [0, 1, 0, 0]
        [0, 1, 0, 0],
        # i=1, j=0: [1, 1, 0, 0]
        [1, 1, 0, 0],
        # i=1, j=1: [0, 0, 1, 0]
        [0, 0, 1, 0],
        # i=2, j=0: [1, -1, 0, 0]
        [1, -1, 0, 0],
        # i=2, j=1: [0, 0, 1, 1]
        [0, 0, 1, 1],
    ]
    
    V_win322 = [
        # j=0, k=0: [1, 0, 0, 0]
        [1, 0, 0, 0],
        # j=0, k=1: [1, 1, 0, 0]
        [1, 1, 0, 0],
        # j=1, k=0: [0, 1, 0, 0]
        [0, 1, 0, 0],
        # j=1, k=1: [0, 0, 1, 1]
        [0, 0, 1, 1],
    ]
    
    W_win322 = [
        # k=0, i=0: [1, 0, 0, 0]
        [1, 0, 0, 0],
        # k=0, i=1: [0, 1, 0, 0]
        [0, 1, 0, 0],
        # k=0, i=2: [0, 0, 0, 1]
        [0, 0, 0, 1],
        # k=1, i=0: [0, 1, 0, 0]
        [0, 1, 0, 0],
        # k=1, i=1: [0, 0, 1, 0]
        [0, 0, 1, 0],
        # k=1, i=2: [0, 0, 1, -1]
        [0, 0, 1, -1],
    ]
    
    # Now combine:
    # Total rank = 7 (for 3x3x3) + 4 (for 3x2x2 first half) + 4 (for 3x2x2 second half) = 15
    # Wait, that's not optimal. Let me reconsider.
    
    # Actually, for 3x4x5, we can split as:
    # A (3x4) * B (4x5) = C (3x5)
    # Split B into B1 (4x3) and B2 (4x2)
    # Then C = A*B1 + A*B2
    # A*B1 is 3x3, A*B2 is 3x2
    # Use Winograd 3x3x3 for A*B1 (rank 7)
    # Use Winograd 3x2x2 for A*B2 (rank 4)
    # Total rank = 7 + 4 = 11
    
    # But we need to handle the column splitting properly.
    # A is 3x4, so for A*B1 where B1 is 4x3, we use all of A.
    # For A*B2 where B2 is 4x2, we also use all of A.
    
    # Let's construct the full decomposition:
    rank_total = 7 + 4
    
    U = [[0] * rank_total for _ in range(n * m)]  # (12, 11)
    V = [[0] * rank_total for _ in range(m * p)]  # (20, 11)
    W = [[0] * rank_total for _ in range(p * n)]  # (15, 11)
    
    # Part 1: Winograd 3x3x3 for A * B1 where B1 = B[:, 0:3]
    # U_win333 maps A (3x3) -> intermediate, but A is 3x4, so we use A[:, 0:3]
    # Actually, Winograd 3x3x3 multiplies 3x3 * 3x3 = 3x3
    # We want A (3x4) * B1 (4x3) = C1 (3x3)
    # This is not directly 3x3x3. We need to think about this differently.
    
    # Let me reconsider the approach. The tensor T[i,j,k] = sum_l A[i,l] * B[l,k]
    # For 3x4x5, we have i in {0,1,2}, j in {0,1,2,3}, k in {0,1,2,3,4}
    
    # A better approach: use the fact that we can decompose the multiplication
    # by splitting the middle dimension.
    
    # Actually, let's just implement the schoolbook for now and verify correctness,
    # then try to optimize.
    
    # Fallback: Schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Check deadline
    if time.time() > deadline:
        return {"U": U_school, "V": V_school, "W": W_school}
    
    # For now, return schoolbook. The Winograd approach needs more careful construction.
    return {"U": U_school, "V": V_school, "W": W_school}
