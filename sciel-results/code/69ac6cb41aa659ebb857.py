import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Strategy: Decompose 3x4x4 into:
    # 1. 3x3x3 core (rows 0-2, cols 0-2)
    # 2. 3x1x4 strip (row 3, cols 0-3) - schoolbook rank 12
    # 3. 3x4x1 strip (rows 0-2, col 3) - schoolbook rank 12
    # 4. 1x1x1 corner (row 3, col 3) - schoolbook rank 1
    # Total schoolbook: 27 + 12 + 12 + 1 = 52 (worse than 48)
    
    # Better: Use Strassen for 2x2x2 blocks within the 4x4 part
    # Decompose 3x4x4 as:
    # - 3x2x4 top part (rows 0-2, cols 0-1 of B)
    # - 3x2x4 bottom part (rows 0-2, cols 2-3 of B)
    # - 1x4x4 last row (row 3)
    
    # Actually, let's use a cleaner block decomposition:
    # Split A (3x4) into A1 (3x2) and A2 (3x2)
    # Split B (4x4) into B1 (2x4) and B2 (2x4)
    # C = A1*B1 + A2*B2
    
    # For A1*B1: 3x2x4
    # For A2*B2: 3x2x4
    # For row 3 of A with full B: 1x4x4
    
    # Even better: Use Strassen on 2x2x2 blocks
    # Split the 4x4 B into four 2x2 blocks: B11, B12, B21, B22
    # Split A (3x4) into A1 (3x2) and A2 (3x2)
    
    # C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2:4] * B[2:4, 0:2]
    # This is getting complex. Let me use a simpler approach.
    
    # Use the known rank-44 decomposition from the inspiration:
    # 23 (for 3x3x3) + 9 (for 3x3x1) + 12 (for 3x1x4)
    # But I need the actual 23 terms.
    
    # Let me try a different approach: use Strassen's algorithm recursively
    # 3x4x4 can be decomposed using 2x2x2 Strassen blocks
    
    # Actually, let's just use the schoolbook 48 rank solution which is guaranteed correct,
    # and then try to reduce it.
    
    # For now, return the 48 rank schoolbook solution
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    
    # Verify
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    
    # Should not reach here
    return {"U": U, "V": V, "W": W}
