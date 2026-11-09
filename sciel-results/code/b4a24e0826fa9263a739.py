"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb, "rank": rank_schoolbook}
    
    # Try block decomposition: split m=5 into 2+3
    # C = A[:, :2] * B[:2, :] + A[:, 2:] * B[2:, :]
    # This gives rank(3,2,5) + rank(3,3,5)
    
    # For (3,2,5): use schoolbook = 30
    # For (3,3,5): try to use a lower-rank construction
    
    # Let's try to find a decomposition for (3,3,5) with rank < 45
    # Using Strassen for the 3x3 part: rank(3,3,3) = 23
    # But we need (3,3,5), not (3,3,3)
    
    # Alternative: split p=5 into 2+3 for the (3,3,5) part
    # rank(3,3,5) = rank(3,3,2) + rank(3,3,3)
    # rank(3,3,2) <= 18 (schoolbook)
    # rank(3,3,3) <= 23 (Strassen)
    # Total: 41
    
    # So total rank = 30 + 41 = 71, which is better than 75!
    
    # Let me construct this explicitly
    
    # First, construct (3,2,5) schoolbook
    U_325 = [[0] * 30 for _ in range(6)]
    V_325 = [[0] * 30 for _ in range(10)]
    W_325 = [[0] * 30 for _ in range(15)]
    col = 0
    for i in range(3):
        for j in range(2):
            for k in range(5):
                U_325[i*2+j][col] = 1
                V_325[j*5+k][col] = 1
                W_325[k*3+i][col] = 1
                col += 1
    
    # Now construct (3,3,5) using split p=5 into 2+3
    # (3,3,2) schoolbook + (3,3,3) Strassen
    
    # (3,3,2) schoolbook: rank = 18
    U_332 = [[0] * 18 for _ in range(9)]
    V_332 = [[0] * 18 for _ in range(6)]
    W_332 = [[0] * 18 for _ in range(6)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U_332[i*3+j][col] = 1
                V_332[j*2+k][col] = 1
                W_332[k*3+i][col] = 1
                col += 1
    
    # (3,3,3) Strassen: rank = 23
    # Strassen's algorithm for 3x3 matrix multiplication
    # This is more complex than 2x2 Strassen
    # Let me use a known construction
    
    # Actually, let me just use schoolbook for (3,3,3) for now: rank = 27
    # Total for (3,3,5): 18 + 27 = 45
    # Total overall: 30 + 45 = 75 (no improvement)
    
    # Let me try a different approach: use the fact that we can share some terms
    
    # For now, let's just return the schoolbook solution
    if time.time() > deadline - 5:
        return {"U": best["U"], "V": best["V"], "W": best["W"]}
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
