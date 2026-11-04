"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook rank n*m*p = 56, then attempt Strassen-based reduction
    # For 2x4x7, we can decompose as 2x4x4 (rank 7 per 2x2x2 block, 4 blocks = 28) + 2x4x3 (rank 7+4=11) = 39
    # But simpler: use block Strassen for 2x4x4 part (rank 7*4=28) and direct for 2x4x3 (rank 4*3=12) = 40
    # Even simpler verified approach: schoolbook rank 56
    
    # First, build a correct schoolbook decomposition as fallback
    def build_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    # Build Strassen-based decomposition for 2x4x7
    # Split into 2x4x4 and 2x4x3
    # For 2x4x4: use 4 independent 2x2x2 Strassen decompositions (rank 7 each) = 28
    # For 2x4x3: use 3 independent 2x4x1 direct decompositions (rank 4 each) = 12
    # Total: 40
    
    rank = 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_entries, v_entries, w_entries):
        nonlocal col
        for idx, val in u_entries:
            if 0 <= idx < len(U):
                U[idx][col] += val
        for idx, val in v_entries:
            if 0 <= idx < len(V):
                V[idx][col] += val
        for idx, val in w_entries:
            if 0 <= idx < len(W):
                W[idx][col] += val
        col += 1
    
    # 2x4x4 part: 4 Strassen 2x2x2 blocks
    # Block (A rows 0-1, B cols 0-1) -> C cols 0-1
    # Strassen for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    add_term([(0, 1), (5, 1)], [(0, 1), (7, 1)], [(0, 1), (3, 1)])
    # M2 = A10(B00+B10)  
    add_term([(4, 1)], [(0, 1), (6, 1)], [(1, 1)])
    # M3 = A00(B01-B11)
    add_term([(0, 1)], [(1, 1), (7, -1)], [(2, 1)])
    # M4 = A11(B10-B00)
    add_term([(5, 1)], [(6, 1), (0, -1)], [(0, 1), (1, 1)])
    # M5 = (A00+A01)B11
    add_term([(0, 1), (1, 1)], [(7, 1)], [(0, -1), (2, 1)])
    # M6 = (A10-A00)(B00+B01)
    add_term([(4, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])
    # M7 = (A01-A11)(B10+B11)
    add_term([(1, 1), (5, -1)], [(6, 1), (7, 1)], [(0, 1)])
    
    # Block (A rows 0-1, B cols 2-3) -> C cols 2-3
    add_term([(0, 1), (5, 1)], [(14, 1), (21, 1)], [(4, 1), (7, 1)])
    add_term([(4, 1)], [(14, 1), (20, 1)], [(5, 1)])
    add_term([(0, 1)], [(15, 1), (21, -1)], [(6, 1)])
    add_term([(5, 1)], [(20, 1), (14, -1)], [(4, 1), (5, 1)])
    add_term([(0, 1), (1, 1)], [(21, 1)], [(4, -1), (6, 1)])
    add_term([(4, 1), (0, -1)], [(14, 1), (15, 1)], [(7, 1)])
    add_term([(1, 1), (5, -1)], [(20, 1), (21, 1)], [(4, 1)])
    
    # Block (A rows 2-3, B cols 0-1) -> C cols 0-1
    add_term([(2, 1), (7, 1)], [(0, 1), (7, 1)], [(0, 1), (3, 1)])
    add_term([(6, 1)], [(0, 1), (6, 1)], [(1, 1)])
    add_term([(2, 1)], [(1, 1), (7, -1)], [(2, 1)])
    add_term([(7, 1)], [(6, 1), (0, -1)], [(0, 1), (1, 1)])
    add_term([(2, 1), (3, 1)], [(7, 1)], [(0, -1), (2, 1)])
    add_term([(6, 1), (2, -1)], [(0, 1), (1, 1)], [(3, 1)])
    add_term([(3, 1), (7, -1)], [(6, 1), (7, 1)], [(0, 1)])
    
    # Block (A rows 2-3, B cols 2-3) -> C cols 2-3
    add_term([(2, 1), (7, 1)], [(14, 1), (21, 1)], [(4, 1), (7, 1)])
    add_term([(6, 1)], [(14, 1), (20, 1)], [(5, 1)])
    add_term([(2, 1)], [(15, 1), (21, -1)], [(6, 1)])
    add_term([(7, 1)], [(20, 1), (14, -1)], [(4, 1), (5, 1)])
    add_term([(2, 1), (3, 1)], [(21, 1)], [(4, -1), (6, 1)])
    add_term([(6, 1), (2, -1)], [(14, 1), (15, 1)], [(7, 1)])
    add_term([(3, 1), (7, -1)], [(20, 1), (21, 1)], [(4, 1)])
    
    # 2x4x3 part: 3 direct 2x4x1 decompositions
    # C col 4: sum_j A[i,j]*B[j,4]
    for j in range(4):
        add_term([(j, 1), (4+j, 1)], [(j, 1)], [(8, 1), (9, 1)])
    
    # C col 5: sum_j A[i,j]*B[j,5]  
    for j in range(4):
        add_term([(j, 1), (4+j, 1)], [(j+4, 1)], [(10, 1), (11, 1)])
    
    # C col 6: sum_j A[i,j]*B[j,6]
    for j in range(4):
        add_term([(j, 1), (4+j, 1)], [(j+8, 1)], [(12, 1), (13, 1)])
    
    # Verify col
    if col != rank:
        # Fall back to schoolbook
        U, V, W, rank = build_schoolbook()
        return {"U": U, "V": V, "W": W}
    
    # Verify the decomposition
    if time.time() - start_time < DEADLINE - 5:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = float(U_np[i*m+j] @ (V_np[j*p+k] * W_np[k*n+i]))
                    if abs(lhs - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            U, V, W, rank = build_schoolbook()
            return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
