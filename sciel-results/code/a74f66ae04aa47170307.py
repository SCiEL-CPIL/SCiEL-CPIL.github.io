import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    _start = time.time()
    _deadline = _start + 130  # 20s safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    # Try to find a lower rank decomposition
    # First, verify the known rank-22 solution works, then try to improve
    
    def build_rank22():
        rank = 22
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val

        # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

        # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

        # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

        return U, V, W, rank
    
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-10:
                        return False
        return True
    
    def is_half_integer(val):
        return val == int(val) or abs(val * 2 - round(val * 2)) < 1e-10
    
    def check_valid(U, V, W):
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if not is_half_integer(val):
                        return False
        return True
    
    # Start with rank-22
    U, V, W, rank = build_rank22()
    
    if not verify(U, V, W, rank):
        raise RuntimeError("Rank-22 decomposition failed verification")
    
    if not check_valid(U, V, W):
        raise RuntimeError("Rank-22 has non-half-integer entries")
    
    # Try to reduce rank by checking if any rank term is redundant
    # A term is redundant if removing it still gives a valid decomposition
    # This is expensive, so we'll do a limited search
    
    best_rank = rank
    best_U, best_V, best_W = U[:], V[:], W[:]
    
    # Try removing each term one at a time
    for r_remove in range(rank):
        if time.time() > _deadline - 30:
            break
        
        U_test = [row[:] for row in U]
        V_test = [row[:] for row in V]
        W_test = [row[:] for row in W]
        
        # Remove term r_remove
        for i in range(n * m):
            U_test[i][r_remove] = 0
        for j in range(m * p):
            V_test[j][r_remove] = 0
        for k in range(p * n):
            W_test[k][r_remove] = 0
        
        # Check if still valid (only check non-zero entries for speed)
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        if r == r_remove:
                            continue
                        s += U_test[i*m+j][r] * V_test[j*p+k][r] * W_test[k*n+i][r]
                    if abs(s - 1) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # This term is redundant, try to remove it and see if we can go lower
            new_rank = rank - 1
            U_new = [[U_test[i][r] for r in range(rank) if r != r_remove] for i in range(n*m)]
            V_new = [[V_test[j][r] for r in range(rank) if r != r_remove] for j in range(m*p)]
            W_new = [[W_test[k][r] for r in range(rank) if r != r_remove] for k in range(p*n)]
            
            if verify(U_new, V_new, new_rank) and check_valid(U_new, V_new, W_new):
                best_rank = new_rank
                best_U, best_V, best_W = U_new, V_new, W_new
                rank = new_rank
                U, V, W = U_new, V_new, W_new
    
    return {"U": best_U, "V": best_V, "W": best_W}
