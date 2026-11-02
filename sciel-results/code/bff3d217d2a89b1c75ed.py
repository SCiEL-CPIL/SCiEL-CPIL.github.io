"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    # Rank is 42 based on Strassen decomposition
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen terms definitions
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # For each group g (0, 1) and sub-block t (0, 1, 2)
    for g in range(2):
        for t in range(3):
            for s in range(7):
                # Check time
                if time.time() > deadline:
                    break
                
                # Initialize local vectors for this term
                u_vec = [0] * (n * m)
                v_vec = [0] * (m * p)
                w_vec = [0] * (p * n)
                
                # A indices
                a_00 = 0 * m + (2 * g)
                a_01 = 0 * m + (2 * g + 1)
                a_10 = 1 * m + (2 * g)
                a_11 = 1 * m + (2 * g + 1)
                
                # B indices
                b_00 = (2 * g) * p + (2 * t)
                b_01 = (2 * g) * p + (2 * t + 1)
                b_10 = (2 * g + 1) * p + (2 * t)
                b_11 = (2 * g + 1) * p + (2 * t + 1)
                
                # C indices
                c_00 = (2 * t) * n + 0
                c_01 = (2 * t + 1) * n + 0
                c_10 = (2 * t) * n + 1
                c_11 = (2 * t + 1) * n + 1
                
                if s == 0:
                    # M0 = (x00+x11)(y00+y11)
                    u_vec[a_00] = 1; u_vec[a_11] = 1
                    v_vec[b_00] = 1; v_vec[b_11] = 1
                    w_vec[c_00] = 1; w_vec[c_11] = 1
                elif s == 1:
                    # M1 = (x10+x11)y00
                    u_vec[a_10] = 1; u_vec[a_11] = 1
                    v_vec[b_00] = 1
                    w_vec[c_10] = 1; w_vec[c_11] = -1
                elif s == 2:
                    # M2 = x00(y01-y11)
                    u_vec[a_00] = 1
                    v_vec[b_01] = 1; v_vec[b_11] = -1
                    w_vec[c_01] = 1; w_vec[c_11] = 1
                elif s == 3:
                    # M3 = x11(y10-y00)
                    u_vec[a_11] = 1
                    v_vec[b_10] = 1; v_vec[b_00] = -1
                    w_vec[c_00] = 1; w_vec[c_10] = 1
                elif s == 4:
                    # M4 = (x00+x01)y11
                    u_vec[a_00] = 1; u_vec[a_01] = 1
                    v_vec[b_11] = 1
                    w_vec[c_00] = -1; w_vec[c_01] = 1
                elif s == 5:
                    # M5 = (x10-x00)(y00+y01)
                    u_vec[a_10] = 1; u_vec[a_00] = -1
                    v_vec[b_00] = 1; v_vec[b_01] = 1
                    w_vec[c_11] = 1
                elif s == 6:
                    # M6 = (x01-x11)(y10+y11)
                    u_vec[a_01] = 1; u_vec[a_11] = -1
                    v_vec[b_10] = 1; v_vec[b_11] = 1
                    w_vec[c_00] = 1
                
                for r in range(n * m):
                    U[r][col] = u_vec[r]
                for r in range(m * p):
                    V[r][col] = v_vec[r]
                for r in range(p * n):
                    W[r][col] = w_vec[r]
                    
                col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
