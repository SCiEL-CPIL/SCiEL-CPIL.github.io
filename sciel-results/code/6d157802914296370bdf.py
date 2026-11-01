import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank = 56

    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    # Strassen coefficients for 2x2x2 multiplication
    # M_s = (sum u_ij A_ij) * (sum v_jk B_jk)
    # C_ik = sum w_ik M_s
    
    # u[s][i][j], v[s][j][k], w[s][i][k]
    # s=0: M0 = (A00+A11)(B00+B11)
    u0 = np.array([[1, 0], [0, 1]])
    v0 = np.array([[1, 0], [0, 1]])
    w0 = np.array([[1, 0], [0, 1]]) # C00+=M0, C11+=M0
    
    # s=1: M1 = (A10+A11)B00
    u1 = np.array([[0, 0], [1, 1]])
    v1 = np.array([[1, 0], [0, 0]])
    w1 = np.array([[0, 0], [1, -1]]) # C10+=M1, C11-=M1
    
    # s=2: M2 = A00(B10-B11)
    u2 = np.array([[1, 0], [0, 0]])
    v2 = np.array([[0, 0], [1, -1]])
    w2 = np.array([[0, 1], [0, 1]]) # C01+=M2, C11+=M2
    
    # s=3: M3 = A11(B01-B00)
    u3 = np.array([[0, 0], [0, 1]])
    v3 = np.array([[-1, 1], [0, 0]])
    w3 = np.array([[1, 0], [1, 0]]) # C00+=M3, C10+=M3
    
    # s=4: M4 = (A00+A01)B11
    u4 = np.array([[1, 1], [0, 0]])
    v4 = np.array([[0, 0], [0, 1]])
    w4 = np.array([[-1, 1], [0, 0]]) # C00-=M4, C01+=M4
    
    # s=5: M5 = (A10-A00)(B00+B10)
    u5 = np.array([[-1, 0], [1, 0]])
    v5 = np.array([[1, 0], [1, 0]])
    w5 = np.array([[0, 0], [0, 1]]) # C11+=M5
    
    # s=6: M6 = (A01-A11)(B11+B01)
    u6 = np.array([[0, 1], [-1, 0]])
    v6 = np.array([[0, 1], [0, 1]])
    w6 = np.array([[1, 0], [0, 0]]) # C00+=M6

    us = [u0, u1, u2, u3, u4, u5, u6]
    vs = [v0, v1, v2, v3, v4, v5, v6]
    ws = [w0, w1, w2, w3, w4, w5, w6]

    # Iterate over 2 sides (L, R) and 4 groups of columns
    for side in range(2):
        for group in range(4):
            # Base indices for this sub-problem
            J_base = 2 * side  # 0 for L, 2 for R
            K_base = 2 * group # 0, 2, 4, 6
            
            for s in range(7):
                r = (side * 4 + group) * 7 + s
                
                # Fill U
                for i in range(2):
                    for j in range(2):
                        I = i
                        J = J_base + j
                        U[I * m + J, r] = us[s][i][j]
                        
                # Fill V
                for j in range(2):
                    for k in range(2):
                        J = J_base + j
                        K = K_base + k
                        V[J * p + K, r] = vs[s][j][k]
                        
                # Fill W
                for k in range(2):
                    for i in range(2):
                        K = K_base + k
                        I = i
                        W[K * n + I, r] = ws[s][i][k]

    # Verify and return
    # The problem asks for integer/half-integer entries. Our entries are integers.
    # We can just return the lists.
    
    if time.time() > deadline:
        # Should not happen, but just in case
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
