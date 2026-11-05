import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    DEADLINE = 145  # seconds safety margin before 150s kill

    # Strategy: Strassen-based 4x4x4 (rank 49) + schoolbook 4x4x1 (rank 16) = rank 65
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65

    # Strassen 2x2x2 base case: 7 multiplications
    # For 2x2 matrices, the 7 Strassen products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01-B11)
    # M4 = (A11)(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # The 2x2x2 Strassen tensor decomposition has 7 terms
    # Each term: (u_indices, v_indices, w_indices)
    # u is indexed by (i,k) for 2x2 A, v by (k,j) for 2x2 B, w by (j,i) for 2x2 C
    
    # Let's define the 7 terms for 2x2x2:
    # Term 0: M1 = (A00+A11)(B00+B11)
    #   U: (0,0):1, (1,1):1
    #   V: (0,0):1, (1,1):1
    #   W: (0,0):1, (1,1):1
    # Term 1: M2 = (A10+A11)(B00)
    #   U: (1,0):1, (1,1):1
    #   V: (0,0):1
    #   W: (1,0):1
    # Term 2: M3 = (A00)(B01-B11)
    #   U: (0,0):1
    #   V: (0,1):1, (1,1):-1
    #   W: (0,0):1
    # Term 3: M4 = (A11)(B10-B00)
    #   U: (1,1):1
    #   V: (1,0):1, (0,0):-1
    #   W: (1,1):1
    # Term 4: M5 = (A00+A01)(B11)
    #   U: (0,0):1, (0,1):1
    #   V: (1,1):1
    #   W: (0,1):1
    # Term 5: M6 = (A10-A00)(B00+B01)
    #   U: (1,0):1, (0,0):-1
    #   V: (0,0):1, (0,1):1
    #   W: (1,0):-1, (1,1):1
    # Term 6: M7 = (A01-A11)(B10+B11)
    #   U: (0,1):1, (1,1):-1
    #   V: (1,0):1, (1,1):1
    #   W: (0,1):-1, (1,1):1

    strassen_2x2 = [
        # (u_dict, v_dict, w_dict) where keys are (row,col) tuples
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,0,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(1,1,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,0,-1),(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,1,-1),(1,1,1)]),
    ]

    col = 0
    # 4x4x4 Strassen: 7 outer Strassen steps, each with 7 inner 2x2x2 Strassen steps
    for outer in range(7):
        outer_u, outer_v, outer_w = strassen_2x2[outer]
        for inner in range(7):
            inner_u, inner_v, inner_w = strassen_2x2[inner]
            
            # Map outer block indices to global 4x4 indices
            # Outer block (br, bc) maps to global rows br*2..br*2+1, cols bc*2..bc*2+1
            # For U: outer (i,k) -> global (br*2+i, bc*2+k) where br,bc from outer_u
            # Actually, let's be more careful.
            # A is 4x4, split into 2x2 blocks of 2x2:
            # A[0:2,0:2] = A00, A[0:2,2:4] = A01, A[2:4,0:2] = A10, A[2:4,2:4] = A11
            # Outer U term (i,k) means: use A[i,k] at block level
            # So global row = i*2 + inner_i, global col = k*2 + inner_k
            
            for (oi, ok, oc) in outer_u:
                for (ii, ik, ic) in inner_u:
                    gi = oi * 2 + ii
                    gk = ok * 2 + ik
                    U[gi * 4 + gk][col] += oc * ic
            
            for (ok, oj, oc) in outer_v:
                for (ik, ij, ic) in inner_v:
                    gk = ok * 2 + ik
                    gj = oj * 2 + ij
                    V[gk * 5 + gj][col] += oc * ic
            
            for (oj, oi, oc) in outer_w:
                for (ij, ii, ic) in inner_w:
                    gj = oj * 2 + ij
                    gi = oi * 2 + ii
                    W[gj * 4 + gi][col] += oc * ic
            
            col += 1

    # Part 2: 4x4x1 schoolbook (rank 16)
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1

    # Verify the decomposition is correct
    if time.time() - start_time < DEADLINE - 10:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        rng = np.random.RandomState(42)
        A_test = rng.rand(4, 4)
        B_test = rng.rand(4, 5)
        
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((4, 5))
        for alpha in range(rank):
            a_part = np.dot(A_test.flatten(), U_arr[:, alpha])
            b_part = np.dot(B_test.flatten(), V_arr[:, alpha])
            C_computed += a_part * b_part * W_arr.reshape(5, 4).T
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # If verification fails, return a simple schoolbook decomposition (rank 80)
            rank = 80
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(4):
                for k in range(4):
                    for j in range(5):
                        U[i * 4 + k][col] = 1
                        V[k * 5 + j][col] = 1
                        W[j * 4 + i][col] = 1
                        col += 1

    return {"U": U, "V": V, "W": W}
