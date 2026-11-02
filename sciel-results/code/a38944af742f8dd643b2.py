"""Tensor decomposition for 2x2 * 2x13 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    n, m, p = 2, 2, 13

    # Schoolbook fallback (rank 52)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U_school[i * m + k, col] = 1.0
                V_school[k * p + j, col] = 1.0
                W_school[j * n + i, col] = 1.0
                col += 1

    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    try:
        # Strassen-based decomposition: 6 blocks of 2x2 (rank 7 each) + 1 block of 2x1 (rank 4)
        rank_strassen = 6 * 7 + 4
        U = np.zeros((n * m, rank_strassen))
        V = np.zeros((m * p, rank_strassen))
        W = np.zeros((p * n, rank_strassen))

        # Strassen terms for 2x2x2x2: (u_t, v_t, w_t)
        # A indices: [a00, a01, a10, a11] corresponding to i=0,k=0; i=0,k=1; i=1,k=0; i=1,k=1
        # B indices: [b00, b01, b10, b11] corresponding to k=0,j=0; k=0,j=1; k=1,j=0; k=1,j=1
        # C indices: [c00, c10, c01, c11] corresponding to j=0,i=0; j=0,i=1; j=1,i=0; j=1,i=1
        # Note: W index is j*n+i. For n=2:
        # j=0, i=0 -> 0
        # j=0, i=1 -> 1
        # j=1, i=0 -> 2
        # j=1, i=1 -> 3
        # The standard Strassen algorithm computes:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C10 = M2 + M4
        # C01 = M1 + M3
        # C11 = M1 - M2 + M3 + M5

        # Let's map these to vectors.
        # A vector: [A00, A01, A10, A11]
        # B vector: [B00, B01, B10, B11]
        # C vector: [C00, C10, C01, C11] (Order: j=0,i=0; j=0,i=1; j=1,i=0; j=1,i=1)
        
        strassen_terms = [
            # M1
            (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
            # M2
            (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0])),
            # M3
            (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 0, 1, 0])),
            # M4
            (np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0])),
            # M5
            (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 0, 0, 1])),
            # M6
            (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 0])), # Wait, M6 doesn't appear in C?
            # Let's re-verify the standard Strassen output combination.
            # C00 = M1 + M4 - M5 + M7
            # C10 = M2 + M4
            # C01 = M1 + M3
            # C11 = M1 - M2 + M3 + M5
            
            # Let's re-derive the W vectors based on the standard formula.
            # W[M1] = [1, 0, 0, 1] (C00+=1, C11+=1)
            # W[M2] = [0, 1, 0, -1] (C10+=1, C11-=-1)
            # W[M3] = [0, 0, 1, 1] (C01+=1, C11+=1)
            # W[M4] = [1, 1, 0, 0] (C00+=1, C10+=1)
            # W[M5] = [-1, 0, 0, 1] (C00-=-1, C11+=1)
            # W[M6] = [0, 0, 0, 0] ? No, M6 is usually (A10-A00)(B00+B01).
            # Let's check C10 = M2 + M4. C11 = M1 - M2 + M3 + M5.
            # Where is M6 and M7?
            # Ah, the standard 7-term Strassen is:
            # P1 = (A00+A11)(B00+B11)
            # P2 = (A10+A11)B00
            # P3 = A00(B01-B11)
            # P4 = A11(B10-B00)
            # P5 = (A00+A01)B11
            # P6 = (A10-A00)(B00+B01)
            # P7 = (A01-A11)(B10+B11)
            
            # C00 = P1 + P4 - P5 + P7
            # C10 = P2 + P4
            # C01 = P1 + P3
            # C11 = P1 - P2 + P3 + P5
            
            # Let's check if P6 is used. It is NOT used in the final sum for 2x2.
            # Wait, standard Strassen uses 7 multiplications.
            # Let's re-read the standard algorithm.
            # Yes, P6 is often defined but not used in the final reconstruction?
            # No, that's wrong. The 7th multiplication is P7.
            # Let's check the rank. It is 7.
            # The terms are P1..P7.
            # P1: U=[1,0,0,1], V=[1,0,0,1]
            # P2: U=[0,0,1,1], V=[1,0,0,0]
            # P3: U=[1,0,0,0], V=[0,1,0,-1]
            # P4: U=[0,0,0,1], V=[-1,0,1,0]
            # P5: U=[1,1,0,0], V=[0,0,0,1]
            # P6: U=[-1,0,1,0], V=[1,1,0,0]  <-- This one is (A10-A00)(B00+B01)
            # P7: U=[0,1,0,-1], V=[0,0,1,1]   <-- This one is (A01-A11)(B10+B11)
            
            # Reconstruction:
            # C00 = P1 + P4 - P5 + P7
            # C10 = P2 + P4
            # C01 = P1 + P3
            # C11 = P1 - P2 + P3 + P5
            
            # Let's check P6. It is not in the sum.
            # Is the rank really 7 if one term is unused?
            # If P6 is unused, the rank would be 6? No, Strassen is 7.
            # Ah, I might be mixing up the specific definitions.
            # Let's use the verified terms from the Current Program which worked.
        ]
        
        # The Current Program used these terms and verified they work.
        # I will copy them exactly.
        strassen_terms = [
            (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
            (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1])),
            (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1])),
            (np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0])),
            (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0])),
            (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])),
            (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0])),
        ]

        col = 0
        for block in range(6):
            j0 = block * 2
            j1 = block * 2 + 1
            
            for t in range(7):
                u_local, v_local, w_local = strassen_terms[t]
                
                # A: global index = i*m + k
                U[0, col] = u_local[0]
                U[1, col] = u_local[1]
                U[2, col] = u_local[2]
                U[3, col] = u_local[3]
                
                # B: global index = k*p + j
                V[0 * p + j0, col] = v_local[0]
                V[0 * p + j1, col] = v_local[1]
                V[1 * p + j0, col] = v_local[2]
                V[1 * p + j1, col] = v_local[3]
                
                # C: global index = j*n + i
                W[j0 * n + 0, col] = w_local[0]
                W[j0 * n + 1, col] = w_local[1]
                W[j1 * n + 0, col] = w_local[2]
                W[j1 * n + 1, col] = w_local[3]
                
                col += 1
        
        # Last column (j=12): schoolbook 2x2 x 2x1, rank 4
        j_last = 12
        for i in range(n):
            for k in range(m):
                U[i * m + k, col] = 1.0
                V[k * p + j_last, col] = 1.0
                W[j_last * n + i, col] = 1.0
                col += 1
        
        # Verify correctness
        T_target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    T_target[i * m + k, k * p + j, j * n + i] = 1.0
                    
        T_approx = np.einsum('it,jt,kt->ijk', U, V, W)
        error = np.max(np.abs(T_approx - T_target))
        
        if error < 1e-10:
            best_rank = rank_strassen
            best_U, best_V, best_W = U, V, W
    
    except Exception:
        pass
    
    if time.time() > deadline:
        pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
