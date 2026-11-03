import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    DEADLINE = 130  # Safety margin before 150s kill
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    
    def elapsed():
        return time.time() - start_time
    
    def time_remaining():
        return DEADLINE - elapsed()
    
    # Laderman 3x3x3 decomposition (rank 23)
    def laderman_3x3():
        U = np.zeros((9, 23))
        V = np.zeros((9, 23))
        W = np.zeros((9, 23))
        
        # M1 = (A11 + A12 + A21 + A22) * (B11 + B12 + B21 + B22)
        U[3, 0] = 1; U[4, 0] = 1; U[6, 0] = 1; U[7, 0] = 1
        V[3, 0] = 1; V[4, 0] = 1; V[6, 0] = 1; V[7, 0] = 1
        W[3, 0] = 1; W[4, 0] = 1; W[6, 0] = 1; W[7, 0] = 1
        
        # M2 = (A11 + A12 - A21 - A22) * (B11 - B12 + B21 - B22)
        U[3, 1] = 1; U[4, 1] = 1; U[6, 1] = -1; U[7, 1] = -1
        V[3, 1] = 1; V[4, 1] = -1; V[6, 1] = 1; V[7, 1] = -1
        W[3, 1] = 1; W[4, 1] = -1; W[6, 1] = 1; W[7, 1] = -1
        
        # M3 = (A11 - A12 + A21 - A22) * (B11 + B12 - B21 - B22)
        U[3, 2] = 1; U[4, 2] = -1; U[6, 2] = 1; U[7, 2] = -1
        V[3, 2] = 1; V[4, 2] = 1; V[6, 2] = -1; V[7, 2] = -1
        W[3, 2] = 1; W[4, 2] = 1; W[6, 2] = -1; W[7, 2] = -1
        
        # M4 = (A11 - A12 - A21 + A22) * (B11 - B12 - B21 + B22)
        U[3, 3] = 1; U[4, 3] = -1; U[6, 3] = -1; U[7, 3] = 1
        V[3, 3] = 1; V[4, 3] = -1; V[6, 3] = -1; V[7, 3] = 1
        W[3, 3] = 1; W[4, 3] = -1; W[6, 3] = -1; W[7, 3] = 1
        
        # M5 = (A11 + A21) * (B11 + B21)
        U[3, 4] = 1; U[6, 4] = 1
        V[3, 4] = 1; V[6, 4] = 1
        W[3, 4] = 1; W[6, 4] = 1
        
        # M6 = (A11 + A22) * (B11 + B22)
        U[3, 5] = 1; U[7, 5] = 1
        V[3, 5] = 1; V[7, 5] = 1
        W[3, 5] = 1; W[7, 5] = 1
        
        # M7 = (A12 + A21) * (B12 + B21)
        U[4, 6] = 1; U[6, 6] = 1
        V[4, 6] = 1; V[6, 6] = 1
        W[4, 6] = 1; W[6, 6] = 1
        
        # M8 = (A12 + A22) * (B12 + B22)
        U[4, 7] = 1; U[7, 7] = 1
        V[4, 7] = 1; V[7, 7] = 1
        W[4, 7] = 1; W[7, 7] = 1
        
        # M9 = (A11 + A12) * (B11 + B12)
        U[3, 8] = 1; U[4, 8] = 1
        V[3, 8] = 1; V[4, 8] = 1
        W[3, 8] = 1; W[4, 8] = 1
        
        # M10 = (A11 + A21) * (B11 + B21) - same as M5, but we need 23 terms.
        # The Laderman algorithm has specific 23 terms. I will use a known valid set.
        # Actually, let's just use schoolbook for 3x3 if Laderman is too complex to verify quickly.
        # But 23 < 27.
        
        # Let's use a simpler, verified Laderman set.
        # Reference: https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm#Laderman%27s_algorithm
        # The Wikipedia article lists 23 terms.
        
        # Term 1: (A11+A12+A21+A22)(B11+B12+B21+B22)
        # Term 2: (A11+A12-A21-A22)(B11-B12+B21-B22)
        # Term 3: (A11-A12+A21-A22)(B11+B12-B21-B22)
        # Term 4: (A11-A12-A21+A22)(B11-B12-B21+B22)
        # Term 5: (A11+A21)(B11+B21)
        # Term 6: (A11+A22)(B11+B22)
        # Term 7: (A12+A21)(B12+B21)
        # Term 8: (A12+A22)(B12+B22)
        # Term 9: (A11+A12)(B11+B12)
        # Term 10: (A11+A21)(B11+B21) -> Duplicate of 5? No, the coefficients in C are different.
        
        # This is getting risky. Let's stick to the 4x4x4 Strassen (rank 49) + schoolbook (36) = 85.
        # Or use 3x3x3 Laderman (23) + schoolbook (54) = 77.
        # I will implement the 3x3x3 Laderman carefully.
        
        # Actually, a simpler improvement is to use Strassen for 4x4x4 (49) and optimize the rest.
        # The rest was 36.
        # Can we reduce the rest?
        # Part 2: j in 0..3, k=4 (4x4x1) - rank 16
        # Part 3: j=4, k in 0..4 (4x1x5) - rank 20
        # Total 36.
        
        # What if we use Strassen for 2x2x2 (7) on the 4x4x4?
        # We did that.
        
        # Let's try to implement Laderman 3x3x3 correctly.
        # If it fails, fallback to 85.
        
        # I will use the 4x4x4 Strassen (49) + schoolbook (36) = 85 for safety, 
        # but try to optimize the schoolbook part if possible.
        # The schoolbook part is:
        # C[i,4] = sum_{j=0}^3 A[i,j] B[j,4] (16 terms)
        # C[i,k] = A[i,4] B[4,k] (20 terms)
        # This is minimal for the non-core parts if we use a 4x4 core.
        
        # So 85 is likely the best for this family.
        # I will return the 85 rank solution.
        
        pass
    
    # Use the 4x4x4 Strassen + schoolbook solution (rank 85)
    U2, V2, W2 = None, None, None
    def strassen_2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1
        
        return U, V, W
    
    U2, V2, W2 = strassen_2x2()
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    strassen_terms = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
    ]
    
    for l in range(7):
        A_blocks, B_blocks, C_blocks = strassen_terms[l]
        for s in range(7):
            col = l * 7 + s
            for (br, bc, coeff) in A_blocks:
                for ii in range(4):
                    val = U2[ii, s]
                    if val != 0:
                        br_local = ii // 2
                        bc_local = ii % 2
                        r = br * 2 + br_local
                        c = bc * 2 + bc_local
                        U4[r * 4 + c, col] += coeff * val
            for (br, bc, coeff) in B_blocks:
                for ii in range(4):
                    val = V2[ii, s]
                    if val != 0:
                        br_local = ii // 2
                        bc_local = ii % 2
                        r = br * 2 + br_local
                        c = bc * 2 + bc_local
                        V4[r * 4 + c, col] += coeff * val
            for (br, bc, coeff) in C_blocks:
                for ii in range(4):
                    val = W2[ii, s]
                    if val != 0:
                        br_local = ii // 2
                        bc_local = ii % 2
                        r = br * 2 + br_local
                        c = bc * 2 + bc_local
                        W4[r * 4 + c, col] += coeff * val
    
    def verify_core():
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    val = np.dot(U4[i*4+j, :], V4[j*4+k, :] * W4[k*4+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    core_ok = verify_core()
    
    if not core_ok:
        U4 = np.zeros((16, 64))
        V4 = np.zeros((16, 64))
        W4 = np.zeros((16, 64))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U4[i*4+j, col] = 1
                    V4[j*4+k, col] = 1
                    W4[k*4+i, col] = 1
                    col += 1
        core_rank = 64
    else:
        core_rank = 49
    
    total_rank = core_rank + 16 + 20
    
    U_full = np.zeros((20, total_rank))
    V_full = np.zeros((25, total_rank))
    W_full = np.zeros((20, total_rank))
    
    for i in range(4):
        for j in range(4):
            U_full[i * 5 + j, :core_rank] = U4[i * 4 + j, :]
    for j in range(4):
        for k in range(4):
            V_full[j * 5 + k, :core_rank] = V4[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W_full[k * 4 + i, :core_rank] = W4[k * 4 + i, :]
    
    for i in range(4):
        for j in range(4):
            r = core_rank + i * 4 + j
            U_full[i * 5 + j, r] = 1
            V_full[j * 5 + 4, r] = 1
            W_full[4 * 4 + i, r] = 1
    
    for i in range(4):
        for k in range(5):
            r = core_rank + 16 + i * 5 + k
            U_full[i * 5 + 4, r] = 1
            V_full[4 * 5 + k, r] = 1
            W_full[k * 4 + i, r] = 1
    
    def verify_full():
        for i in range(4):
            for j in range(5):
                for k in range(5):
                    val = np.dot(U_full[i * 5 + j, :], V_full[j * 5 + k, :] * W_full[k * 4 + i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if verify_full():
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    else:
        rank = 100
        U = np.zeros((20, rank))
        V = np.zeros((25, rank))
        W = np.zeros((20, rank))
        col = 0
        for i in range(4):
            for j in range(5):
                for k in range(5):
                    U[i * 5 + j, col] = 1
                    V[j * 5 + k, col] = 1
                    W[k * 4 + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
