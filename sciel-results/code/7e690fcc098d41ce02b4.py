"""Matrix multiplication 4x5x6 - rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 6
    deadline = time.time() + 140  # safety margin before 150s kill
    
    num_U, num_V, num_W = n * m, m * p, p * n  # 20, 30, 24
    
    def verify_full(U, V, W):
        """Full verification: all valid tensor entries must match, invalid must be zero."""
        rank = U.shape[1]
        valid = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid.add((i*m+j, j*p+k, k*n+i))
        
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Check all valid entries
        for (a, b, c) in valid:
            s = np.sum(U_arr[a,:] * V_arr[b,:] * W_arr[c,:])
            if abs(s - 1) > 1e-9:
                return False
        
        # Check invalid entries (sample for speed)
        import random
        rng = random.Random(42)
        checked = 0
        while checked < 1000:
            a = rng.randint(0, num_U-1)
            b = rng.randint(0, num_V-1)
            c = rng.randint(0, num_W-1)
            if (a,b,c) in valid:
                continue
            s = np.sum(U_arr[a,:] * V_arr[b,:] * W_arr[c,:])
            if abs(s) > 1e-9:
                return False
            checked += 1
        
        return True
    
    # Schoolbook fallback
    rank_school = n * m * p  # 120
    U_school = np.zeros((num_U, rank_school), dtype=float)
    V_school = np.zeros((num_V, rank_school), dtype=float)
    W_school = np.zeros((num_W, rank_school), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_factors = (U_school, V_school, W_school)
    best_rank = rank_school
    
    # --- Block decomposition approach ---
    # Split j into {0,1,2,3} and {4}, k into {0,1,2,3,4,5}
    # C = A*B, A is 4x5, B is 5x6
    # C = A_left * B_top + A_right * B_bottom
    # where A_left is 4x4, A_right is 4x1, B_top is 4x6, B_bottom is 1x6
    
    # Part 1: A_right * B_bottom (4x1 * 1x6)
    # This contributes to C[i,k] = A_right[i,0] * B_bottom[0,k]
    # In tensor terms: for i in 0..3, k in 0..5: T[i*5+4, 24+k, k*4+i] = 1
    # Rank 4 decomposition: one term per i
    # U[i*5+4, r] = 1 if i==r else 0
    # V[24+k, r] = 1 for all k
    # W[k*4+i, r] = 1 if i==r else 0
    
    # Part 2: A_left * B_top (4x4 * 4x6)
    # Split B_top into B_top_left (4x4) and B_top_right (4x2)
    # C = A_left * B_top_left + A_left * B_top_right
    
    # Part 2a: A_left * B_top_left (4x4 * 4x4)
    # Use Strassen-like decomposition with rank 28
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # 
    # Using Strassen for each 2x2*2x2*2x2:
    # M1 = (A21+A22)*B11
    # M2 = A22*(B12-B11)
    # M3 = (A11+A12)*B22
    # M4 = A11*(B21-B22)
    # M5 = (A11+A21)*(B11+B22)
    # M6 = (A12-A21)*(B11+B12)
    # M7 = (A11-A22)*(B21+B22)
    #
    # C11 = M5 + M4 - M3 + M7
    # C12 = M1 + M2
    # C21 = M3 + M4
    # C22 = M1 - M5 + M2 + M7
    #
    # Each Mx is a 2x2*2x2 product, which has rank 4 (schoolbook) or 7 (Strassen)
    # But we're computing them as intermediate, so we need to decompose each 2x2*2x2*2x2
    #
    # For a 2x2*2x2*2x2 with Strassen (rank 7):
    # For 2x2 matrices P, Q, R:
    # P = [[p11, p12], [p21, p22]]
    # Q = [[q11, q12], [q21, q22]]
    # R = [[r11, r12], [r21, r22]]
    #
    # M1 = (p21+p22)*q11*r11 + (p21+p22)*q11*r22 + (p21+p22)*q22*r11 + (p21+p22)*q22*r22
    #     + ... this gets complicated
    #
    # Let me use a simpler approach: just use schoolbook for the 4x4*4x4 part with rank 16
    # and see if the total is better than 120.
    
    # Actually, let me try a different strategy: use the fact that we can share factors
    # across the different parts of the decomposition.
    
    # Total rank estimate:
    # Part 1 (4x1*1x6): rank 4
    # Part 2a (4x4*4x4): rank 16 (schoolbook)
    # Part 2b (4x4*4x2): rank 8 (schoolbook)
    # Total: 4 + 16 + 8 = 28
    
    # Let me implement this.
    
    U = np.zeros((num_U, 28), dtype=float)
    V = np.zeros((num_V, 28), dtype=float)
    W = np.zeros((num_W, 28), dtype=float)
    col = 0
    
    # Part 1: j=4 (A_right * B_bottom), rank 4
    for r in range(4):
        # U[i*5+4, r] = 1 if i==r else 0
        U[r*5+4, col] = 1
        # V[24+k, r] = 1 for all k in 0..5
        for k in range(6):
            V[24+k, col] = 1
        # W[k*4+i, r] = 1 if i==r else 0
        for k in range(6):
            W[k*4+r, col] = 1
        col += 1
    
    # Part 2a: j=0..3, k=0..3 (A_left * B_top_left, 4x4*4x4), rank 16
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a = i*5 + j
                b = j*6 + k
                c = k*4 + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    # Part 2b: j=0..3, k=4..5 (A_left * B_top_right, 4x4*4x2), rank 8
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                a = i*5 + j
                b = j*6 + k
                c = k*4 + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    # Verify
    if col != 28:
        # Fallback to schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    if verify_full(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If verification failed, return schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
