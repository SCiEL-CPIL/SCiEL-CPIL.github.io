"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy import linalg

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # Rank 39 decomposition using Hybrid Strassen/Schoolbook
    # Split j (0..2) into {0,1} (Strassen part) and {2} (Schoolbook part)
    # Strassen part: 2x2 x 2x7 -> 3 pairs of cols (rank 7 each) + 1 single col (rank 4) = 21 + 4 = 25
    # Schoolbook part: 2x1 x 1x7 -> 2*7 = 14
    # Total rank = 25 + 14 = 39
    
    rank = 39
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    
    # Part 1: Schoolbook for j=2 (index 2)
    # A[i, 2] * B[2, k] for i in 0,1 and k in 0..6
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
            
    # Part 2: Strassen for j in {0,1}
    # A is 2x2 (rows 0,1; cols 0,1)
    # B is 2x7 (rows 0,1; cols 0..6)
    # We process columns of B in pairs and singles
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A10)B11
    # M6 = (A11 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M6
    # C11 = M1 - M3 + M6 + M7
    
    # Define the 7 steps as (U_coeffs, V_coeffs, W_coeffs)
    # U indices: 0:A00, 1:A01, 2:A10, 3:A11 (relative to i in 0,1 and j in 0,1)
    # Actually, global U indices for i in 0,1 and j in 0,1 are:
    # i=0, j=0 -> 0*3+0 = 0
    # i=0, j=1 -> 0*3+1 = 1
    # i=1, j=0 -> 1*3+0 = 3
    # i=1, j=1 -> 1*3+1 = 4
    
    # V indices for i in 0,1 and k in {k1, k2}
    # i=0, k=k1 -> 0*7+k1
    # i=0, k=k2 -> 0*7+k2
    # i=1, k=k1 -> 1*7+k1
    # i=1, k=k2 -> 1*7+k2
    
    # W indices for i in 0,1 and k in {k1, k2}
    # i=0, k=k1 -> k1*2+0
    # i=0, k=k2 -> k2*2+0
    # i=1, k=k1 -> k1*2+1
    # i=1, k=k2 -> k2*2+1
    
    strassen_steps = [
        # Step 1: M1
        ({0: 1, 4: 1}, {0: 1, 1: 1}, {0: 1, 3: 1}), # (A00+A11)(B00+B11) -> C00+C11
        # Step 2: M2
        ({3: 1, 4: 1}, {0: 1}, {2: 1}),             # (A10+A11)B00 -> C10
        # Step 3: M3
        ({0: 1}, {2: 1, 3: -1}, {1: 1, 3: -1}),     # A00(B10-B11) -> C01-C11
        # Step 4: M4
        ({4: 1}, {1: 1, 0: -1}, {0: -1, 1: 1}),     # A11(B01-B00) -> -C00+C01
        # Step 5: M5
        ({0: 1, 3: 1}, {3: 1}, {0: 1}),             # (A00+A10)B11 -> C00
        # Step 6: M6
        ({4: 1, 0: -1}, {0: 1, 1: 1}, {2: 1, 3: 1}),# (A11-A00)(B00+B01) -> C10+C11
        # Step 7: M7
        ({1: 1, 4: -1}, {2: 1, 3: 1}, {0: 1, 3: 1}),# (A01-A11)(B10+B11) -> C00+C11
    ]
    
    # Note: The W coefficients in the list above are relative to the 4 outputs of a 2x2x2 block.
    # We need to map these relative indices (0:C00, 1:C01, 2:C10, 3:C11) to global W indices.
    
    pairs = [(0, 1), (2, 3), (4, 5)]
    
    for k1, k2 in pairs:
        # Map relative V indices (0:B00, 1:B01, 2:B10, 3:B11) to global
        # B00 is row 0, col k1 -> 0*7+k1
        # B01 is row 0, col k2 -> 0*7+k2
        # B10 is row 1, col k1 -> 1*7+k1
        # B11 is row 1, col k2 -> 1*7+k2
        v_map = {
            0: 0 * p + k1,
            1: 0 * p + k2,
            2: 1 * p + k1,
            3: 1 * p + k2
        }
        
        # Map relative W indices (0:C00, 1:C01, 2:C10, 3:C11) to global
        # C00 is row 0, col k1 -> k1*2+0
        # C01 is row 0, col k2 -> k2*2+0
        # C10 is row 1, col k1 -> k1*2+1
        # C11 is row 1, col k2 -> k2*2+1
        w_map = {
            0: k1 * n + 0,
            1: k2 * n + 0,
            2: k1 * n + 1,
            3: k2 * n + 1
        }
        
        # Map relative U indices (0:A00, 1:A01, 2:A10, 3:A11) to global
        # A00 is i=0, j=0 -> 0*3+0 = 0
        # A01 is i=0, j=1 -> 0*3+1 = 1
        # A10 is i=1, j=0 -> 1*3+0 = 3
        # A11 is i=1, j=1 -> 1*3+1 = 4
        u_map = {
            0: 0,
            1: 1,
            2: 3,
            3: 4
        }
        
        for u_coeffs, v_coeffs, w_coeffs in strassen_steps:
            for u_rel, val in u_coeffs.items():
                U[u_map[u_rel], col] += val
            for v_rel, val in v_coeffs.items():
                V[v_map[v_rel], col] += val
            for w_rel, val in w_coeffs.items():
                W[w_map[w_rel], col] += val
            col += 1
            
    # Single column k=6
    # C06 = A00 B06 + A01 B16
    # C16 = A10 B06 + A11 B16
    
    # Term 1: A00 B06
    U[0, col] = 1
    V[0 * p + 6, col] = 1
    W[6 * n + 0, col] = 1
    col += 1
    
    # Term 2: A01 B16
    U[1, col] = 1
    V[1 * p + 6, col] = 1
    W[6 * n + 0, col] = 1
    col += 1
    
    # Term 3: A10 B06
    U[3, col] = 1
    V[0 * p + 6, col] = 1
    W[6 * n + 1, col] = 1
    col += 1
    
    # Term 4: A11 B16
    U[4, col] = 1
    V[1 * p + 6, col] = 1
    W[6 * n + 1, col] = 1
    col += 1
    
    # Verify
    # target tensor
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
                
    R = np.zeros((nm, mp, pn))
    for c in range(rank):
        R += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        
    if np.max(np.abs(R - target)) > 1e-6:
        # Fallback to schoolbook if verification fails (should not happen)
        rank_school = n * m * p
        U_s = np.zeros((nm, rank_school))
        V_s = np.zeros((mp, rank_school))
        W_s = np.zeros((pn, rank_school))
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, col_s] = 1
                    V_s[j * p + k, col_s] = 1
                    W_s[k * n + i, col_s] = 1
                    col_s += 1
        return {
            "U": U_s.tolist(),
            "V": V_s.tolist(),
            "W": W_s.tolist()
        }
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
