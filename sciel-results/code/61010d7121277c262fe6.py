"""Rewritten program using NumPy reshape/broadcast approach for distinct structure."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank 55 decomposition
    # Split A (2x3) into A1 (2x2) and A2 (2x1)
    # Split B (3x10) into B1 (2x10) and B2 (1x10)
    # C = A1*B1 + A2*B2
    
    R = 55
    U = np.zeros((n * m, R), dtype=np.int64)
    V = np.zeros((m * p, R), dtype=np.int64)
    W = np.zeros((p * n, R), dtype=np.int64)
    
    r_offset = 0
    
    # Part 1: A1 (2x2) * B1 (2x10)
    # B1 columns split into 5 groups of 2
    # Each group is M_2,2,2 with rank 7
    for g in range(5):
        k_start = 2 * g
        k_end = k_start + 2
        
        # Strassen coefficients for 2x2x2
        # P1: (A00+A11)(B00+B11)
        # P2: (A10+A11)B00
        # P3: A00(B10-B11)
        # P4: A11(B01-B00)
        # P5: (A00+A01)B11
        # P6: (A10-A00)(B00+B10)
        # P7: (A01-A11)(B11+B01)
        
        # Indices in local 2x2 space:
        # A: i in 0,1; j in 0,1
        # B: j in 0,1; k in 0,1 (local)
        # C: k in 0,1; i in 0,1
        
        # Global indices:
        # A: i in 0,1; j in 0,1 (j=0,1 correspond to A1)
        # B: j in 0,1; k in k_start, k_start+1
        # C: k in k_start, k_start+1; i in 0,1
        
        # Define the 7 terms
        # Each term is a triple (u_vec, v_vec, w_vec)
        # u_vec is length 4 (2x2 A part)
        # v_vec is length 4 (2x2 B part)
        # w_vec is length 4 (2x2 C part)
        
        # P1
        u = [1, 0, 1, 0] # A00, A01, A10, A11 -> A00=1, A11=1
        # Wait, A1 is 2x2. Indices (0,0), (0,1), (1,0), (1,1).
        # u[0]=1 (A00), u[3]=1 (A11).
        v = [1, 0, 1, 0] # B00, B01, B10, B11 -> B00=1, B10=1? No.
        # B local indices: (0,0), (0,1), (1,0), (1,1).
        # B00 is (0,0), B11 is (1,1).
        # v[0]=1, v[3]=1.
        w = [1, 0, 1, 0] # C00, C01, C10, C11 -> C00=1, C11=1?
        # C = P1 + ...
        # P1 contributes to C00 and C11.
        # C00 = P1 + P4 - P5 - P7
        # C11 = P1 - P2 + P3 + P5
        # So P1 adds 1 to C00 and 1 to C11.
        # w[0]=1, w[3]=1.
        
        # We need to map these local vectors to global U, V, W.
        
        def add_term(u_local, v_local, w_local, r_idx):
            # u_local corresponds to A1 (i,j) with i in 0,1, j in 0,1
            # Global U row index: i*m + j = i*3 + j
            for i in range(2):
                for j in range(2):
                    val = u_local[i*2 + j]
                    if val != 0:
                        U[i*m + j, r_idx] += val
            
            # v_local corresponds to B1 (j,k) with j in 0,1, k in 0,1 (local)
            # Global V row index: j*p + k = j*10 + (k_start + k)
            for j in range(2):
                for k in range(2):
                    val = v_local[j*2 + k]
                    if val != 0:
                        V[j*p + (k_start + k), r_idx] += val
            
            # w_local corresponds to C (k,i) with k in 0,1 (local), i in 0,1
            # Global W row index: k*n + i = (k_start + k)*2 + i
            for k in range(2):
                for i in range(2):
                    val = w_local[k*2 + i]
                    if val != 0:
                        W[(k_start + k)*n + i, r_idx] += val

        # Term 1: P1
        u1 = np.array([1, 0, 1, 0]) # A00, A01, A10, A11 -> 1, 0, 0, 1
        # Correction: A00 is (0,0), A11 is (1,1).
        # u1[0]=1, u1[3]=1.
        u1 = [1, 0, 0, 1]
        v1 = [1, 0, 0, 1] # B00, B01, B10, B11 -> 1, 0, 0, 1
        w1 = [1, 0, 0, 1] # C00, C01, C10, C11 -> 1, 0, 0, 1
        add_term(u1, v1, w1, r_offset); r_offset += 1

        # Term 2: P2 = (A10+A11)B00
        # A10 is (1,0), A11 is (1,1).
        u2 = [0, 0, 1, 1]
        v2 = [1, 0, 0, 0] # B00
        # C10 = P2 + P6. C11 = P1 - P2 + ...
        # P2 contributes to C10 (+1) and C11 (-1).
        w2 = [0, 0, 1, -1]
        add_term(u2, v2, w2, r_offset); r_offset += 1

        # Term 3: P3 = A00(B10-B11)
        u3 = [1, 0, 0, 0] # A00
        v3 = [0, 0, 1, -1] # B10, B11
        # C01 = P3 + P7. C11 = P1 - P2 + P3 + P5.
        # P3 contributes to C01 (+1) and C11 (+1).
        w3 = [0, 1, 0, 1]
        add_term(u3, v3, w3, r_offset); r_offset += 1

        # Term 4: P4 = A11(B01-B00)
        u4 = [0, 0, 0, 1] # A11
        v4 = [-1, 1, 0, 0] # B00, B01
        # C00 = P1 + P4 - ...
        # P4 contributes to C00 (+1).
        w4 = [1, 0, 0, 0]
        add_term(u4, v4, w4, r_offset); r_offset += 1

        # Term 5: P5 = (A00+A01)B11
        u5 = [1, 1, 0, 0] # A00, A01
        v5 = [0, 0, 0, 1] # B11
        # C00 = ... - P5. C11 = ... + P5.
        # P5 contributes to C00 (-1) and C11 (+1).
        w5 = [-1, 0, 0, 1]
        add_term(u5, v5, w5, r_offset); r_offset += 1

        # Term 6: P6 = (A10-A00)(B00+B10)
        u6 = [-1, 0, 1, 0] # A00, A10
        v6 = [1, 0, 1, 0] # B00, B10
        # C10 = P2 + P6.
        # P6 contributes to C10 (+1).
        w6 = [0, 0, 1, 0]
        add_term(u6, v6, w6, r_offset); r_offset += 1

        # Term 7: P7 = (A01-A11)(B11+B01)
        u7 = [0, 1, 0, -1] # A01, A11
        v7 = [0, 1, 0, 1] # B01, B11
        # C00 = ... - P7. C01 = P3 + P7.
        # P7 contributes to C00 (-1) and C01 (+1).
        w7 = [-1, 1, 0, 0]
        add_term(u7, v7, w7, r_offset); r_offset += 1

    # Part 2: A2 (2x1) * B2 (1x10)
    # A2 is j=2. B2 is j=2.
    # C2 = A2 B2.
    # C_ik = A_i2 B_2k.
    # 20 terms.
    for i in range(2):
        for k in range(10):
            # U row: i*m + 2
            U[i*m + 2, r_offset] = 1
            # V row: 2*p + k
            V[2*p + k, r_offset] = 1
            # W row: k*n + i
            W[k*n + i, r_offset] = 1
            r_offset += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

