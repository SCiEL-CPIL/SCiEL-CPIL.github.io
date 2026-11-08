"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    n, m, p = DIMENSIONS
    # Target rank is 45 based on the block decomposition strategy
    # 3 * Rank(2x2x2) + 2 * (Rank(2x2x2) + Rank(2x2x1) + Rank(2x1x3))
    # 3 * 7 + 2 * (7 + 4 + 1) = 21 + 24 = 45
    
    U = []
    V = []
    W = []
    
    # Helper to add a term
    def add_term(u_dict, v_dict, w_dict):
        # u_dict: {global_idx: coeff}
        # v_dict: {global_idx: coeff}
        # w_dict: {global_idx: coeff}
        
        max_u = max(u_dict.keys()) if u_dict else -1
        max_v = max(v_dict.keys()) if v_dict else -1
        max_w = max(w_dict.keys()) if w_dict else -1
        
        # We need to pad to the size of the largest index seen so far?
        # No, we build the full vectors at the end or maintain size.
        # It's easier to store as dicts and convert at the end, 
        # but we need to know the max index to create the vectors.
        # Max U index: n*m - 1 = 9
        # Max V index: m*p - 1 = 29
        # Max W index: p*n - 1 = 11
        
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        
        for idx, coeff in u_dict.items():
            u_vec[idx] = coeff
        for idx, coeff in v_dict.items():
            v_vec[idx] = coeff
        for idx, coeff in w_dict.items():
            w_vec[idx] = coeff
            
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)

    # Indices:
    # A[i, j] -> U index: i * m + j
    # B[j, k] -> V index: j * p + k
    # C[i, k] -> W index: k * n + i
    
    def a_idx(i, j):
        return i * m + j
        
    def b_idx(j, k):
        return j * p + k
        
    def c_idx(i, k):
        return k * n + i

    # --- Block 1: A1 (2x2) * B1 (2x6) ---
    # A1 uses i in {0,1}, j in {0,1}
    # B1 uses j in {0,1}, k in {0..5}
    # C uses i in {0,1}, k in {0..5}
    
    # Split B1 into 3 blocks of 2x2
    for t in range(3):
        k_offset = t * 2
        k0 = k_offset
        k1 = k_offset + 1
        
        # Strassen 7 terms for 2x2x2
        # Local A: i in {0,1}, j in {0,1}
        # Local B: j in {0,1}, k in {k0, k1}
        # Local C: i in {0,1}, k in {k0, k1}
        
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        # u: A00=1, A11=1
        # v: B0k0=1, B1k1=1
        # w: C0k0=1, C0k1=1, C1k0=1, C1k1=1
        add_term(
            {a_idx(0,0): 1, a_idx(1,1): 1},
            {b_idx(0,k0): 1, b_idx(1,k1): 1},
            {c_idx(0,k0): 1, c_idx(0,k1): 1, c_idx(1,k0): 1, c_idx(1,k1): 1}
        )
        
        # M2 = (A10 + A11) * B0k0
        # u: A10=1, A11=1
        # v: B0k0=1
        # w: C1k0=1, C1k1=1
        add_term(
            {a_idx(1,0): 1, a_idx(1,1): 1},
            {b_idx(0,k0): 1},
            {c_idx(1,k0): 1, c_idx(1,k1): 1}
        )
        
        # M3 = A00 * (B0k1 - B1k1)
        # u: A00=1
        # v: B0k1=1, B1k1=-1
        # w: C0k1=1, C1k1=1
        add_term(
            {a_idx(0,0): 1},
            {b_idx(0,k1): 1, b_idx(1,k1): -1},
            {c_idx(0,k1): 1, c_idx(1,k1): 1}
        )
        
        # M4 = A11 * (B1k0 - B0k0)
        # u: A11=1
        # v: B1k0=1, B0k0=-1
        # w: C0k0=1, C1k0=1
        add_term(
            {a_idx(1,1): 1},
            {b_idx(1,k0): 1, b_idx(0,k0): -1},
            {c_idx(0,k0): 1, c_idx(1,k0): 1}
        )
        
        # M5 = (A00 + A01) * B1k1
        # u: A00=1, A01=1
        # v: B1k1=1
        # w: C0k0=1, C0k1=1
        add_term(
            {a_idx(0,0): 1, a_idx(0,1): 1},
            {b_idx(1,k1): 1},
            {c_idx(0,k0): 1, c_idx(0,k1): 1}
        )
        
        # M6 = (A10 - A00) * (B0k0 + B0k1)
        # u: A10=1, A00=-1
        # v: B0k0=1, B0k1=1
        # w: C1k1=1
        add_term(
            {a_idx(1,0): 1, a_idx(0,0): -1},
            {b_idx(0,k0): 1, b_idx(0,k1): 1},
            {c_idx(1,k1): 1}
        )
        
        # M7 = (A01 - A11) * (B1k0 + B1k1)
        # u: A01=1, A11=-1
        # v: B1k0=1, B1k1=1
        # w: C0k0=1
        add_term(
            {a_idx(0,1): 1, a_idx(1,1): -1},
            {b_idx(1,k0): 1, b_idx(1,k1): 1},
            {c_idx(0,k0): 1}
        )

    # --- Block 2: A2 (2x3) * B2 (3x6) ---
    # A2 uses i in {0,1}, j in {2,3,4}
    # B2 uses j in {2,3,4}, k in {0..5}
    
    # Split B2 into 2 blocks of 3x3
    for t in range(2):
        k_offset = t * 3
        k0 = k_offset
        k1 = k_offset + 1
        k2 = k_offset + 2
        
        # Sub-block: A2 (2x3) * B_block (3x3)
        # Split A2 into A21 (2x2, cols 2,3) and A22 (2x1, col 4)
        # Split B_block into B21 (2x3, rows 2,3) and B22 (1x3, row 4)
        
        # Part 1: A21 (2x2) * B21 (2x3)
        # Split output 3 into 2 (k0,k1) and 1 (k2)
        
        # 1a. 2x2x2 part (k0, k1)
        # A cols: 2,3. B rows: 2,3. C cols: k0, k1.
        
        # M1 = (A02 + A13) * (B2k0 + B3k1)
        add_term(
            {a_idx(0,2): 1, a_idx(1,3): 1},
            {b_idx(2,k0): 1, b_idx(3,k1): 1},
            {c_idx(0,k0): 1, c_idx(0,k1): 1, c_idx(1,k0): 1, c_idx(1,k1): 1}
        )
        
        # M2 = (A12 + A13) * B2k0
        add_term(
            {a_idx(1,2): 1, a_idx(1,3): 1},
            {b_idx(2,k0): 1},
            {c_idx(1,k0): 1, c_idx(1,k1): 1}
        )
        
        # M3 = A02 * (B2k1 - B3k1)
        add_term(
            {a_idx(0,2): 1},
            {b_idx(2,k1): 1, b_idx(3,k1): -1},
            {c_idx(0,k1): 1, c_idx(1,k1): 1}
        )
        
        # M4 = A13 * (B3k0 - B2k0)
        add_term(
            {a_idx(1,3): 1},
            {b_idx(3,k0): 1, b_idx(2,k0): -1},
            {c_idx(0,k0): 1, c_idx(1,k0): 1}
        )
        
        # M5 = (A02 + A03) * B3k1
        add_term(
            {a_idx(0,2): 1, a_idx(0,3): 1},
            {b_idx(3,k1): 1},
            {c_idx(0,k0): 1, c_idx(0,k1): 1}
        )
        
        # M6 = (A12 - A02) * (B2k0 + B2k1)
        add_term(
            {a_idx(1,2): 1, a_idx(0,2): -1},
            {b_idx(2,k0): 1, b_idx(2,k1): 1},
            {c_idx(1,k1): 1}
        )
        
        # M7 = (A03 - A13) * (B3k0 + B3k1)
        add_term(
            {a_idx(0,3): 1, a_idx(1,3): -1},
            {b_idx(3,k0): 1, b_idx(3,k1): 1},
            {c_idx(0,k0): 1}
        )
        
        # 1b. 2x2x1 part (k2)
        # C_i, k2 = A_i2 B_2k2 + A_i3 B_3k2
        # Term 1: A02 B2k2 -> C0k2
        add_term(
            {a_idx(0,2): 1},
            {b_idx(2,k2): 1},
            {c_idx(0,k2): 1}
        )
        # Term 2: A12 B2k2 -> C1k2
        add_term(
            {a_idx(1,2): 1},
            {b_idx(2,k2): 1},
            {c_idx(1,k2): 1}
        )
        # Term 3: A03 B3k2 -> C0k2
        add_term(
            {a_idx(0,3): 1},
            {b_idx(3,k2): 1},
            {c_idx(0,k2): 1}
        )
        # Term 4: A13 B3k2 -> C1k2
        add_term(
            {a_idx(1,3): 1},
            {b_idx(3,k2): 1},
            {c_idx(1,k2): 1}
        )
        
        # Part 2: A22 (2x1) * B22 (1x3)
        # A col: 4. B row: 4. C cols: k0, k1, k2.
        # C_i, k = A_i4 B_4k
        # Rank 1 term:
        add_term(
            {a_idx(0,4): 1, a_idx(1,4): 1},
            {b_idx(4,k0): 1, b_idx(4,k1): 1, b_idx(4,k2): 1},
            {c_idx(0,k0): 1, c_idx(0,k1): 1, c_idx(0,k2): 1, c_idx(1,k0): 1, c_idx(1,k1): 1, c_idx(1,k2): 1}
        )

    # Convert lists of lists to the required format
    # U, V, W are already lists of lists
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
