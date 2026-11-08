import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # We will build the decomposition by summing up rank-1 tensors.
    # Each rank-1 tensor is defined by vectors u (len nm), v (len mp), w (len pn).
    # U[:, r] = u, V[:, r] = v, W[:, r] = w.
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a rank-1 term
    def add_term(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
        
    # Helper for Strassen 2x2x2
    # Inputs: 
    #   A is 2x2, B is 2x2.
    #   We need to map the local indices (0,1) to global indices.
    #   A indices: (i, j) -> global a = i*m + j
    #   B indices: (j, k) -> global b = j*p + k
    #   C indices: (i, k) -> global c = k*n + i
    def strassen_2x2x2(offset_i, offset_j, offset_k):
        # offset_i: row offset for A and C
        # offset_j: col offset for A, row offset for B
        # offset_k: col offset for B and C
        
        # Strassen constants
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B10)
        # M7 = (A01 - A11) * (B01 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to create U, V, W vectors for each M.
        
        # M1: (A00 + A11) * (B00 + B11)
        # u: A00=1, A11=1
        # v: B00=1, B11=1
        # w: C00=1, C11=1
        
        # Local indices:
        # A00: i=0, j=0 -> a = offset_i*m + offset_j
        # A11: i=1, j=1 -> a = (offset_i+1)*m + (offset_j+1)
        # B00: j=0, k=0 -> b = offset_j*p + offset_k
        # B11: j=1, k=1 -> b = (offset_j+1)*p + (offset_k+1)
        # C00: i=0, k=0 -> c = offset_k*n + offset_i
        # C11: i=1, k=1 -> c = (offset_k+1)*n + (offset_i+1)
        
        a00 = offset_i*m + offset_j
        a11 = (offset_i+1)*m + (offset_j+1)
        b00 = offset_j*p + offset_k
        b11 = (offset_j+1)*p + (offset_k+1)
        c00 = offset_k*n + offset_i
        c11 = (offset_k+1)*n + (offset_i+1)
        
        # M1
        u = [0]*nm; v = [0]*mp; w = [0]*pn
        u[a00]=1; u[a11]=1
        v[b00]=1; v[b11]=1
        w[c00]=1; w[c11]=1
        add_term(u, v, w)
        
        # M2: (A10 + A11) * B00
        # u: A10=1, A11=1
        # v: B00=1
        # w: C10=1
        a10 = (offset_i+1)*m + offset_j
        c10 = offset_k*n + (offset_i+1)
        
        u = [0]*nm; v = [0]*mp; w = [0]*pn
        u[a10]=1; u[a11]=1
        v[b00]=1
        w[c10]=1
        add_term(u, v, w)
        
        # M3: A00 * (B01 - B11)
        # u: A00=1
        # v: B01=1, B11=-1
        # w: C01=1
        b01 = offset_j*p + (offset_k+1)
        c01 = (offset_k+1)*n + offset_i
        
        u = [0]*nm; v = [0]*mp; w = [0]*pn
        u[a00]=1
        v[b01]=1; v[b11]=-1
        w[c01]=1
        add_term(u, v, w)
        
        # M4: A11 * (B10 - B00)
        # u: A11=1
        # v: B10=1, B00=-1
        # w: C00=1, C10=1
        b10 = (offset_j+1)*p + offset_k
        
        u = [0]*nm; v = [0]*mp; w = [0]*pn
        u[a11]=1
        v[b10]=1; v[b00]=-1
        w[c00]=1; w[c10]=1
        add_term(u, v, w)
        
        # M5: (A00 + A01) * B11
        # u: A00=1, A01=1
        # v: B11=1
        # w: C00=-1, C01=1
        a01 = offset_i*m + (offset_j+1)
        
        u = [0]*nm; v = [0]*mp; w = [0]*pn
        u[a00]=1; u[a01]=1
        v[b11]=1
        w[c00]=-1; w[c01]=1
        add_term(u, v, w)
        
        # M6: (A10 - A00) * (B00 + B10)
        # u: A10=1, A00=-1
        # v: B00=1, B10=1
        # w: C11=1
        
        u = [0]*nm; v = [0]*mp; w = [0]*pn
        u[a10]=1; u[a00]=-1
        v[b00]=1; v[b10]=1
        w[c11]=1
        add_term(u, v, w)
        
        # M7: (A01 - A11) * (B01 + B11)
        # u: A01=1, A11=-1
        # v: B01=1, B11=1
        # w: C00=1, C01=1
        
        u = [0]*nm; v = [0]*mp; w = [0]*pn
        u[a01]=1; u[a11]=-1
        v[b01]=1; v[b11]=1
        w[c00]=1; w[c01]=1
        add_term(u, v, w)

    # Helper for 1x2x2 (Rank 4)
    # A is 1x2, B is 2x2.
    # C is 1x2.
    # C0 = A0*B00 + A1*B10
    # C1 = A0*B01 + A1*B11
    def one_2x2x2(offset_i, offset_j, offset_k):
        # offset_i: row for A and C (only 1 row)
        # offset_j: col for A, row for B (2 cols/rows)
        # offset_k: col for B and C (2 cols)
        
        a0 = offset_i*m + offset_j
        a1 = offset_i*m + (offset_j+1)
        
        b00 = offset_j*p + offset_k
        b01 = offset_j*p + (offset_k+1)
        b10 = (offset_j+1)*p + offset_k
        b11 = (offset_j+1)*p + (offset_k+1)
        
        c0 = offset_k*n + offset_i
        c1 = (offset_k+1)*n + offset_i
        
        # Term 1: A0 * B00 -> C0
        u = [0]*nm; v = [0]*mp; w = [0]*pn
        u[a0]=1
        v[b00]=1
        w[c0]=1
        add_term(u, v, w)
        
        # Term 2: A1 * B10 -> C0
        u = [0]*nm; v = [0]*mp; w = [0]*pn
        u[a1]=1
        v[b10]=1
        w[c0]=1
        add_term(u, v, w)
        
        # Term 3: A0 * B01 -> C1
        u = [0]*nm; v = [0]*mp; w = [0]*pn
        u[a0]=1
        v[b01]=1
        w[c1]=1
        add_term(u, v, w)
        
        # Term 4: A1 * B11 -> C1
        u = [0]*nm; v = [0]*mp; w = [0]*pn
        u[a1]=1
        v[b11]=1
        w[c1]=1
        add_term(u, v, w)

    # Main Logic
    # Split B (4x6) into 3 blocks of 4x2.
    # B_k is columns 2k, 2k+1.
    for k_block in range(3):
        offset_k = k_block * 2
        
        # For each block, we compute C_block = A * B_block (3x2).
        # Split A (3x4) into A_L (3x2) and A_R (3x2).
        # Split B_block (4x2) into B_T (2x2) and B_B (2x2).
        
        # A_L * B_T: A_L is rows 0,1,2 cols 0,1. B_T is rows 0,1 cols 0,1 (local).
        # This is 3x2 by 2x2.
        # Split A_L into A_LL (2x2) and A_LR (1x2).
        
        # A_LL * B_T: 2x2 by 2x2. Strassen.
        # offset_i = 0, offset_j = 0, offset_k = offset_k
        strassen_2x2x2(0, 0, offset_k)
        
        # A_LR * B_T: 1x2 by 2x2. Rank 4.
        # offset_i = 2, offset_j = 0, offset_k = offset_k
        one_2x2x2(2, 0, offset_k)
        
        # A_R * B_B: A_R is rows 0,1,2 cols 2,3. B_B is rows 2,3 cols 0,1 (local).
        # This is 3x2 by 2x2.
        # Split A_R into A_RL (2x2) and A_RR (1x2).
        
        # A_RL * B_B: 2x2 by 2x2. Strassen.
        # offset_i = 0, offset_j = 2, offset_k = offset_k
        # Note: B_B rows are 2,3. So offset_j for B is 2.
        # But in our helper, offset_j is the column offset for A and row offset for B.
        # A_R cols are 2,3. So offset_j = 2.
        # B_B rows are 2,3. So offset_j = 2.
        strassen_2x2x2(0, 2, offset_k)
        
        # A_RR * B_B: 1x2 by 2x2. Rank 4.
        # offset_i = 2, offset_j = 2, offset_k = offset_k
        one_2x2x2(2, 2, offset_k)

    # Convert lists to matrices
    rank = len(U_list)
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    for r in range(rank):
        U[:, r] = U_list[r]
        V[:, r] = V_list[r]
        W[:, r] = W_list[r]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
