import numpy as np
import time

# Keep the exact entry-point name and return interface.
# Strategy: keep the rank-85 Strassen-based design family (49 + 16 + 16 + 4)
# and refine it to be numerically clean, fast, and robust. We will:
# - Build U, V, W as numpy arrays (integers) for speed and clarity.
# - Add a lightweight self-check in development? No, keep it deterministic and simple.
# - Ensure we do not exceed time: construction is O(1) with tiny constants.

def construct():
    # Track start for a safety guard (not needed here, but included for clarity).
    t0 = time.time()
    
    n, m, p = 4, 5, 5
    # Dimensions:
    # U: (n*m, rank) = (20, 85)
    # V: (m*p, rank) = (25, 85)
    # W: (p*n, rank) = (20, 85)
    # The "output factor follows the official trace convention W[k*n+i, r]"
    # This means for C[i, k], the index in W is k*n + i.
    # Our W matrix has rows indexed by (k, i) flattened as k*n + i.
    # Columns are the rank dimension.
    
    rank = 85
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    # Helper to get 2x2x2 Strassen factors
    # Returns U, V, W of shape 4x7
    # U indices: 0:00, 1:01, 2:10, 3:11
    # V indices: 0:00, 1:01, 2:10, 3:11
    # W indices: 0:00, 1:01, 2:10, 3:11 (using k*n+i, n=2 -> 0:00, 1:01, 2:10, 3:11)
    
    s_u = np.zeros((4, 7), dtype=np.int8)
    s_v = np.zeros((4, 7), dtype=np.int8)
    s_w = np.zeros((4, 7), dtype=np.int8)
    
    # Term 0: (a00+a11)(b00+b11)
    s_u[0, 0]=1; s_u[3, 0]=1
    s_v[0, 0]=1; s_v[3, 0]=1
    s_w[0, 0]=1; s_w[3, 0]=1
    
    # Term 1: (a10+a11)b00
    s_u[2, 1]=1; s_u[3, 1]=1
    s_v[0, 1]=1
    s_w[2, 1]=1; s_w[3, 1]=-1
    
    # Term 2: a00(b01-b11)
    s_u[0, 2]=1
    s_v[1, 2]=1; s_v[3, 2]=-1
    s_w[1, 2]=1; s_w[3, 2]=1
    
    # Term 3: a11(b10-b00)
    s_u[3, 3]=1
    s_v[2, 3]=1; s_v[0, 3]=-1
    s_w[0, 3]=1; s_w[2, 3]=1
    
    # Term 4: (a00+a01)b11
    s_u[0, 4]=1; s_u[1, 4]=1
    s_v[3, 4]=1
    s_w[0, 4]=-1; s_w[1, 4]=1
    
    # Term 5: (a10-a00)(b00+b01)
    s_u[0, 5]=-1; s_u[2, 5]=1
    s_v[0, 5]=1; s_v[1, 5]=1
    s_w[3, 5]=1
    
    # Term 6: (a01-a11)(b10+b11)
    s_u[1, 6]=1; s_u[3, 6]=-1
    s_v[2, 6]=1; s_v[3, 6]=1
    s_w[0, 6]=1
    
    # 4x4x4 part: 49 terms
    # We define the 7 Strassen 4x4 terms.
    # For each term, we have:
    # - a_blocks: dict of (br, bc) -> coeff
    # - b_blocks: dict of (br, bc) -> coeff
    # - c_blocks: dict of (br, bc) -> coeff
    
    terms = [
        # M1 = (A11+A22)(B11+B22)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2 = (A21+A22)B11
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3 = A11(B12-B22)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4 = A22(B21-B11)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5 = (A11+A12)B22
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6 = (A21-A11)(B11+B12)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7 = (A12-A22)(B21+B22)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    for t_idx, (a_coeff, b_coeff, c_coeff) in enumerate(terms):
        start_col = t_idx * 7
        for inner_k in range(7):
            col = start_col + inner_k
            u_vec = s_u[:, inner_k] # shape (4,)
            v_vec = s_v[:, inner_k] # shape (4,)
            w_vec = s_w[:, inner_k] # shape (4,)
            
            # Distribute U
            for (a_br, a_bc), ac in a_coeff.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        val = u_vec[local_idx]
                        if val != 0:
                            gi = 2 * a_br + lr
                            gj = 2 * a_bc + lc
                            U[gi * 5 + gj, col] += ac * val
            
            # Distribute V
            for (b_br, b_bc), bc in b_coeff.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        val = v_vec[local_idx]
                        if val != 0:
                            gj = 2 * b_br + lr
                            gk = 2 * b_bc + lc
                            V[gj * 5 + gk, col] += bc * val
            
            # Distribute W
            for (c_br, c_bc), cc in c_coeff.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        val = w_vec[local_idx]
                        if val != 0:
                            gi = 2 * c_br + lr
                            gk = 2 * c_bc + lc
                            # W index: k*n + i = gk * 4 + gi
                            W[gk * 4 + gi, col] += cc * val
    
    col = 49
    
    # 4x4x1 part: A_L (4x4) * B_LR (4x1) -> C (4x1)
    # A rows 0..3, cols 0..3. B rows 0..3, col 4. C rows 0..3, col 4.
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W[4 * 4 + i, col] = 1 # k=4, i in 0..3 -> 16 + i
            col += 1
            
    # 4x1x4 part: A_R (4x1) * B_RL (1x4) -> C (4x4)
    # A rows 0..3, col 4. B row 4, cols 0..3. C rows 0..3, cols 0..3.
    for i in range(4):
        for k in range(4):
            U[i * 5 + 4, col] = 1
            V[4 * 5 + k, col] = 1
            W[k * 4 + i, col] = 1
            col += 1
            
    # 4x1x1 part: A_R (4x1) * B_RR (1x1) -> C (4x1)
    # A rows 0..3, col 4. B row 4, col 4. C rows 0..3, col 4.
    for i in range(4):
        U[i * 5 + 4, col] = 1
        V[4 * 5 + 4, col] = 1
        W[4 * 4 + i, col] = 1
        col += 1
        
    # Safety check (should not trigger)
    if time.time() - t0 > 110:
        # If somehow slow, return what we have.
        pass

    return {"U": U, "V": V, "W": W}
