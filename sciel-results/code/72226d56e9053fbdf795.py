"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-55 exact tensor decomposition for 2x3 x 3x10 matrix multiplication.
    
    Strategy:
    1. Split m=3 into m1=2 (left), m2=1 (right).
    2. Split B into B_top (2x10) and B_bottom (1x10).
    3. C = A_left B_top + A_right B_bottom.
    
    Part 1 (A_right B_bottom):
    - A_right is 2x1, B_bottom is 1x10.
    - 20 independent products. Rank = 20.
    
    Part 2 (A_left B_top):
    - A_left is 2x2, B_top is 2x10.
    - Split B_top into 5 blocks of 2x2.
    - Apply Strassen's algorithm to each 2x2 x 2x2 block.
    - 5 blocks * 7 mults/block = 35 mults. Rank = 35.
    
    Total Rank = 55.
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    rank = 55
    
    # Initialize factor matrices
    # U: (n*m, rank) = (6, 55)
    # V: (m*p, rank) = (30, 55)
    # W: (p*n, rank) = (20, 55)
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    
    # --- Part 1: A_right (A[:, 2]) * B_bottom (B[2, :]) ---
    # A indices: A[0,2]=2, A[1,2]=5
    # B indices: B[2,k] = 20+k
    # C indices: C[i,k] -> W[k*n+i]
    
    for i in range(n):
        a_idx = i * m + 2  # 2 for i=0, 5 for i=1
        for k in range(p):
            b_idx = 2 * p + k  # 20..29
            c_idx = k * n + i  # 2k+i
            
            U[a_idx, col] = 1.0
            V[b_idx, col] = 1.0
            W[c_idx, col] = 1.0
            col += 1
            
            if time.time() - start_time > 110:
                # Safety margin, though this loop is very fast
                break
    if col != 20:
        # This should never happen in Part 1
        pass

    # --- Part 2: A_left (A[:, :2]) * B_top (B[:2, :]) ---
    # A indices:
    # x00 = A[0,0] = 0
    # x01 = A[0,1] = 1
    # x10 = A[1,0] = 3
    # x11 = A[1,1] = 4
    
    x00, x01, x10, x11 = 0, 1, 3, 4
    
    for j in range(5):
        # B block j covers columns 2j, 2j+1 of B_top (rows 0 and 1)
        # y00 = B[0, 2j]   -> index 2j
        # y01 = B[0, 2j+1] -> index 2j+1
        # y10 = B[1, 2j]   -> index 10+2j
        # y11 = B[1, 2j+1] -> index 10+2j+1
        
        y00 = 2 * j
        y01 = 2 * j + 1
        y10 = 10 + 2 * j
        y11 = 10 + 2 * j + 1
        
        # C block j covers columns 2j, 2j+1 of C
        # C[0, 2j]   -> W[2j*2 + 0] = 4j
        # C[1, 2j]   -> W[2j*2 + 1] = 4j+1
        # C[0, 2j+1] -> W[(2j+1)*2 + 0] = 4j+2
        # C[1, 2j+1] -> W[(2j+1)*2 + 1] = 4j+3
        
        z00 = 4 * j
        z10 = 4 * j + 1
        z01 = 4 * j + 2
        z11 = 4 * j + 3
        
        # Strassen's 7 products for 2x2 * 2x2
        # M1 = (X00 + X11) * (Y00 + Y11)
        U[x00, col] += 1.0
        U[x11, col] += 1.0
        V[y00, col] += 1.0
        V[y11, col] += 1.0
        W[z00, col] += 1.0
        W[z11, col] += 1.0
        col += 1
        
        # M2 = (X10 + X11) * Y00
        U[x10, col] += 1.0
        U[x11, col] += 1.0
        V[y00, col] += 1.0
        W[z10, col] += 1.0
        W[z11, col] -= 1.0
        col += 1
        
        # M3 = X00 * (Y01 - Y11)
        U[x00, col] += 1.0
        V[y01, col] += 1.0
        V[y11, col] -= 1.0
        W[z01, col] += 1.0
        W[z11, col] += 1.0
        col += 1
        
        # M4 = X11 * (Y10 - Y00)
        U[x11, col] += 1.0
        V[y10, col] += 1.0
        V[y00, col] -= 1.0
        W[z00, col] += 1.0
        W[z10, col] += 1.0
        col += 1
        
        # M5 = (X00 + X01) * Y11
        U[x00, col] += 1.0
        U[x01, col] += 1.0
        V[y11, col] += 1.0
        W[z00, col] -= 1.0
        W[z01, col] += 1.0
        col += 1
        
        # M6 = (X10 - X00) * (Y00 + Y01)
        U[x10, col] += 1.0
        U[x00, col] -= 1.0
        V[y00, col] += 1.0
        V[y01, col] += 1.0
        W[z11, col] += 1.0
        col += 1
        
        # M7 = (X01 - X11) * (Y10 + Y11)
        U[x01, col] += 1.0
        U[x11, col] -= 1.0
        V[y10, col] += 1.0
        V[y11, col] += 1.0
        W[z00, col] += 1.0
        col += 1
        
        if time.time() - start_time > 110:
            break

    if col != rank:
        # If we broke early due to timeout, we might have an incomplete matrix.
        # However, the loops are tiny (55 iterations total), so this is extremely unlikely.
        # We return what we have, but it might be invalid.
        # To be safe, if incomplete, we could try to return a known valid smaller rank?
        # No, better to just return the potentially invalid one if timeout is imminent, 
        # but given the speed, it's fine.
        pass

    # Quick validation (optional, but good for debugging during development)
    # In production, this adds ~1ms.
    if col == rank:
        np.random.seed(42)
        A_test = np.random.rand(n, m)
        B_test = np.random.rand(m, p)
        C_expected = A_test @ B_test
        
        # Compute C via decomposition
        C_calc = np.zeros((n, p))
        for r in range(rank):
            u_vec = U[:, r].reshape(n, m)
            v_vec = V[:, r].reshape(m, p)
            P_r = u_vec @ v_vec
            for i in range(n):
                for k in range(p):
                    C_calc[i, k] += P_r[i, k] * W[k * n + i, r]
                    
        if not np.allclose(C_calc, C_expected, atol=1e-9):
            # If validation fails, there is a bug in the logic.
            # We will still return the matrices, but the score will be 0.
            pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
