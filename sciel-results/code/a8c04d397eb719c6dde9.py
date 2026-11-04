import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n, m, p = 4, 5, 6
    
    # Rank 49 for 4x4x4
    rank_444 = 49
    # Rank 32 for 4x4x2
    rank_442 = 4 * 4 * 2
    # Rank 24 for 4x1x6
    rank_416 = 4 * 1 * 6
    
    total_rank = rank_444 + rank_442 + rank_416
    
    # Initialize U, V, W with zeros
    # U shape: (n*m, rank) = (20, 105)
    # V shape: (m*p, rank) = (30, 105)
    # W shape: (p*n, rank) = (24, 105)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: 4x4x4 using Recursive Strassen (Rank 49) ---
    
    # We define the 2x2x2 Strassen decomposition matrices.
    # Indices: U (i,k), V (k,j), W (j,i) with i,k,j in {0,1}.
    # U index: i*2 + k
    # V index: k*2 + j
    # W index: j*2 + i
    
    # Strassen Formulas:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 0: M0
    # U: A00(1), A11(1)
    U2[0, 0] = 1.0; U2[3, 0] = 1.0
    # V: B00(1), B11(1)
    V2[0, 0] = 1.0; V2[3, 0] = 1.0
    # W: C00(1), C11(1)
    W2[0, 0] = 1.0; W2[3, 0] = 1.0
    
    # Term 1: M1
    # U: A10(1), A11(1)
    U2[2, 1] = 1.0; U2[3, 1] = 1.0
    # V: B00(1)
    V2[0, 1] = 1.0
    # W: C10(1)
    W2[1, 1] = 1.0
    
    # Term 2: M2
    # U: A00(1)
    U2[0, 2] = 1.0
    # V: B01(1), B11(-1)
    V2[1, 2] = 1.0; V2[3, 2] = -1.0
    # W: C01(1), C11(1)
    W2[2, 2] = 1.0; W2[3, 2] = 1.0
    
    # Term 3: M3
    # U: A11(1)
    U2[3, 3] = 1.0
    # V: B10(1), B00(-1)
    V2[2, 3] = 1.0; V2[0, 3] = -1.0
    # W: C00(1), C10(1)
    W2[0, 3] = 1.0; W2[1, 3] = 1.0
    
    # Term 4: M4
    # U: A00(1), A01(1)
    U2[0, 4] = 1.0; U2[1, 4] = 1.0
    # V: B11(1)
    V2[3, 4] = 1.0
    # W: C01(1), C00(-1)
    W2[2, 4] = 1.0; W2[0, 4] = -1.0
    
    # Term 5: M5
    # U: A10(1), A00(-1)
    U2[2, 5] = 1.0; U2[0, 5] = -1.0
    # V: B00(1), B01(1)
    V2[0, 5] = 1.0; V2[1, 5] = 1.0
    # W: C11(1)
    W2[3, 5] = 1.0
    
    # Term 6: M6
    # U: A01(1), A11(-1)
    U2[1, 6] = 1.0; U2[3, 6] = -1.0
    # V: B10(1), B11(1)
    V2[2, 6] = 1.0; V2[3, 6] = 1.0
    # W: C00(1)
    W2[0, 6] = 1.0
    
    # Generate 4x4x4 terms by applying 2x2 Strassen at block level and local level.
    # Global indices: i, k, j in 0..3.
    # Block indices: ib, kb, jb in 0..1.
    # Local indices: il, kl, jl in 0..1.
    # i = ib*2 + il, k = kb*2 + kl, j = jb*2 + jl.
    
    # For each block term bt (0..6) and local term lt (0..6):
    for bt in range(7):
        for lt in range(7):
            u_acc = [0.0] * 16 # i*4+k
            v_acc = [0.0] * 16 # k*4+j
            w_acc = [0.0] * 16 # j*4+i
            
            # Accumulate U coefficients
            for ib in range(2):
                for kb in range(2):
                    ub = U2[ib*2+kb, bt]
                    if ub == 0: continue
                    for il in range(2):
                        for kl in range(2):
                            ul = U2[il*2+kl, lt]
                            if ul == 0: continue
                            i_g = ib*2 + il
                            k_g = kb*2 + kl
                            u_acc[i_g*4 + k_g] += ub * ul
            
            # Accumulate V coefficients
            for kb in range(2):
                for jb in range(2):
                    vb = V2[kb*2+jb, bt]
                    if vb == 0: continue
                    for kl in range(2):
                        for jl in range(2):
                            vl = V2[kl*2+jl, lt]
                            if vl == 0: continue
                            k_g = kb*2 + kl
                            j_g = jb*2 + jl
                            v_acc[k_g*4 + j_g] += vb * vl
            
            # Accumulate W coefficients
            for jb in range(2):
                for ib in range(2):
                    wb = W2[jb*2+ib, bt]
                    if wb == 0: continue
                    for jl in range(2):
                        for il in range(2):
                            wl = W2[jl*2+il, lt]
                            if wl == 0: continue
                            j_g = jb*2 + jl
                            i_g = ib*2 + il
                            w_acc[j_g*4 + i_g] += wb * wl
            
            # Add to global matrices
            # U index: i*m + k. Here m=5, but for the 4x4x4 part, k is 0..3.
            # Note: The global m is 5. The indices k in the 4x4x4 part are 0..3.
            for idx in range(16):
                val_u = u_acc[idx]
                if val_u != 0:
                    i_g, k_g = divmod(idx, 4)
                    U[i_g * m + k_g][col] = val_u
                
                val_v = v_acc[idx]
                if val_v != 0:
                    k_g, j_g = divmod(idx, 4)
                    V[k_g * p + j_g][col] = val_v
                
                val_w = w_acc[idx]
                if val_w != 0:
                    j_g, i_g = divmod(idx, 4)
                    W[j_g * n + i_g][col] = val_w
            
            col += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # If we ran out of time in Part 1, we should return a valid schoolbook result for the rest
    # or just ensure the partial result is valid? 
    # Given the small size, Part 1 (49 terms) should take negligible time (< 0.1s).
    # So we proceed to Part 2 and 3.

    # --- Part 2: 4x4x2 (Schoolbook) ---
    # i in 0..3, k in 0..3, j in 4..5
    for i in range(4):
        for k in range(4):
            for j in range(4, 6):
                U[i * m + k][col] = 1.0
                V[k * p + j][col] = 1.0
                W[j * n + i][col] = 1.0
                col += 1
                
    # --- Part 3: 4x1x6 (Schoolbook) ---
    # i in 0..3, k=4, j in 0..5
    for i in range(4):
        for j in range(6):
            U[i * m + 4][col] = 1.0
            V[4 * p + j][col] = 1.0
            W[j * n + i][col] = 1.0
            col += 1
            
    # Convert to lists if needed, but numpy arrays are accepted? 
    # The task says "factor matrices of shapes...". Lists of lists work for the interface.
    return {"U": U, "V": V, "W": W}

