import time
import numpy as np

def construct():
    start_time = time.time()
    t_limit = 115.0  # Safety margin before 120s limit

    n, m, p = 2, 4, 8
    
    # Rank 56 decomposition based on 8 independent 2x2x2 Strassen blocks
    # Each block contributes rank 7
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    # Define the local 2x2x2 Strassen factors
    # r=0 (M1) = (a00+a11)(b00+b11) -> c00 += M1, c11 += M1
    # r=1 (M2) = (a10+a11)b00       -> c10 += M2, c11 -= M2
    # r=2 (M3) = a00(b01-b11)       -> c01 += M3, c11 += M3
    # r=3 (M4) = a11(b10-b00)       -> c00 += M4, c10 += M4
    # r=4 (M5) = (a00+a01)b11       -> c00 -= M5, c01 += M5
    # r=5 (M6) = (a10-a00)(b00+b01) -> c10 += M6
    # r=6 (M7) = (a01-a11)(b10+b11) -> c01 += M7

    # Local U (4x7), V (4x7), W (4x7) for 2x2x2
    # Indices: 0=a00, 1=a01, 2=a10, 3=a11
    #           0=b00, 1=b01, 2=b10, 3=b11
    #           0=c00, 1=c01, 2=c10, 3=c11

    U_loc = np.zeros((4, 7))
    V_loc = np.zeros((4, 7))
    W_loc = np.zeros((4, 7))

    # M1
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1

    # M2
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1

    # M3
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1

    # M4
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1

    # M5
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1

    # M6
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1

    # M7
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1

    # We decompose the 2x4 * 4x8 multiplication into 8 independent 2x2 * 2x2 multiplications?
    # No, the previous logic mapped 8 blocks.
    # Let's trace the indices from the original working code to ensure correctness.
    # The original code iterated j in 0..3 and s in 0..1.
    # This implies 8 blocks.
    # The mapping of A (2x4) to local 2x2 blocks:
    # s=0: rows 0,1 of A map to local rows 0,1. Cols 0,1 for j=0?
    # Actually, the original code logic was:
    # for j in range(4):
    #   for s in range(2):
    #      base_r = (j * 2 + s) * 7
    #
    #      # A mapping (2x4 global -> 2x2 local)
    #      if s == 0:
    #          # Global A indices:
    #          # a_map[0] (local a00) = global 0*4 + 0 = 0
    #          # a_map[1] (local a01) = global 0*4 + 1 = 1
    #          # a_map[2] (local a10) = global 1*4 + 0 = 4
    #          # a_map[3] (local a11) = global 1*4 + 1 = 5
    #      else:
    #          # a_map[0] = global 0*4 + 2 = 2
    #          # a_map[1] = global 0*4 + 3 = 3
    #          # a_map[2] = global 1*4 + 2 = 6
    #          # a_map[3] = global 1*4 + 3 = 7
    #
    #      # B mapping (4x8 global -> 2x2 local)
    #      if s == 0:
    #          # b_map[0] (local b00) = global 0*8 + (2*j)
    #          # b_map[1] (local b01) = global 0*8 + (2*j+1)
    #          # b_map[2] (local b10) = global 1*8 + (2*j)
    #          # b_map[3] (local b11) = global 1*8 + (2*j+1)
    #      else:
    #          # b_map[0] = global 2*8 + (2*j)
    #          # b_map[1] = global 2*8 + (2*j+1)
    #          # b_map[2] = global 3*8 + (2*j)
    #          # b_map[3] = global 3*8 + (2*j+1)
    #
    #      # C mapping (8x2 global? No, p*n = 8*2=16. W is p*n x rank.
    #      # Trace convention W[k*n+i, r] where k is col of B (0..7), i is row of A (0..1)?
    #      # Standard C = AB. C is 2x8.
    #      # Tensor T_{i,j,k} = sum_l A_{i,l} B_{l,k}.
    #      # Decomposition: T = sum_r U_{i,l,r} V_{l,k,r} W_{k,i,r}
    #      # Here indices:
    #      # U: (n*m, rank) -> indices (i*m + l)? No, U is (n*m, rank).
    #      # Usually U_{i,l} is flattened to index i*m + l.
    #      # V: (m*p, rank) -> indices (l*p + k).
    #      # W: (p*n, rank) -> indices (k*n + i).
    #
    #      # Let's check the original W mapping:
    #      # c_map[0] (local c00, i=0, k=0 in local block)
    #      # Local block corresponds to specific global i, k.
    #      # The original code:
    #      # c_map[0] = (2*j) * 2 + 0  => k=2*j, i=0
    #      # c_map[1] = (2*j+1) * 2 + 0 => k=2*j+1, i=0
    #      # c_map[2] = (2*j) * 2 + 1  => k=2*j, i=1
    #      # c_map[3] = (2*j+1) * 2 + 1 => k=2*j+1, i=1
    #
    #      # This covers all 8 k values (0..7) and both i values (0,1).
    #      # So for each (j,s), we compute a 2x2 block of C?
    #      # For fixed j, k is 2j, 2j+1. i is 0,1.
    #      # So we are computing C[0:2, 2j:2j+2].
    #      # The A part used:
    #      # If s=0: A[0:2, 0:2] (cols 0,1). B part: B[0:2, 2j:2j+2] (rows 0,1).
    #      # If s=1: A[0:2, 2:4] (cols 2,3). B part: B[2:4, 2j:2j+2] (rows 2,3).
    #      
    #      # Summing over s:
    #      # C[0:2, 2j:2j+2] = A[0:2, 0:2] * B[0:2, 2j:2j+2] + A[0:2, 2:4] * B[2:4, 2j:2j+2]
    #      # This is exactly the matrix multiplication for columns 2j, 2j+1 of C.
    #      # Since j goes 0..3, we cover all 8 columns of C.
    #      # Each term is a 2x2 * 2x2 multiplication, decomposed by Strassen (rank 7).
    #      # Total rank = 4 (j) * 2 (s) * 7 = 56.

    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7
            
            # Global indices for A (2x4) -> local (2x2)
            # Local 0: i=0, l=0
            # Local 1: i=0, l=1
            # Local 2: i=1, l=0
            # Local 3: i=1, l=1
            # Global A index = i * 4 + l
            if s == 0:
                a_idx = [0*4+0, 0*4+1, 1*4+0, 1*4+1]
            else:
                a_idx = [0*4+2, 0*4+3, 1*4+2, 1*4+3]
            
            # Global indices for B (4x8) -> local (2x2)
            # Local 0: l=0, k=0
            # Local 1: l=0, k=1
            # Local 2: l=1, k=0
            # Local 3: l=1, k=1
            # Global B index = l * 8 + k
            # k corresponds to 2*j and 2*j+1
            # l corresponds to 0,1 for s=0 and 2,3 for s=1
            if s == 0:
                b_idx = [0*8+(2*j), 0*8+(2*j+1), 1*8+(2*j), 1*8+(2*j+1)]
            else:
                b_idx = [2*8+(2*j), 2*8+(2*j+1), 3*8+(2*j), 3*8+(2*j+1)]
            
            # Global indices for W (p*n = 8*2) -> local (2x2)
            # W index = k * n + i = k * 2 + i
            # Local 0: i=0, k=0 (local) -> global k=2*j, i=0
            # Local 1: i=0, k=1 (local) -> global k=2*j+1, i=0
            # Local 2: i=1, k=0 (local) -> global k=2*j, i=1
            # Local 3: i=1, k=1 (local) -> global k=2*j+1, i=1
            w_idx = [(2*j)*2 + 0, (2*j+1)*2 + 0, (2*j)*2 + 1, (2*j+1)*2 + 1]
            
            # Populate U, V, W
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_loc[la, l]
                    if val != 0:
                        U[a_idx[la], r] += val
                for lb in range(4):
                    val = V_loc[lb, l]
                    if val != 0:
                        V[b_idx[lb], r] += val
                for lc in range(4):
                    val = W_loc[lc, l]
                    if val != 0:
                        W[w_idx[lc], r] += val

    if time.time() - start_time > t_limit:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
