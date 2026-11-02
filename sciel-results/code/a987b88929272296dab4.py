"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time

def construct():
    """Rank-42 Strassen decomposition for 2x2 * 2x12 matrix multiplication.
    
    Partition 12 columns into 6 groups of 2. For each 2x2 * 2x2 block,
    use Strassen's rank-7 decomposition.
    Total rank = 6 * 7 = 42.
    
    Indexing conventions:
      U[i*m+j, r]: row = i*m+j (i in 0..n-1, j in 0..m-1) for A[i][j]
      V[j*p+k, r]: row = j*p+k (j in 0..m-1, k in 0..p-1) for B[j][k]
      W[k*n+i, r]: row = k*n+i (k in 0..p-1, i in 0..n-1) for C[i][k] (trace convention)
    """
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    num_blocks = p // 2  # 6
    rank = num_blocks * 7  # 42
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 24 x 42
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Strassen 7-product decomposition for 2x2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[C00,C01],[C10,C11]]
    # Local indices:
    # A: 0=a, 1=b, 2=c, 3=d
    # B: 0=e, 1=f, 2=g, 3=h
    # C: 0=C00, 1=C01, 2=C10, 3=C11
    
    # The 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define coefficients for each term (M1..M7)
    # Each term is (U_coeffs, V_coeffs, W_coeffs)
    strassen_terms = [
        # M1 = (a+d)(e+h)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (c+d)e
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = a(f-h)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = d(g-e)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (a+b)h
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (c-a)(e+f)
        ({0: -1, 2: 1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (b-d)(g+h)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    for t in range(num_blocks):
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
        for s in range(7):
            col = t * 7 + s
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[s]
            
            # U: A indices are global (0..3), no offset
            for a_idx, val in u_coeffs.items():
                U[a_idx][col] = val
            
            # V: B indices are local to the block
            # Local B index 0 -> B[0, 2t] -> global index 0*p + 2t = 2t
            # Local B index 1 -> B[0, 2t+1] -> global index 0*p + 2t+1 = 2t+1
            # Local B index 2 -> B[1, 2t] -> global index 1*p + 2t = 12+2t
            # Local B index 3 -> B[1, 2t+1] -> global index 1*p + 2t+1 = 12+2t+1
            for local_b, val in v_coeffs.items():
                if local_b == 0:
                    global_b = 2 * t
                elif local_b == 1:
                    global_b = 2 * t + 1
                elif local_b == 2:
                    global_b = m * p // 2 + 2 * t  # 12 + 2t
                else:  # local_b == 3
                    global_b = m * p // 2 + 2 * t + 1  # 12 + 2t + 1
                V[global_b][col] = val
            
            # W: C indices are local to the block
            # Local C index 0 -> C[0, 2t] -> W index k*n+i = (2t)*2 + 0 = 4t
            # Local C index 1 -> C[0, 2t+1] -> W index (2t+1)*2 + 0 = 4t+2
            # Local C index 2 -> C[1, 2t] -> W index (2t)*2 + 1 = 4t+1
            # Local C index 3 -> C[1, 2t+1] -> W index (2t+1)*2 + 1 = 4t+3
            for local_c, val in w_coeffs.items():
                if local_c == 0:
                    global_w = (2 * t) * n + 0
                elif local_c == 1:
                    global_w = (2 * t + 1) * n + 0
                elif local_c == 2:
                    global_w = (2 * t) * n + 1
                else:  # local_c == 3
                    global_w = (2 * t + 1) * n + 1
                W[global_w][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
