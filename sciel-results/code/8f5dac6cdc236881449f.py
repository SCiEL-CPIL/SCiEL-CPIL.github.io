"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Try WFTT approach first, then fall back to block-Strassen 84
    # WFTT: Use FFT over Z_4 for rows/cols of A, and Z_6 for cols of B
    
    # Strategy: Decompose 4x4*4x6 into spectral components
    # For 4x4 matrices, use FFT over Z_4 (4 points)
    # For 4x6, the column dimension of B is 6, use FFT over Z_6 (6 points)
    # The rank would be product of spectral supports
    
    # Let's try a systematic approach: build the tensor and find minimal rank
    # via checking if lower-rank decompositions exist
    
    # First, try the known block-Strassen rank 84 as baseline
    def build_block_strassen():
        rank = 84
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(2):
            for k in range(3):
                for j in range(2):
                    a_idx = 2*i * m + 2*j
                    b_idx = 2*i * m + 2*j + 1
                    c_idx = (2*i+1) * m + 2*j
                    d_idx = (2*i+1) * m + 2*j + 1
                    
                    e_idx = 2*j * p + 2*k
                    f_idx = 2*j * p + 2*k + 1
                    g_idx = (2*j+1) * p + 2*k
                    h_idx = (2*j+1) * p + 2*k + 1
                    
                    c11_idx = (2*k) * n + 2*i
                    c12_idx = (2*k+1) * n + 2*i
                    c21_idx = (2*k) * n + 2*i + 1
                    c22_idx = (2*k+1) * n + 2*i + 1
                    
                    # M1 = (a+d)(e+h)
                    U[a_idx][col] = 1; U[d_idx][col] = 1
                    V[e_idx][col] = 1; V[h_idx][col] = 1
                    W[c11_idx][col] = 1; W[c22_idx][col] = 1
                    col += 1
                    
                    # M2 = (c+d)e
                    U[c_idx][col] = 1; U[d_idx][col] = 1
                    V[e_idx][col] = 1
                    W[c21_idx][col] = 1; W[c22_idx][col] = -1
                    col += 1
                    
                    # M3 = a(f-h)
                    U[a_idx][col] = 1
                    V[f_idx][col] = 1; V[h_idx][col] = -1
                    W[c12_idx][col] = 1; W[c22_idx][col] = 1
                    col += 1
                    
                    # M4 = d(g-e)
                    U[d_idx][col] = 1
                    V[g_idx][col] = 1; V[e_idx][col] = -1
                    W[c11_idx][col] = 1; W[c21_idx][col] = 1
                    col += 1
                    
                    # M5 = (a+b)h
                    U[a_idx][col] = 1; U[b_idx][col] = 1
                    V[h_idx][col] = 1
                    W[c11_idx][col] = -1; W[c12_idx][col] = 1
                    col += 1
                    
                    # M6 = (c-a)(e+f)
                    U[c_idx][col] = 1; U[a_idx][col] = -1
                    V[e_idx][col] = 1; V[f_idx][col] = 1
                    W[c22_idx][col] = 1
                    col += 1
                    
                    # M7 = (b-d)(g+h)
                    U[b_idx][col] = 1; U[d_idx][col] = -1
                    V[g_idx][col] = 1; V[h_idx][col] = 1
                    W[c11_idx][col] = 1
                    col += 1
        
        return U, V, W, rank
    
    def verify_decomposition(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        T_expected = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_expected[i*m+j, j*p+k, k*n+i] = 1
        
        T_decomp = np.zeros_like(T_expected)
        for r in range(rank):
            T_decomp += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
        
        return np.allclose(T_decomp, T_expected, atol=1e-10)
    
    # Build and verify block-Strassen 84
    U_bs, V_bs, W_bs, rank_bs = build_block_strassen()
    if verify_decomposition(U_bs, V_bs, W_bs, rank_bs):
        # Try to find lower rank using WFTT-inspired approach
        # The key insight: for 4x4*4x6, we can potentially use
        # 4-point FFT on the 4x4 blocks and 6-point FFT on the 4x6 blocks
        
        # Let's try rank 72: 4*4*4.5 = 72? No, need integer.
        # Try various ranks below 84
        
        # Actually, let's think about this more carefully.
        # The tensor is T[i,j,k] = delta_{i*m+j, j*p+k, k*n+i}
        # 
        # WFTT approach: If we can diagonalize the multiplication structure
        # using Fourier transforms, the rank becomes the number of non-zero
        # spectral components.
        
        # For 4x4*4x6:
        # - A is 4x4: can use 4-point FFT on both row and col indices
        # - B is 4x6: can use 4-point FFT on row, 6-point FFT on col
        # - C is 4x6: same as B
        #
        # The multiplication in spectral domain: 
        # A_hat * B_hat = C_hat
        # where the multiplication is pointwise in some sense.
        
        # This is complex. Let me try a different approach:
        # Use the fact that rank(4x4*4x6) <= rank(4x4*4x4) + rank(4x4*4x2)
        # rank(4x4*4x4) = 49 (Strassen-based: 7*7=49? No, that's not right)
        # Actually rank(4x4*4x4) is known to be at most 49 via Strassen
        # But we need 4x4*4x6.
        
        # Let me try: decompose into 4x4*4x4 + 4x4*4x2
        # rank(4x4*4x4) = 49 (using Strassen: 2^3 * 7 = 56? or 7^2=49?)
        # Actually, for 4x4*4x4, using Strassen on 2x2 blocks:
        # 4 block products, each with 7 mults = 28? No...
        # 2x2 blocks: 4 block positions, each 2x2*2x2 needs 7 mults = 28
        # But that's not optimal. Known bound is 49 for 4x4*4x4.
        
        # For 4x4*4x2: 
        # This is like multiplying a 4x4 by a 4x2.
        # Can view as 2 separate 4x4*4x1 problems? No.
        # Or use Strassen-like: 2x2*2x1 blocks.
        
        # Let me just try to verify if rank 84 is indeed valid and return it.
        # The WFTT idea is promising but implementing it correctly is complex.
        
        # For now, return the verified block-Strassen 84.
        return {"U": U_bs, "V": V_bs, "W": W_bs}
    else:
        # Fallback to naive rank-96
        rank_naive = n * m * p
        U_naive = [[0] * rank_naive for _ in range(n * m)]
        V_naive = [[0] * rank_naive for _ in range(m * p)]
        W_naive = [[0] * rank_naive for _ in range(p * n)]
        col_n = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_naive[i*m+j][col_n] = 1
                    V_naive[j*p+k][col_n] = 1
                    W_naive[k*n+i][col_n] = 1
                    col_n += 1
        return {"U": U_naive, "V": V_naive, "W": W_naive}
# EVOLVE-BLOCK-END
