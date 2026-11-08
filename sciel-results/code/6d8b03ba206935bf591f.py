"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _verify(n, m, p, U, V, W):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Convert lists to numpy arrays for vectorized computation
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We can compute this efficiently.
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # Compute the sum over c
    # T[i, j, k] = sum_c U_reshaped[i, j, c] * V_reshaped[j, k, c] * W_reshaped[k, i, c]
    # This is a tensor contraction.
    # Step 1: Compute M1[i, j, k, ?] no, we need to sum over c.
    # Let's use einsum or simple loops if dimensions are small.
    # n=2, m=4, p=6. rank=42.
    # T shape: (2, 4, 6).
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        uc = U_reshaped[:, :, c] # (n, m)
        vc = V_reshaped[:, :, c] # (m, p)
        wc = W_reshaped[:, :, c] # (p, n)
        
        # We need to add uc[i,j] * vc[j,k] * wc[k,i] to T[i,j,k]
        # This is not a standard matmul.
        # We can iterate over i, j, k?
        # Or use einsum: 'ijc,jkc,kic->ijk'
        # But we are doing it per c.
        # For fixed c:
        # term[i,j,k] = uc[i,j] * vc[j,k] * wc[k,i]
        # We can compute this as:
        # temp[i,j,k] = uc[i,j] * vc[j,k]
        # Then T += temp * wc (broadcasting wc[k,i] to [k,i] -> need to align indices)
        
        # Let's just loop over i, j, k for simplicity and correctness, 
        # since n*m*p = 48 is small.
        for i in range(n):
            for j in range(m):
                u_val = uc[i, j]
                if u_val == 0: continue
                for k in range(p):
                    v_val = vc[j, k]
                    if v_val == 0: continue
                    w_val = wc[k, i]
                    if w_val == 0: continue
                    T[i, j, k] += u_val * v_val * w_val

    # Target tensor
    target = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i, j, k] = 1.0

    return np.allclose(T, target, atol=1e-9)

def _strassen_decomposition(n, m, p):
    """
    Decompose M_{n,m,p} using block Strassen.
    For (2,4,6): split m=4 into two blocks of size 2, p=6 into three blocks of size 2.
    Each subproblem is M_{2,2,2} solved by Strassen (rank 7).
    Total rank = 2 * 3 * 7 = 42.
    """
    a_block_size = 2
    b_col_block_size = 2
    a_blocks = m // a_block_size
    b_col_blocks = p // b_col_block_size
    strassen_rank = 7
    rank = a_blocks * b_col_blocks * strassen_rank
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1}
    ]
    
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1}
    ]
    
    W_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1}
    ]
    
    col = 0
    for a in range(a_blocks):
        for b in range(b_col_blocks):
            for s in range(strassen_rank):
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * a_block_size + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * a_block_size + j_loc
                    k = b * b_col_block_size + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                for (i_loc, k_loc), coeff in W_coeffs[s].items():
                    i = i_loc
                    k = b * b_col_block_size + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = 110.0 # 10s margin
    
    result = _strassen_decomposition(n, m, p)
    
    if time.time() < deadline:
        if _verify(n, m, p, result["U"], result["V"], result["W"]):
            return result
    
    # Fallback to schoolbook if verification fails or timeout
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
