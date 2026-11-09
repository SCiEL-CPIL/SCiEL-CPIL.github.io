"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if len(U) > 0 else 0
    if rank == 0:
        return False
        
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T[(i,j), (jp,k), (kp,ip)] = delta(i,ip)*delta(j,jp)*delta(k,kp)
    # Check: sum_c U[i*m+j,c] * V[jp*p+k,c] * W[kp*n+ip,c] = delta(i,ip)*delta(j,jp)*delta(k,kp)
    
    # Vectorized check
    # We can check all combinations at once or in batches.
    # n=2, m=4, p=6.
    # Total checks: 2*4*4*6*6*2 = 2304.
    # This is small, so a loop is fine, but vectorized is better.
    
    # Create indices
    # We want to compute S[i, j, jp, k, kp, ip] = sum_c U[i*m+j, c] V[jp*p+k, c] W[kp*n+ip, c]
    
    # Reshape U, V, W to (dim1, dim2, rank)
    # U: (n*m, rank) -> (n, m, rank)
    U_reshaped = U_arr.reshape(n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # We need to sum over c.
    # S[i, j, jp, k, kp, ip] = sum_c U_reshaped[i, j, c] * V_reshaped[jp, k, c] * W_reshaped[kp, ip, c]
    
    # This is a tensor contraction.
    # We can use np.einsum
    # 'ijc,jkc,kpc -> ...' no, indices are mixed.
    # U is indexed by i, j.
    # V is indexed by jp, k.
    # W is indexed by kp, ip.
    
    # Let's compute the product for all i, j, jp, k, kp, ip.
    # This might be memory heavy if not careful, but n,m,p are small.
    
    # U_reshaped: (n, m, rank)
    # V_reshaped: (m, p, rank)
    # W_reshaped: (p, n, rank)
    
    # We want T[i, j, jp, k, kp, ip]
    # Term = U[i,j,c] * V[jp,k,c] * W[kp,ip,c]
    
    # Use einsum:
    # 'ijc,jkc,kpc -> ijjpkp' ? No.
    # i, j from U.
    # jp, k from V.
    # kp, ip from W.
    # Sum over c.
    
    # Result shape: (n, m, m, p, p, n)
    try:
        T = np.einsum('ijc,jkc,kpc->ijjjkkii', U_reshaped, V_reshaped, W_reshaped) 
        # Wait, einsum indices must be unique letters.
        # i, j, c
        # jp, k, c -> use a, b, c
        # kp, ip, c -> use d, e, c
        # Result: i, j, a, b, d, e
        T = np.einsum('ijc,abc,dec->ijabde', U_reshaped, V_reshaped, W_reshaped)
        
        # T[i, j, a, b, d, e] corresponds to i, j, jp=a, k=b, kp=d, ip=e
        # We want to check if T[i, j, a, b, d, e] == 1 if i==e, j==a, b==d else 0.
        
        # Create target tensor
        # Target[i, j, a, b, d, e] = 1 if i==e and j==a and b==d else 0
        
        # We can construct this using broadcasting.
        # i, j, a, b, d, e
        # i: 0..n-1
        # j: 0..m-1
        # a: 0..m-1
        # b: 0..p-1
        # d: 0..p-1
        # e: 0..n-1
        
        idx_i = np.arange(n)[:, None, None, None, None, None]
        idx_j = np.arange(m)[None, :, None, None, None, None]
        idx_a = np.arange(m)[None, None, :, None, None, None]
        idx_b = np.arange(p)[None, None, None, :, None, None]
        idx_d = np.arange(p)[None, None, None, None, :, None]
        idx_e = np.arange(n)[None, None, None, None, None, :]
        
        target = (idx_i == idx_e) & (idx_j == idx_a) & (idx_b == idx_d)
        target = target.astype(float)
        
        if np.allclose(T, target, atol=1e-5):
            return True
        return False
    except Exception:
        return False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Schoolbook decomposition
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # No search is performed to ensure reliability and speed.
    # The schoolbook decomposition is the known safe baseline.
    
    return best_result
# EVOLVE-BLOCK-END
