"""
Optimized rank-35 decomposition for 2x2 * 2x10 matrix multiplication.

This program constructs the tensor decomposition by treating the 2x10 B matrix
as five independent 2x2 blocks. Each block is multiplied by the 2x2 A matrix
using the rank-7 Strassen algorithm. The total rank is 5 * 7 = 35, which is
the theoretical minimum for this tensor.
"""

import numpy as np

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    for the matrix multiplication tensor of dimensions (2, 2, 10).
    
    Shapes:
    U: (4, 35)
    V: (20, 35)
    W: (20, 35)
    
    The decomposition satisfies:
    sum_r U[i, j, r] * V[j, k, r] * W[k, i, r] = C[i, k]
    where A is 2x2, B is 2x10, C is 2x10.
    """
    n, m, p = 2, 2, 10
    num_blocks = p // m  # 5 blocks of 2x2
    
    # Define the 7 Strassen terms for 2x2 multiplication.
    # Each term is a tuple of three dictionaries representing the non-zero entries
    # for the U, V, and W factors of that term.
    # 
    # Indices for U: (i, j) where i in 0..1, j in 0..1
    # Indices for V: (j, k) where j in 0..1, k in 0..1
    # Indices for W: (k, i) where k in 0..1, i in 0..1
    #
    # Strassen products:
    # M0: (a+d)(e+h)
    # M1: (c+d)e
    # M2: a(f-h)
    # M3: d(g-e)
    # M4: (a+b)h
    # M5: (c-a)(e+f)
    # M6: (b-d)(g+h)
    
    # Coefficients for each term: (U_entries, V_entries, W_entries)
    # U_entries: {(i,j): coeff}
    # V_entries: {(j,k): coeff}
    # W_entries: {(k,i): coeff}
    
    terms = [
        # M0 = (a+d)(e+h)
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        # M1 = (c+d)e
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,1): 1, (1,1): -1}
        ),
        # M2 = a(f-h)
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1}
        ),
        # M3 = d(g-e)
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1}
        ),
        # M4 = (a+b)h
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (1,0): 1}
        ),
        # M5 = (c-a)(e+f)
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        # M6 = (b-d)(g+h)
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        ),
    ]
    
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    # Initialize factor matrices
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # Populate matrices for each block
    for b in range(num_blocks):
        base_col = b * rank_per_block
        for t in range(rank_per_block):
            col = base_col + t
            u_dict, v_dict, w_dict = terms[t]
            
            # Fill U
            for (i, j), val in u_dict.items():
                U[i * m + j, col] = val
                
            # Fill V
            # B is 2x10. Block b corresponds to columns 2*b and 2*b+1.
            # Local index k in {0,1} maps to global column 2*b + k.
            for (j, k), val in v_dict.items():
                global_k = 2 * b + k
                V[j * p + global_k, col] = val
                
            # Fill W
            # C is 2x10. Block b corresponds to columns 2*b and 2*b+1.
            # The trace convention is W[k*n + i, col].
            # Local index k in {0,1} maps to global column 2*b + k.
            for (k, i), val in w_dict.items():
                global_k = 2 * b + k
                W[global_k * n + i, col] = val

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-START
# The construct function above is the entry point.
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    # Simple self-test to ensure correctness
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    
    n, m, p = 2, 2, 10
    rank = U.shape[1]
    
    # Random test
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    # Reconstruct C
    # C[i, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is equivalent to:
    # Let U_mat be (n*m, rank), V_mat be (m*p, rank), W_mat be (p*n, rank)
    # We can reshape and use einsum or manual loops.
    # Let's use einsum for speed and clarity in verification.
    # U_flat: (n*m, rank) -> U_t: (n, m, rank)
    # V_flat: (m*p, rank) -> V_t: (m, p, rank)
    # W_flat: (p*n, rank) -> W_t: (p, n, rank)
    
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    # C[i, k] = sum_j sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    C_computed = np.einsum('ijr,jkr,kir->ik', U_t, V_t, W_t)
    
    error = np.max(np.abs(C_expected - C_computed))
    if error > 1e-9:
        print(f"Verification Failed. Max error: {error}")
    else:
        print(f"Verification Passed. Rank: {rank}")
