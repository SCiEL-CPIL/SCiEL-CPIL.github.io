"""Rewritten program using Strassen's algorithm for 4x4 matrix multiplication."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define 2x2 Strassen coefficients
    # U_in: 7 products, each is a vector of 4 coefficients for A (flattened row-major)
    U_in = [
        [1, 0, 0, 1], # M0: A00+A11
        [0, 0, 1, 1], # M1: A10+A11
        [1, 0, 0, 0], # M2: A00
        [0, 0, 0, 1], # M3: A11
        [1, 1, 0, 0], # M4: A00+A01
        [-1, 0, 1, 0],# M5: A10-A00
        [0, 1, 0, -1] # M6: A01-A11
    ]
    
    # V_in: 7 products, each is a vector of 4 coefficients for B (flattened row-major)
    V_in = [
        [1, 0, 0, 1], # M0: B00+B11
        [1, 0, 0, 0], # M1: B00
        [0, 1, 0, -1],# M2: B01-B11
        [-1, 0, 1, 0],# M3: B10-B00
        [0, 0, 0, 1], # M4: B11
        [1, 1, 0, 0], # M5: B00+B01
        [0, 0, 1, 1]  # M6: B10+B11
    ]
    
    # W_in: 7 products, each is a vector of 4 coefficients for C (flattened k*n+i)
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    W_in = [
        [1, 0, 0, 1], # M0: C00+1, C11+1
        [0, 0, 1, -1],# M1: C10+1, C11-1
        [0, 1, 0, 1], # M2: C01+1, C11+1
        [1, 0, 1, 0], # M3: C00+1, C10+1
        [-1, 1, 0, 0],# M4: C00-1, C01+1
        [0, 0, 0, 1], # M5: C11+1
        [1, 0, 0, 0]  # M6: C00+1
    ]
    
    # Construct 4x4 matrices
    # Rank index r = s1 * 7 + s2
    # s1: outer Strassen step (block level)
    # s2: inner Strassen step (sub-block level)
    
    for r in range(rank):
        s1 = r // 7
        s2 = r % 7
        
        # Outer coefficients for blocks
        # U_out[s1] gives coefficients for blocks A00, A01, A10, A11
        # We need to map these to the specific U_in, V_in, W_in structures.
        # Actually, the structure is:
        # U[i*4+j, r] = U_outer[I, J, s1] * U_inner[i', j', s2]
        # where I=i//2, J=j//2, i'=i%2, j'=j%2
        
        # Let's precompute U_outer, V_outer, W_outer as 7x4 vectors (flattened block indices)
        # Block indices: 0=A00, 1=A01, 2=A10, 3=A11
        
        # We can reuse U_in, V_in, W_in for the outer step too, because the 
        # Strassen algorithm is self-similar. The "scalars" in the outer step 
        # are the blocks, and the coefficients are the same.
        
        U_outer = U_in[s1]
        V_outer = V_in[s1]
        W_outer = W_in[s1]
        
        # Iterate over all elements in the 4x4 matrices
        for i in range(4):
            for j in range(4):
                I = i // 2
                J = j // 2
                i_prime = i % 2
                j_prime = j % 2
                
                # Index in flattened 2x2 block
                idx_block = I * 2 + J
                idx_sub = i_prime * 2 + j_prime
                
                coeff = U_outer[idx_block] * U_in[s2][idx_sub]
                U[i * 4 + j][r] = coeff
        
        for j in range(4):
            for k in range(4):
                J = j // 2
                K = k // 2
                j_prime = j % 2
                k_prime = k % 2
                
                idx_block = J * 2 + K
                idx_sub = j_prime * 2 + k_prime
                
                coeff = V_outer[idx_block] * V_in[s2][idx_sub]
                V[j * 4 + k][r] = coeff
        
        for k in range(4):
            for i in range(4):
                K = k // 2
                I = i // 2
                k_prime = k % 2
                i_prime = i % 2
                
                idx_block = K * 2 + I
                idx_sub = k_prime * 2 + i_prime
                
                coeff = W_outer[idx_block] * W_in[s2][idx_sub]
                W[k * 4 + i][r] = coeff
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
