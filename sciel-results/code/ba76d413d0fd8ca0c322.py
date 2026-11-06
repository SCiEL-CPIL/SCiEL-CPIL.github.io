import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block size 2
    block_size = 2
    num_blocks = n // block_size
    
    # Total rank
    # 27 block combinations * 7 Strassen multiplications
    rank = (num_blocks ** 3) * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen coefficients for 2x2
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # Note: The task uses cyclic W index k*n+i.
    # Standard Strassen outputs c00, c01, c10, c11.
    # c_ij corresponds to W index (k=i, j=j)? No.
    # W index is k*n + i.
    # In the block context, local indices are i', j', k'.
    # W local index is k' + i'? No, global.
    
    # Let's map the Strassen output c_{i', k'} to W.
    # W is indexed by (k, i).
    # So c_{i', k'} contributes to W at global index (k_global, i_global).
    # k_global = K * 2 + k'
    # i_global = I * 2 + i'
    # W_idx = k_global * n + i_global
    
    for I in range(num_blocks):
        for J in range(num_blocks):
            for K in range(num_blocks):
                # Base indices for this block combination
                # A block (I, J)
                # B block (J, K)
                # C block (K, I)
                
                # Process 7 Strassen terms
                for s in range(7):
                    c_idx = col + s
                    
                    # Define U, V, W contributions
                    # We need to add to U, V, W at specific rows
                    
                    if s == 0:
                        # M1 = (a00+a11)(b00+b11)
                        # U: a00, a11
                        # V: b00, b11
                        # W: c00 (+1), c11 (+1)
                        
                        # U
                        i0 = I * 2 + 0; j0 = J * 2 + 0
                        i1 = I * 2 + 1; j1 = J * 2 + 1
                        U[i0 * m + j0][c_idx] += 1
                        U[i1 * m + j1][c_idx] += 1
                        
                        # V
                        j0 = J * 2 + 0; k0 = K * 2 + 0
                        j1 = J * 2 + 1; k1 = K * 2 + 1
                        V[j0 * p + k0][c_idx] += 1
                        V[j1 * p + k1][c_idx] += 1
                        
                        # W
                        # c00 -> i'=0, k'=0 -> W idx (k0, i0)
                        # c11 -> i'=1, k'=1 -> W idx (k1, i1)
                        k0 = K * 2 + 0; i0 = I * 2 + 0
                        k1 = K * 2 + 1; i1 = I * 2 + 1
                        W[k0 * n + i0][c_idx] += 1
                        W[k1 * n + i1][c_idx] += 1
                        
                    elif s == 1:
                        # M2 = (a10+a11)b00
                        # U: a10, a11
                        # V: b00
                        # W: c10 (+1), c11 (-1)
                        
                        i0 = I * 2 + 1; j0 = J * 2 + 0
                        i1 = I * 2 + 1; j1 = J * 2 + 1
                        U[i0 * m + j0][c_idx] += 1
                        U[i1 * m + j1][c_idx] += 1
                        
                        j0 = J * 2 + 0; k0 = K * 2 + 0
                        V[j0 * p + k0][c_idx] += 1
                        
                        # c10 -> i'=1, k'=0
                        # c11 -> i'=1, k'=1
                        k0 = K * 2 + 0; i1 = I * 2 + 1
                        k1 = K * 2 + 1
                        W[k0 * n + i1][c_idx] += 1
                        W[k1 * n + i1][c_idx] -= 1

                    elif s == 2:
                        # M3 = a00(b01-b11)
                        # U: a00
                        # V: b01, -b11
                        # W: c01 (+1), c11 (+1)
                        
                        i0 = I * 2 + 0; j0 = J * 2 + 0
                        U[i0 * m + j0][c_idx] += 1
                        
                        j0 = J * 2 + 0; k1 = K * 2 + 1
                        j1 = J * 2 + 1
                        V[j0 * p + k1][c_idx] += 1
                        V[j1 * p + k1][c_idx] -= 1
                        
                        # c01 -> i'=0, k'=1
                        # c11 -> i'=1, k'=1
                        k1 = K * 2 + 1; i0 = I * 2 + 0
                        i1 = I * 2 + 1
                        W[k1 * n + i0][c_idx] += 1
                        W[k1 * n + i1][c_idx] += 1

                    elif s == 3:
                        # M4 = a11(b10-b00)
                        # U: a11
                        # V: b10, -b00
                        # W: c00 (+1), c10 (+1)
                        
                        i1 = I * 2 + 1; j1 = J * 2 + 1
                        U[i1 * m + j1][c_idx] += 1
                        
                        j1 = J * 2 + 1; k0 = K * 2 + 0
                        j0 = J * 2 + 0
                        V[j1 * p + k0][c_idx] += 1
                        V[j0 * p + k0][c_idx] -= 1
                        
                        # c00 -> i'=0, k'=0
                        # c10 -> i'=1, k'=0
                        k0 = K * 2 + 0; i0 = I * 2 + 0
                        i1 = I * 2 + 1
                        W[k0 * n + i0][c_idx] += 1
                        W[k0 * n + i1][c_idx] += 1

                    elif s == 4:
                        # M5 = (a00+a01)b11
                        # U: a00, a01
                        # V: b11
                        # W: c00 (-1), c01 (+1)
                        
                        i0 = I * 2 + 0; j0 = J * 2 + 0
                        j1 = J * 2 + 1
                        U[i0 * m + j0][c_idx] += 1
                        U[i0 * m + j1][c_idx] += 1
                        
                        j1 = J * 2 + 1; k1 = K * 2 + 1
                        V[j1 * p + k1][c_idx] += 1
                        
                        # c00 -> i'=0, k'=0
                        # c01 -> i'=0, k'=1
                        k0 = K * 2 + 0; i0 = I * 2 + 0
                        k1 = K * 2 + 1
                        W[k0 * n + i0][c_idx] -= 1
                        W[k1 * n + i0][c_idx] += 1

                    elif s == 5:
                        # M6 = (a10-a00)(b00+b01)
                        # U: a10, -a00
                        # V: b00, b01
                        # W: c11 (+1)
                        
                        i1 = I * 2 + 1; j0 = J * 2 + 0
                        i0 = I * 2 + 0
                        U[i1 * m + j0][c_idx] += 1
                        U[i0 * m + j0][c_idx] -= 1
                        
                        j0 = J * 2 + 0; k0 = K * 2 + 0
                        k1 = K * 2 + 1
                        V[j0 * p + k0][c_idx] += 1
                        V[j0 * p + k1][c_idx] += 1
                        
                        # c11 -> i'=1, k'=1
                        k1 = K * 2 + 1; i1 = I * 2 + 1
                        W[k1 * n + i1][c_idx] += 1

                    elif s == 6:
                        # M7 = (a01-a11)(b10+b11)
                        # U: a01, -a11
                        # V: b10, b11
                        # W: c00 (+1)
                        
                        i0 = I * 2 + 0; j1 = J * 2 + 1
                        i1 = I * 2 + 1
                        U[i0 * m + j1][c_idx] += 1
                        U[i1 * m + j1][c_idx] -= 1
                        
                        j1 = J * 2 + 1; k0 = K * 2 + 0
                        k1 = K * 2 + 1
                        V[j1 * p + k0][c_idx] += 1
                        V[j1 * p + k1][c_idx] += 1
                        
                        # c00 -> i'=0, k'=0
                        k0 = K * 2 + 0; i0 = I * 2 + 0
                        W[k0 * n + i0][c_idx] += 1

                col += 7
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
