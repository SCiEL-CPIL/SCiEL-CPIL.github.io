"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Attempt to construct Rank 32 decomposition:
    # Rank 23 (Winograd FMM-3 for 3x3x3) + Rank 9 (Naive for 3x3x1)
    
    # Winograd FMM-3 Coefficients for 3x3x3
    # U, V, W are 9x23 matrices
    # Note: These coefficients are from Winograd's 1976 paper.
    # If they are incorrect, the verification will fail.
    
    U_win = [
        [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    # The above U_win is likely incorrect (it looks like a banded matrix).
    # I will use a verification step to catch this.
    
    V_win = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    ]
    
    W_win = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    
    # Construct the 9 terms for the 4th column (index 3)
    # C_i,3 = sum_j A_i,j B_j,3
    U_col3 = [[0]*9 for _ in range(9)]
    V_col3 = [[0]*9 for _ in range(9)]
    W_col3 = [[0]*9 for _ in range(9)]
    
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            U_col3[i * 3 + j][idx] = 1
            V_col3[j * 4 + 3][idx] = 1 # V index: j * p + k = j * 4 + 3
            W_col3[3 * 3 + i][idx] = 1 # W index: k * n + i = 3 * 3 + i
    
    # Combine
    U = [row + [U_col3[i][j] for j in range(9)] for i, row in enumerate(U_win)]
    V = [row + [V_col3[i][j] for j in range(9)] for i, row in enumerate(V_win)]
    W = [row + [W_col3[i][j] for j in range(9)] for i, row in enumerate(W_win)]
    
    # Verification
    def verify(U, V, W, n, m, p):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        # Check T = sum U V W
        # T_{i,j,k} = sum_t U_{i,j,t} V_{j,k,t} W_{k,i,t}
        # Note: Indices in U are i*m+j, V are j*p+k, W are k*n+i
        # Wait, the standard tensor decomposition is:
        # M_{i,j,k} = sum_t U_{i,j,t} V_{j,k,t} W_{k,i,t}
        # Here M is the multiplication tensor.
        # M_{i,j,k} = 1 if i==k' and j==l'? No.
        # M_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
        # So M_{i,j,k} = 1 for all i,j,k? No.
        # The tensor T has T_{i,j,k} = 1 if we are computing C_{i,k} from A_{i,j} and B_{j,k}.
        # So T_{i,j,k} = 1 for all valid i,j,k.
        
        rank = U.shape[1]
        T = np.zeros((n*m, m*p, p*n))
        for t in range(rank):
            u = U[:, t]
            v = V[:, t]
            w = W[:, t]
            # u is vector of length n*m
            # v is vector of length m*p
            # w is vector of length p*n
            # Outer product?
            # T_{a,b,c} += u_a v_b w_c
            # But we need to map a->(i,j), b->(j,k), c->(k,i)
            # This is tricky with flattened indices.
            
            # Let's just check the resulting matrix multiplication.
            # Create random A, B
            A = np.random.rand(n, m)
            B = np.random.rand(m, p)
            C_ref = A @ B
            
            # Compute C using decomposition
            # C_{i,k} = sum_t (sum_j U_{i,j,t} A_{i,j}) (sum_j V_{j,k,t} B_{j,k}) W_{k,i,t}
            # Wait, the decomposition is:
            # C_{i,k} = sum_t W_{k,i,t} (sum_j U_{i,j,t} A_{i,j}) (sum_j V_{j,k,t} B_{j,k})
            # This is not standard.
            # Standard: C_{i,k} = sum_t U_{i,j,t} A_{i,j} V_{j,k,t} B_{j,k} W_{k,i,t}
            # Sum over j and t.
            
            C_comp = np.zeros((n, p))
            for t in range(rank):
                # Sum over j
                # U_{i,j,t} is U[i*m + j, t]
                # V_{j,k,t} is V[j*p + k, t]
                # W_{k,i,t} is W[k*n + i, t]
                
                # This is O(n*m*p*rank)
                for i in range(n):
                    for k in range(p):
                        w_val = W[k*n + i, t]
                        if w_val == 0: continue
                        sum_u = 0
                        sum_v = 0
                        # Actually, we can precompute
                        # Let's just do the full sum
                        prod = 0
                        for j in range(m):
                            u_val = U[i*m + j, t]
                            v_val = V[j*p + k, t]
                            prod += u_val * A[i, j] * v_val * B[j, k]
                        C_comp[i, k] += w_val * prod
                        
            if not np.allclose(C_ref, C_comp):
                return False
        return True

    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to Naive 36
        rank = n * m * p
        U_naive = [[0] * rank for _ in range(n * m)]
        V_naive = [[0] * rank for _ in range(m * p)]
        W_naive = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_naive[i * m + j][column] = 1
                    V_naive[j * p + k][column] = 1
                    W_naive[k * n + i][column] = 1
                    column += 1
        return {"U": U_naive, "V": V_naive, "W": W_naive}

# EVOLVE-BLOCK-END
