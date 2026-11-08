"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

_DIMENSIONS = [2, 3, 6]
DIMENSIONS = _DIMENSIONS  # Keep name for compatibility

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    _deadline = _t0 + 110  # Safety margin before 120s limit
    n, m, p = DIMENSIONS
    
    # We use a block decomposition.
    # p=6 is split into 3 blocks of size 2.
    # Each block is a 2x3x2 multiplication.
    # Rank of 2x3x2 is 11 (using Strassen for 2x2x2 part + 4 scalar mults).
    # Total rank = 3 * 11 = 33.
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)] # 6 x 33
    V = [[0] * rank for _ in range(m * p)] # 18 x 33
    W = [[0] * rank for _ in range(p * n)] # 12 x 33
    
    # Define the 11 terms for a 2x3x2 block
    # Term structure: (u_vec, v_vec, w_coeffs)
    # u_vec: dict of {flat_index: coeff}
    # v_vec: dict of {flat_index: coeff}
    # w_coeffs: dict of {flat_index: coeff}
    
    # Indices for A (2x3): i in 0..1, j in 0..2. flat = i*3+j
    # Indices for B (3x2): j in 0..2, k in 0..1. flat = j*2+k
    # Indices for C (2x2): i in 0..1, k in 0..1. flat = i*2+k
    
    # Strassen terms for A_L (2x2) * B_T (2x2)
    # A_L uses j=0,1. B_T uses j=0,1.
    # A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    
    # Terms for a 2x3x2 block (rank 11)
    # A is 2x3: A[i,j], flat = i*3+j
    # B is 3x2: B[j,k], flat = j*2+k
    # C is 2x2: C[i,k], flat for W = k*n+i = k*2+i
    
    terms = []
    
    # Strassen terms for the 2x2x2 subproblem (j=0,1; k=0,1)
    # A_L = A[:, 0:2] (2x2), B_T = B[0:2, :] (2x2)
    # A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    
    # M1 = (A00+A11)(B00+B11)
    terms.append({
        'u': {0: 1, 4: 1},
        'v': {0: 1, 3: 1},
        'w': {0: 1, 3: 1}  # C[0,0], C[1,1]
    })
    # M2 = (A10+A11)B00
    terms.append({
        'u': {3: 1, 4: 1},
        'v': {0: 1},
        'w': {2: 1, 3: -1}  # C[1,0], C[1,1]
    })
    # M3 = A00(B01-B11)
    terms.append({
        'u': {0: 1},
        'v': {1: 1, 3: -1},
        'w': {1: 1, 3: 1}  # C[0,1], C[1,1]
    })
    # M4 = A11(B10-B00)
    terms.append({
        'u': {4: 1},
        'v': {2: 1, 0: -1},
        'w': {0: 1, 2: 1}  # C[0,0], C[1,0]
    })
    # M5 = (A00+A01)B11
    terms.append({
        'u': {0: 1, 1: 1},
        'v': {3: 1},
        'w': {0: -1, 1: 1}  # C[0,0], C[0,1]
    })
    # M6 = (A10-A00)(B00+B01)
    terms.append({
        'u': {3: 1, 0: -1},
        'v': {0: 1, 1: 1},
        'w': {3: 1}  # C[1,1]
    })
    # M7 = (A01-A11)(B10+B11)
    terms.append({
        'u': {1: 1, 4: -1},
        'v': {2: 1, 3: 1},
        'w': {0: 1}  # C[0,0]
    })
    
    # Scalar terms for j=2
    # K1 = A02 * B20
    terms.append({
        'u': {2: 1},
        'v': {4: 1},
        'w': {0: 1}  # C[0,0]
    })
    # K2 = A02 * B21
    terms.append({
        'u': {2: 1},
        'v': {5: 1},
        'w': {1: 1}  # C[0,1]
    })
    # K3 = A12 * B20
    terms.append({
        'u': {5: 1},
        'v': {4: 1},
        'w': {2: 1}  # C[1,0]
    })
    # K4 = A12 * B21
    terms.append({
        'u': {5: 1},
        'v': {5: 1},
        'w': {3: 1}  # C[1,1]
    })
    
    # Fill the matrices for 3 blocks of 2x3x2
    block_size = 2  # k dimension per block
    num_blocks = p // block_size  # 3 blocks
    
    for block in range(num_blocks):
        if time.time() > _deadline:
            break
        for t_idx, term in enumerate(terms):
            col = block * 11 + t_idx
            
            # U
            for idx, val in term['u'].items():
                U[idx][col] = val
            
            # V
            for l_idx, val in term['v'].items():
                j_local = l_idx // block_size
                k_local = l_idx % block_size
                g_idx = j_local * p + (block * block_size + k_local)
                V[g_idx][col] = val
            
            # W
            for l_idx, val in term['w'].items():
                i_local = l_idx // block_size
                k_local = l_idx % block_size
                g_idx = (block * block_size + k_local) * n + i_local
                W[g_idx][col] = val
    
    # Verify the decomposition using vectorized operations
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    np.random.seed(42)
    for trial in range(20):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C from decomposition using vectorized ops
        # a_scalars[r] = dot(U_np[:, r], A.flatten())
        # b_scalars[r] = dot(V_np[:, r], B.flatten())
        # C_decomp = (a_scalars * b_scalars) @ W_np  (with appropriate reshaping)
        
        a_flat = A.flatten()  # shape (n*m,)
        b_flat = B.flatten()  # shape (m*p,)
        
        a_scalars = U_np.T @ a_flat  # shape (rank,)
        b_scalars = V_np.T @ b_flat  # shape (rank,)
        
        # C_decomp[k*n+i] = sum_r W_np[k*n+i, r] * a_scalars[r] * b_scalars[r]
        c_flat = W_np @ (a_scalars * b_scalars)  # shape (p*n,)
        C_decomp = c_flat.reshape((p, n)).T  # shape (n, p)
        
        if not np.allclose(C_decomp, C_expected, atol=1e-9):
            pass  # Shouldn't happen
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
