import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 26, 26
    
    # Target rank: 6 blocks of 2x2x2 (rank 7) + 1 block of 2x2x1 (rank 4) = 46
    R = 46
    U = np.zeros((nm, R), dtype=float)
    V = np.zeros((mp, R), dtype=float)
    W = np.zeros((pn, R), dtype=float)
    
    def A_idx(i, j): return i * m + j
    def B_idx(j, k): return j * p + k
    def C_idx(i, k): return k * n + i
    
    # Strassen coefficients for 2x2x2
    # U, V, W are 4-vectors for the 4 entries of the 2x2 matrices
    # Indices for U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for V: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for W: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # M1: (a00+a11)(b00+b11) -> c00 +=, c11 +=
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
        # M2: (a10+a11)b00 -> c10 +=, c11 -=
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, -1])),
        # M3: a00(b01-b11) -> c01 +=, c11 +=
        (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 1, 0, 1])),
        # M4: a11(b10-b00) -> c00 +=, c10 +=
        (np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0]), np.array([1, 0, 1, 0])),
        # M5: (a00+a01)b11 -> c00 -=, c01 +=
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 1, 0, 0])),
        # M6: (a10-a00)(b00+b01) -> c11 +=
        (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])),
        # M7: (a01-a11)(b10+b11) -> c00 +=
        (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0])),
    ]
    
    col = 0
    
    # Precompute index arrays for faster access
    U_indices = np.array([A_idx(i, j) for i in range(n) for j in range(m)])  # [0,1,2,3]
    
    # Process 6 blocks of 2 columns (k=0..11)
    for block in range(6):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # V indices for this block: [B_idx(0,k0), B_idx(0,k1), B_idx(1,k0), B_idx(1,k1)]
        V_block_indices = np.array([B_idx(0, k0), B_idx(0, k1), B_idx(1, k0), B_idx(1, k1)])
        # W indices for this block: [C_idx(0,k0), C_idx(1,k0), C_idx(0,k1), C_idx(1,k1)]
        # Wait, W local index: 0:(i=0,k_off=0), 1:(i=0,k_off=1), 2:(i=1,k_off=0), 3:(i=1,k_off=1)
        # So W_block_indices should be: [C_idx(0,k0), C_idx(0,k1), C_idx(1,k0), C_idx(1,k1)]
        W_block_indices = np.array([C_idx(0, k0), C_idx(0, k1), C_idx(1, k0), C_idx(1, k1)])
        
        for t in range(7):
            u_loc, v_loc, w_loc = strassen_terms[t]
            
            # U: 2x2 matrix A - all 4 entries
            U[U_indices, col] += u_loc
            
            # V: 2x2 submatrix of B
            V[V_block_indices, col] += v_loc
            
            # W: 2x2 submatrix of C
            W[W_block_indices, col] += w_loc
            
            col += 1
    
    # Process last column k=12 (Schoolbook 2x2x1)
    k_last = 12
    # Terms: A[i][j] * B[j][k] -> C[i][k]
    # 4 terms
    for i in range(n):
        for j in range(m):
            U[A_idx(i, j), col] += 1.0
            V[B_idx(j, k_last), col] += 1.0
            W[C_idx(i, k_last), col] += 1.0
            col += 1
    
    assert col == R, f"Column count mismatch: {col} vs {R}"
    
    # Verify using vectorized operations
    if time.time() < deadline:
        # Build index arrays for verification
        # For each (i,j,k) triple, compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        i_vals, j_vals, k_vals = np.meshgrid(
            np.arange(n), np.arange(m), np.arange(p), indexing='ij'
        )
        i_flat = i_vals.ravel()
        j_flat = j_vals.ravel()
        k_flat = k_vals.ravel()
        
        U_idx = i_flat * m + j_flat
        V_idx = j_flat * p + k_flat
        W_idx = k_flat * n + i_flat
        
        # Compute all 52 products at once: shape (52, R)
        U_sel = U[U_idx, :]  # (52, R)
        V_sel = V[V_idx, :]  # (52, R)
        W_sel = W[W_idx, :]  # (52, R)
        
        products = U_sel * V_sel * W_sel  # (52, R)
        sums = products.sum(axis=1)  # (52,)
        
        valid = np.all(np.abs(sums - 1.0) < 1e-9)
        
        if not valid:
            # Fallback to schoolbook (Rank 52)
            R0 = n * m * p
            U0 = np.zeros((nm, R0), dtype=float)
            V0 = np.zeros((mp, R0), dtype=float)
            W0 = np.zeros((pn, R0), dtype=float)
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U0[A_idx(i, j), c] = 1.0
                        V0[B_idx(j, k), c] = 1.0
                        W0[C_idx(i, k), c] = 1.0
                        c += 1
            return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
        
        # Snap to half-integers
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Time out, return best effort (likely valid if construction is deterministic)
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
