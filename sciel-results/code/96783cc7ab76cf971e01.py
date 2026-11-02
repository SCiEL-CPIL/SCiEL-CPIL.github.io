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
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2: (a10+a11)b00 -> c10 +=, c11 -=
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3: a00(b01-b11) -> c01 +=, c11 +=
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4: a11(b10-b00) -> c00 +=, c10 +=
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # M5: (a00+a01)b11 -> c00 -=, c01 +=
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6: (a10-a00)(b00+b01) -> c11 +=
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7: (a01-a11)(b10+b11) -> c00 +=
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    col = 0
    
    # Process 6 blocks of 2 columns (k=0..11)
    for block in range(6):
        k0 = block * 2
        k1 = block * 2 + 1
        
        for t in range(7):
            u_loc, v_loc, w_loc = strassen_terms[t]
            
            # U: 2x2 matrix A
            for i in range(n):
                for j in range(m):
                    U[A_idx(i, j), col] += u_loc[i * m + j]
            
            # V: 2x2 submatrix of B (rows j=0,1; cols k0,k1)
            # V index in full matrix is j*p + k
            # Local V index 0:(0,k0), 1:(0,k1), 2:(1,k0), 3:(1,k1)
            # We need to map local (j, k_offset) to global
            for j in range(m):
                for k_off in range(2):
                    k = k0 + k_off
                    local_idx = j * 2 + k_off
                    V[B_idx(j, k), col] += v_loc[local_idx]
            
            # W: 2x2 submatrix of C (rows i=0,1; cols k0,k1)
            # W index in full matrix is k*n + i
            # Local W index 0:(k0,0), 1:(k0,1), 2:(k1,0), 3:(k1,1)
            # Note: Strassen W coefficients are for C entries c_ij where i,j are local indices.
            # c_00 (local) corresponds to C[0][k0] -> Global C_idx(0, k0)
            # c_01 (local) corresponds to C[0][k1] -> Global C_idx(0, k1)
            # c_10 (local) corresponds to C[1][k0] -> Global C_idx(1, k0)
            # c_11 (local) corresponds to C[1][k1] -> Global C_idx(1, k1)
            for i in range(n):
                for k_off in range(2):
                    k = k0 + k_off
                    local_idx = k_off * n + i  # Wait, local W is 2x2.
                    # Local W array is [c00, c01, c10, c11]
                    # Index 0: i=0, k_off=0
                    # Index 1: i=1, k_off=0 ? No, standard row-major for C:
                    # C is 2x2. C[0][0], C[0][1], C[1][0], C[1][1].
                    # So local_idx = i * 2 + k_off.
                    local_idx = i * 2 + k_off
                    W[C_idx(i, k), col] += w_loc[local_idx]
            
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
    
    # Verify
    if time.time() < deadline:
        valid = True
        # Check all target entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over rank
                    val = 0.0
                    for r in range(R):
                        val += U[A_idx(i, j), r] * V[B_idx(j, k), r] * W[C_idx(i, k), r]
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        
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
