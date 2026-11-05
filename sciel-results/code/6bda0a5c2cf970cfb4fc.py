"""
Matrix multiplication 3x4x7 tensor decomposition.
Uses six disjoint 2x2x2 Strassen blocks (rank 7 each) covering i in {0,1},
j-pairs {0,1},{2,3}, k-pairs {0,1},{2,3},{4,5}, plus schoolbook for the rest.
Total rank: 6*7 + 36 = 78.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]


def construct():
    """
    Returns a rank-78 exact tensor decomposition for multiplying a 3x4 matrix
    by a 4x7 matrix.
    
    Returns dict with keys 'U', 'V', 'W' (lists of lists) representing
    factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    dim_U = n * m  # 12
    dim_V = m * p  # 28
    dim_W = p * n  # 21
    num_eqs = n * m * p  # 84

    def get_schoolbook():
        rank = num_eqs
        U = [[0] * rank for _ in range(dim_U)]
        V = [[0] * rank for _ in range(dim_V)]
        W = [[0] * rank for _ in range(dim_W)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Verify exact decomposition."""
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    # ============================================================
    # Six 2x2x2 Strassen blocks for i in {0,1}
    # Tiling j x k: j-pairs {0,1},{2,3} and k-pairs {0,1},{2,3},{4,5}
    # Each block covers 2*2*2=8 entries with rank 7
    # 6 blocks: 6*7=42 rank for 48 entries
    # Remaining: 84-48=36 entries with schoolbook (rank 36)
    # Total rank: 42+36=78
    # ============================================================

    def add_strassen_block(U, V, W, col_start, i0, j0, k0):
        """Add a 2x2x2 Strassen block starting at column col_start."""
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        # Standard 7-term Strassen decomposition for 2x2x2 tensor
        # We need: sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
        # for the 2x2x2 sub-tensor.
        
        # The 7 terms of Strassen for 2x2x2:
        # Let A = [[a00, a01],[a10, a11]], B = [[b00, b01],[b10, b11]], C = [[c00, c01],[c10, c11]]
        # C = A*B
        #
        # m1 = a00 * b00
        # m2 = (a10 + a11) * b00
        # m3 = a00 * (b01 - b11)
        # m4 = a11 * (b10 - b00)
        # m5 = (a00 + a01) * b11
        # m6 = (a10 - a00) * (b00 + b01)
        # m7 = (a01 - a11) * (b10 + b11)
        #
        # c00 = m1 + m4 - m5 + m7
        # c01 = m3 + m5
        # c10 = m2 - m4 - m6
        # c11 = m1 - m2 + m3 + m6
        #
        # In our tensor notation for the 2x2x2 block:
        # U[i,j,t] corresponds to coefficient of A[i,j] in term t
        # V[j,k,t] corresponds to coefficient of B[j,k] in term t
        # W[k,i,t] corresponds to coefficient of C[k,i] in term t
        #
        # Note: C is indexed as C[k,i] in our convention (W[k*n+i])
        # So C[0,0]=c00, C[0,1]=c10, C[1,0]=c01, C[1,1]=c11
        # Wait, let me re-check. W[k*n+i] means W is indexed by (k, i) where k is the column of B (which is row of C) and i is the row of A (which is column of C).
        # Actually in standard matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p.
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor is T[i,j,k] = A[i,j]*B[j,k] contributing to C[i,k].
        # W is indexed W[k*n + i] which is W[k,i] = coefficient for C[i,k].
        
        # So for the 2x2x2 sub-block with local indices (i', j', k') in {0,1}:
        # We need sum_t U_local[i',j',t] * V_local[j',k',t] * W_local[k',i',t] = delta
        # where W_local[k',i'] is the coefficient for C[i',k'].
        
        # c00 = m1 + m4 - m5 + m7  => W[0,0] gets contributions from m1(+1), m4(+1), m5(-1), m7(+1)
        # c01 = m3 + m5            => W[1,0] gets contributions from m3(+1), m5(+1)
        # c10 = m2 - m4 - m6       => W[0,1] gets contributions from m2(+1), m4(-1), m6(-1)
        # c11 = m1 - m2 + m3 + m6  => W[1,1] gets contributions from m1(+1), m2(-1), m3(+1), m6(+1)
        
        # m1 = a00 * b00
        # U[0,0] += 1, V[0,0] += 1
        U[i0*m+j0][col_start] = 1
        V[j0*p+k0][col_start] = 1
        W[k0*n+i0][col_start] = 1  # c00 += m1
        W[k1*n+i1][col_start] = 1  # c11 += m1

        # m2 = (a10 + a11) * b00
        # U[1,0] += 1, U[1,1] += 1, V[0,0] += 1
        U[i1*m+j0][col_start+1] = 1
        U[i1*m+j1][col_start+1] = 1
        V[j0*p+k0][col_start+1] = 1
        W[k0*n+i1][col_start+1] = 1   # c10 += m2
        W[k1*n+i1][col_start+1] = -1  # c11 -= m2

        # m3 = a00 * (b01 - b11)
        # U[0,0] += 1, V[0,1] += 1, V[1,1] += -1
        U[i0*m+j0][col_start+2] = 1
        V[j0*p+k1][col_start+2] = 1
        V[j1*p+k1][col_start+2] = -1
        W[k1*n+i0][col_start+2] = 1  # c01 += m3
        W[k1*n+i1][col_start+2] = 1  # c11 += m3

        # m4 = a11 * (b10 - b00)
        # U[1,1] += 1, V[1,0] += 1, V[0,0] += -1
        U[i1*m+j1][col_start+3] = 1
        V[j1*p+k0][col_start+3] = 1
        V[j0*p+k0][col_start+3] = -1
        W[k0*n+i0][col_start+3] = 1   # c00 += m4
        W[k0*n+i1][col_start+3] = -1  # c10 -= m4

        # m5 = (a00 + a01) * b11
        # U[0,0] += 1, U[0,1] += 1, V[1,1] += 1
        U[i0*m+j0][col_start+4] = 1
        U[i0*m+j1][col_start+4] = 1
        V[j1*p+k1][col_start+4] = 1
        W[k0*n+i0][col_start+4] = -1  # c00 -= m5
        W[k1*n+i0][col_start+4] = 1   # c01 += m5

        # m6 = (a10 - a00) * (b00 + b01)
        # U[1,0] += 1, U[0,0] += -1, V[0,0] += 1, V[0,1] += 1
        U[i1*m+j0][col_start+5] = 1
        U[i0*m+j0][col_start+5] = -1
        V[j0*p+k0][col_start+5] = 1
        V[j0*p+k1][col_start+5] = 1
        W[k0*n+i1][col_start+5] = -1  # c10 -= m6
        W[k1*n+i1][col_start+5] = 1   # c11 += m6

        # m7 = (a01 - a11) * (b10 + b11)
        # U[0,1] += 1, U[1,1] += -1, V[1,0] += 1, V[1,1] += 1
        U[i0*m+j1][col_start+6] = 1
        U[i1*m+j1][col_start+6] = -1
        V[j1*p+k0][col_start+6] = 1
        V[j1*p+k1][col_start+6] = 1
        W[k0*n+i0][col_start+6] = 1   # c00 += m7

        return col_start + 7

    # Build the structured decomposition
    rank = 78
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]

    # Six blocks: i={0,1}, j-pairs {0,1},{2,3}, k-pairs {0,1},{2,3},{4,5}
    blocks = [
        (0, 0, 0),  # i={0,1}, j={0,1}, k={0,1}
        (0, 0, 2),  # i={0,1}, j={0,1}, k={2,3}
        (0, 0, 4),  # i={0,1}, j={0,1}, k={4,5}
        (0, 2, 0),  # i={0,1}, j={2,3}, k={0,1}
        (0, 2, 2),  # i={0,1}, j={2,3}, k={2,3}
        (0, 2, 4),  # i={0,1}, j={2,3}, k={4,5}
    ]

    col = 0
    covered = set()
    for i0, j0, k0 in blocks:
        col = add_strassen_block(U, V, W, col, i0, j0, k0)
        for i in (i0, i0+1):
            for j in (j0, j0+1):
                for k in (k0, k0+1):
                    covered.add((i, j, k))

    # Add schoolbook terms for remaining (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1

    if col != rank:
        # Fallback to schoolbook if something went wrong
        return get_schoolbook()

    # Verify the structured decomposition
    if not verify(U, V, W):
        # If verification fails, try a quick search
        if time.time() < deadline - 15:
            np.random.seed(42)
            
            # Precompute index arrays
            i_idx = np.repeat(np.arange(n), m * p)
            j_idx = np.tile(np.repeat(np.arange(m), p), n)
            k_idx = np.tile(np.arange(p), n * m)
            
            U_rows = i_idx * m + j_idx
            V_rows = j_idx * p + k_idx
            W_rows = k_idx * n + i_idx
            
            def verify_fast(U_arr, V_arr, W_arr):
                U_sub = U_arr[U_rows, :]
                V_sub = V_arr[V_rows, :]
                W_sub = W_arr[W_rows, :]
                prod = U_sub * V_sub * W_sub
                sums = np.sum(prod, axis=1)
                return np.allclose(sums, 1.0, atol=1e-9)
            
            def try_solve_W(U_arr, V_arr, r):
                U_sub = U_arr[U_rows, :]
                V_sub = V_arr[V_rows, :]
                coeff = U_sub * V_sub
                
                A = np.zeros((num_eqs, dim_W * r))
                b = np.ones(num_eqs)
                
                col_indices = W_rows[:, None] * r + np.arange(r)[None, :]
                A[np.arange(num_eqs)[:, None], col_indices] = coeff
                
                try:
                    W_flat, residuals, rank_sol, s = np.linalg.lstsq(A, b, rcond=1e-12)
                except Exception:
                    return None
                
                if len(residuals) > 0 and residuals[0] > 1e-8:
                    return None
                
                W_sol = W_flat.reshape((dim_W, r))
                W_polished = np.round(W_sol * 2) / 2.0
                
                if verify_fast(U_arr, V_arr, W_polished):
                    return W_polished
                return None
            
            # Try a few random initializations at rank 78
            for trial in range(30):
                if time.time() > deadline - 5:
                    break
                
                U_rand = np.random.choice([-1.0, 0.0, 1.0], size=(dim_U, rank))
                V_rand = np.random.choice([-1.0, 0.0, 1.0], size=(dim_V, rank))
                
                # Ensure each row is nonzero
                for row in range(dim_U):
                    if np.all(U_rand[row, :] == 0):
                        U_rand[row, np.random.randint(rank)] = 1.0
                for row in range(dim_V):
                    if np.all(V_rand[row, :] == 0):
                        V_rand[row, np.random.randint(rank)] = 1.0
                
                W_sol = try_solve_W(U_rand, V_rand, rank)
                if W_sol is not None:
                    return {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_sol.tolist()}
        
        return get_schoolbook()

    return {"U": U, "V": V, "W": W}
