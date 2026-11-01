import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal tensor decomposition for 2x2 * 2x14 matrix multiplication.
    Tries to find rank < 49 via structured search; falls back to rank-49 Strassen.
    """
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS  # n=2, m=2, p=14

    # First, build the known rank-49 Strassen solution as fallback
    def build_strassen_49():
        num_blocks = p // 2   # 7
        strassen_rank = 7
        total_rank = num_blocks * strassen_rank  # 49

        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]

        terms = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1}),
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
        ]

        for t in range(num_blocks):
            k0 = 2 * t
            k1 = 2 * t + 1
            for s in range(strassen_rank):
                col = t * strassen_rank + s
                u_coeffs, v_local, w_local = terms[s]
                for u_idx, coeff in u_coeffs.items():
                    U[u_idx][col] = coeff
                for v_loc, coeff in v_local.items():
                    j = v_loc // 2
                    k_col = k0 + (v_loc % 2)
                    v_idx = j * p + k_col
                    V[v_idx][col] = coeff
                for w_loc, coeff in w_local.items():
                    i = w_loc % 2
                    k_col = k0 + (w_loc // 2)
                    w_idx = k_col * n + i
                    W[w_idx][col] = coeff

        return U, V, W

    # Verify a decomposition
    def verify(U, V, W, tol=1e-9):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_np[i * m + j], V_np[j * p + k] * W_np[k * n + i])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err <= tol

    # Try to find a lower-rank decomposition using a structured approach
    # Key insight: The U matrix only has 4 rows, and in the Strassen solution,
    # each column of U corresponds to one of 7 Strassen patterns repeated 7 times.
    # We can try to find a basis for U that allows sharing across blocks.
    
    # Attempt: Use the fact that A is 2x2 and B is 2x14.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is a bilinear form. Let's try a direct rank search.
    
    # For small ranks, we can set up the equations and solve.
    # The number of equations is n*m*p = 56.
    # Unknowns: rank * (n*m + m*p + p*n) = rank * (4 + 28 + 28) = 60*rank
    # For rank < 56/60... this is nonlinear. Let's use a different approach.
    
    # Try rank values from 48 down to 40
    best_result = None
    for target_rank in range(48, 39, -1):
        if time.time() > deadline - 30:
            break
            
        # Use a structured initialization based on Strassen but with potential sharing
        # Initialize with random values and try to solve via alternating least squares
        np.random.seed(42)
        
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        # Initialize with Strassen-like structure
        for r in range(target_rank):
            # Assign a Strassen pattern
            s = r % 7
            t = r // 7
            terms = [
                ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
                ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}),
                ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1}),
                ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1}),
                ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),
                ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
                ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
            ]
            u_c, v_c, w_c = terms[s]
            for idx, coeff in u_c.items():
                U[idx, r] = coeff
            for v_loc, coeff in v_c.items():
                j = v_loc // 2
                k_col = t * 2 + (v_loc % 2)
                if k_col < p:
                    V[j * p + k_col, r] = coeff
            for w_loc, coeff in w_c.items():
                i = w_loc % 2
                k_col = t * 2 + (w_loc // 2)
                if k_col < p:
                    W[k_col * n + i, r] = coeff
        
        # Try alternating minimization
        success = False
        for iteration in range(50):
            if time.time() > deadline - 20:
                break
            
            # Update U: for each (i,j), solve U[i*m+j, :] = (V[j*p+0,:]*W[0*n+i,:] + ... + V[j*p+p-1,:]*W[(p-1)*n+i,:])^T / ...
            # Actually, the constraint is: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
            # This is: U[i*m+j,:] * diag(V[j*p+k,:] * W[k*n+i,:]) = 1 for all k
            # Let's form the linear system for U[i*m+j,:]:
            # For each k: sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1
            # This is: (V[j*p+0,:]*W[0*n+i,:] + ... ) is not quite right.
            
            # Let M[i,j,k] = V[j*p+k,:] * W[k*n+i,:] (element-wise product, shape (rank,))
            # Then we need: sum_r U[i*m+j,r] * M[i,j,k,r] = 1 for all k
            # This is: U[i*m+j,:] @ M[i,j,:,:].T = ones(p)
            # Where M[i,j,:,:] has shape (p, rank)
            
            # Build M for fixed i,j
            M_mat = np.zeros((p, target_rank))
            for k in range(p):
                M_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
            
            # Solve: U[i*m+j,:] @ M_mat.T = ones(p)
            # This is underdetermined if target_rank > p. Use least squares.
            try:
                U_row, residuals, rank_m, sv = np.linalg.lstsq(M_mat.T, np.ones(p), rcond=None)
                U[i * m + j, :] = U_row
            except:
                break
            
            # Similarly update V and W
            # For V[j*p+k,:]: sum_r V[j*p+k,r] * (U[i*m+j,r] * W[k*n+i,r]) = 1 for all i
            # M2[i, r] = U[i*m+j,r] * W[k*n+i,r], shape (n, rank)
            # V[j*p+k,:] @ M2.T = ones(n)
            for k in range(p):
                M2 = np.zeros((n, target_rank))
                for i in range(n):
                    M2[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    V_row, _, _, _ = np.linalg.lstsq(M2.T, np.ones(n), rcond=None)
                    V[j * p + k, :] = V_row
                except:
                    break
            
            # For W[k*n+i,:]: sum_r W[k*n+i,r] * (U[i*m+j,r] * V[j*p+k,r]) = 1 for all j
            # M3[j, r] = U[i*m+j,r] * V[j*p+k,r], shape (m, rank)
            # W[k*n+i,:] @ M3.T = ones(m)
            for k in range(p):
                for i in range(n):
                    M3 = np.zeros((m, target_rank))
                    for j in range(m):
                        M3[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W_row, _, _, _ = np.linalg.lstsq(M3.T, np.ones(m), rcond=None)
                        W[k * n + i, :] = W_row
                    except:
                        break
        
        # Verify
        if verify(U.tolist(), V.tolist(), W.tolist()):
            best_result = (U.tolist(), V.tolist(), W.tolist())
            break
    
    if best_result is not None:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    # Fallback to rank-49 Strassen
    U, V, W = build_strassen_49()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
