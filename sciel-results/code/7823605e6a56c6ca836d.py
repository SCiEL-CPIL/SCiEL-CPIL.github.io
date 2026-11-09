import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T of shape (n, m, p) with all ones
    T = np.ones((n, m, p))
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Deadline guard
    start_time = time.time()
    deadline = 120.0
    
    # Try to find a decomposition with rank < schoolbook_rank
    # We'll try ranks from schoolbook_rank - 1 down to 50
    # If we find one, we return it.
    
    # Helper to check if a decomposition is valid
    def is_valid(U, V, W, T):
        # Compute sum of outer products
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        
        # Compute tensor product
        # T_approx[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
        # Note: W index is k*n + i, so W[k, i, c]
        
        # Efficient computation
        # Sum over c
        # We can use einsum
        # 'ijc,jkc,kic->ijk'
        try:
            T_approx = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        except Exception:
            return False, None, None, None
            
        # Check if T_approx is close to T
        if np.allclose(T_approx, T, atol=1e-9):
            return True, U, V, W
        return False, None, None, None

    # Helper to round to half-integers
    def round_to_half(x):
        return np.round(x * 2) / 2.0

    # ALS solver for fixed rank r
    def solve_als(r, max_iter=100, tol=1e-12):
        # Initialize randomly
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Normalize
        for c in range(r):
            norm_U = np.linalg.norm(U[:, c])
            norm_V = np.linalg.norm(V[:, c])
            norm_W = np.linalg.norm(W[:, c])
            if norm_U > 0: U[:, c] /= norm_U
            if norm_V > 0: V[:, c] /= norm_V
            if norm_W > 0: W[:, c] /= norm_W
            
        for it in range(max_iter):
            if time.time() > deadline - 5:
                return None, None, None
                
            # Solve for U
            # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
            # For fixed V, W, this is linear in U.
            # Let A[i, j, c] = sum_k V[j, k, c] W[k, i, c]
            # Then T[i, j, k] = sum_c U[i, j, c] A[i, j, c]? No.
            # T[i, j, k] is fixed.
            # We want to minimize || T - sum_c u_c v_c w_c ||_F
            # This is equivalent to solving a linear system for each (i, j)
            # T[i, j, :] = sum_c U[i, j, c] (V[j, :, c] * W[:, i, c])?
            # Let's flatten.
            
            # Construct matrix B of size (n*m) x (p) for each c?
            # Actually, for each c, the term is u_c (n*m) * v_c (m*p) * w_c (p*n)
            # This is not a simple matrix multiplication.
            
            # Let's use the einsum formulation.
            # T_approx = einsum('ijc,jkc,kic->ijk', U, V, W)
            # Derivative w.r.t U[i, j, c]:
            # dT/dU[i, j, c] = V[j, k, c] W[k, i, c] (summed over k? No, for each k)
            # So for fixed i, j, the equation is:
            # T[i, j, k] = sum_c U[i, j, c] * (sum_k' V[j, k', c] W[k', i, c] delta_{k', k}?)
            # No, W[k, i, c] depends on k.
            # So T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
            
            # This is a linear system in U[i, j, c] for each k?
            # No, U[i, j, c] is multiplied by V[j, k, c] W[k, i, c].
            # For a fixed i, j, we have p equations (one for each k) and r unknowns (U[i, j, c] for c=1..r).
            # Let M[i, j] be a p x r matrix where M[i, j][k, c] = V[j, k, c] W[k, i, c]
            # Then T[i, j, :] = M[i, j] @ U[i, j, :]
            # We can solve this least squares problem for each i, j.
            
            new_U = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # Build M[i, j] of shape (p, r)
                    M_ij = np.zeros((p, r))
                    for k in range(p):
                        for c in range(r):
                            M_ij[k, c] = V[j * p + k, c] * W[k * n + i, c]
                    # Solve M_ij @ U[i, j, :] = T[i, j, :]
                    # T[i, j, :] is a vector of length p (all ones)
                    try:
                        U_ij, residuals, rank, sv = np.linalg.lstsq(M_ij, np.ones(p), rcond=None)
                        new_U[i * m + j, :] = U_ij
                    except Exception:
                        new_U[i * m + j, :] = U[i * m + j, :]
            U = new_U
            
            # Solve for V
            new_V = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
                    # For fixed j, k, we have n equations (one for each i) and r unknowns (V[j, k, c]).
                    # M[j, k][i, c] = U[i, j, c] W[k, i, c]
                    # T[:, j, k] = M[j, k] @ V[j, k, :]
                    M_jk = np.zeros((n, r))
                    for i in range(n):
                        for c in range(r):
                            M_jk[i, c] = U[i * m + j, c] * W[k * n + i, c]
                    try:
                        V_jk, residuals, rank, sv = np.linalg.lstsq(M_jk, np.ones(n), rcond=None)
                        new_V[j * p + k, :] = V_jk
                    except Exception:
                        new_V[j * p + k, :] = V[j * p + k, :]
            V = new_V
            
            # Solve for W
            new_W = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
                    # For fixed k, i, we have m equations (one for each j) and r unknowns (W[k, i, c]).
                    # M[k, i][j, c] = U[i, j, c] V[j, k, c]
                    # T[i, :, k] = M[k, i] @ W[k, i, :]
                    M_ki = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            M_ki[j, c] = U[i * m + j, c] * V[j * p + k, c]
                    try:
                        W_ki, residuals, rank, sv = np.linalg.lstsq(M_ki, np.ones(m), rcond=None)
                        new_W[k * n + i, :] = W_ki
                    except Exception:
                        new_W[k * n + i, :] = W[k * n + i, :]
            W = new_W
            
            # Check convergence
            U_reshaped = U.reshape(n, m, r)
            V_reshaped = V.reshape(m, p, r)
            W_reshaped = W.reshape(p, n, r)
            T_approx = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
            err = np.linalg.norm(T_approx - T)
            if err < tol:
                return U, V, W
                
        return U, V, W

    # Try ranks from 104 down to 80
    for r in range(schoolbook_rank - 1, 79, -1):
        if time.time() > deadline - 10:
            break
            
        # Try a few random restarts
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
                
            U, V, W = solve_als(r, max_iter=50)
            if U is None:
                continue
                
            # Check if valid
            valid, U_v, V_v, W_v = is_valid(U, V, W, T)
            if valid:
                # Round to half-integers
                U_r = round_to_half(U_v)
                V_r = round_to_half(V_v)
                W_r = round_to_half(W_v)
                
                # Verify rounded version
                valid_r, U_r_v, V_r_v, W_r_v = is_valid(U_r, V_r, W_r, T)
                if valid_r:
                    best_rank = r
                    best_U = U_r_v
                    best_V = V_r_v
                    best_W = W_r_v
                    break
        if best_rank < schoolbook_rank:
            break

    if best_rank < schoolbook_rank:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Return schoolbook
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
