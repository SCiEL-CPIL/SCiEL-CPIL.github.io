"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, tol=1e-8):
    n, m, p = DIMENSIONS
    # Random test matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    # Compute C from factors
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    R = U.shape[1]
    C_comp = np.zeros((n, p))
    
    # Reshape U to (n, m, R)
    U_reshaped = U.reshape(n, m, R)
    S_A = np.einsum('ijr,ij->ir', U_reshaped, A)
    
    # Reshape V to (m, p, R)
    V_reshaped = V.reshape(m, p, R)
    S_B = np.einsum('jkr,jk->kr', V_reshaped, B)
    
    # Reshape W to (p, n, R)
    W_reshaped = W.reshape(p, n, R)
    
    # C_ik = sum_r W_reshaped[k, i, r] * S_A[i, r] * S_B[k, r]
    C_comp = np.einsum('kir,ir,kr->ik', W_reshaped, S_A, S_B)
    
    return np.allclose(C_true, C_comp, atol=tol)

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    best_result = None
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s margin for safety

    # We will try ranks from a lower bound up to schoolbook
    # Known lower bound for 3x4x7 is likely around 70-80.
    # Let's try from 70 to 83.
    # If we find a solution, we return it.
    
    # To save time, we can start from a rank slightly below schoolbook and go down.
    # Or start from a reasonable guess and go down.
    
    # Let's try ranks from 83 down to 70.
    for R in range(rank_schoolbook - 1, 69, -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations for this rank
        found = False
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W with random values
            # Scale is important.
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # ALS Iterations
            success = False
            for it in range(100):
                if time.time() > deadline:
                    break
                    
                # Update U
                # For each i, j, solve A_ij u_ij = 1
                # A_ij[k, r] = V[j*p+k, r] * W[k*n+i, r]
                for i in range(n):
                    for j in range(m):
                        # Construct A_ij (p x R)
                        # V is (m*p, R), W is (p*n, R)
                        # V[j*p + k, r] for k in 0..p-1
                        # W[k*n + i, r] for k in 0..p-1
                        A_ij = np.zeros((p, R))
                        for k in range(p):
                            A_ij[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        # Solve A_ij u = 1
                        # Using least squares
                        u_sol, residuals, rank, sv = np.linalg.lstsq(A_ij, np.ones(p), rcond=None)
                        U[i * m + j, :] = u_sol
                        
                # Update V
                # For each j, k, solve B_jk v_jk = 1
                # B_jk[i, r] = U[i*m+j, r] * W[k*n+i, r]
                for j in range(m):
                    for k in range(p):
                        B_jk = np.zeros((n, R))
                        for i in range(n):
                            B_jk[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        v_sol, residuals, rank, sv = np.linalg.lstsq(B_jk, np.ones(n), rcond=None)
                        V[j * p + k, :] = v_sol
                        
                # Update W
                # For each k, i, solve C_ki w_ki = 1
                # C_ki[j, r] = U[i*m+j, r] * V[j*p+k, r]
                for k in range(p):
                    for i in range(n):
                        C_ki = np.zeros((m, R))
                        for j in range(m):
                            C_ki[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        w_sol, residuals, rank, sv = np.linalg.lstsq(C_ki, np.ones(m), rcond=None)
                        W[k * n + i, :] = w_sol
                        
                # Check residual
                # Compute T_comp
                U_r = U.reshape(n, m, R)
                V_r = V.reshape(m, p, R)
                W_r = W.reshape(p, n, R)
                T_comp = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                residual = np.linalg.norm(T_comp - 1.0)
                
                if residual < 1e-8:
                    # Verify
                    if _verify(U, V, W):
                        return {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                    else:
                        # If verification fails but residual is low, might be numerical issue
                        # Try to polish with least_squares
                        pass

    # Fallback to schoolbook
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
