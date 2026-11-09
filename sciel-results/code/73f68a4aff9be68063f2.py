import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook solution
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower rank solution
    # We know rank is at least max(n*m, m*p, p*n) ? No, but let's try 27, 28, 29...
    # Actually, let's try ranks from 25 to 35.
    
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Define the target tensor T
    # T[i, j, k] = 1 if this is a valid term? No, T is the multiplication tensor.
    # T[i, j, k] is the coefficient of x_i y_j z_k in (sum x_i A_ij y_j) (sum y_j B_jk z_k)?
    # No, the tensor for matrix multiplication is defined such that
    # C_ik = sum_j A_ij B_jk.
    # The tensor T has indices i, j, k.
    # T[i, j, k] = 1.
    # Wait, the decomposition is for the tensor of the multiplication.
    # The tensor T_{ijk} is 1 for all i, j, k.
    # Because C_{ik} = sum_j A_{ij} B_{jk}.
    # The product term is A_{ij} B_{jk}.
    # So the tensor is T_{ijk} = 1.
    
    T = np.ones((n, m, p))
    
    for R in range(25, 40):
        if time.time() > deadline:
            break
            
        # Try to find a decomposition of rank R
        found = False
        for trial in range(5): # 5 random restarts
            if time.time() > deadline:
                break
                
            # Initialize
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # ALS iterations
            for iter in range(50):
                if time.time() > deadline:
                    break
                
                # Solve for U
                # U[i*m+j, r]
                # For each i, j:
                # T[i, j, :] = sum_r U[i*m+j, r] * V[j*p + :, r] * W[:, i, r]?
                # No, W index is k*n + i.
                # W[k*n + i, r]
                
                # Let's precompute the "basis" matrices for V and W
                # For a fixed i, j, we want to solve for U[i*m+j, :]
                # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                
                # This is a linear system in U[i*m+j, r].
                # Let u = U[i*m+j, :] (vector of size R)
                # Let v_k = V[j*p+k, :] (vector of size R)
                # Let w_k = W[k*n+i, :] (vector of size R)
                # T[i, j, k] = sum_r u_r (v_k)_r (w_k)_r
                # This is not linear in u if v and w are fixed?
                # Yes it is linear in u.
                # T[i, j, k] = sum_r u_r * (v_k[r] * w_k[r])
                # Let M_k = v_k * w_k (element-wise product)
                # T[i, j, k] = u . M_k
                # So for fixed i, j, we have p equations for R unknowns u_r.
                # A_{k, r} = V[j*p+k, r] * W[k*n+i, r]
                # b_k = T[i, j, k]
                # Solve A u = b
                
                for i in range(n):
                    for j in range(m):
                        # Build A (p x R)
                        A = np.zeros((p, R))
                        b = np.zeros(p)
                        for k in range(p):
                            v_k = V[j*p+k, :]
                            w_k = W[k*n+i, :]
                            A[k, :] = v_k * w_k
                            b[k] = T[i, j, k]
                        
                        # Least squares
                        u, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        U[i*m+j, :] = u
                        
                # Solve for V
                # V[j*p+k, r]
                # For fixed j, k:
                # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                # Let v = V[j*p+k, :]
                # T[i, j, k] = sum_r v_r * (U[i*m+j, r] * W[k*n+i, r])
                # A_{i, r} = U[i*m+j, r] * W[k*n+i, r]
                # b_i = T[i, j, k]
                # Solve A v = b
                
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, R))
                        b = np.zeros(n)
                        for i in range(n):
                            u_i = U[i*m+j, :]
                            w_k = W[k*n+i, :]
                            A[i, :] = u_i * w_k
                            b[i] = T[i, j, k]
                        
                        v, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        V[j*p+k, :] = v
                        
                # Solve for W
                # W[k*n+i, r]
                # For fixed k, i:
                # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                # Let w = W[k*n+i, :]
                # T[i, j, k] = sum_r w_r * (U[i*m+j, r] * V[j*p+k, r])
                # A_{j, r} = U[i*m+j, r] * V[j*p+k, r]
                # b_j = T[i, j, k]
                # Solve A w = b
                
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, R))
                        b = np.zeros(m)
                        for j in range(m):
                            u_ij = U[i*m+j, :]
                            v_jk = V[j*p+k, :]
                            A[j, :] = u_ij * v_jk
                            b[j] = T[i, j, k]
                        
                        w, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        W[k*n+i, :] = w
                        
            # Check error
            # Reconstruct T
            T_recon = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
            
            error = np.linalg.norm(T - T_recon)
            if error < 1e-6:
                found = True
                break
                
        if found:
            # Round to nearest half-integer?
            # The problem says integer/half-integer entries.
            # We can try to round U, V, W to nearest 0.5
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Check if rounded version still works
            T_recon_rounded = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon_rounded[i, j, k] = np.dot(U_rounded[i*m+j, :], V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                        
            error_rounded = np.linalg.norm(T - T_recon_rounded)
            
            if error_rounded < 1e-6:
                # Use rounded version
                best_U = U_rounded.tolist()
                best_V = V_rounded.tolist()
                best_W = W_rounded.tolist()
                best_rank = R
                break
            else:
                # Use floating point version?
                # The problem says "integer/half-integer entries".
                # If floating point doesn't satisfy this, it might be invalid.
                # But if the true solution is rational, the rounded version should be close.
                # If error_rounded is small but not 0, maybe we need more precision?
                # Or maybe the solution is not exactly half-integers?
                # "integer/half-integer" usually implies the entries are in {0, 0.5, 1, 1.5, ...}
                # Let's assume if error is small, we can use the rounded version.
                # If not, we skip this rank.
                pass

    return {"U": best_U, "V": best_V, "W": best_W}
