"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with the known rank-55 Strassen-based decomposition
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part A: j=0 (first column of A, first row of B)
    # A[i,0] * B[0,k] -> C[i,k]
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
    # 5 blocks of 2x2 Strassen
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2
        
        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1
        
        # M1 = (a11 + a22)(b11 + b22)
        U[u_11][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M2 = (a21 + a22)b11
        U[u_21][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1
        
        # M3 = a11(b12 - b22)
        U[u_11][col] += 1
        V[v_12][col] += 1
        V[v_22][col] += -1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M4 = a22(b21 - b11)
        U[u_22][col] += 1
        V[v_21][col] += 1
        V[v_11][col] += -1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1
        
        # M5 = (a11 + a12)b22
        U[u_11][col] += 1
        U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1
        
        # M6 = (a21 - a11)(b11 + b12)
        U[u_21][col] += 1
        U[u_11][col] += -1
        V[v_11][col] += 1
        V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M7 = (a12 - a22)(b21 + b22)
        U[u_12][col] += 1
        U[u_22][col] += -1
        V[v_21][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1
    
    # Convert to numpy arrays for efficient operations
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    dim_u = n * m  # 6
    dim_v = m * p  # 30
    dim_w = p * n  # 20
    
    # Build the true tensor
    T_true = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T_true[alpha, beta, gamma] = 1
    
    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-10):
        """Verify that the decomposition exactly matches T_true."""
        if time.time() > deadline - 5:
            return False
        T_rec = np.zeros((dim_u, dim_v, dim_w))
        for c in range(U_mat.shape[1]):
            T_rec += np.einsum('i,j,k->ijk', U_mat[:, c], V_mat[:, c], W_mat[:, c])
        return np.allclose(T_rec, T_true, atol=tol)
    
    current_U = U_np.copy()
    current_V = V_np.copy()
    current_W = W_np.copy()
    current_rank = rank
    
    # Attempt peeling: check each column for linear dependence
    # Strategy: try to find if any component can be expressed as a linear combination of others
    # For exact decompositions, we look for components where the V-factor has at most 2 non-zeros
    # and try to merge them
    
    peeled = True
    while peeled and time.time() < deadline - 15:
        peeled = False
        for c in range(current_rank):
            if time.time() > deadline - 15:
                break
            # Check if column c can be removed (is linearly dependent)
            # A component is removable if there exist coefficients for other components
            # such that the rank-1 tensor equals a linear combination of others
            
            # Build the system: for each (i,j,k) in T_true, 
            # U[i,c]*V[j,c]*W[k,c] = sum_{c'!=c} lambda_{c'} * U[i,c']*V[j,c']*W[k,c']
            
            # This is a linear system in lambda_{c'}
            # Number of equations: dim_u * dim_v * dim_w = 6*30*20 = 3600
            # Number of unknowns: current_rank - 1
            
            # Only attempt if the system is feasible (more equations than unknowns is fine)
            other_cols = [i for i in range(current_rank) if i != c]
            num_unknowns = len(other_cols)
            
            if num_unknowns < 2:
                continue
            
            # Build the linear system A_lambda * lambda = b
            # b_{i,j,k} = U[i,c]*V[j,c]*W[k,c]
            # A_{(i,j,k), c'} = U[i,c']*V[j,c']*W[k,c']
            
            # This is a 3600 x (rank-1) system. Use least squares.
            # To save time, only check a subset of columns
            
            if c % 3 != 0:  # Check every 3rd column
                continue
            
            try:
                # Build the matrix A (flattened)
                # A has shape (dim_u*dim_v*dim_w, num_unknowns)
                A_mat = np.zeros((dim_u * dim_v * dim_w, num_unknowns))
                b_vec = np.zeros(dim_u * dim_v * dim_w)
                
                idx = 0
                for i in range(dim_u):
                    for j in range(dim_v):
                        for k in range(dim_w):
                            b_vec[idx] = current_U[i, c] * current_V[j, c] * current_W[k, c]
                            for oc_idx, oc in enumerate(other_cols):
                                A_mat[idx, oc_idx] = current_U[i, oc] * current_V[j, oc] * current_W[k, oc]
                            idx += 1
                
                # Solve least squares: minimize ||A*lambda - b||
                # If the residual is near zero, the component is linearly dependent
                lambda_sol, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                # Check if the residual is small
                if residuals is not None and len(residuals) > 0:
                    residual_norm = np.sqrt(residuals[0])
                    b_norm = np.linalg.norm(b_vec)
                    if b_norm > 1e-10 and residual_norm / b_norm < 1e-8:
                        # Component c is linearly dependent, remove it
                        current_U = np.delete(current_U, c, axis=1)
                        current_V = np.delete(current_V, c, axis=1)
                        current_W = np.delete(current_W, c, axis=1)
                        current_rank -= 1
                        peeled = True
                        break
            except Exception:
                continue
    
    # Verify the peeled decomposition
    if current_rank < rank:
        if verify_decomposition(current_U, current_V, current_W):
            # Convert back to list format
            U_final = current_U.tolist()
            V_final = current_V.tolist()
            W_final = current_W.tolist()
            return {"U": U_final, "V": V_final, "W": W_final}
    
    # Fallback: verify original and return
    if verify_decomposition(U_np, V_np, W_np):
        return {"U": U, "V": V, "W": W}
    
    # Final fallback to schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col_s = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col_s] = 1
                V_school[j * p + k][col_s] = 1
                W_school[k * n + i][col_s] = 1
                col_s += 1
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
