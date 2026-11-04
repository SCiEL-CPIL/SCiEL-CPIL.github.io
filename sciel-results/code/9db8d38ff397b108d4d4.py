import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115

    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Tensor construction
    # T[a, b, c] where a in [0, nm), b in [0, mp), c in [0, pn)
    # a = i*m + j, b = j*p + k, c = k*n + i
    # T[a, b, c] = 1 if j_a == j_b and k_b == k_c and i_c == i_a else 0
    
    S_vec = np.zeros((p * n, n * m * m * p)) # Shape (pn, nm * mp)
    # Actually, Z row is vec(u v^T) of size nm * mp.
    # Let's use dimensions:
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    dim_z = dim_u * dim_v
    
    for c in range(dim_w):
        k = c // n
        i = c % n
        for j in range(m):
            a = i * m + j
            b = j * p + k
            S_vec[c, a * dim_v + b] = 1.0

    best_result = schoolbook()
    best_rank = n * m * p

    # Try to find lower rank
    # Lower bound is max(nm, mp, pn) = 21
    # We try ranks from 25 up to 62
    
    for target_rank in range(25, best_rank):
        if time.time() > deadline - 10:
            break
            
        R = target_rank
        # Try a few random W matrices
        found = False
        for trial in range(10):
            if time.time() > deadline - 5:
                break
                
            # Random W with integers
            W = np.random.randint(-1, 2, size=(dim_w, R))
            
            # Ensure W^T has full rank
            # If not, re-roll
            if np.linalg.matrix_rank(W.T) < dim_w:
                continue
                
            # Solve W^T Z = S_vec
            # Z is R x dim_z
            # Z0 = pinv(W.T) @ S_vec
            Z0 = np.linalg.pinv(W.T) @ S_vec
            
            # Null space of W.T
            # W.T is R x dim_w. Null space is in R-dim.
            # Basis N is R x (R - dim_w)
            try:
                N = np.linalg.svd(W.T, full_matrices=True)[0][:, dim_w:]
            except:
                continue
                
            # Parameterize Z = Z0 + N @ Y
            # Y is (R - dim_w) x dim_z
            dim_y = R - dim_w
            
            def objective(y_flat):
                Y = y_flat.reshape(dim_y, dim_z)
                Z = Z0 + N @ Y
                # Reshape Z rows to (dim_u, dim_v)
                # Compute sum of squares of smallest dim_u singular values
                cost = 0.0
                for r in range(R):
                    M_r = Z[r].reshape(dim_u, dim_v)
                    # SVD
                    try:
                        s = np.linalg.svd(M_r, compute_uv=False)
                        # We want rank 1, so s[1:] should be 0
                        cost += np.sum(s[1:]**2)
                    except:
                        cost += 1e6
                return cost

            # Initial guess y = 0
            y0 = np.zeros(dim_y * dim_z)
            
            # Optimize
            # Use BFGS or L-BFGS-B
            res = minimize(objective, y0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-8})
            
            if res.fun < 1e-6:
                # Check if we can snap to half-integers
                Y = res.x.reshape(dim_y, dim_z)
                Z = Z0 + N @ Y
                
                # Extract U, V
                U = np.zeros((dim_u, R))
                V = np.zeros((dim_v, R))
                
                valid = True
                for r in range(R):
                    M_r = Z[r].reshape(dim_u, dim_v)
                    # SVD to get u, v
                    U_svd, s_svd, Vt_svd = np.linalg.svd(M_r, full_matrices=False)
                    if s_svd[0] < 1e-6:
                        # Rank 0? Should not happen if S_vec is not 0
                        # But if M_r is 0, it's valid rank 1 (0)
                        U[:, r] = 0
                        V[:, r] = 0
                        continue
                    
                    u_vec = U_svd[:, 0] * np.sqrt(s_svd[0])
                    v_vec = Vt_svd[0, :] * np.sqrt(s_svd[0])
                    
                    U[:, r] = u_vec
                    V[:, r] = v_vec
                    
                    # Snap to nearest 0.5
                    U_snapped = np.round(U[:, r] * 2) / 2
                    V_snapped = np.round(V[:, r] * 2) / 2
                    W_snapped = np.round(W[:, r] * 2) / 2 # W is already integer, so snapped is same
                    
                    # Verify with snapped values
                    # Reconstruct tensor
                    # This check is expensive, do it only once at the end if all rows are good
                    # For now, assume valid if objective was low
                    
                # Final verification with snapped values
                U_final = np.zeros((dim_u, R))
                V_final = np.zeros((dim_v, R))
                W_final = np.zeros((dim_w, R))
                
                for r in range(R):
                    M_r = Z[r].reshape(dim_u, dim_v)
                    U_svd, s_svd, Vt_svd = np.linalg.svd(M_r, full_matrices=False)
                    if s_svd[0] < 1e-6:
                        continue
                    u_vec = U_svd[:, 0] * np.sqrt(s_svd[0])
                    v_vec = Vt_svd[0, :] * np.sqrt(s_svd[0])
                    
                    U_final[:, r] = np.round(u_vec * 2) / 2
                    V_final[:, r] = np.round(v_vec * 2) / 2
                    W_final[:, r] = W[:, r]
                    
                # Verify exactness
                # T_reconstructed[c] = sum_r W_final[c, r] * (U_final[:, r] * V_final[:, r])
                # Check against S_vec
                # This is a linear check
                # Z_check = np.zeros((R, dim_z))
                # for r in range(R):
                #     Z_check[r] = (U_final[:, r] * V_final[:, r]).flatten()
                # T_check = W_final.T @ Z_check
                # if np.allclose(T_check, S_vec, atol=1e-5):
                #     return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
                
                # Faster verification
                T_check = W_final.T @ (U_final * V_final).T # (dim_w, R) @ (R, dim_z) -> (dim_w, dim_z)
                # Note: (U_final * V_final).T is not correct broadcasting for outer product sum.
                # We need sum_r U_final[a, r] V_final[b, r] W_final[c, r]
                # This is elementwise product of U and V, then dot with W?
                # Let M = U_final * V_final (elementwise, R columns)
                # T[a, b, c] = sum_r M[a, b, r] W_final[c, r]
                # T_vec[c, a*dim_v+b] = sum_r W_final[c, r] * (U_final[a, r] * V_final[b, r])
                
                # Construct T_check
                T_check = np.zeros((dim_w, dim_z))
                for c in range(dim_w):
                    # T_check[c, :] = sum_r W_final[c, r] * (U_final[:, r] * V_final[:, r])
                    # This is (W_final[c, :] @ (U_final * V_final).T)
                    T_check[c] = W_final[c] @ (U_final * V_final).T
                    
                if np.allclose(T_check, S_vec, atol=1e-5):
                    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

    return best_result
