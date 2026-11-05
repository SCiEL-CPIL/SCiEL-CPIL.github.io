import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin

    # Precompute T_target
    # T[a, b, g] = 1 if a=i*m+j, b=j*p+k, g=k*n+i
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                g = k * n + i
                T_target[a, b, g] = 1.0

    def compute_tensor(U, V, W):
        return np.einsum('ac,bc,dc->abd', U, V, W)

    def verify(U, V, W):
        # Snap to half-integers
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        
        T = compute_tensor(U_s, V_s, W_s)
        
        # Check max error
        if np.max(np.abs(T - T_target)) > 1e-9:
            return False
        return True

    def to_lists(U, V, W):
        U = np.round(np.asarray(U, dtype=float) * 2) / 2
        V = np.round(np.asarray(V, dtype=float) * 2) / 2
        W = np.round(np.asarray(W, dtype=float) * 2) / 2
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return to_lists(U, V, W), rank

    # ALS Implementation with vectorization
    def als_decompose(rank, max_iter=100, tol=1e-7):
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Normalize to avoid scaling issues
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)

        prev_loss = np.inf
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Update U
            # T[a,b,g] = sum_c U[a,c] V[b,c] W[g,c]
            # Fix b,g. Sum over c.
            # P[b,g,c] = V[b,c] * W[g,c]
            # T_flat[a, b*g] = sum_c U[a,c] P[b,g,c]
            
            # Reshape V and W to compute P
            # V is (m*p, rank), W is (p*n, rank)
            # We need P of shape (m*p, p*n, rank)
            P = V[:, np.newaxis, :] * W[np.newaxis, :, :]
            # P shape: (m*p, p*n, rank)
            
            # T_target reshaped to (n*m, m*p*p*n)
            T_flat = T_target.reshape(n*m, -1)
            
            # We want to solve for U.
            # T_flat[a, :] = U[a, :] @ P_reshaped
            # P_reshaped shape: (rank, m*p*p*n)
            P_reshaped = P.reshape(m*p * p*n, rank).T
            
            # Solve least squares: U[a, :] @ P_reshaped = T_flat[a, :]
            # U[a, :] = T_flat[a, :] @ P_reshaped^T @ (P_reshaped @ P_reshaped^T)^-1
            # Or use lstsq
            try:
                # U_new = np.linalg.lstsq(P_reshaped, T_flat.T, rcond=None)[0].T
                # This solves P_reshaped @ U_new.T = T_flat.T
                U_new = np.linalg.lstsq(P_reshaped, T_flat.T, rcond=None)[0].T
            except:
                return None, np.inf
                
            # Update V
            # T[a,b,g] = sum_c U[a,c] V[b,c] W[g,c]
            # Fix a,g.
            # Q[a,g,c] = U[a,c] * W[g,c]
            Q = U[:, np.newaxis, :] * W[np.newaxis, :, :]
            # Q shape: (n*m, p*n, rank)
            Q_reshaped = Q.reshape(n*m * p*n, rank).T
            
            # T_target reshaped to (m*p, n*m*p*n)
            # We need T_target[a, b, g] for all a,g for a fixed b.
            # T_target shape: (n*m, m*p, p*n)
            # Transpose to (m*p, n*m, p*n)
            T_V = T_target.transpose(1, 0, 2).reshape(m*p, -1)
            
            try:
                V_new = np.linalg.lstsq(Q_reshaped, T_V.T, rcond=None)[0].T
            except:
                return None, np.inf
                
            # Update W
            # T[a,b,g] = sum_c U[a,c] V[b,c] W[g,c]
            # Fix a,b.
            # R[a,b,c] = U[a,c] * V[b,c]
            R = U[:, np.newaxis, :] * V[np.newaxis, :, :]
            # R shape: (n*m, m*p, rank)
            R_reshaped = R.reshape(n*m * m*p, rank).T
            
            # T_target reshaped to (p*n, n*m*m*p)
            T_W = T_target.transpose(2, 0, 1).reshape(p*n, -1)
            
            try:
                W_new = np.linalg.lstsq(R_reshaped, T_W.T, rcond=None)[0].T
            except:
                return None, np.inf
                
            U, V, W = U_new, V_new, W_new
            
            # Calculate loss
            T_curr = compute_tensor(U, V, W)
            loss = np.sum((T_curr - T_target)**2)
            
            if abs(prev_loss - loss) < tol:
                break
            prev_loss = loss
        
        # Check convergence
        T_curr = compute_tensor(U, V, W)
        max_dev = np.max(np.abs(T_curr - T_target))
        
        if max_dev < 1e-5:
            # Snap and verify
            if verify(U, V, W):
                return (U, V, W), loss
        return None, loss

    best_result = None
    best_rank = n * m * p
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Try ranks from low to high
    ranks_to_try = [19, 20, 21, 22, 23]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        max_seeds = 50 if rank <= 20 else 20
        
        for seed in range(max_seeds):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(seed)
            result, loss = als_decompose(rank, max_iter=200, tol=1e-8)
            
            if result is not None:
                U, V, W = result
                best_result = to_lists(U, V, W)
                best_rank = rank
                if rank <= 20:
                    break
        
        if best_rank <= 20:
            break

    if best_result is None:
        best_result = to_lists(best_U, best_V, best_W)
        
    return best_result

