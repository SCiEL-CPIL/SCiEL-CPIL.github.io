import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Build the target tensor T
    # T[a, b, c] = 1 if A[i,j] * B[j,k] contributes to C[i,k]
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Schoolbook decomposition (rank 45)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    def compute_error(U, V, W, target_rank):
        """Compute sum of squared errors between T and the decomposition."""
        T_approx = np.zeros((n*m, m*p, p*n))
        for r in range(target_rank):
            T_approx += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.sum((T - T_approx)**2)
    
    def als_decompose(target_rank, seed, max_iter=200, tol=1e-10):
        """Alternating Least Squares for tensor decomposition."""
        np.random.seed(seed)
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Solve for U: for each column a, U[a,:] should satisfy
            # sum_r U[a,r] * V[:,r] * W[:,r] ≈ T[a,:,:]
            # This is a linear system: for each a,
            # (V ⊙ W)^T * U[:,a] = T[a,:,:].reshape(-1)
            # where ⊙ is element-wise product, but actually:
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # So for fixed a: T[a,:,:] = sum_r U[a,r] * (V[:,r] * W[:,r])
            # Let M[a] be the matrix with M[a][b,c,r] = V[b,r]*W[c,r]
            # Then T[a,:,:].flatten() = M[a].flatten() @ U[:,a]
            
            # More efficiently: for each r, define w_r = V[:,r] * W[:,r] (shape m*p, p*n)
            # T[a,:,:] = sum_r U[a,r] * w_r
            # So: T_flat[a] = sum_r U[a,r] * w_r_flat
            # This is: T_flat = U @ W_flat^T where W_flat[r] = w_r.flatten()
            
            # Actually, let's use the standard ALS approach:
            # Fix V and W, solve for U:
            # For each a: T[a,:,:] = sum_r U[a,r] * (V[:,r] * W[:,r])
            # Let B_r = V[:,r] * W[:,r] (shape (m*p, p*n))
            # T[a] = sum_r U[a,r] * B_r
            # In matrix form: T.reshape(n*m, -1) = U @ (B_1, B_2, ..., B_r).T
            # where each B_r is flattened
            
            # Build the system
            # For each r, B_r = outer(V[:,r], W[:,r])
            # T[a].flatten() = sum_r U[a,r] * B_r.flatten()
            
            # Solve: for each a, we have a linear system in U[a,:]
            # T[a].flatten() = sum_r U[a,r] * (V[:,r] * W[:,r]).flatten()
            
            # Let's build the matrix M where M[a, r] = <T[a], V[:,r] * W[:,r]>
            # No wait, that's not right either.
            
            # Standard ALS for CP decomposition:
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # Fix V, W: solve for U
            # For each a: T[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r])
            # This is a linear system: T[a].flatten() = U[a,:] @ (outer(V[:,r], W[:,r]).flatten() for all r)
            
            # Build matrix A_u where A_u[:, r] = (V[:,r] * W[:,r]).flatten()
            A_u = np.zeros((m*p * p*n, target_rank))
            for r in range(target_rank):
                A_u[:, r] = (V[:, r] * W[:, r]).flatten()
            
            # T_flat: shape (n*m, m*p*p*n)
            T_flat = T.reshape(n*m, -1)
            
            # Solve T_flat = U @ A_u^T
            # U = T_flat @ (A_u^T)^+
            # Using least squares: U = lstsq(A_u, T_flat.T).T
            try:
                U_new, _, _, _ = np.linalg.lstsq(A_u, T_flat.T, rcond=None)
                U = U_new.T
            except:
                pass
            
            # Fix U, W: solve for V
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For each b: T[:,b,:] = sum_r V[b,r] * outer(U[:,r], W[:,r])
            A_v = np.zeros((n*m * p*n, target_rank))
            for r in range(target_rank):
                A_v[:, r] = (U[:, r] * W[:, r]).flatten()
            T_flat_v = T.reshape(-1, m*p).T  # shape (m*p, n*m*p*n)
            T_flat_v = T.reshape(m*p, -1)
            try:
                V_new, _, _, _ = np.linalg.lstsq(A_v, T_flat_v.T, rcond=None)
                V = V_new.T
            except:
                pass
            
            # Fix U, V: solve for W
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For each c: T[:,:,c] = sum_r W[c,r] * outer(U[:,r], V[:,r])
            A_w = np.zeros((n*m * m*p, target_rank))
            for r in range(target_rank):
                A_w[:, r] = (U[:, r] * V[:, r]).flatten()
            T_flat_w = T.reshape(p*n, -1)
            try:
                W_new, _, _, _ = np.linalg.lstsq(A_w, T_flat_w.T, rcond=None)
                W = W_new.T
            except:
                pass
            
            error = compute_error(U, V, W, target_rank)
            if error < tol:
                break
        
        return U, V, W, compute_error(U, V, W, target_rank)
    
    # Try ranks from 44 down to 30
    for target_rank in range(44, 29, -1):
        if time.time() > deadline - 10:
            break
        
        success = False
        attempts = 30
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
            
            seed = attempt * 1000 + target_rank
            U, V, W, error = als_decompose(target_rank, seed, max_iter=100, tol=1e-12)
            
            if error < 1e-8:
                # Try to snap to half-integers
                U_q = np.round(U * 2) / 2
                V_q = np.round(V * 2) / 2
                W_q = np.round(W * 2) / 2
                
                error_q = compute_error(U_q, V_q, W_q, target_rank)
                if error_q < 1e-8:
                    success = True
                    best_rank = target_rank
                    best_U = U_q
                    best_V = V_q
                    best_W = W_q
                    break
                else:
                    # Try integer snap
                    U_q2 = np.round(U)
                    V_q2 = np.round(V)
                    W_q2 = np.round(W)
                    error_q2 = compute_error(U_q2, V_q2, W_q2, target_rank)
                    if error_q2 < 1e-8:
                        success = True
                        best_rank = target_rank
                        best_U = U_q2
                        best_V = V_q2
                        best_W = W_q2
                        break
        
        if success:
            break
    
    if best_rank < rank_school:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
