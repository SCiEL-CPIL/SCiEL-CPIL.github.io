import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Target tensor T is 1s for all i, j, k
    # U shape (n*m, R), V shape (m*p, R), W shape (p*n, R)
    
    def error_and_grad(params, R):
        # params is a flat vector of length n*m*R + m*p*R + p*n*R
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        U = U_flat.reshape(n, m, R)
        V = V_flat.reshape(m, p, R)
        W = W_flat.reshape(p, n, R)
        
        # Compute M[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Using einsum: 'ijr,jkr,kr->ijk' ? No, indices are i,j,k
        # U[i,j,r], V[j,k,r], W[k,i,r]
        # M = np.einsum('ijr,jkr,kr->ijk', U, V, W) is wrong because W is (p,n,R) -> (k,i,r)
        # W shape is (p, n, R) which is (k, i, r).
        # So M[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        # Let's construct M
        # Temp = U * W[None, :, None, :] ? No.
        
        # Efficient way:
        # For each r, U[:, :, r] (n,m) * V[:, :, r] (m,p) * W[:, :, r] (p,n)
        # This is a trace-like operation but element-wise in i,j,k.
        
        # M[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        
        # We can do this by looping over r? R is small (21-50).
        # Or use einsum.
        # U: (n, m, R)
        # V: (m, p, R)
        # W: (p, n, R)
        # We want M (n, m, p)
        
        # Let's try:
        # A = U * W[:, :, None, :] # (n, m, R) * (p, n, R) -> needs alignment
        # W is (p, n, R). We need W[k, i, r].
        # So W_transposed = W.T (n, p, R)? No, W[k,i,r].
        # W is indexed by k (0..p-1), i (0..n-1), r.
        # So W[k, i, r].
        
        # M[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        # Let's compute for each i, j, k.
        # This is a batched operation.
        
        # Method 1: Loop over r (R is small)
        M = np.zeros((n, m, p))
        for r in range(R):
            # U[:, :, r] is (n, m)
            # V[:, :, r] is (m, p)
            # W[:, :, r] is (p, n)
            
            # We need M[i,j,k] += U[i,j,r] * V[j,k,r] * W[k,i,r]
            # This is:
            # Term = U[:, :, r] * W[r, :, :]? No.
            # W[k,i,r] is W[:, :, r][k, i].
            
            # Let U_r = U[:, :, r] (n, m)
            # Let V_r = V[:, :, r] (m, p)
            # Let W_r = W[:, :, r] (p, n)
            
            # M[i,j,k] += U_r[i,j] * V_r[j,k] * W_r[k,i]
            
            # This can be done as:
            # Temp1[i,j,k] = U_r[i,j] * V_r[j,k]
            # Temp1 = U_r[:, :, None] * V_r[None, :, :] # (n, m, p)
            # M += Temp1 * W_r[None, :, :]? No, W_r[k,i] depends on k and i.
            # W_r is (p, n). We need W_r[k, i].
            # So we need to broadcast W_r to (n, m, p) such that index is (i, j, k) -> W_r[k, i].
            
            # W_r[k, i] can be accessed by W_r.T[i, k]? No, W_r[k, i] is row k, col i.
            # So W_r.T[i, k] is row i, col k.
            # Let W_rot = W_r.T # (n, p) where W_rot[i, k] = W_r[k, i]
            
            # M += U_r[:, :, None] * V_r[None, :, :] * W_rot[:, None, :]
            
            U_r = U[:, :, r]
            V_r = V[:, :, r]
            W_rot = W[:, :, r].T # (n, p)
            
            M += U_r[:, :, None] * V_r[None, :, :] * W_rot[:, None, :]
            
        # Error
        E = np.sum((M - 1.0)**2)
        
        # Gradient
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # dE/dU[i,j,r] = 2 * sum_k (M[i,j,k] - 1) * V[j,k,r] * W[k,i,r]
        # dE/dV[j,k,r] = 2 * sum_i (M[i,j,k] - 1) * U[i,j,r] * W[k,i,r]
        # dE/dW[k,i,r] = 2 * sum_j (M[i,j,k] - 1) * U[i,j,r] * V[j,k,r]
        
        # Let's compute these efficiently.
        # diff = M - 1 (n, m, p)
        diff = M - 1.0
        
        for r in range(R):
            U_r = U[:, :, r]
            V_r = V[:, :, r]
            W_r = W[:, :, r]
            W_rot = W_r.T # (n, p)
            
            # grad_U[i,j,r] = 2 * sum_k diff[i,j,k] * V_r[j,k] * W_rot[i,k]
            # grad_U[:, :, r] = 2 * np.sum(diff * V_r[None, :, :] * W_rot[:, None, :], axis=2)
            
            # grad_V[j,k,r] = 2 * sum_i diff[i,j,k] * U_r[i,j] * W_rot[i,k]
            # grad_V[:, :, r] = 2 * np.sum(diff * U_r[:, :, None] * W_rot[:, None, :], axis=0)
            # Wait, U_r is (n, m), so U_r[i,j]. diff is (n, m, p).
            # U_r[:, :, None] is (n, m, 1). Broadcast to (n, m, p).
            # W_rot[:, None, :] is (n, 1, p). Broadcast to (n, m, p).
            
            # grad_W[k,i,r] = 2 * sum_j diff[i,j,k] * U_r[i,j] * V_r[j,k]
            # grad_W[:, :, r] = 2 * np.sum(diff * U_r[:, :, None] * V_r[None, :, :], axis=1)
            # Result shape (n, p). We need to assign to W[:, :, r] which is (p, n).
            # So grad_W[:, :, r] = (2 * np.sum(...)).T
            
            grad_U[:, :, r] = 2 * np.sum(diff * V_r[None, :, :] * W_rot[:, None, :], axis=2)
            grad_V[:, :, r] = 2 * np.sum(diff * U_r[:, :, None] * W_rot[:, None, :], axis=0)
            
            # For grad_W:
            # sum_j diff[i,j,k] * U_r[i,j] * V_r[j,k]
            # diff (n, m, p) * U_r (n, m, 1) * V_r (1, m, p)
            # Sum over j (axis=1) -> (n, p)
            # This gives grad for W_rot[i,k].
            # W_rot[i,k] = W_r[k,i].
            # So grad_W_rot = 2 * np.sum(diff * U_r[:, :, None] * V_r[None, :, :], axis=1)
            # grad_W[:, :, r] = grad_W_rot.T
            
            grad_W_rot = 2 * np.sum(diff * U_r[:, :, None] * V_r[None, :, :], axis=1)
            grad_W[:, :, r] = grad_W_rot.T
            
        # Flatten gradients
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return E, grad

    best_rank = None
    best_solution = None
    
    # Try ranks from 21 upwards
    for R in range(21, 51):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Random init
            np.random.seed(attempt * 1000 + R)
            U0 = np.random.randn(n, m, R)
            V0 = np.random.randn(m, p, R)
            W0 = np.random.randn(p, n, R)
            
            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            # Optimize
            # Use L-BFGS-B
            result = opt.minimize(
                lambda x: error_and_grad(x, R)[0],
                params0,
                jac=lambda x: error_and_grad(x, R)[1],
                method='L-BFGS-B',
                options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-12}
            )
            
            if result.success:
                E = result.fun
                if E < 1e-6:
                    # Snap to integers/half-integers
                    U_opt = result.x[:n*m*R].reshape(n, m, R)
                    V_opt = result.x[n*m*R:n*m*R+m*p*R].reshape(m, p, R)
                    W_opt = result.x[n*m*R+m*p*R:].reshape(p, n, R)
                    
                    U_snap = np.round(U_opt * 2) / 2
                    V_snap = np.round(V_opt * 2) / 2
                    W_snap = np.round(W_opt * 2) / 2
                    
                    # Verify exactness
                    M_check = np.zeros((n, m, p))
                    for r in range(R):
                        U_r = U_snap[:, :, r]
                        V_r = V_snap[:, :, r]
                        W_rot = W_snap[:, :, r].T
                        M_check += U_r[:, :, None] * V_r[None, :, :] * W_rot[:, None, :]
                    
                    if np.allclose(M_check, 1.0, atol=1e-9):
                        # Found a valid decomposition
                        best_rank = R
                        best_solution = (U_snap.flatten(), V_snap.flatten(), W_snap.flatten())
                        break
                        
        if best_rank is not None:
            break

    if best_solution is not None:
        U_flat, V_flat, W_flat = best_solution
        U = U_flat.reshape(n*m, R)
        V = V_flat.reshape(m*p, R)
        W = W_flat.reshape(p*n, R)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
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
