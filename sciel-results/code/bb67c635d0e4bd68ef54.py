# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = 5, 5, 5
    start_time = time.time()
    deadline = start_time + 145.0

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify(U, V, W, rank):
        # Vectorized verification
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # We need sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1 for all i,j,k
        # This is a tensor contraction.
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        
        # Sum over rank axis
        # Product U[i,j,c] * V[j,k,c] * W[k,i,c]
        # We can do this by iterating over c and accumulating into a (n, p) matrix?
        # Or using einsum.
        # np.einsum('ijc,jkc,kic->ik', U_t, V_t, W_t)
        # This computes C[i,k] = sum_j sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # Wait, the tensor definition is T[i,j,k] = 1.
        # The decomposition is T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
        # So we need the result to be 1 for ALL i,j,k.
        # The einsum above sums over j as well? No, j is an index in V and U.
        # Let's check indices:
        # U[i, j, c], V[j, k, c], W[k, i, c].
        # Summing over c gives T[i, j, k].
        # So we need np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t) to be all 1s.
        
        T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        return np.max(np.abs(T - 1.0))

    def round_to_half(x):
        return np.round(x * 2) / 2.0

    def try_rank(rank, max_restarts=5, max_iters=200):
        if time.time() > deadline - 15:
            return None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                return None
            
            np.random.seed(rank * 100 + restart)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Normalize to prevent overflow
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    return None
                
                # Update U
                # For each i, j: min_u || Z u - 1 ||^2 where Z[k, c] = V[j,k,c] * W[k,i,c]
                U_t = U.reshape(n, m, rank)
                V_t = V.reshape(m, p, rank)
                W_t = W.reshape(p, n, rank)
                
                for i in range(n):
                    for j in range(m):
                        # Z is (p, rank)
                        Z = V_t[j, :, :] * W_t[:, i, :] # Broadcasting: (p, rank) * (p, rank) -> (p, rank)
                        try:
                            U_t[i, j, :] = np.linalg.lstsq(Z, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        Z = U_t[:, j, :] * W_t[k, :, :] # (n, rank) * (n, rank) -> (n, rank)
                        try:
                            V_t[j, k, :] = np.linalg.lstsq(Z, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        Z = U_t[:, :, i] * V_t[:, k, :] # Wait, indices.
                        # W[k, i, c]. 
                        # Term: U[i, j, c] V[j, k, c].
                        # For fixed k, i: sum_c W[k, i, c] (U[i, j, c] V[j, k, c]) = 1 for all j.
                        # Z[j, c] = U[i, j, c] * V[j, k, c]
                        Z = U_t[i, :, :] * V_t[:, k, :] # (m, rank) * (m, rank) -> (m, rank)
                        try:
                            W_t[k, i, :] = np.linalg.lstsq(Z, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                U = U_t.reshape(n*m, rank)
                V = V_t.reshape(m*p, rank)
                W = W_t.reshape(p*n, rank)
                
                # Check convergence
                err = verify(U, V, W, rank)
                if err < 1e-8:
                    # Polish
                    # Flatten
                    params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                    
                    def objective(params):
                        U_p = params[:n*m*rank].reshape(n, m, rank)
                        V_p = params[n*m*rank:2*n*m*rank].reshape(m, p, rank)
                        W_p = params[2*n*m*rank:].reshape(p, n, rank)
                        T = np.einsum('ijc,jkc,kic->ijk', U_p, V_p, W_p)
                        return (T - 1.0).flatten()
                    
                    try:
                        res = least_squares(objective, params, method='lm', max_nfev=100)
                        U_p = res.x[:n*m*rank].reshape(n, m, rank)
                        V_p = res.x[n*m*rank:2*n*m*rank].reshape(m, p, rank)
                        W_p = res.x[2*n*m*rank:].reshape(p, n, rank)
                        
                        U_r = round_to_half(U_p)
                        V_r = round_to_half(V_p)
                        W_r = round_to_half(W_p)
                        
                        err_r = verify(U_r, V_r, W_r, rank)
                        if err_r < 1e-8:
                            return {
                                "U": U_r.tolist(),
                                "V": V_r.tolist(),
                                "W": W_r.tolist()
                            }
                    except:
                        pass

                    # Fallback to direct rounding if polish fails
                    U_r = round_to_half(U)
                    V_r = round_to_half(V)
                    W_r = round_to_half(W)
                    err_r = verify(U_r, V_r, W_r, rank)
                    if err_r < 1e-8:
                        return {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
        
        return None

    for rank in [49, 50, 55, 60, 70, 80, 100, 125]:
        if time.time() > deadline - 20:
            break
        result = try_rank(rank, max_restarts=5, max_iters=100)
        if result is not None:
            return result

    return schoolbook()
# EVOLVE-BLOCK-END
