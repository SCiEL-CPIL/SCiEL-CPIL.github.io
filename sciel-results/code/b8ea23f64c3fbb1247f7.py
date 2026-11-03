import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    total_vars_factor = nm + mp + pn
    
    # Deadline
    start_time = time.time()
    deadline = start_time + 100  # 100s limit, 20s buffer
    
    # Helper to verify a decomposition
    def verify(U, V, W):
        # U, V, W are numpy arrays of shape (nm, r), (mp, r), (pn, r)
        r = U.shape[1]
        T = np.zeros((n, m, p))
        for c in range(r):
            U_mat = U[:, c].reshape(n, m)
            V_mat = V[:, c].reshape(m, p)
            W_mat = W[:, c].reshape(p, n)
            # T_ijk += U_mat[i,j] * V_mat[j,k] * W_mat[k,i]
            # Efficient way:
            # Sum over c of (U_c \otimes V_c \otimes W_c)
            # We can use einsum
            # T += np.einsum('ij,jk,ki->ijk', U_mat, V_mat, W_mat)
            T += np.einsum('ij,jk,ki->ijk', U_mat, V_mat, W_mat)
        
        return np.allclose(T, 1.0, atol=1e-9)

    # Helper to get schoolbook
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {'U': U, 'V': V, 'W': W}

    # Solver for a specific rank
    def solve_rank(r, num_restarts=5):
        total_vars = r * total_vars_factor
        
        def residuals(params):
            U = params[:r*nm].reshape(r, nm).T
            V = params[r*nm:r*nm+r*mp].reshape(r, mp).T
            W = params[r*nm+r*mp:].reshape(r, pn).T
            
            T = np.zeros((n, m, p))
            for c in range(r):
                U_mat = U[:, c].reshape(n, m)
                V_mat = V[:, c].reshape(m, p)
                W_mat = W[:, c].reshape(p, n)
                T += np.einsum('ij,jk,ki->ijk', U_mat, V_mat, W_mat)
            return (T - 1).flatten()

        best_res = None
        best_cost = float('inf')
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                return None
                
            # Random initialization
            np.random.seed(restart + r)
            x0 = np.random.randn(total_vars) * 0.5
            
            try:
                res = opt.least_squares(residuals, x0, method='lm', max_nfev=5000)
                if res.cost < best_cost:
                    best_cost = res.cost
                    best_res = res
                    
                    if res.cost < 1e-8:
                        break
            except Exception:
                continue

        if best_res is None or best_cost > 1e-6:
            return None

        # Extract U, V, W
        U = best_res.x[:r*nm].reshape(nm, r)
        V = best_res.x[r*nm:r*nm+r*mp].reshape(mp, r)
        W = best_res.x[r*nm+r*mp:].reshape(pn, r)
        
        # Try to snap to half-integers
        U2 = np.round(U * 2) / 2
        V2 = np.round(V * 2) / 2
        W2 = np.round(W * 2) / 2
        
        if verify(U2, V2, W2):
            return {'U': U2, 'V': V2, 'W': W2}
            
        # Try to snap to integers
        U1 = np.round(U)
        V1 = np.round(V)
        W1 = np.round(W)
        
        if verify(U1, V1, W1):
            return {'U': U1, 'V': V1, 'W': W1}
            
        # If snapping fails, maybe the solution is valid but not close to half-integers?
        # The problem requires integer/half-integer entries.
        # So we must snap.
        # If snapping fails, this rank is not achieved with this solution.
        return None

    # Search for minimum rank
    # We know 42 is valid.
    # We try to find r < 42.
    # Likely candidates: 30, 32, 34, 36, 38, 40, 41.
    # We will try in increasing order to find the minimum.
    # But to save time, we can try a few "likely" ones first.
    # However, finding the *minimum* is the goal.
    # Let's iterate from 25 to 41.
    
    best_result = None
    best_rank = 42
    
    # To be efficient, we can try ranks that are multiples of something?
    # No, just iterate.
    
    # We will limit the number of restarts for lower ranks if time is tight.
    
    for r in range(25, 42):
        if time.time() > deadline:
            break
            
        # Reduce restarts for higher r if we are running out of time?
        # No, lower r is harder, needs more restarts.
        # But we start from 25.
        
        res = solve_rank(r, num_restarts=10 if r < 35 else 5)
        
        if res is not None:
            best_rank = r
            best_result = res
            # If we found a very low rank, we might stop?
            # But we want the minimum.
            # If r is close to lower bound (21), we can stop.
            if r <= 23:
                break

    if best_result is not None:
        return {
            'U': best_result['U'].tolist(),
            'V': best_result['V'].tolist(),
            'W': best_result['W'].tolist()
        }
    else:
        schoolbook = get_schoolbook()
        return {
            'U': schoolbook['U'].tolist(),
            'V': schoolbook['V'].tolist(),
            'W': schoolbook['W'].tolist()
        }
