import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 115.0  # Leave 5s margin for 120s limit (or 150s if that's the hard kill, prompt says 120s finish, 150s kill. I'll use 115 to be safe for the 120s requirement)
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_schoolbook
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    def validate(U, V, W, rank):
        # Check if sum_c U[i,j,c] V[j,k,c] W[k,i,c] == 1 for all i,j,k
        # U shape (nm, r), V shape (mp, r), W shape (pn, r)
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        try:
            U_reshaped = U.reshape(n, m, rank)
            V_reshaped = V.reshape(m, p, rank)
            W_reshaped = W.reshape(p, n, rank)
            
            # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
            T = np.einsum('ijc,jkc,kin->ijk', U_reshaped, V_reshaped, W_reshaped)
            error = np.max(np.abs(T - 1.0))
            return error
        except Exception:
            return float('inf')

    def try_rank(r, max_restarts=5, max_iters=100):
        if r >= best_rank:
            return None
            
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
                
            # Initialize
            # Scale is important. 
            # Let's try initializing with small random values
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # ALS Loop
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update W
                # For each (k, i), solve M_w x = b_w
                # M_w is (m, r), b_w is (m,)
                # M_w[j, c] = U[i*m+j, c] * V[j*p+k, c]
                # We can vectorize this somewhat, but loops over n*p are small (15)
                for i in range(n):
                    for k in range(p):
                        # Construct M_w
                        # U[i, :, :] is (m, r)
                        # V[:, k, :] is (m, r) -- wait, V is (m, p, r)
                        # V_reshaped = V.reshape(m, p, r)
                        # So V_reshaped[:, k, :] is (m, r)
                        # Element-wise product: U[i, :, :] * V_reshaped[:, k, :] -> (m, r)
                        U_slice = U.reshape(n, m, r)[i, :, :]
                        V_slice = V.reshape(m, p, r)[:, k, :]
                        M_w = U_slice * V_slice
                        b_w = np.ones(m)
                        try:
                            sol, _, _, _ = np.linalg.lstsq(M_w, b_w, rcond=None)
                            W.reshape(p, n, r)[k, i, :] = sol
                        except:
                            pass

                # Update V
                for j in range(m):
                    for k in range(p):
                        # M_v is (n, r)
                        # M_v[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        U_slice = U.reshape(n, m, r)[:, j, :] # (n, r)
                        W_slice = W.reshape(p, n, r)[k, :, :] # (n, r)
                        M_v = U_slice * W_slice
                        b_v = np.ones(n)
                        try:
                            sol, _, _, _ = np.linalg.lstsq(M_v, b_v, rcond=None)
                            V.reshape(m, p, r)[j, k, :] = sol
                        except:
                            pass

                # Update U
                for i in range(n):
                    for j in range(m):
                        # M_u is (p, r)
                        # M_u[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        V_slice = V.reshape(m, p, r)[j, :, :] # (p, r)
                        W_slice = W.reshape(p, n, r)[:, i, :] # (p, r)
                        M_u = V_slice * W_slice
                        b_u = np.ones(p)
                        try:
                            sol, _, _, _ = np.linalg.lstsq(M_u, b_u, rcond=None)
                            U.reshape(n, m, r)[i, j, :] = sol
                        except:
                            pass

            # After ALS, check error
            err = validate(U, V, W, r)
            if err < 1e-4:
                # Try to snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                err_snap = validate(U_snap, V_snap, W_snap, r)
                if err_snap < 1e-4:
                    return U_snap, V_snap, W_snap
                else:
                    # If snapping fails, maybe try integer snapping?
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    err_int = validate(U_int, V_int, W_int, r)
                    if err_int < 1e-4:
                        return U_int, V_int, W_int
                    else:
                        # If no integer/half-integer solution found, but float is good,
                        # we might return the float solution? 
                        # Prompt says "integer/half-integer entries".
                        # If we can't find one, we should probably not return this rank.
                        # Or we can do a local search around the snapped values.
                        pass
        
        return None

    # Try ranks from 59 down to 60? No, 59, 58, 57...
    # Schoolbook is 60.
    # We want to find < 60.
    # Start from 59 down to 50 (arbitrary cutoff to save time)
    for r in range(59, 49, -1):
        if time.time() > deadline:
            break
        result = try_rank(r, max_restarts=10, max_iters=50)
        if result is not None:
            U, V, W = result
            err = validate(U, V, W, r)
            if err < 1e-4:
                best_rank = r
                best_U = U
                best_V = V
                best_W = W
                break

    if best_rank == rank_schoolbook:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
