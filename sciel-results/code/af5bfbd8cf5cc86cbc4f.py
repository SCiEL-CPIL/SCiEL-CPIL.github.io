import numpy as np
import time
import scipy.optimize

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Construct the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    T_flat = T.flatten()

    # Schoolbook rank
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, col] = 1
                V_sb[b, col] = 1
                W_sb[c, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb

    start_time = time.time()
    deadline = start_time + 100 # 20s margin

    # Precompute indices for faster residual?
    # No, einsum is likely fine for small sizes.

    def residual(params, R):
        # params is flat array of length R*(nm+mp+pn)
        U = params[:R*nm].reshape(nm, R)
        V = params[R*nm:R*nm+R*mp].reshape(mp, R)
        W = params[R*nm+R*mp:].reshape(pn, R)
        
        # T_recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return (T_recon - T).flatten()

    # We search for R from rank_sb - 1 down to 1
    # To save time, we might skip some ranks if we don't find a solution quickly
    # But we want the minimal rank.
    
    for R in range(rank_sb - 1, 0, -1):
        if time.time() > deadline:
            break
            
        # Number of restarts
        # If R is high, the solution is dense, maybe easier?
        # If R is low, it's harder.
        # Let's try 3 restarts for high R, 10 for low R?
        # Actually, low R is more likely to be the answer if it exists.
        # But high R is closer to schoolbook.
        
        max_restarts = 5
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialization
            # Random normal
            params_init = np.random.randn(R * (nm + mp + pn))
            
            try:
                # Use Levenberg-Marquardt
                # It requires the Jacobian or approximates it.
                # For 2700 variables, numerical Jacobian is expensive.
                # Maybe use 'trf' or 'dogbox'?
                # 'lm' is fastest for small problems.
                res = scipy.optimize.least_squares(residual, params_init, args=(R,), method='lm', max_nfev=2000)
                
                if res.success:
                    # Check cost
                    if res.cost < 1e-8:
                        U = params_init[:R*nm].reshape(nm, R) # Wait, res.x is the solution
                        U = res.x[:R*nm].reshape(nm, R)
                        V = res.x[R*nm:R*nm+R*mp].reshape(mp, R)
                        W = res.x[R*nm+R*mp:].reshape(pn, R)
                        
                        # Final verification
                        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                        if np.allclose(T_recon, T, atol=1e-8):
                            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                            best_rank = R
                            found = True
                            # We continue to search for lower R
                            
            except Exception:
                pass

    return best_result
