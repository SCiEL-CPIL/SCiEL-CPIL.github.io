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
    deadline = start_time + 130 # 20s margin

    def residual(params, R):
        U = params[:R*nm].reshape(nm, R)
        V = params[R*nm:R*nm+R*mp].reshape(mp, R)
        W = params[R*nm+R*mp:].reshape(pn, R)
        
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return (T_recon - T).flatten()

    # Search for lower ranks
    # Start from rank_sb - 1 down to 1
    # But realistically, only rank_sb - 1 is worth trying given the time.
    
    for R in range(rank_sb - 1, 0, -1):
        if time.time() > deadline:
            break
            
        # If R is significantly lower, the probability of finding a solution is low.
        # We will focus most effort on R = rank_sb - 1.
        
        max_restarts = 5
        if R < rank_sb - 2:
            max_restarts = 1 # Just a quick check
            
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialization
            # Random normal
            np.random.seed(restart + R) # Reproducibility for debugging, but random is better for search
            # Actually, let's use random seed for variety
            params_init = np.random.randn(R * (nm + mp + pn))
            
            try:
                # Use Levenberg-Marquardt
                # 'lm' is good for small problems
                # max_nfev limits the time per call
                res = scipy.optimize.least_squares(residual, params_init, args=(R,), method='lm', max_nfev=500)
                
                if res.success:
                    # Check cost
                    if res.cost < 1e-10: # Tighter tolerance for "exact"
                        U = res.x[:R*nm].reshape(nm, R)
                        V = res.x[R*nm:R*nm+R*mp].reshape(mp, R)
                        W = res.x[R*nm+R*mp:].reshape(pn, R)
                        
                        # Final verification
                        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                        if np.allclose(T_recon, T, atol=1e-10):
                            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                            best_rank = R
                            # We found a better rank, continue to search for even lower?
                            # Yes, but break out of restart loop for this R
                            break
                            
            except Exception:
                pass

    return best_result
