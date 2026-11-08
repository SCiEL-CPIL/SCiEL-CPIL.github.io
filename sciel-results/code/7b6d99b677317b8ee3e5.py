import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 7]
DEADLINE = time.time() + 110  # 10s safety margin

def construct():
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p
    
    # Trivial solution
    U_triv = np.zeros((n * m, trivial_rank))
    V_triv = np.zeros((m * p, trivial_rank))
    W_triv = np.zeros((p * n, trivial_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i * m + j, col] = 1
                V_triv[j * p + k, col] = 1
                W_triv[k * n + i, col] = 1
                col += 1
                
    best_rank = trivial_rank
    best_U = U_triv
    best_V = V_triv
    best_W = W_triv

    # Define objective function
    def objective(params, R):
        # params is a flat array of length R * (nm + mp + pn)
        nm, mp, pn = n * m, m * p, p * n
        U_flat = params[:R * nm].reshape(R, nm).T
        V_flat = params[R * nm:R * (nm + mp)].reshape(R, mp).T
        W_flat = params[R * (nm + mp):].reshape(R, pn).T
        
        # Calculate error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over r
                    val = np.dot(U_flat[i * m + j, :], V_flat[j * p + k, :] * W_flat[k * n + i, :])
                    error += (val - 1.0) ** 2
        return error

    def is_exact(U, V, W, R):
        # Check if entries are close to half-integers and if decomposition is exact
        # Rounding to nearest 0.5
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        # Check exactness with rounded values
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_round[i * m + j, :], V_round[j * p + k, :] * W_round[k * n + i, :])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        return True

    # Search for lower ranks
    # We try a few ranks. Lower bound is n*p = 21.
    # We prioritize ranks that are powers of 2 or specific known structures? 
    # Just iterate downwards from trivial_rank - 1 or upwards from lower bound?
    # Upwards is better because lower ranks are harder to find but more valuable.
    # However, finding rank 21 is extremely unlikely.
    # Let's try a few random restarts for R in a small set of candidates.
    
    candidates = []
    # Add some strategic candidates
    for r in range(21, trivial_rank):
        candidates.append(r)
        
    # Sort candidates by likelihood? Lower rank is better score.
    # But finding low rank is harder.
    # Let's just try a few low ranks and some mid ranks.
    # To save time, we only try a subset.
    # Let's try R = 21, 22, ..., 30 and then some random ones.
    
    search_space = list(range(21, 31)) + list(range(50, 60)) + list(range(90, 105))
    search_space = sorted(set(search_space))
    
    for R in search_space:
        if time.time() > DEADLINE - 10:
            break
            
        # If R is already worse than best, skip? 
        # No, we want to find ANY R < best_rank.
        if R >= best_rank:
            continue
            
        # Try a few restarts
        max_restarts = 3
        for restart in range(max_restarts):
            if time.time() > DEADLINE - 5:
                break
                
            # Initialize
            nm, mp, pn = n * m, m * p, p * n
            total_vars = R * (nm + mp + pn)
            
            # Random initialization
            x0 = np.random.randn(total_vars) * 0.5
            
            # Optimize
            res = minimize(objective, x0, args=(R,), method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.success or res.fun < 1e-6:
                # Extract U, V, W
                U_opt = res.x[:R * nm].reshape(R, nm).T
                V_opt = res.x[R * nm:R * (nm + mp)].reshape(R, mp).T
                W_opt = res.x[R * (nm + mp):].reshape(R, pn).T
                
                if is_exact(U_opt, V_opt, W_opt, R):
                    # Found a better rank!
                    best_rank = R
                    best_U = np.round(U_opt * 2) / 2
                    best_V = np.round(V_opt * 2) / 2
                    best_W = np.round(W_opt * 2) / 2
                    break # Move to next R? Or stop if we found a very low one?
                    # Let's break to try even lower R? 
                    # The loop iterates R in increasing order (21, 22...). 
                    # So if we find R=25, we break inner loop, continue to R=26? 
                    # No, we want the MINIMAL rank. 
                    # So we should iterate R from low to high. 
                    # If we find R, we update best. 
                    # We should stop if we find R close to lower bound? 
                    # Let's just continue to see if we can find lower. 
                    # But the loop is `for R in search_space` which is sorted. 
                    # So it tries 21, then 22. 
                    # If it finds 25, it updates best_rank=25. 
                    # Then it tries 26? No, 26 > 25. 
                    # We should only try R < best_rank. 
                    # The `if R >= best_rank: continue` handles this. 
                    pass

    # Return the best found
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    # Self-test
    res = construct()
    print("Rank:", len(res['U'][0]))
