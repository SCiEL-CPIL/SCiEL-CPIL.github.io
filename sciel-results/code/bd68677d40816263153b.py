"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135  # 135s deadline, 15s safety margin
    
    n, m, p = 3, 4, 4
    nm, mp, pn = n*m, m*p, p*n  # 12, 16, 12
    
    # Try to find rank-47 decomposition
    rank = 47
    
    # The tensor has 48 entries, all equal to 1
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    
    # Parameterize: flatten U, V, W into a single vector
    # x = [U.flatten(), V.flatten(), W.flatten()]
    # Length: 12*47 + 16*47 + 12*47 = 564 + 752 + 564 = 1880
    
    # Residual: 48 values, each should be 1
    
    def residuals(x):
        U = x[:nm*rank].reshape(nm, rank)
        V = x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
        W = x[nm*rank+mp*rank:].reshape(pn, rank)
        
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over c of U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    res[idx] = val - 1.0
                    idx += 1
        return res
    
    # Initialize with schoolbook-like structure
    x0 = np.zeros(nm*rank + mp*rank + pn*rank)
    
    # Set up schoolbook for first 47 terms
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if idx < rank:
                    U_idx = (i*m + j) * rank + idx
                    V_idx = nm*rank + (j*p + k) * rank + idx
                    W_idx = nm*rank + mp*rank + (k*n + i) * rank + idx
                    x0[U_idx] = 1.0
                    x0[V_idx] = 1.0
                    x0[W_idx] = 1.0
                    idx += 1
    
    # Try to solve
    try:
        result = least_squares(residuals, x0, method='lm', max_nfev=10000, 
                               xtol=1e-12, ftol=1e-12, gtol=1e-12)
        
        if time.time() > deadline:
            raise TimeoutError()
        
        if result.success or result.cost < 1e-10:
            # Extract solution
            U_sol = result.x[:nm*rank].reshape(nm, rank)
            V_sol = result.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
            W_sol = result.x[nm*rank+mp*rank:].reshape(pn, rank)
            
            # Round to nearest rational (try small denominators)
            # First check if it's close to integers
            U_round = np.round(U_sol)
            V_round = np.round(V_sol)
            W_round = np.round(W_sol)
            
            # Verify
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_round[i*m+j, :] * V_round[j*p+k, :] * W_round[k*n+i, :])
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-8:
                # Convert to lists of integers
                U_list = [[int(U_round[i,j]) for j in range(rank)] for i in range(nm)]
                V_list = [[int(V_round[i,j]) for j in range(rank)] for i in range(mp)]
                W_list = [[int(W_round[i,j]) for j in range(rank)] for i in range(pn)]
                return {"U": U_list, "V": V_list, "W": W_list}
            
            # Try rounding to half-integers
            U_half = np.round(U_sol * 2) / 2
            V_half = np.round(V_sol * 2) / 2
            W_half = np.round(W_sol * 2) / 2
            
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_half[i*m+j, :] * V_half[j*p+k, :] * W_half[k*n+i, :])
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-8:
                # Convert to lists of half-integers (as fractions)
                # The task says integer/half-integer entries
                U_list = [[U_half[i,j] for j in range(rank)] for i in range(nm)]
                V_list = [[V_half[i,j] for j in range(rank)] for i in range(mp)]
                W_list = [[W_half[i,j] for j in range(rank)] for i in range(pn)]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    except Exception:
        pass
    
    # Fallback: return schoolbook
    rank_school = n*m*p
    U_s = [[0]*rank_school for _ in range(nm)]
    V_s = [[0]*rank_school for _ in range(mp)]
    W_s = [[0]*rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j][col] = 1
                V_s[j*p+k][col] = 1
                W_s[k*n+i][col] = 1
                col += 1
    return {"U": U_s, "V": V_s, "W": W_s}
# EVOLVE-BLOCK-END
