import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin
    
    # Target tensor: T[a, b, c] = 1 if indices consistent, 0 otherwise
    # a = i*m + j, b = j'*p + k, c = k'*n + i'
    # T[a,b,c] = 1 iff j == j' and k == k' and i == i'
    nm, mp, pn = n*m, m*p, p*n
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_rank = n*m*p
    best_result = None
    
    # Try various ranks
    for rank in range(best_rank - 1, 50, -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        max_iters = 200
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            rng = np.random.default_rng(attempt * 1000 + rank)
            U = rng.standard_normal((nm, rank)) * 0.1
            V = rng.standard_normal((mp, rank)) * 0.1
            W = rng.standard_normal((pn, rank)) * 0.1
            
            # Alternating least squares
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                
                # Update U: for each (i,j), solve U[i*m+j, :] * (V[:, r] * W[:, r])^T = T[i*m+j, :, :]
                for idx in range(nm):
                    # Build RHS and LHS
                    # U[idx, r] * sum_r V[b, r] * W[c, r] = T[idx, b, c]
                    # For fixed idx, we need: sum_r U[idx, r] * (V[:, r] * W[:, r])[b,c] = T[idx, :, :]
                    # This is: U[idx, :] @ M = T[idx, :].reshape(-1)
                    # where M[b,c,r] = V[b,r] * W[c,r]
                    M = np.einsum('br,cr->bcr', V, W).reshape(-1, rank)
                    rhs = T[idx].reshape(-1)
                    # Solve U[idx, :] @ M = rhs
                    try:
                        U[idx] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                    except:
                        pass
                
                # Update V
                for idx in range(mp):
                    M = np.einsum('ar,cr->acr', U, W).reshape(-1, rank)
                    rhs = T[:, idx, :].reshape(-1)
                    try:
                        V[idx] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                    except:
                        pass
                
                # Update W
                for idx in range(pn):
                    M = np.einsum('ar,br->abr', U, V).reshape(-1, rank)
                    rhs = T[:, :, idx].reshape(-1)
                    try:
                        W[idx] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                    except:
                        pass
                
                # Check error
                if it % 10 == 0:
                    recon = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.linalg.norm(recon - T)
                    if err < 1e-8:
                        # Found a solution! Try to round
                        break
                
                if time.time() > deadline - 10:
                    break
            
            # Check if we found a solution
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(recon - T)
            if err < 1e-8:
                # Round to nearest integer or half-integer
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                recon_r = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                err_r = np.linalg.norm(recon_r - T)
                if err_r < 1e-8:
                    # Convert to lists
                    U_list = [[float(x) for x in row] for row in U_rounded]
                    V_list = [[float(x) for x in row] for row in V_rounded]
                    W_list = [[float(x) for x in row] for row in W_rounded]
                    best_rank = rank
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    break
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    else:
        # Fall back to schoolbook
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
