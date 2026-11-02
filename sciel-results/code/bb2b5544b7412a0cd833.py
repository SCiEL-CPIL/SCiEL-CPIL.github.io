"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36
    
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    deadline = time.time() + 100
    
    # For each candidate rank, try to find a decomposition
    for rank in range(schoolbook_rank - 1, 25, -1):
        if time.time() > deadline:
            break
        
        success = False
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 1000 + rank + 42)
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            for it in range(300):
                if time.time() > deadline:
                    break
                
                # Update W using all constraints
                # For each c_idx, we have constraints from all (a,b) pairs:
                # sum_r U[a,r]*V[b,r]*W[c_idx,r] = T[a,b,c_idx]
                # This is: (U[a,:]*V[b,:]) @ W[c_idx,:] = T[a,b,c_idx] for all (a,b)
                # So we have a system: A_full @ W[c_idx,:] = rhs_full
                # where A_full has one row per (a,b) pair, and rhs_full is T[a,b,c_idx]
                
                new_W = np.zeros_like(W)
                for c_idx in range(p * n):
                    # Determine which (k,i) this c_idx corresponds to
                    k = c_idx // n
                    i = c_idx % n
                    
                    # Build the full system for W[c_idx,:]
                    # Constraints: for all a in [0, n*m), b in [0, m*p):
                    #   sum_r U[a,r]*V[b,r]*W[c_idx,r] = T[a,b,c_idx]
                    # T[a,b,c_idx] = 1 if there exists j such that a=i*m+j, b=j*p+k, else 0
                    
                    A_full = np.zeros((n*m * m*p, rank))
                    rhs_full = np.zeros(n*m * m*p)
                    row = 0
                    for a in range(n*m):
                        for b in range(m*p):
                            A_full[row, :] = U[a, :] * V[b, :]
                            # Check if T[a,b,c_idx] = 1
                            # a = i'*m+j', b = j''*p+k', c_idx = k''*n+i''
                            # T=1 if i'=i, j'=j'', k'=k
                            i_a = a // m
                            j_a = a % m
                            j_b = b // p
                            k_b = b % p
                            if i_a == i and j_a == j_b and k_b == k:
                                rhs_full[row] = 1.0
                            row += 1
                    
                    # Solve least squares
                    sol, residuals, rank_c, sv = np.linalg.lstsq(A_full, rhs_full, rcond=None)
                    new_W[c_idx, :] = sol
                
                # Similarly update U and V with full constraints
                new_U = np.zeros_like(U)
                for a in range(n*m):
                    i = a // m
                    j = a % m
                    A_full = np.zeros((m*p * p*n, rank))
                    rhs_full = np.zeros(m*p * p*n)
                    row = 0
                    for b in range(m*p):
                        for c_idx in range(p*n):
                            A_full[row, :] = V[b, :] * W[c_idx, :]
                            # Check if T[a,b,c_idx] = 1
                            j_b = b // p
                            k_b = b % p
                            k_c = c_idx // n
                            i_c = c_idx % n
                            if j == j_b and i == i_c and k_b == k_c:
                                rhs_full[row] = 1.0
                            row += 1
                    sol, residuals, rank_u, sv = np.linalg.lstsq(A_full, rhs_full, rcond=None)
                    new_U[a, :] = sol
                
                new_V = np.zeros_like(V)
                for b in range(m*p):
                    j = b // p
                    k = b % p
                    A_full = np.zeros((n*m * p*n, rank))
                    rhs_full = np.zeros(n*m * p*n)
                    row = 0
                    for a in range(n*m):
                        for c_idx in range(p*n):
                            A_full[row, :] = U[a, :] * W[c_idx, :]
                            i_a = a // m
                            j_a = a % m
                            k_c = c_idx // n
                            i_c = c_idx % n
                            if j_a == j and i_a == i_c and k == k_c:
                                rhs_full[row] = 1.0
                            row += 1
                    sol, residuals, rank_v, sv = np.linalg.lstsq(A_full, rhs_full, rcond=None)
                    new_V[b, :] = sol
                
                diff = np.max(np.abs(new_U - U)) + np.max(np.abs(new_V - V)) + np.max(np.abs(new_W - W))
                U, V, W = new_U, new_V, new_W
                if diff < 1e-12:
                    break
            
            # Verify
            valid = True
            for a in range(n*m):
                for b in range(m*p):
                    for c_idx in range(p*n):
                        val = np.dot(U[a, :] * V[b, :], W[c_idx, :])
                        i_a = a // m
                        j_a = a % m
                        j_b = b // p
                        k_b = b % p
                        k_c = c_idx // n
                        i_c = c_idx % n
                        expected = 1.0 if (j_a == j_b and i_a == i_c and k_b == k_c) else 0.0
                        if abs(val - expected) > 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                success = True
                break
        
        if success:
            break
    
    if best_rank < schoolbook_rank and best_U is not None:
        U_list = [[round(float(x), 6) for x in row] for row in best_U]
        V_list = [[round(float(x), 6) for x in row] for row in best_V]
        W_list = [[round(float(x), 6) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        rank = schoolbook_rank
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
# EVOLVE-BLOCK-END

