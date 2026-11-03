import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    # Build the multiplication tensor
    # T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i
    # T = 1 if consistent, 0 otherwise
    
    # We'll search for rank R decompositions
    # Start with a reasonable guess
    
    best_rank = n * m * p  # schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook as fallback
    U_school = np.zeros((n*m, best_rank))
    V_school = np.zeros((m*p, best_rank))
    W_school = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank decompositions
    # Use the fact that the tensor has structure
    
    # For each target rank, try to find a decomposition
    for target_rank in range(1, best_rank):
        if time.time() > deadline - 20:
            break
        
        # Initialize
        np.random.seed(42)
        U = np.random.randn(n*m, target_rank) * 0.1
        V = np.random.randn(m*p, target_rank) * 0.1
        W = np.random.randn(p*n, target_rank) * 0.1
        
        # Iteratively optimize
        for iteration in range(50):
            if time.time() > deadline - 20:
                break
            
            # Fix U and V, solve for W
            # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # For fixed a,b: sum_r (U[a,r]*V[b,r]) * W[c,r] = T[a,b,c]
            # This is a linear system in W
            
            # Build the system: for each (a,b), we have p*n equations in target_rank unknowns
            # But we only need the equations where T[a,b,c] is nonzero or where consistency matters
            
            # Actually, let's just do a simple gradient descent or alternating least squares
            
            # ALS: alternate fixing two factors and solving for the third
            for _ in range(3):  # Cycle through U, V, W
                if time.time() > deadline - 20:
                    break
                
                # Solve for W: for each c, W[c,:] is determined by
                # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all a,b
                # This is: (U^T * V_flattened) @ W[c,:] = T_flattened[c,:]
                # More precisely: for each c, the vector (sum_r U[a,r]*V[b,r]*W[c,r])_{a,b} should equal T[:,:,c]
                # So: (U ⊙ V) @ W[c,:] = T[:,:,c] where ⊙ is some product
                
                # Let's form the matrix M where M[(a,b), r] = U[a,r] * V[b,r]
                # Then M @ W[c,:] = T[:,:,c]
                
                M = np.zeros((n*m*m*p, target_rank))
                idx = 0
                for a in range(n*m):
                    for b in range(m*p):
                        for r in range(target_rank):
                            M[idx, r] = U[a, r] * V[b, r]
                        idx += 1
                
                # T flattened: T_flat[c, idx] = T[a,b,c]
                T_flat = np.zeros((p*n, n*m*m*p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m + j
                            b = j*p + k
                            c = k*n + i
                            T_flat[c, a*m*p + b] = 1
                
                # Solve M @ W[c,:] = T_flat[c,:] for each c
                for c in range(p*n):
                    try:
                        W[c, :] = np.linalg.lstsq(M, T_flat[c, :], rcond=None)[0]
                    except:
                        pass
                
                # Similarly for U and V
                # Solve for U: for each a, U[a,:] determined by
                # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                # M2[(b,c), r] = V[b,r]*W[c,r], M2 @ U[a,:] = T[a,:,:]
                
                M2 = np.zeros((m*p*p*n, target_rank))
                idx = 0
                for b in range(m*p):
                    for c in range(p*n):
                        for r in range(target_rank):
                            M2[idx, r] = V[b, r] * W[c, r]
                        idx += 1
                
                T_flat2 = np.zeros((n*m, m*p*p*n))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m + j
                            b = j*p + k
                            c = k*n + i
                            T_flat2[a, b*p*n + c] = 1
                
                for a in range(n*m):
                    try:
                        U[a, :] = np.linalg.lstsq(M2, T_flat2[a, :], rcond=None)[0]
                    except:
                        pass
                
                # Solve for V: for each b, V[b,:] determined by
                # M3[(a,c), r] = U[a,r]*W[c,r], M3 @ V[b,:] = T[:,:,b]
                
                M3 = np.zeros((n*m*p*n, target_rank))
                idx = 0
                for a in range(n*m):
                    for c in range(p*n):
                        for r in range(target_rank):
                            M3[idx, r] = U[a, r] * W[c, r]
                        idx += 1
                
                T_flat3 = np.zeros((m*p, n*m*p*n))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m + j
                            b = j*p + k
                            c = k*n + i
                            T_flat3[b, a*p*n + c] = 1
                
                for b in range(m*p):
                    try:
                        V[b, :] = np.linalg.lstsq(M3, T_flat3[b, :], rcond=None)[0]
                    except:
                        pass
        
        # Check if we found an exact decomposition
        # Round to nearest half-integer
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        # Verify
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    val = np.sum(U_round[a, :] * V_round[b, :] * W_round[c, :])
                    error += abs(val - 1)
        
        # Also check zero entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if not (j_a == j_b and k_b == k_c and i_a == i_c):
                        val = np.sum(U_round[a, :] * V_round[b, :] * W_round[c, :])
                        error += abs(val)
        
        if error < 1e-6:
            best_rank = target_rank
            best_U, best_V, best_W = U_round, V_round, W_round
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
