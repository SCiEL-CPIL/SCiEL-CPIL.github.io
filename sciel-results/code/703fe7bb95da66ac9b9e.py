import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Theoretical lower bound
    min_rank = max(n*m, m*p, p*n)
    
    # We try to find a decomposition with rank R
    # Start from min_rank and go up
    # Given time constraints, we might not be able to search too far
    # But for 3x5x6, known ranks are likely < 90
    
    best_result = None
    best_rank = n * m * p # Schoolbook rank
    
    # Pre-compute schoolbook for fallback
    U_school = [[0] * (n*m*p) for _ in range(n*m)]
    V_school = [[0] * (n*m*p) for _ in range(m*p)]
    W_school = [[0] * (n*m*p) for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    start_time = time.time()
    time_limit = 110 # Leave 10s margin
    
    # Try ranks from min_rank to best_rank - 1
    for R in range(min_rank, best_rank):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random initializations for this rank
        found = False
        for trial in range(5):
            if time.time() - start_time > time_limit:
                break
                
            np.random.seed(42 + R + trial)
            
            # Initialize U, V, W
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # ALS Iterations
            for it in range(50):
                if time.time() - start_time > time_limit:
                    break
                    
                # Update U
                for i in range(n):
                    for j in range(m):
                        # Solve for U[i*m+j, :]
                        # Target: 1 if j'=j, i''=i, k''=k'
                        # Equation: sum_r U[i,j,r] V[j',k',r] W[k'',i'',r] = delta
                        # Construct matrix A of size (m*p*n) x R
                        # Rows indexed by (j', k', i'')
                        
                        # We can vectorize this
                        # V is (m*p, R). Reshape to (m, p, R)
                        # W is (p*n, R). Reshape to (p, n, R)
                        
                        # For fixed i, j:
                        # We need A_{(j',k',i''), r} = V_{j',k',r} * W_{i'',k',r}?
                        # Wait, W index is (k'', i'').
                        # So W_{k'', i'', r}.
                        # In the sum, the index of W is (k'', i'').
                        # The index of V is (j', k').
                        # The condition is k'' = k'?
                        # No, the condition is k = k''.
                        # In the sum over r, the indices j', k', i'' are free.
                        # The target is 1 if j'=j, i''=i, k''=k' (since k=k'').
                        
                        # So A_{(j',k',i''), r} = V_{j',k',r} * W_{k',i'',r}?
                        # No, W is indexed by (k'', i'').
                        # So W_{k'', i'', r}.
                        # The product is V_{j',k',r} * W_{k'',i'',r}.
                        
                        # We want to solve A U_col = b.
                        # b_{(j',k',i'')} = 1 if j'=j, i''=i, k''=k'.
                        
                        # Let's build A
                        # V_reshaped: (m, p, R)
                        # W_reshaped: (p, n, R) -> indices (k'', i'')
                        
                        # A has rows (j', k', i'').
                        # A_{(j',k',i''), r} = V[j', k', r] * W[k'', i'', r]?
                        # Wait, the index of W in the product is (k'', i'').
                        # But the condition links k'' and k'?
                        # No, k'' is the column of C. k' is the column of B.
                        # The condition is k'' = k (where k is the column of B).
                        # So k'' = k'.
                        # So we sum over r: U[i,j,r] V[j',k',r] W[k',i'',r].
                        
                        # So A_{(j',k',i''), r} = V[j',k',r] * W[k',i'',r].
                        
                        V_r = V.reshape(m, p, R)
                        W_r = W.reshape(p, n, R)
                        
                        # Construct A
                        # Size (m*p*n) x R
                        # Loop over j', k', i''
                        # A_flat = np.zeros((m*p*n, R))
                        # b_flat = np.zeros(m*p*n)
                        
                        # To speed up, we can precompute some structures
                        # But for small sizes, direct construction is fine.
                        
                        A = np.zeros((m*p*n, R))
                        b = np.zeros(m*p*n)
                        
                        idx = 0
                        for jp in range(m):
                            for kp in range(p):
                                for ip in range(n):
                                    # V index: jp, kp
                                    # W index: kp, ip (since k''=kp, i''=ip)
                                    # A[idx, r] = V[jp, kp, r] * W[kp, ip, r]
                                    A[idx, :] = V_r[jp, kp, :] * W_r[kp, ip, :]
                                    
                                    # b
                                    if jp == j and ip == i:
                                        # k'' must equal k' (which is kp)
                                        # So b is 1 for all kp?
                                        # No, the target is 1 if k''=k'.
                                        # So for fixed jp=j, ip=i, b is 1 for all kp.
                                        b[idx] = 1
                                    idx += 1
                                    
                        # Solve least squares
                        # U[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass

                # Update V
                for j in range(m):
                    for k in range(p):
                        # Solve for V[j*p+k, :]
                        # Equation: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta
                        # A_{(i,j',k''), r} = U[i,j,r] * W[k'',i,r]?
                        # Condition: i=i'', j'=j, k''=k.
                        # So A_{(i,j',k''), r} = U[i,j,r] * W[k'',i,r].
                        # b_{(i,j',k'')} = 1 if i=i, j'=j, k''=k.
                        
                        U_r = U.reshape(n, m, R)
                        W_r = W.reshape(p, n, R)
                        
                        A = np.zeros((n*m*p, R))
                        b = np.zeros(n*m*p)
                        
                        idx = 0
                        for ip in range(n):
                            for jp in range(m):
                                for kp in range(p):
                                    # U index: ip, jp
                                    # W index: kp, ip
                                    A[idx, :] = U_r[ip, jp, :] * W_r[kp, ip, :]
                                    
                                    if jp == j and kp == k:
                                        # i can be anything?
                                        # No, condition is i=i'' (ip), j'=j (jp), k''=k (kp).
                                        # So b is 1 for all ip.
                                        b[idx] = 1
                                    idx += 1
                                    
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass

                # Update W
                for k in range(p):
                    for i in range(n):
                        # Solve for W[k*n+i, :]
                        # Equation: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta
                        # A_{(i,j,k'), r} = U[i,j,r] * V[j,k,r]?
                        # Condition: i=i'', j'=j, k''=k.
                        # So A_{(i,j,k'), r} = U[i,j,r] * V[j,k,r].
                        # b_{(i,j,k')} = 1 if i=i, j=j, k'=k.
                        
                        U_r = U.reshape(n, m, R)
                        V_r = V.reshape(m, p, R)
                        
                        A = np.zeros((n*m*p, R))
                        b = np.zeros(n*m*p)
                        
                        idx = 0
                        for ip in range(n):
                            for jp in range(m):
                                for kp in range(p):
                                    # U index: ip, jp
                                    # V index: jp, kp
                                    A[idx, :] = U_r[ip, jp, :] * V_r[jp, kp, :]
                                    
                                    if ip == i and jp == j and kp == k:
                                        b[idx] = 1
                                    idx += 1
                                    
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass

            # Check residual
            # Compute T_reconstructed
            # T_rec[i,j, j',k', i'',k''] = sum_r U[i,j,r] V[j',k',r] W[k'',i'',r]
            # We only need to check if it matches T_target.
            # T_target is 1 if i=i'', j=j', k''=k'.
            
            U_r = U.reshape(n, m, R)
            V_r = V.reshape(m, p, R)
            W_r = W.reshape(p, n, R)
            
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for kp in range(p):
                            for ip in range(n):
                                for kp2 in range(p):
                                    # Check condition
                                    # Target: 1 if i==ip, j==jp, kp==kp2
                                    val = 0
                                    for r in range(R):
                                        val += U_r[i, j, r] * V_r[jp, kp, r] * W_r[kp2, ip, r]
                                    
                                    if i == ip and j == jp and kp == kp2:
                                        err = abs(val - 1)
                                    else:
                                        err = abs(val)
                                    
                                    if err > max_err:
                                        max_err = err
                                        
            if max_err < 1e-8:
                # Success
                best_rank = R
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                found = True
                break
                
        if found:
            break

    return best_result

# EVOLVE-BLOCK-END
