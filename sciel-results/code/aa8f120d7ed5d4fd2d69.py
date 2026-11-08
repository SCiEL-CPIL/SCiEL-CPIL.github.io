"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    deadline = time.time() + 140  # safety margin
    
    # The tensor T[a,b,c] = 1 if a=(i,j), b=(j',k), c=(k',i') with i=i',j=j',k=k' else 0
    # We need: sum_c U[a,c]*V[b,c]*W[c_idx,c] = T[a,b,c]
    # where c_idx = k'*n + i'
    
    # Build the target tensor as a list of (a, b, c_idx) -> value
    target = {}
    for i in range(n):
        for j in range(m):
            for jp in range(m):  # j'
                for k in range(p):
                    for kp in range(p):  # k'
                        for ip in range(n):  # i'
                            a = i*m + j
                            b = jp*p + k
                            c = kp*n + ip
                            val = 1 if (i==ip and j==jp and k==kp) else 0
                            if val != 0:
                                target[(a,b,c)] = val
    # Actually, T is sparse: only 3*4*6 = 72 non-zero entries (when i=i', j=j', k=k')
    # T[(i,j),(j,k),(k,i)] = 1 for all i,j,k
    
    # So we have 72 equations: for each (i,j,k):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # And 5184 - 72 = 5112 equations: sum_c ... = 0
    
    # Let's try ranks from 24 upward
    best_result = None
    best_rank = n*m*p  # schoolbook
    
    for r in range(24, best_rank + 1):
        if time.time() > deadline - 5:
            break
        
        # Try to find a decomposition of rank r
        # Use random initialization and optimization
        success = False
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(attempt * 1000 + r)
            U = np.random.randn(nm, r) * 0.5
            V = np.random.randn(mp, r) * 0.5
            W = np.random.randn(pn, r) * 0.5
            
            # Optimize using gradient descent
            lr = 0.01
            for step in range(2000):
                if time.time() > deadline - 5:
                    break
                
                # Compute current tensor values
                # For each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                # This is expensive to compute directly, so use matrix operations
                
                # Build the residual for the 72 non-zero entries
                # and a sample of the zero entries
                
                # Compute all 72 non-zero entries
                res_nonzero = np.zeros(72)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m + j
                            b = j*p + k
                            c = k*n + i
                            # sum_c U[a,c]*V[b,c]*W[c,c]
                            res_nonzero[idx] = np.sum(U[a,:] * V[b,:] * W[c,:])
                            idx += 1
                
                # Compute a sample of zero entries
                # There are 5112 zero entries, sample 100
                zero_entries = []
                all_entries = set()
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            all_entries.add((i*m+j, j*p+k, k*n+i))
                
                # Sample zero entries
                zero_samples = []
                for _ in range(100):
                    a = np.random.randint(nm)
                    b = np.random.randint(mp)
                    c = np.random.randint(pn)
                    if (a,b,c) not in all_entries:
                        zero_samples.append((a,b,c))
                
                res_zero = np.zeros(len(zero_samples))
                for idx2, (a,b,c) in enumerate(zero_samples):
                    res_zero[idx2] = np.sum(U[a,:] * V[b,:] * W[c,:])
                
                # Loss
                loss = np.sum((res_nonzero - 1)**2) + np.sum(res_zero**2)
                
                # Compute gradients
                # d(loss)/d(U[a,c]) = 2 * (res_nonzero[a_idx] - 1) * V[b,c] * W[c,c] for non-zero
                #                + 2 * res_zero[sample] * V[b,c] * W[c,c] for zero samples
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m + j
                            b = j*p + k
                            c = k*n + i
                            err = res_nonzero[idx] - 1
                            dU[a,:] += 2 * err * V[b,:] * W[c,:]
                            dV[b,:] += 2 * err * U[a,:] * W[c,:]
                            dW[c,:] += 2 * err * U[a,:] * V[b,:]
                            idx += 1
                
                for idx2, (a,b,c) in enumerate(zero_samples):
                    err = res_zero[idx2]
                    dU[a,:] += 2 * err * V[b,:] * W[c,:]
                    dV[b,:] += 2 * err * U[a,:] * W[c,:]
                    dW[c,:] += 2 * err * U[a,:] * V[b,:]
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Check convergence
                if loss < 1e-10:
                    success = True
                    break
            
            if success:
                # Verify the solution
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m+j
                            b = j*p+k
                            c = k*n+i
                            val = np.sum(U[a,:] * V[b,:] * W[c,:])
                            max_err = max(max_err, abs(val - 1))
                for _ in range(200):
                    a = np.random.randint(nm)
                    b = np.random.randint(mp)
                    c = np.random.randint(pn)
                    if (a,b,c) not in all_entries:
                        val = np.sum(U[a,:] * V[b,:] * W[c,:])
                        max_err = max(max_err, abs(val))
                
                if max_err < 1e-8:
                    best_rank = r
                    best_result = (U.copy(), V.copy(), W.copy())
                    break
        
        if best_result is not None:
            break
    
    if best_result is None:
        # Fallback to schoolbook
        rank = n*m*p
        U = [[0]*rank for _ in range(nm)]
        V = [[0]*rank for _ in range(mp)]
        W = [[0]*rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][column] = 1
                    V[j*p+k][column] = 1
                    W[k*n+i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    else:
        U, V, W = best_result
        # Convert to lists
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
