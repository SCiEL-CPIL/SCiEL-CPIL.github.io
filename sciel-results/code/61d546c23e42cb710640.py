import time
import numpy as np
from itertools import product

TIME_LIMIT = 140

def construct():
    n, m, p = 3, 4, 6
    
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - 5  # 5s safety margin
    
    # Build the multiplication tensor T[i,j,k] = 1
    # Flattened: T_flat[(i*m+j)*p+k] = 1 for all i,j,k
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Current best: rank 66 from block Strassen + schoolbook
    def build_block_decomp():
        terms = []
        
        def add_schoolbook(i, j, k):
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            terms.append({'U': {u_idx: 1}, 'V': {v_idx: 1}, 'W': {w_idx: 1}})
        
        blocks_222 = []
        for bj in [0, 2]:
            for bk in [0, 2, 4]:
                blocks_222.append((0, bj, bk))
        
        blocks_122 = []
        for bj in [0, 2]:
            for bk in [0, 2, 4]:
                blocks_122.append((2, bj, bk))
        
        strassen_terms = [
            ({(0,0): 1, (1,1): -1}, {(1,1): 1, (0,0): -1}, {(0,0): 1, (1,1): 1}),
            ({(0,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): -1, (0,1): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,1): 1, (1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): -1, (1,0): 1})
        ]
        
        for i0, j0, k0 in blocks_222:
            for u_vals, v_vals, w_vals in strassen_terms:
                term = {'U': {}, 'V': {}, 'W': {}}
                for (u, v), val in u_vals.items():
                    i = i0 + u
                    j = j0 + v
                    term['U'][i*m + j] = val
                for (x, y), val in v_vals.items():
                    j = j0 + x
                    k = k0 + y
                    term['V'][j*p + k] = val
                for (k_l, i_l), val in w_vals.items():
                    k = k0 + k_l
                    i = i0 + i_l
                    term['W'][k*n + i] = val
                terms.append(term)
        
        for i0, j0, k0 in blocks_122:
            for v in range(2):
                for y in range(2):
                    i = i0
                    j = j0 + v
                    k = k0 + y
                    add_schoolbook(i, j, k)
        
        rank = len(terms)
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        for r, term in enumerate(terms):
            for idx, val in term['U'].items():
                U[idx, r] = val
            for idx, val in term['V'].items():
                V[idx, r] = val
            for idx, val in term['W'].items():
                W[idx, r] = val
                
        return U, V, W
    
    U_best, V_best, W_best = build_block_decomp()
    best_rank = U_best.shape[1]
    
    # Verify the decomposition
    def verify(U, V, W):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :])
                    val = np.dot(val, W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U_best, V_best, W_best):
        # Fallback to pure schoolbook
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append({'U': {i*m+j: 1}, 'V': {j*p+k: 1}, 'W': {k*n+i: 1}})
        U_best = np.zeros((n*m, n*m*p))
        V_best = np.zeros((m*p, n*m*p))
        W_best = np.zeros((p*n, n*m*p))
        for r, term in enumerate(terms):
            for idx, val in term['U'].items():
                U_best[idx, r] = val
            for idx, val in term['V'].items():
                V_best[idx, r] = val
            for idx, val in term['W'].items():
                W_best[idx, r] = val
        best_rank = n*m*p
    
    # Try ALS search for lower rank
    def als_search(rank, max_iters=200, n_restarts=3):
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            
            # Random initialization
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U: for each row i*m+j, U[i*m+j, :] = argmin_U ||T_slice - V_slice @ W_slice||
                for i in range(n):
                    for j in range(m):
                        # T[i,j,:] should equal sum_r U[i*m+j,r] * V[j*p:(j+1)*p, r] * W[?, r]
                        # Actually: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                        # For fixed i,j: T[i,j,k] for k=0..p-1
                        # = sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r])
                        # Let A[k,r] = V[j*p+k,r] * W[k*n+i,r]
                        # Then T[i,j,:] = U[i*m+j,:] @ A
                        # So U[i*m+j,:] = T[i,j,:] @ A^+ (pseudo-inverse)
                        
                        A = np.zeros((p, rank))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        
                        try:
                            U_new = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                        except:
                            continue
                        U[i*m+j, :] = U_new
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        # T[:,j,k] for i=0..n-1 = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                        # Let B[i,r] = U[i*m+j,r] * W[k*n+i,r]
                        # T[:,j,k] = V[j*p+k,:] @ B
                        B = np.zeros((n, rank))
                        for i in range(n):
                            B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            V_new = np.linalg.lstsq(B.T, np.ones(n), rcond=None)[0]
                        except:
                            continue
                        V[j*p+k, :] = V_new
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        # T[i,:,k] for j=0..m-1 = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                        # Let C[j,r] = U[i*m+j,r] * V[j*p+k,r]
                        # T[i,:,k] = W[k*n+i,:] @ C
                        C = np.zeros((m, rank))
                        for j in range(m):
                            C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            W_new = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                        except:
                            continue
                        W[k*n+i, :] = W_new
                
                # Check convergence
                if verify(U, V, W):
                    # Try to snap to rational values
                    return U, V, W
            
            # After max_iters, check if close
            if verify(U, V, W):
                return U, V, W
        
        return None
    
    # Try ranks below best_rank
    for rank in range(best_rank - 1, max(best_rank - 10, 50), -1):
        if time.time() > deadline:
            break
        
        result = als_search(rank, max_iters=100, n_restarts=2)
        if result is not None:
            U_new, V_new, W_new = result
            if verify(U_new, V_new, W_new):
                best_rank = rank
                U_best, V_best, W_best = U_new, V_new, W_new
                break
    
    # Convert to list format
    U_out = U_best.tolist()
    V_out = V_best.tolist()
    W_out = W_best.tolist()
    
    return {'U': U_out, 'V': V_out, 'W': W_out}
