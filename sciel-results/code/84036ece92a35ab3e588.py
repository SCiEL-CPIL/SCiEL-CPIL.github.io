import numpy as np
import time

def construct():
    n, m, p = 3, 5, 5
    t0 = time.time()
    deadline = t0 + 110.0
    
    # Helper to verify
    def _verify(U, V, W, n, m, p, tol=1e-6):
        r = U.shape[1]
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = np.dot(W[k*n+i, :], U[i*m+j1, :] * V[j2*p+k, :])
                        target = 1.0 if j1 == j2 else 0.0
                        if abs(s - target) > tol:
                            return False
        return True
    
    # Get Strassen terms for a 2x2x2 block
    def get_strassen_terms(i1, i2, j1, j2, k1, k2):
        terms = []
        
        # M1 = (A11+A22)(B11+B22)
        terms.append({
            'U': {i1*m+j1: 1, i2*m+j2: 1},
            'V': {j1*p+k1: 1, j2*p+k2: 1},
            'W': {k1*n+i1: 1, k2*n+i2: 1}
        })
        
        # M2 = (A21+A22)B11
        terms.append({
            'U': {i2*m+j1: 1, i2*m+j2: 1},
            'V': {j1*p+k1: 1},
            'W': {k1*n+i2: 1, k2*n+i2: -1}
        })
        
        # M3 = A11(B21-B22)
        terms.append({
            'U': {i1*m+j1: 1},
            'V': {j2*p+k1: 1, j2*p+k2: -1},
            'W': {k2*n+i1: 1, k2*n+i2: 1}
        })
        
        # M4 = A22(B12-B11)
        terms.append({
            'U': {i2*m+j2: 1},
            'V': {j1*p+k2: 1, j1*p+k1: -1},
            'W': {k1*n+i1: 1, k1*n+i2: 1}
        })
        
        # M5 = (A11+A12)B22
        terms.append({
            'U': {i1*m+j1: 1, i1*m+j2: 1},
            'V': {j2*p+k2: 1},
            'W': {k1*n+i1: -1, k2*n+i1: 1}
        })
        
        # M6 = (A21-A11)(B11+B12)
        terms.append({
            'U': {i2*m+j1: 1, i1*m+j1: -1},
            'V': {j1*p+k1: 1, j1*p+k2: 1},
            'W': {k2*n+i2: 1}
        })
        
        # M7 = (A12-A22)(B21+B22)
        terms.append({
            'U': {i1*m+j2: 1, i2*m+j2: -1},
            'V': {j2*p+k1: 1, j2*p+k2: 1},
            'W': {k1*n+i1: 1}
        })
        
        return terms
    
    # Define the 4 disjoint 2x2x2 blocks
    blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
    ]
    
    # Build the decomposition
    n_terms_not_in_blocks = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                in_block = False
                for (i1,i2,j1,j2,k1,k2) in blocks:
                    if i in (i1,i2) and j in (j1,j2) and k in (k1,k2):
                        in_block = True
                        break
                if not in_block:
                    n_terms_not_in_blocks += 1
    
    n_strassen_terms = len(blocks) * 7
    total_rank = n_terms_not_in_blocks + n_strassen_terms
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    
    # Add schoolbook terms not in blocks
    for i in range(n):
        for j in range(m):
            for k in range(p):
                in_block = False
                for (i1,i2,j1,j2,k1,k2) in blocks:
                    if i in (i1,i2) and j in (j1,j2) and k in (k1,k2):
                        in_block = True
                        break
                if not in_block:
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    # Add Strassen terms for each block
    for (i1,i2,j1,j2,k1,k2) in blocks:
        terms = get_strassen_terms(i1, i2, j1, j2, k1, k2)
        for term in terms:
            for row, val in term['U'].items():
                U[row, col] += val
            for row, val in term['V'].items():
                V[row, col] += val
            for row, val in term['W'].items():
                W[row, col] += val
            col += 1
    
    # Verify
    if _verify(U, V, W, n, m, p):
        best_rank = total_rank
        best_U = U.copy()
        best_V = V.copy()
        best_W = W.copy()
    else:
        # Fallback to schoolbook
        rank0 = n * m * p
        U0 = np.zeros((n*m, rank0))
        V0 = np.zeros((m*p, rank0))
        W0 = np.zeros((p*n, rank0))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U0[i*m+j, col] = 1
                    V0[j*p+k, col] = 1
                    W0[k*n+i, col] = 1
                    col += 1
        best_rank = rank0
        best_U = U0
        best_V = V0
        best_W = W0
    
    # Try to reduce further using ALS (optional, if time permits)
    if time.time() < deadline - 30:
        def _als(U, V, W, n, m, p, max_iters=100, tol=1e-8):
            r = U.shape[1]
            prev_err = float('inf')
            for it in range(max_iters):
                if time.time() > deadline - 10.0:
                    break
                    
                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m*m, r))
                        for j1 in range(m):
                            for j2 in range(m):
                                A[j1*m+j2, :] = U[i*m+j1, :] * V[j2*p+k, :]
                        b = np.zeros(m*m)
                        for j in range(m):
                            b[j*m+j] = 1.0
                        w, *_ = np.linalg.lstsq(A, b, rcond=None)
                        W[k*n+i, :] = w
                        
                # Update U
                for i in range(n):
                    for j1 in range(m):
                        A = np.zeros((p*m, r))
                        b = np.zeros(p*m)
                        for k in range(p):
                            for j2 in range(m):
                                A[k*m+j2, :] = W[k*n+i, :] * V[j2*p+k, :]
                                b[k*m+j2] = 1.0 if j1 == j2 else 0.0
                        u, *_ = np.linalg.lstsq(A, b, rcond=None)
                        U[i*m+j1, :] = u
                        
                # Update V
                for j2 in range(m):
                    for k in range(p):
                        A = np.zeros((n*m, r))
                        b = np.zeros(n*m)
                        for i in range(n):
                            for j1 in range(m):
                                A[i*m+j1, :] = W[k*n+i, :] * U[i*m+j1, :]
                                b[i*m+j1] = 1.0 if j1 == j2 else 0.0
                        v, *_ = np.linalg.lstsq(A, b, rcond=None)
                        V[j2*p+k, :] = v
                        
                # Calculate error
                err = 0.0
                for i in range(n):
                    for k in range(p):
                        for j1 in range(m):
                            for j2 in range(m):
                                s = np.dot(W[k*n+i, :], U[i*m+j1, :] * V[j2*p+k, :])
                                target = 1.0 if j1 == j2 else 0.0
                                err += (s - target)**2
                                
                if err < tol:
                    break
                if err > prev_err * 1.01 and it > 10:
                    break
                prev_err = err
                
            return U, V, W, err
        
        # Try to reduce rank
        for r in range(best_rank - 1, 50, -1):
            if time.time() > deadline - 20.0:
                break
            
            n_restarts = 2
            found = False
            for restart in range(n_restarts):
                if time.time() > deadline - 15.0:
                    break
                    
                np.random.seed(42 + r * 100 + restart)
                U_rand = np.random.randn(n*m, r) * 0.5
                V_rand = np.random.randn(m*p, r) * 0.5
                W_rand = np.random.randn(p*n, r) * 0.5
                
                U_r, V_r, W_r, err = _als(U_rand, V_rand, W_rand, n, m, p, max_iters=100)
                
                if err < 1e-8:
                    U_snap = np.round(U_r * 2) / 2
                    V_snap = np.round(V_r * 2) / 2
                    W_snap = np.round(W_r * 2) / 2
                    
                    if _verify(U_snap, V_snap, W_snap, n, m, p):
                        best_rank = r
                        best_U = U_snap.copy()
                        best_V = V_snap.copy()
                        best_W = W_snap.copy()
                        found = True
                        break
            
            if found:
                break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
