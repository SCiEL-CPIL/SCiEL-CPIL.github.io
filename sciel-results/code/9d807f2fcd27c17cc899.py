import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135
    
    # Build target: T[i*m+j, j*p+k, k*n+i] = 1
    # Non-zero entries
    nz_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nz_entries.append((a, b, c))
    
    def tensor_value(a, b, c):
        """Get T[a,b,c]"""
        i = a // m
        j = a % m
        # b must equal j*p+k for some k, and c must equal k*n+i
        if b < j * p or b >= (j + 1) * p:
            return 0
        k = b - j * p
        expected_c = k * n + i
        if c == expected_c:
            return 1
        return 0
    
    def compute_error(U, V, W, sample_size=200):
        """Compute error on non-zero entries and sampled zero entries"""
        rank = U.shape[1]
        err = 0.0
        # Non-zero entries
        for (a, b, c) in nz_entries:
            s = 0.0
            for r in range(rank):
                s += U[a, r] * V[b, r] * W[c, r]
            err += (s - 1.0) ** 2
        # Sample zero entries
        import random
        rng = np.random.RandomState(0)
        for _ in range(sample_size):
            a = rng.randint(0, n * m)
            b = rng.randint(0, m * p)
            c = rng.randint(0, p * n)
            tv = tensor_value(a, b, c)
            if tv == 0:
                s = 0.0
                for r in range(rank):
                    s += U[a, r] * V[b, r] * W[c, r]
                err += s ** 2
        return err
    
    def try_rank_als(r, n_init=30, max_iter=100):
        """ALS search for rank-r decomposition"""
        best_err = float('inf')
        best = None
        
        for init in range(n_init):
            if time.time() > deadline:
                break
            
            rng = np.random.RandomState(init * 12345 + 67890)
            U = rng.randn(n * m, r) * 0.3
            V = rng.randn(m * p, r) * 0.3
            W = rng.randn(p * n, r) * 0.3
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Update U row by row
                for a in range(n * m):
                    # X shape: (m*p*p*n, r), X[(b,c), r] = V[b,r]*W[c,r]
                    # y shape: (m*p*p*n,)
                    # Use only non-zero + a few zero entries for speed
                    # Non-zero for row a: i=a//m, j=a%m, k in range(p): b=j*p+k, c=k*n+i
                    i = a // m
                    j = a % m
                    idx = []
                    vals = []
                    for k in range(p):
                        b = j * p + k
                        c = k * n + i
                        idx.append((b, c))
                        vals.append(1.0)
                    # Add some zero entries
                    rng2 = np.random.RandomState(a * 1000 + it)
                    for _ in range(50):
                        b = rng2.randint(0, m * p)
                        c = rng2.randint(0, p * n)
                        if tensor_value(a, b, c) == 0:
                            idx.append((b, c))
                            vals.append(0.0)
                    
                    X = np.array([[V[b, r] * W[c, r] for r in range(r)] for (b, c) in idx])
                    y = np.array(vals)
                    U[a, :], res, rk, sv = np.linalg.lstsq(X, y, rcond=None)
                
                # Update V
                for b in range(m * p):
                    j = b // p
                    k = b % p
                    idx = []
                    vals = []
                    for i in range(n):
                        a = i * m + j
                        c = k * n + i
                        idx.append((a, c))
                        vals.append(1.0)
                    rng2 = np.random.RandomState(b * 1000 + it + 1000)
                    for _ in range(50):
                        a = rng2.randint(0, n * m)
                        c = rng2.randint(0, p * n)
                        if tensor_value(a, b, c) == 0:
                            idx.append((a, c))
                            vals.append(0.0)
                    
                    X = np.array([[U[a, r] * W[c, r] for r in range(r)] for (a, c) in idx])
                    y = np.array(vals)
                    V[b, :], res, rk, sv = np.linalg.lstsq(X, y, rcond=None)
                
                # Update W
                for c in range(p * n):
                    k = c // n
                    i = c % n
                    idx = []
                    vals = []
                    for j in range(m):
                        a = i * m + j
                        b = j * p + k
                        idx.append((a, b))
                        vals.append(1.0)
                    rng2 = np.random.RandomState(c * 1000 + it + 2000)
                    for _ in range(50):
                        a = rng2.randint(0, n * m)
                        b = rng2.randint(0, m * p)
                        if tensor_value(a, b, c) == 0:
                            idx.append((a, b))
                            vals.append(0.0)
                    
                    X = np.array([[U[a, r] * V[b, r] for r in range(r)] for (a, b) in idx])
                    y = np.array(vals)
                    W[c, :], res, rk, sv = np.linalg.lstsq(X, y, rcond=None)
            
            err = compute_error(U, V, W, sample_size=100)
            if err < best_err:
                best_err = err
                best = (U.copy(), V.copy(), W.copy())
        
        if best is not None and best_err < 1e-6:
            return best, best_err
        return best, best_err
    
    # Try ranks
    best_result = None
    best_rank = n * m * p
    
    # Try ranks from low to high
    for r in range(40, 90):
        if time.time() > deadline:
            break
        result, err = try_rank_als(r, n_init=15, max_iter=50)
        if result is not None and err < 1e-6:
            U, V, W = result
            # Try to round to half-integers
            U_h = np.round(U * 2) / 2
            V_h = np.round(V * 2) / 2
            W_h = np.round(W * 2) / 2
            err_h = compute_error(U_h, V_h, W_h, sample_size=200)
            if err_h < 1e-8:
                best_result = (U_h, V_h, W_h)
                best_rank = r
                break
            else:
                U_i = np.round(U)
                V_i = np.round(V)
                W_i = np.round(W)
                err_i = compute_error(U_i, V_i, W_i, sample_size=200)
                if err_i < 1e-8:
                    best_result = (U_i, V_i, W_i)
                    best_rank = r
                    break
                else:
                    best_result = (U, V, W)
                    best_rank = r
                    break
    
    if best_result is None:
        # Schoolbook fallback
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        best_result = (U, V, W)
    
    U, V, W = best_result
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
