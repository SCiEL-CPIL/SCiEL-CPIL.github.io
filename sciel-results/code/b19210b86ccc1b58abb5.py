import numpy as np
import time

def construct():
    n, m, p = 2, 2, 7
    nm, mp, pn = n*m, m*p, p*n  # 4, 14, 14
    
    # Build valid terms
    valid_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_terms.append((a, b, c))
    
    # Precompute invalid term list for efficiency
    invalid_terms = []
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                i1 = a // m
                j1 = a % m
                j2 = b // p
                k1 = b % p
                i2 = c % n
                k2 = c // n
                if j1 == j2 and i1 == i2 and k1 == k2:
                    continue
                invalid_terms.append((a, b, c))
    
    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for (a, b, c) in valid_terms:
            s = 0.0
            for r_idx in range(r):
                s += U[a, r_idx] * V[b, r_idx] * W[c, r_idx]
            if abs(s - 1.0) > tol:
                return False
        for (a, b, c) in invalid_terms:
            s = 0.0
            for r_idx in range(r):
                s += U[a, r_idx] * V[b, r_idx] * W[c, r_idx]
            if abs(s) > tol:
                return False
        return True
    
    def compute_error_fast(U, V, W):
        """Compute error using vectorized operations."""
        r = U.shape[1]
        err = 0.0
        # Valid terms
        for (a, b, c) in valid_terms:
            s = np.dot(U[a, :], V[b, :] * W[c, :])
            err += (s - 1.0) ** 2
        # Invalid terms
        for (a, b, c) in invalid_terms:
            s = np.dot(U[a, :], V[b, :] * W[c, :])
            err += s ** 2
        return err
    
    # Known rank-25 Strassen solution
    def make_strassen_solution():
        U = np.zeros((nm, 25), dtype=float)
        V = np.zeros((mp, 25), dtype=float)
        W = np.zeros((pn, 25), dtype=float)
        
        def add_strassen_block(U, V, W, col_start, k_start):
            U[0, col_start+0] += 1; U[3, col_start+0] += 1
            V[7*0 + k_start, col_start+0] += 1; V[7*1 + (k_start+1), col_start+0] += 1
            W[2*k_start + 0, col_start+0] += 1; W[2*(k_start+1) + 1, col_start+0] += 1
            U[2, col_start+1] += 1; U[3, col_start+1] += 1
            V[7*0 + k_start, col_start+1] += 1
            W[2*k_start + 1, col_start+1] += 1; W[2*(k_start+1) + 1, col_start+1] -= 1
            U[0, col_start+2] += 1
            V[7*0 + (k_start+1), col_start+2] += 1; V[7*1 + (k_start+1), col_start+2] -= 1
            W[2*(k_start+1) + 0, col_start+2] += 1; W[2*(k_start+1) + 1, col_start+2] += 1
            U[3, col_start+3] += 1
            V[7*1 + k_start, col_start+3] += 1; V[7*0 + k_start, col_start+3] -= 1
            W[2*k_start + 0, col_start+3] += 1; W[2*k_start + 1, col_start+3] += 1
            U[0, col_start+4] += 1; U[1, col_start+4] += 1
            V[7*1 + (k_start+1), col_start+4] += 1
            W[2*k_start + 0, col_start+4] -= 1; W[2*(k_start+1) + 0, col_start+4] += 1
            U[2, col_start+5] += 1; U[0, col_start+5] -= 1
            V[7*0 + k_start, col_start+5] += 1; V[7*0 + (k_start+1), col_start+5] += 1
            W[2*(k_start+1) + 1, col_start+5] += 1
            U[1, col_start+6] += 1; U[3, col_start+6] -= 1
            V[7*1 + k_start, col_start+6] += 1; V[7*1 + (k_start+1), col_start+6] += 1
            W[2*k_start + 0, col_start+6] += 1
        
        add_strassen_block(U, V, W, 0, 0)
        add_strassen_block(U, V, W, 7, 2)
        add_strassen_block(U, V, W, 14, 4)
        
        col_start = 21
        k = 6
        U[0, col_start+0] += 1; V[7*0 + k, col_start+0] += 1; W[2*k + 0, col_start+0] += 1
        U[1, col_start+1] += 1; V[7*1 + k, col_start+1] += 1; W[2*k + 0, col_start+1] += 1
        U[2, col_start+2] += 1; V[7*0 + k, col_start+2] += 1; W[2*k + 1, col_start+2] += 1
        U[3, col_start+3] += 1; V[7*1 + k, col_start+3] += 1; W[2*k + 1, col_start+3] += 1
        
        return U, V, W
    
    U25, V25, W25 = make_strassen_solution()
    
    best_U, best_V, best_W = U25.copy(), V25.copy(), W25.copy()
    best_rank = 25
    
    start_time = time.time()
    deadline = start_time + 125
    
    def try_optimize(U_init, V_init, W_init, max_iter=2000):
        """Adam-style optimizer."""
        rank = U_init.shape[1]
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        lr = 0.005
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        
        mU = np.zeros_like(U); vU = np.zeros_like(U)
        mV = np.zeros_like(V); vV = np.zeros_like(V)
        mW = np.zeros_like(W); vW = np.zeros_like(W)
        
        t = 0
        prev_err = compute_error_fast(U, V, W)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            t += 1
            lr_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
            
            # Compute gradients
            gU = np.zeros_like(U)
            gV = np.zeros_like(V)
            gW = np.zeros_like(W)
            
            for (a, b, c) in valid_terms:
                S = np.dot(U[a, :], V[b, :] * W[c, :])
                diff = S - 1.0
                if abs(diff) < 1e-10:
                    continue
                vw = V[b, :] * W[c, :]
                gU[a, :] += 2 * diff * vw
                gV[b, :] += 2 * diff * U[a, :] * W[c, :]
                gW[c, :] += 2 * diff * U[a, :] * V[b, :]
            
            for (a, b, c) in invalid_terms:
                S = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(S) < 1e-10:
                    continue
                vw = V[b, :] * W[c, :]
                gU[a, :] += 2 * S * vw
                gV[b, :] += 2 * S * U[a, :] * W[c, :]
                gW[c, :] += 2 * S * U[a, :] * V[b, :]
            
            # Adam update
            mU = beta1 * mU + (1 - beta1) * gU
            vU = beta2 * vU + (1 - beta2) * gU**2
            mUh = mU / (1 - beta1**t)
            vUh = vU / (1 - beta2**t)
            U -= lr_t * mUh / (np.sqrt(vUh) + eps)
            
            mV = beta1 * mV + (1 - beta1) * gV
            vV = beta2 * vV + (1 - beta2) * gV**2
            mVh = mV / (1 - beta1**t)
            vVh = vV / (1 - beta2**t)
            V -= lr_t * mVh / (np.sqrt(vVh) + eps)
            
            mW = beta1 * mW + (1 - beta1) * gW
            vW = beta2 * vW + (1 - beta2) * gW**2
            mWh = mW / (1 - beta1**t)
            vWh = vW / (1 - beta2**t)
            W -= lr_t * mWh / (np.sqrt(vWh) + eps)
            
            if it % 200 == 0:
                err = compute_error_fast(U, V, W)
                if err < 1e-8:
                    break
        
        err = compute_error_fast(U, V, W)
        return U, V, W, err
    
    # Try to find rank < 25
    for target_rank in [24, 23]:
        if time.time() > deadline:
            break
        
        found = False
        n_attempts = 30 if target_rank == 24 else 20
        
        for attempt in range(n_attempts):
            if time.time() > deadline:
                break
            
            np.random.seed(target_rank * 1000 + attempt)
            
            U_init = np.zeros((nm, target_rank), dtype=float)
            V_init = np.zeros((mp, target_rank), dtype=float)
            W_init = np.zeros((pn, target_rank), dtype=float)
            
            for r in range(target_rank):
                # Sparse random initialization
                n_u = np.random.randint(2, 5)
                idx_u = np.random.choice(nm, size=n_u, replace=False)
                U_init[idx_u, r] = np.random.choice([-1, 1], size=n_u)
                
                n_v = np.random.randint(2, 5)
                idx_v = np.random.choice(mp, size=n_v, replace=False)
                V_init[idx_v, r] = np.random.choice([-1, 1], size=n_v)
                
                n_w = np.random.randint(2, 5)
                idx_w = np.random.choice(pn, size=n_w, replace=False)
                W_init[idx_w, r] = np.random.choice([-1, 1], size=n_w)
            
            U_opt, V_opt, W_opt, err = try_optimize(U_init, V_init, W_init, max_iter=1500)
            
            if err < 1e-6:
                # Try rounding
                U_int = np.round(U_opt).astype(int)
                V_int = np.round(V_opt).astype(int)
                W_int = np.round(W_opt).astype(int)
                
                if verify(U_int, V_int, W_int, tol=1e-9):
                    best_U, best_V, best_W = U_int, V_int, W_int
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    # Final verification
    if not verify(best_U, best_V, best_W, tol=1e-9):
        best_U, best_V, best_W = U25, V25, W25
        best_rank = 25
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
