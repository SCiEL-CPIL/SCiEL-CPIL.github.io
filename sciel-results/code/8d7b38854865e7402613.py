import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 12
    np_ = n * p  # 8
    
    deadline = time.time() + 100
    
    # Precompute the index mappings
    valid_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_triples.append((a, b, c))
    
    # Invalid triples (where tensor entry should be 0)
    all_triples = set()
    for a in range(nm):
        for b in range(mp):
            for c in range(np_):
                all_triples.add((a, b, c))
    
    valid_set = set(valid_triples)
    invalid_triples = [t for t in all_triples if t not in valid_set]
    
    best_result = None
    best_rank = None
    
    for rank in range(12, 24):
        if time.time() > deadline - 20:
            break
        
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            
            U0 = np.random.randn(nm, rank)
            V0 = np.random.randn(mp, rank)
            W0 = np.random.randn(np_, rank)
            
            def residuals(params):
                U = params[:nm*rank].reshape(nm, rank)
                V = params[nm*rank:2*nm*rank].reshape(mp, rank)
                W = params[2*nm*rank:].reshape(np_, rank)
                
                res = []
                for (a, b, c) in valid_triples:
                    val = np.dot(U[a], V[b] * W[c])
                    res.append(val - 1.0)
                
                # Sample some invalid triples for regularization
                if len(invalid_triples) > 100:
                    sample_idx = np.random.choice(len(invalid_triples), 100, replace=False)
                    for idx in sample_idx:
                        a, b, c = invalid_triples[idx]
                        val = np.dot(U[a], V[b] * W[c])
                        res.append(val)
                
                return np.array(res)
            
            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                result = least_squares(residuals, params0, method='lm', max_nfev=5000)
            except:
                continue
            
            if result.cost < 1e-10:
                U_sol = result.x[:nm*rank].reshape(nm, rank)
                V_sol = result.x[nm*rank:2*nm*rank].reshape(mp, rank)
                W_sol = result.x[2*nm*rank:].reshape(np_, rank)
                
                # Check all valid triples
                max_err = 0
                for (a, b, c) in valid_triples:
                    val = np.dot(U_sol[a], V_sol[b] * W_sol[c])
                    max_err = max(max_err, abs(val - 1.0))
                
                # Check all invalid triples
                for (a, b, c) in invalid_triples:
                    val = np.dot(U_sol[a], V_sol[b] * W_sol[c])
                    max_err = max(max_err, abs(val))
                
                if max_err < 1e-8:
                    # Try to round to half-integers
                    U_r = np.round(U_sol * 2) / 2
                    V_r = np.round(V_sol * 2) / 2
                    W_r = np.round(W_sol * 2) / 2
                    
                    max_err_r = 0
                    for (a, b, c) in valid_triples:
                        val = np.dot(U_r[a], V_r[b] * W_r[c])
                        max_err_r = max(max_err_r, abs(val - 1.0))
                    for (a, b, c) in invalid_triples:
                        val = np.dot(U_r[a], V_r[b] * W_r[c])
                        max_err_r = max(max_err_r, abs(val))
                    
                    if max_err_r < 1e-6:
                        best_rank = rank
                        best_result = (U_r.tolist(), V_r.tolist(), W_r.tolist())
                        break
        
        if best_result is not None:
            break
    
    if best_result is None:
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(np_)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
