import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _make_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_rec = np.einsum('ic,jc,kc->ijk', U, V, W)
    if not np.allclose(T_rec, T, atol=tol):
        return False
    return True

def _als_decompose(T, rank, n_iter=100, tol=1e-10, seed=123):
    n1, n2, n3 = T.shape
    np.random.seed(seed)
    
    U = np.random.randn(n1, rank)
    V = np.random.randn(n2, rank)
    W = np.random.randn(n3, rank)
    
    for it in range(n_iter):
        if it % 3 == 0:
            K = np.einsum('jc,kc->jkc', V, W).reshape(n2*n3, rank)
            T_mat = T.reshape(n1, n2*n3)
            Kt = K.T
            U = T_mat @ np.linalg.pinv(Kt)
        elif it % 3 == 1:
            UW = np.einsum('ic,kc->ikc', U, W).reshape(n1*n3, rank)
            T_mat = T.transpose(1,0,2).reshape(n2, n1*n3)
            Ut = UW.T
            V = T_mat @ np.linalg.pinv(Ut)
        else:
            UV = np.einsum('ic,jc->ijc', U, V).reshape(n1*n2, rank)
            T_mat = T.transpose(2,0,1).reshape(n3, n1*n2)
            Ut = UV.T
            W = T_mat @ np.linalg.pinv(Ut)
        
        if it > 0:
            T_rec = np.einsum('ic,jc,kc->ijk', U, V, W)
            err = np.linalg.norm(T_rec - T)
            if err < tol:
                break
    
    return U, V, W

def _round_to_rational(mat, denom=2):
    # Scale, round, scale back
    scaled = mat * denom
    rounded = np.round(scaled)
    return rounded / denom

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    T = _make_tensor(n, m, p)
    
    # Try ranks from 85 down to 80
    best_result = None
    
    for target_rank in range(85, 79, -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        for seed in range(20):
            if time.time() > deadline - 20:
                break
            
            U, V, W = _als_decompose(T, target_rank, n_iter=200, tol=1e-10, seed=seed)
            
            # Try rounding to 1/2 and 1/4
            for denom in [2, 4]:
                U_r = _round_to_rational(U, denom)
                V_r = _round_to_rational(V, denom)
                W_r = _round_to_rational(W, denom)
                
                if _verify_decomposition(U_r, V_r, W_r, T):
                    best_result = {'U': U_r.tolist(), 'V': V_r.tolist(), 'W': W_r.tolist()}
                    found = True
                    break
            
            if found:
                break
        
        if found:
            break
    
    if best_result:
        return best_result
    
    # Fallback to schoolbook (Rank 100)
    rank_sb = n*m*p
    U_fb = np.zeros((n*m, rank_sb))
    V_fb = np.zeros((m*p, rank_sb))
    W_fb = np.zeros((p*n, rank_sb))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, c] = 1
                V_fb[j*p+k, c] = 1
                W_fb[k*n+i, c] = 1
                c += 1
    return {'U': U_fb.tolist(), 'V': V_fb.tolist(), 'W': W_fb.tolist()}
