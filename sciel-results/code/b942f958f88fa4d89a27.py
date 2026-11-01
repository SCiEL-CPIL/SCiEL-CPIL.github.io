import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 145

def construct():
    start_time = time.time()
    n, m, p = 2, 3, 10
    
    def build_rank_55():
        rank = 55
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        a00, a01, a10, a11, a02, a12 = 0, 1, 3, 4, 2, 5
        
        for b in range(5):
            k0, k1 = 2*b, 2*b+1
            b00, b01 = k0, k1
            b10, b11 = 10+k0, 10+k1
            c00, c01 = 2*k0, 2*k1
            c10, c11 = 2*k0+1, 2*k1+1
            
            U[a00,col]+=1; U[a11,col]+=1; V[b00,col]+=1; V[b11,col]+=1; W[c00,col]+=1; W[c11,col]+=1; col+=1
            U[a10,col]+=1; U[a11,col]+=1; V[b00,col]+=1; W[c10,col]+=1; W[c11,col]-=1; col+=1
            U[a00,col]+=1; V[b01,col]+=1; V[b11,col]-=1; W[c01,col]+=1; W[c11,col]+=1; col+=1
            U[a11,col]+=1; V[b10,col]+=1; V[b00,col]-=1; W[c00,col]+=1; W[c10,col]+=1; col+=1
            U[a00,col]+=1; U[a01,col]+=1; V[b11,col]+=1; W[c00,col]-=1; W[c01,col]+=1; col+=1
            U[a10,col]+=1; U[a00,col]-=1; V[b00,col]+=1; V[b01,col]+=1; W[c11,col]+=1; col+=1
            U[a01,col]+=1; U[a11,col]-=1; V[b10,col]+=1; V[b11,col]+=1; W[c00,col]+=1; col+=1
        
        for k in range(p):
            b2k = 2*p+k
            U[a02,col]+=1; V[b2k,col]+=1; W[2*k,col]+=1; col+=1
            U[a12,col]+=1; V[b2k,col]+=1; W[2*k+1,col]+=1; col+=1
        
        return U, V, W
    
    U55, V55, W55 = build_rank_55()
    best_U, best_V, best_W = U55, V55, W55
    
    def verify(U, V, W, tol=1e-10):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i*m+j
                    idx_v = j*p+k
                    idx_w = k*n+i
                    val = np.dot(U[idx_u], V[idx_v] * W[idx_w])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find rank-54 by merging pairs
    def try_merge(U, V, W, i, j):
        if time.time() - start_time > DEADLINE - 5:
            return None
        
        mask = np.ones(55, dtype=bool)
        mask[i] = False
        mask[j] = False
        U_rem = U[:, mask]
        V_rem = V[:, mask]
        W_rem = W[:, mask]
        
        u1, u2 = U[:, i], U[:, j]
        v1, v2 = V[:, i], V[:, j]
        w1, w2 = W[:, i], W[:, j]
        
        # Try simple combinations
        candidates = [
            ((u1+u2)/2, (v1+v2)/2, (w1+w2)/2),
            (u1+u2, v1, w1),
            (u1, v1+v2, w1),
            (u1, v1, w1+w2),
            (u1-u2, v1, w1),
            (u1, v1-v2, w1),
            (u1, v1, w1-w2),
        ]
        
        for u_c, v_c, w_c in candidates:
            U_test = np.column_stack([U_rem, u_c])
            V_test = np.column_stack([V_rem, v_c])
            W_test = np.column_stack([W_rem, w_c])
            if verify(U_test, V_test, W_test):
                return U_test, V_test, W_test
        
        return None
    
    found = False
    for i in range(55):
        for j in range(i+1, 55):
            if time.time() - start_time > DEADLINE - 5:
                found = True
                break
            result = try_merge(U55, V55, W55, i, j)
            if result is not None:
                best_U, best_V, best_W = result
                found = True
                break
        if found:
            break
    
    if time.time() - start_time > DEADLINE:
        return {"U": U55.tolist(), "V": V55.tolist(), "W": W55.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
