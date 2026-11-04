import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 16
    dim_v = m * p  # 28
    dim_w = p * n  # 28
    
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    deadline = time.time() + 130.0
    schoolbook_rank = n * m * p
    
    def _schoolbook():
        U = [[0] * schoolbook_rank for _ in range(dim_u)]
        V = [[0] * schoolbook_rank for _ in range(dim_v)]
        W = [[0] * schoolbook_rank for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    for seed in range(500):
        if time.time() > deadline:
            break
        
        np.random.seed(seed)
        R = T.copy().astype(np.float64)
        U_list = []
        V_list = []
        W_list = []
        
        success = True
        for step in range(schoolbook_rank):
            if time.time() > deadline:
                success = False
                break
            
            if not np.any(R):
                break
            
            best_score = -1e18
            best_u = best_v = best_w = None
            
            for restart in range(20):
                if time.time() > deadline:
                    break
                
                u = np.random.choice([-1, 0, 1], size=dim_u)
                v = np.random.choice([-1, 0, 1], size=dim_v)
                w = np.random.choice([-1, 0, 1], size=dim_w)
                
                for _iter in range(30):
                    M_u = np.einsum('abc,b,c->a', R, v, w)
                    new_u = np.sign(M_u).astype(np.float64)
                    M_v = np.einsum('abc,a,c->b', R, u, w)
                    new_v = np.sign(M_v).astype(np.float64)
                    M_w = np.einsum('abc,a,b->c', R, u, v)
                    new_w = np.sign(M_w).astype(np.float64)
                    
                    if np.array_equal(new_u, u) and np.array_equal(new_v, v) and np.array_equal(new_w, w):
                        break
                    u, v, w = new_u, new_v, new_w
                
                score = np.einsum('abc,a,b,c', R, u, v, w)
                if score > best_score:
                    best_score = score
                    best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
            
            if best_score <= 0:
                success = False
                break
            
            R -= np.einsum('a,b,c->abc', best_u, best_v, best_w)
            U_list.append(best_u)
            V_list.append(best_v)
            W_list.append(best_w)
        
        if success and np.allclose(R, 0, atol=1e-6):
            rank = len(U_list)
            if rank < best_rank:
                best_rank = rank
                best_U = [x.tolist() for x in np.array(U_list).T]
                best_V = [x.tolist() for x in np.array(V_list).T]
                best_W = [x.tolist() for x in np.array(W_list).T]
                if rank < schoolbook_rank:
                    return {"U": best_U, "V": best_V, "W": best_W}
    
    if best_U is not None and best_rank < schoolbook_rank:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return _schoolbook()
# EVOLVE-BLOCK-END
