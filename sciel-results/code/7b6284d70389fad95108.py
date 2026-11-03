import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    rank_schoolbook = n * m * p
    
    # Schoolbook decomposition
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    # Target rank
    target_rank = 28
    
    # Build Tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def solve_for_U(V, W, R):
        A = np.zeros((30, nm*R))
        b = np.ones(30)
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    for r in range(R):
                        A[eq_idx, a*R + r] = V[b_idx, r] * W[c_idx, r]
                    eq_idx += 1
        U, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        return U.reshape(nm, R)

    def solve_for_V(U, W, R):
        A = np.zeros((30, mp*R))
        b = np.ones(30)
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    for r in range(R):
                        A[eq_idx, b_idx*R + r] = U[a, r] * W[c_idx, r]
                    eq_idx += 1
        V, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        return V.reshape(mp, R)

    def solve_for_W(U, V, R):
        A = np.zeros((30, pn*R))
        b = np.ones(30)
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    for r in range(R):
                        A[eq_idx, c_idx*R + r] = U[a, r] * V[b_idx, r]
                    eq_idx += 1
        W, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        return W.reshape(pn, R)

    def check_decomposition(U, V, W, R):
        Rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.sum((Rec - T)**2)
        if err > 1e-8:
            return False, err
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        if np.allclose(U2, np.round(U2), atol=1e-6) and \
           np.allclose(V2, np.round(V2), atol=1e-6) and \
           np.allclose(W2, np.round(W2), atol=1e-6):
            U_out = [[float(x/2) for x in row] for row in U2]
            V_out = [[float(x/2) for x in row] for row in V2]
            W_out = [[float(x/2) for x in row] for row in W2]
            return True, err
        return False, err

    start_time = time.time()
    deadline = start_time + 140.0
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    R = target_rank
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        np.random.seed(42 + restart)
        U = np.random.randn(nm, R)
        V = np.random.randn(mp, R)
        W = np.random.randn(pn, R)
        
        for r in range(R):
            U[:, r] /= np.linalg.norm(U[:, r])
            V[:, r] /= np.linalg.norm(V[:, r])
            W[:, r] /= np.linalg.norm(W[:, r])
            
        for step in range(50):
            if time.time() > deadline:
                break
                
            U = solve_for_U(V, W, R)
            V = solve_for_V(U, W, R)
            W = solve_for_W(U, V, R)
            
            Rec = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.sum((Rec - T)**2)
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            if err < 1e-10:
                break
                
        if best_err < 1e-6:
            valid, err = check_decomposition(best_U, best_V, best_W, R)
            if valid:
                return {"U": best_U, "V": best_V, "W": best_W}

    return {"U": U_sb, "V": V_sb, "W": W_sb}
