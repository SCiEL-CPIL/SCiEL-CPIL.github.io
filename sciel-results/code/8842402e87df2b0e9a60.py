import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target multiplication tensor T[u, v, w]
    # u = i*m + j, v = j*p + k, w = k*n + i
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def compute_error(U, V, W):
        Rec = np.einsum('ur,vr,wr->uvw', U, V, W)
        diff = Rec - T
        return np.linalg.norm(diff)

    def compute_grads(U, V, W):
        Rec = np.einsum('ur,vr,wr->uvw', U, V, W)
        diff = Rec - T
        
        # Gradients:
        # dE/dU[u,r] = 2 * sum_v,w diff[u,v,w] * V[v,r] * W[w,r]
        # dE/dV[v,r] = 2 * sum_u,w diff[u,v,w] * U[u,r] * W[w,r]
        # dE/dW[w,r] = 2 * sum_u,v diff[u,v,w] * U[u,r] * V[v,r]
        
        # Using einsum for efficiency
        # grad_U[u, r] = 2 * sum_v,w diff[u, v, w] V[v, r] W[w, r]
        # We can compute this as:
        # temp1 = einsum('uvw, vr -> uw r', diff, V) -> shape (nm, pn, R)
        # grad_U = 2 * einsum('uwr, wr -> ur', temp1, W)
        
        # Or more directly:
        # grad_U = 2 * einsum('uvw, vr, wr -> ur', diff, V, W)
        
        gU = 2.0 * np.einsum('uvw,vr,wr->ur', diff, V, W)
        gV = 2.0 * np.einsum('uvw,ur,wr->vr', diff, U, W)
        gW = 2.0 * np.einsum('uvw,ur,vr->wr', diff, U, V)
        
        return gU, gV, gW

    def optimize_gd(U, V, W, max_iter=5000, lr=0.01, momentum=0.9, deadline=0):
        R = U.shape[1]
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            gU, gV, gW = compute_grads(U, V, W)
            
            vU = momentum * vU + (1 - momentum) * gU
            vV = momentum * vV + (1 - momentum) * gV
            vW = momentum * vW + (1 - momentum) * gW
            
            U -= lr * vU
            V -= lr * vV
            W -= lr * vW
            
            err = compute_error(U, V, W)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if err < 1e-8:
                break
                
        return best_U, best_V, best_W, best_err

    deadline = time.time() + 110

    best_solution = None
    best_rank = n * m * p

    # Try Rank 44
    R_target = 44
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialization: Random normal, scaled
        scale = 1.0
        U = np.random.randn(nm, R_target) * scale
        V = np.random.randn(mp, R_target) * scale
        W = np.random.randn(pn, R_target) * scale
        
        U, V, W, err = optimize_gd(U, V, W, max_iter=2000, lr=0.05, momentum=0.9, deadline=deadline)
        
        if err < 1e-5:
            best_solution = (U, V, W)
            best_rank = R_target
            break

    if best_solution is None:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1.0
                    V[j * p + k, column] = 1.0
                    W[k * n + i, column] = 1.0
                    column += 1
        best_solution = (U, V, W)
        best_rank = rank

    U, V, W = best_solution
    
    def to_list(matrix):
        L = []
        for row in matrix:
            r = []
            for val in row:
                nearest = round(val * 2) / 2.0
                if abs(val - nearest) < 1e-6:
                    r.append(nearest)
                else:
                    r.append(float(val))
            L.append(r)
        return L

    return {
        "U": to_list(U),
        "V": to_list(V),
        "W": to_list(W)
    }
