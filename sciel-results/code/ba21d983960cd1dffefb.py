"""Matrix multiplication rank decomposition for 2x5 x 5x6."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    
    # Try to find a lower rank decomposition using a structured approach
    # For 2x5x6, attempt rank reduction by finding linear dependencies
    
    # The tensor equation: sum_c U[i*m+j,c]*V[j'*p+k,c]*W[k*n+i,c] = delta_{j,j'}
    # Try to solve for lower r using random search with restarts
    
    def tensor_residual(U, V, W):
        r = U.shape[1]
        max_res = 0.0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(r):
                            val += U[i*m+j, c] * V[jp*p+k, c] * W[k*n+i, c]
                        expected = 1.0 if j == jp else 0.0
                        res = abs(val - expected)
                        if res > max_res:
                            max_res = res
        return max_res
    
    def search_lower_rank(r, n_restarts=3, max_iters=100):
        if time.time() > deadline - 10:
            return None
        rng = np.random.default_rng(r)
        best_U = best_V = best_W = None
        best_res = np.inf
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Initialize from schoolbook projection
            U0, V0, W0 = best["U"], best["V"], best["W"]
            assign = rng.integers(0, r, size=schoolbook_rank)
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            for c0 in range(schoolbook_rank):
                c1 = assign[c0]
                U[:, c1] += U0[:, c0]
                V[:, c1] += V0[:, c0]
                W[:, c1] += W0[:, c0]
            
            res = tensor_residual(U, V, W)
            if res < best_res:
                best_res = res
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if res < 1e-9:
                    break
            
            # Simple local search
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                lr = 0.05
                # Perturb U
                idx = rng.choice(n*m * r)
                old = U[idx // r, idx % r]
                U[idx // r, idx % r] += rng.standard_normal() * lr
                new_res = tensor_residual(U, V, W)
                if new_res < res:
                    res = new_res
                    if res < best_res:
                        best_res = res
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        if res < 1e-9:
                            return best_U, best_V, best_W
                else:
                    U[idx // r, idx % r] = old
                
                # Perturb V
                idx = rng.choice(m*p * r)
                old = V[idx // r, idx % r]
                V[idx // r, idx % r] += rng.standard_normal() * lr
                new_res = tensor_residual(U, V, W)
                if new_res < res:
                    res = new_res
                    if res < best_res:
                        best_res = res
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        if res < 1e-9:
                            return best_U, best_V, best_W
                else:
                    V[idx // r, idx % r] = old
                
                # Perturb W
                idx = rng.choice(p*n * r)
                old = W[idx // r, idx % r]
                W[idx // r, idx % r] += rng.standard_normal() * lr
                new_res = tensor_residual(U, V, W)
                if new_res < res:
                    res = new_res
                    if res < best_res:
                        best_res = res
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        if res < 1e-9:
                            return best_U, best_V, best_W
                else:
                    W[idx // r, idx % r] = old
        
        if best_res < 1e-8 and best_U is not None:
            return best_U, best_V, best_W
        return None
    
    # Try a few ranks
    for r in [58, 59]:
        if time.time() > deadline - 15:
            break
        result = search_lower_rank(r)
        if result is not None:
            U, V, W = result
            # Verify with actual matrix multiplication
            rng = np.random.default_rng(42)
            ok = True
            for _ in range(3):
                A = rng.standard_normal((n, m))
                B = rng.standard_normal((m, p))
                C_exp = A @ B
                C_dec = np.zeros((n, p))
                for c in range(r):
                    f = np.array([sum(U[i*m+j, c]*A[i,j] for j in range(m)) for i in range(n)])
                    g = np.array([sum(V[j*p+k, c]*B[j,k] for j in range(m)) for k in range(p)])
                    for i in range(n):
                        for k in range(p):
                            C_dec[i,k] += f[i]*g[k]*W[k*n+i, c]
                if not np.allclose(C_dec, C_exp, atol=1e-8):
                    ok = False
                    break
            if ok:
                return {"U": U, "V": V, "W": W}
    
    return best
# EVOLVE-BLOCK-END
