import numpy as np
import time

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor T is sparse
    # T[u, v, w] = 1 if u = i*m+j, v = j*p+k, w = k*n+i
    # Else 0
    
    def get_target_tensor():
        T = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    T[u, v, w] = 1.0
        return T

    T = get_target_tensor()
    
    best_rank = nm * mp # Schoolbook
    best_solution = None
    
    # Try ranks from 17 upwards (17 is the target)
    # We know 20 works. 17 is the goal.
    # We will try to find a solution for rank 17.
    
    deadline = time.time() + 110.0 # Leave 10s margin
    
    def check_and_round(U, V, W, rank):
        # Check if current U, V, W are close to integer/half-integer
        # and if they exactly decompose T
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify exactness
        # Reconstruct tensor
        # T_recon[u, v, w] = sum_c U_r[u, c] * V_r[v, c] * W_r[w, c]
        # We can do this efficiently by iterating c
        T_recon = np.zeros((nm, mp, pn))
        for c in range(rank):
            T_recon += np.einsum('u,v,w->uvw', U_r[:, c], V_r[:, c], W_r[:, c])
        
        # Check if T_recon matches T
        if np.allclose(T_recon, T, atol=1e-9):
            return U_r, V_r, W_r
        return None

    def run_als(rank, max_iters=500):
        # Initialize
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            # For each u, solve A_u U_u = T_u
            # A_u[v * pn + w, c] = V[v, c] * W[w, c]
            # T_u[v * pn + w] = T[u, v, w]
            for u in range(nm):
                # Construct A_u
                A_u = np.zeros((mp * pn, rank))
                for v in range(mp):
                    for w in range(pn):
                        idx = v * pn + w
                        A_u[idx, :] = V[v, :] * W[w, :]
                b_u = T[u, :, :].flatten()
                # Solve least squares
                U[u, :], residuals, rank_u, sv = np.linalg.lstsq(A_u, b_u, rcond=None)
            
            # Update V
            for v in range(mp):
                A_v = np.zeros((nm * pn, rank))
                for u in range(nm):
                    for w in range(pn):
                        idx = u * pn + w
                        A_v[idx, :] = U[u, :] * W[w, :]
                b_v = T[:, v, :].flatten()
                V[v, :], residuals, rank_v, sv = np.linalg.lstsq(A_v, b_v, rcond=None)
                
            # Update W
            for w in range(pn):
                A_w = np.zeros((nm * mp, rank))
                for u in range(nm):
                    for v in range(mp):
                        idx = u * mp + v
                        A_w[idx, :] = U[u, :] * V[v, :]
                b_w = T[:, :, w].flatten()
                W[w, :], residuals, rank_w, sv = np.linalg.lstsq(A_w, b_w, rcond=None)
            
            # Check convergence
            # Compute error
            T_recon = np.zeros((nm, mp, pn))
            for c in range(rank):
                T_recon += np.einsum('u,v,w->uvw', U[:, c], V[:, c], W[:, c])
            err = np.linalg.norm(T_recon - T)
            if err < 1e-6:
                res = check_and_round(U, V, W, rank)
                if res:
                    return res
        
        return None

    # Try rank 17 with multiple restarts
    np.random.seed(42)
    for attempt in range(20):
        if time.time() > deadline:
            break
        res = run_als(17, max_iters=200)
        if res:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook rank 20
    rank = nm * mp
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
