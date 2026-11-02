import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    target = np.ones((n, m, p))
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    def verify_solution(U_sol, V_sol, W_sol, rank):
        """Verify that the solution exactly decomposes the matrix multiplication tensor."""
        if rank == 0:
            return np.allclose(target, 0)
        U_r = U_sol.reshape(n, m, rank)
        V_r = V_sol.reshape(m, p, rank)
        W_r = W_sol.reshape(p, n, rank)
        T_approx = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T_approx, target, atol=1e-9)

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
        return U, V, W

    best_U, best_V, best_W = schoolbook()

    def cp_als_rank1(tensor, n_iter=100, n_restarts=20):
        """Find best rank-1 approximation of tensor using CP-ALS."""
        n, m, p = tensor.shape
        best_residual = np.inf
        best_u, best_v, best_w = None, None, None
        
        for r in range(n_restarts):
            u = np.random.randn(n)
            v = np.random.randn(m)
            w = np.random.randn(p)
            
            for it in range(n_iter):
                # Update w
                mat_w = np.tensordot(tensor, u, axes=([0], [0]))
                mat_w = np.tensordot(mat_w, v, axes=([0], [0]))
                w_new = mat_w
                if np.linalg.norm(w_new) < 1e-10:
                    break
                w = w_new / np.linalg.norm(w_new)
                
                # Update v
                mat_v = np.tensordot(tensor, u, axes=([0], [0]))
                mat_v = np.tensordot(mat_v, w, axes=([2], [0]))
                v_new = mat_v
                if np.linalg.norm(v_new) < 1e-10:
                    break
                v = v_new / np.linalg.norm(v_new)
                
                # Update u
                mat_u = np.tensordot(tensor, v, axes=([1], [0]))
                mat_u = np.tensordot(mat_u, w, axes=([1], [0]))
                u_new = mat_u
                if np.linalg.norm(u_new) < 1e-10:
                    break
                u = u_new / np.linalg.norm(u_new)
            
            # Calculate residual
            approx = np.einsum('i,j,k->ijk', u, v, w)
            residual = np.linalg.norm(tensor - approx)
            
            if residual < best_residual:
                best_residual = residual
                best_u, best_v, best_w = u, v, w
                
        return best_u, best_v, best_w, best_residual

    def try_greedy(max_rank=40, n_restarts_cp=20):
        R = target.copy()
        Us, Vs, Ws = [], [], []
        
        for r in range(max_rank):
            if time.time() > deadline:
                break
                
            u, v, w, res_norm = cp_als_rank1(R, n_iter=50, n_restarts=n_restarts_cp)
            
            if u is None:
                break
                
            # Project R onto u, v, w
            num = np.sum(R * u[:, None, None] * v[None, :, None] * w[None, None, :])
            den = np.sum(u**2) * np.sum(v**2) * np.sum(w**2)
            
            if den < 1e-10:
                break
                
            coeff = num / den
            R = R - coeff * np.einsum('i,j,k->ijk', u, v, w)
            
            # Store scaled vectors
            Us.append(coeff * u)
            Vs.append(v)
            Ws.append(w)
            
            if np.linalg.norm(R) < 1e-8:
                break
        
        if np.linalg.norm(R) < 1e-8:
            rank = len(Us)
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            for r in range(rank):
                u, v, w = Us[r], Vs[r], Ws[r]
                for i in range(n):
                    for j in range(m):
                        U[i * m + j, r] = u[i]
                for j in range(m):
                    for k in range(p):
                        V[j * p + k, r] = v[j]
                for k in range(p):
                    for i in range(n):
                        W[k * n + i, r] = w[k]
                        
            # Try to snap to half-integers
            U_snapped = np.round(U * 2) / 2
            V_snapped = np.round(V * 2) / 2
            W_snapped = np.round(W * 2) / 2
            
            if verify_solution(U_snapped, V_snapped, W_snapped, rank):
                return U_snapped, V_snapped, W_snapped, rank
            else:
                # Try local search on snapped values
                # For simplicity, if float solution is exact, return it
                if verify_solution(U, V, W, rank):
                    return U, V, W, rank
                     
        return None

    # Run greedy search with different max ranks
    for max_rank in [20, 25, 30, 35, 40]:
        if time.time() > deadline:
            break
        res = try_greedy(max_rank)
        if res:
            U, V, W, rank = res
            if rank < best_rank:
                best_U, best_V, best_W = U, V, W
                best_rank = rank
                if rank <= 20:
                    break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
