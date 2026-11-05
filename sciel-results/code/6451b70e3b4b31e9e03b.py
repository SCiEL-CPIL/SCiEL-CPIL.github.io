import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 6]

def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 3x4 matrix by a 4x6 matrix.
    
    The decomposition satisfies:
        A * B = sum_{t=0}^{rank-1} W[:,t] * outer(U[:,t], V[:,t])
    where:
        U is (n*m, rank) = (12, rank)
        V is (m*p, rank) = (24, rank)
        W is (p*n, rank) = (18, rank)
    """
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72
    NM, MP, PN = n * m, m * p, p * n  # 12, 24, 18
    
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin before 150s limit
    
    # Construct the target flattened tensor T_flat
    # T_flat[u, v, w] = 1 if u=(i,j), v=(j,k), w=(i,k) are consistent
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    T_flat = np.zeros((NM, MP, PN), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_flat[u, v, w] = 1.0

    def check_exact(U, V, W, rank):
        """Check if U, V, W with given rank exactly reconstruct T_flat"""
        # T_rec[u, v, w] = sum_t U[u, t] V[v, t] W[w, t]
        # Use einsum for speed
        T_rec = np.einsum('ut,vt,wt->uvw', U, V, W)
        return np.allclose(T_rec, T_flat, atol=1e-9)

    def build_schoolbook(rank=None):
        """Build the schoolbook decomposition"""
        if rank is None:
            rank = rank_schoolbook
        U = np.zeros((NM, rank))
        V = np.zeros((MP, rank))
        W = np.zeros((PN, rank))
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if column < rank:
                        u = i * m + j
                        v = j * p + k
                        w = k * n + i
                        U[u, column] = 1
                        V[v, column] = 1
                        W[w, column] = 1
                    column += 1
        
        return U, V, W, rank

    def als_decompose(rank, max_iters=200, seed=0, init=None):
        """Alternating Least Squares for the flattened tensor"""
        np.random.seed(seed)
        
        if init is not None:
            U, V, W = init
        else:
            # Random initialization
            U = np.random.randn(NM, rank)
            V = np.random.randn(MP, rank)
            W = np.random.randn(PN, rank)
            
        prev_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = np.inf
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Compute current error
            T_rec = np.einsum('ut,vt,wt->uvw', U, V, W)
            err = T_rec - T_flat
            current_err = np.sum(err**2)
            
            if current_err < best_err:
                best_err = current_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if abs(prev_err - current_err) < 1e-16:
                break
            if current_err < 1e-12:
                break
            prev_err = current_err
            
            # Solve for U: U[:, t] = argmin ||T_flat - U V W||
            # Flatten T_flat to (NM, MP*PN)
            # This is expensive. Use iterative updates.
            
            # Update U
            # For each t, U[:, t] is coupled.
            # Better: Fix V, W. Update U.
            # T_flat[u, v, w] = sum_t U[u, t] V[v, t] W[w, t]
            # Let M_u = sum_t U[u, t] ...
            
            # Efficient ALS for 3-tensor:
            # U_new = T_flat * (V W^T)^+ ? No.
            
            # We can use the fact that:
            # U[:, t] = argmin || T_flat - sum_t' U[:, t'] V[t', :] W[t', :] ||
            
            # Simpler: Use the matricized form.
            # T_(1) is NM x (MP*PN).
            # T_(1)[u, v*w] = T_flat[u, v, w].
            # T_(1) = U * (V (Kronecker) W)^T ?
            # Actually, T_flat = sum_t U[:, t] outer(V[:, t], W[:, t]).T?
            # T_flat[u, v, w] = sum_t U[u, t] V[v, t] W[w, t].
            # This is a sum of outer products of U[:, t] and (V[:, t] * W[:, t])? No.
            # It's a sum of rank-1 tensors in the 3D space.
            
            # Update U:
            # Fix V, W.
            # For each u, T_flat[u, :, :] = sum_t U[u, t] outer(V[:, t], W[:, t]).
            # This is a linear system for U[u, :].
            # Let A_u be a matrix of size (MP*PN) x rank.
            # A_u[v*w, t] = V[v, t] * W[w, t].
            # Then T_flat[u, :].flatten() = A_u @ U[u, :].
            
            # Construct A_u for all u?
            # A is (NM, MP*PN, rank).
            # A[u, v*w, t] = V[v, t] * W[w, t].
            
            # This might be memory heavy but NM=12 is small.
            # MP*PN = 24*18 = 432.
            # A size: 12 * 432 * rank.
            
            # We can compute U by solving least squares for each u.
            for u in range(NM):
                # Construct A_u
                # A_u is (MP*PN, rank)
                # A_u[idx_vw, t] = V[v, t] * W[w, t]
                # idx_vw = v * PN + w
                A_u = np.zeros((MP * PN, rank))
                for v in range(MP):
                    for w in range(PN):
                        idx = v * PN + w
                        A_u[idx, :] = V[v, :] * W[w, :]
                
                b_u = T_flat[u, :, :].flatten()
                try:
                    u_sol, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                    U[u, :] = u_sol
                except:
                    pass
            
            # Update V
            # Fix U, W.
            # T_flat[u, v, w] = sum_t U[u, t] V[v, t] W[w, t].
            # For each v, T_flat[:, v, :].flatten() = A_v @ V[v, :].
            # A_v[u*w, t] = U[u, t] * W[w, t].
            for v in range(MP):
                A_v = np.zeros((NM * PN, rank))
                for u in range(NM):
                    for w in range(PN):
                        idx = u * PN + w
                        A_v[idx, :] = U[u, :] * W[w, :]
                
                b_v = T_flat[:, v, :].flatten()
                try:
                    v_sol, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                    V[v, :] = v_sol
                except:
                    pass
            
            # Update W
            # Fix U, V.
            # For each w, T_flat[:, :, w].flatten() = A_w @ W[w, :].
            # A_w[u*v, t] = U[u, t] * V[v, t].
            for w in range(PN):
                A_w = np.zeros((NM * MP, rank))
                for u in range(NM):
                    for v in range(MP):
                        idx = u * MP + v
                        A_w[idx, :] = U[u, :] * V[v, :]
                
                b_w = T_flat[:, :, w].flatten()
                try:
                    w_sol, _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
                    W[w, :] = w_sol
                except:
                    pass
        
        U, V, W = best_U, best_V, best_W
        
        # Final check
        T_rec = np.einsum('ut,vt,wt->uvw', U, V, W)
        final_err = np.sum((T_rec - T_flat)**2)
        
        if final_err < 1e-6:
            # Snap to integers/half-integers if possible
            U_s = np.round(U * 2) / 2.0
            V_s = np.round(V * 2) / 2.0
            W_s = np.round(W * 2) / 2.0
            if check_exact(U_s, V_s, W_s, rank):
                return U_s, V_s, W_s
            
            return U, V, W
        
        return None

    best_result = None
    best_rank = rank_schoolbook
    
    # Try Rank 66 first, as it is a known bound
    candidate_ranks = [66, 64, 60, 56, 52, 48, 44, 40, 36]
    
    for rank in candidate_ranks:
        if time.time() > deadline - 20:
            break
        
        # Try multiple seeds for ALS
        found = False
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            result = als_decompose(rank, max_iters=100, seed=seed)
            if result is not None:
                U, V, W = result
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                found = True
                break
        
        if found:
            break

    # Fallback to schoolbook
    if best_result is None:
        U, V, W, rank = build_schoolbook()
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return best_result
