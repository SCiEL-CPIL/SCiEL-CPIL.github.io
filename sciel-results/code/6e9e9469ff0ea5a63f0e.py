import numpy as np
import time

def construct():
    n, m, p = 2, 3, 3
    U_size, V_size, W_size = n * m, m * p, p * n  # 6, 9, 6
    
    # Build target tensor
    T = np.zeros((U_size, V_size, W_size))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    deadline = time.time() + 140.0  # safety margin for 150s limit
    
    def verify(U, V, W):
        rank = U.shape[1]
        T_calc = np.einsum('as,bs,cs->abc', U, V, W)
        return np.allclose(T_calc, T, atol=1e-9)
    
    def snap_and_verify(U, V, W):
        U_snap = np.round(U * 2) / 2.0
        V_snap = np.round(V * 2) / 2.0
        W_snap = np.round(W * 2) / 2.0
        if verify(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap
        return None
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((U_size, rank))
        V = np.zeros((V_size, rank))
        W = np.zeros((W_size, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def als_decompose(rank, U0=None, V0=None, W0=None, max_iter=200, tol=1e-14):
        """Alternating least squares with optional initial values."""
        if U0 is not None:
            U, V, W = U0.copy(), V0.copy(), W0.copy()
        else:
            U = np.random.randn(U_size, rank) * 0.5
            V = np.random.randn(V_size, rank) * 0.5
            W = np.random.randn(W_size, rank) * 0.5
        
        prev_cost = np.inf
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Update U: for each row i, solve (V⊗W)^T * U_i = T_i
            # T_i is T[i,:,:].flatten(), (V⊗W) has shape (V_size*W_size, rank)
            VW = np.einsum('jc,kc->jkc', V, W).reshape(V_size * W_size, rank)
            T_flat = T.reshape(U_size, V_size * W_size)
            try:
                U_new = np.linalg.lstsq(VW.T, T_flat.T, rcond=None)[0].T
            except Exception:
                return None
            
            # Update V: for each row j, solve (U⊗W)^T * V_j = T_j
            UW = np.einsum('ic,kc->ikc', U_new, W).reshape(U_size * W_size, rank)
            T_v = T.reshape(V_size, U_size * W_size)
            try:
                V_new = np.linalg.lstsq(UW.T, T_v.T, rcond=None)[0].T
            except Exception:
                return None
            
            # Update W: for each row k, solve (U⊗V)^T * W_k = T_k
            UV = np.einsum('ic,jc->ijc', U_new, V_new).reshape(U_size * V_size, rank)
            T_w = T.reshape(W_size, U_size * V_size)
            try:
                W_new = np.linalg.lstsq(UV.T, T_w.T, rcond=None)[0].T
            except Exception:
                return None
            
            # Compute cost
            T_calc = np.einsum('as,bs,cs->abc', U_new, V_new, W_new)
            cost = np.sum((T_calc - T) ** 2)
            
            if cost < tol:
                return U_new, V_new, W_new
            if cost > prev_cost * (1 + 1e-10):
                break
            prev_cost = cost
            U, V, W = U_new, V_new, W_new
        
        # Return final even if not converged (might be close)
        T_calc = np.einsum('as,bs,cs->abc', U, V, W)
        cost = np.sum((T_calc - T) ** 2)
        if cost < 1e-8:
            return U, V, W
        return None
    
    def structured_init(rank):
        """Create structured initialization exploiting the block structure."""
        U = np.zeros((U_size, rank))
        V = np.zeros((V_size, rank))
        W = np.zeros((W_size, rank))
        
        # The tensor has 18 non-zero entries. Try to assign them to rank-1 components.
        # Each component (u_s, v_s, w_s) can cover multiple entries if they share structure.
        # For now, use a heuristic: distribute the 18 entries across rank components.
        
        # Non-zero entries of T
        entries = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    entries.append((i * m + j, j * p + k, k * n + i))
        
        # Try to group entries that share indices
        # Simple approach: one component per entry, but with some sharing
        for s in range(min(rank, len(entries))):
            a, b, c = entries[s]
            U[a, s] = 1.0
            V[b, s] = 1.0
            W[c, s] = 1.0
        
        # Add some random perturbation for remaining components
        for s in range(min(rank, len(entries)), rank):
            U[:, s] = np.random.randn(U_size) * 0.1
            V[:, s] = np.random.randn(V_size) * 0.1
            W[:, s] = np.random.randn(W_size) * 0.1
        
        return U, V, W
    
    # Try ranks from 17 down to 10
    best_result = None
    
    for rank in range(17, 9, -1):
        if time.time() > deadline - 5:
            break
        
        # Try structured initialization first
        np.random.seed(123 + rank)
        U0, V0, W0 = structured_init(rank)
        result = als_decompose(rank, U0, V0, W0, max_iter=300)
        if result is not None:
            U_opt, V_opt, W_opt = result
            snapped = snap_and_verify(U_opt, V_opt, W_opt)
            if snapped:
                U_final, V_final, W_final = snapped
                return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
        
        # Try multiple random restarts
        max_restarts = 50 if rank >= 15 else 80
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(1000 + rank * 100 + restart * 7)
            U0 = np.random.randn(U_size, rank) * 0.5
            V0 = np.random.randn(V_size, rank) * 0.5
            W0 = np.random.randn(W_size, rank) * 0.5
            
            result = als_decompose(rank, U0, V0, W0, max_iter=200)
            if result is not None:
                U_opt, V_opt, W_opt = result
                snapped = snap_and_verify(U_opt, V_opt, W_opt)
                if snapped:
                    U_final, V_final, W_final = snapped
                    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
    
    # Fallback: schoolbook
    return get_schoolbook()
