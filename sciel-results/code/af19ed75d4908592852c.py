import numpy as np
import time
import sys

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Build the target tensor T
    # T[a, b, c] where a = i*m + j, b = j*p + k, c = k*n + i
    # a in [0, nm), b in [0, mp), c in [0, pn)
    nm = n * m
    mp = m * p
    pn = p * n
    
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def check_and_snap(U, V, W, r):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Snap to nearest half-integer
        U_snap = np.round(U_np * 2) / 2
        V_snap = np.round(V_np * 2) / 2
        W_snap = np.round(W_np * 2) / 2
        
        # Reconstruct
        T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
        
        # Check exactness
        if not np.allclose(T_recon, T_target, atol=1e-9):
            return None
            
        # Convert back to lists
        U_out = [[float(x) for x in row] for row in U_snap]
        V_out = [[float(x) for x in row] for row in V_snap]
        W_out = [[float(x) for x in row] for row in W_snap]
        
        return {"U": U_out, "V": V_out, "W": W_out}

    def als_decompose(r, seed, max_iters=50):
        """
        Alternating Least Squares with snapping.
        """
        if time.time() > deadline:
            return None
            
        np.random.seed(seed)
        
        # Sparse random initialization
        # Entries are 0, 0.5, 1, -0.5, -1
        def sparse_init(shape, density=0.5):
            U = np.zeros(shape)
            mask = np.random.rand(*shape) < density
            U[mask] = np.random.choice([-1, -0.5, 0.5, 1], size=np.sum(mask))
            return U

        U = sparse_init((nm, r))
        V = sparse_init((mp, r))
        W = sparse_init((pn, r))
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
                
            # Update U: Fix V, W.
            # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # For fixed a, we want U[a, :] to minimize || T[a, :, :] - U[a, :] (V * W)^T ||
            # Let M = V * W (element-wise? No, it's outer product sum)
            # T[a] is a vector of length mp*pn.
            # U[a] is a vector of length r.
            # We need U[a] @ (V[:, r] * W[:, r])?
            # T[a, b, c] = sum_r U[a, r] * (V[b, r] * W[c, r])
            # Let Z_r[b, c] = V[b, r] * W[c, r]
            # T[a] = sum_r U[a, r] Z_r
            # This is a linear system: U[a] Z = T[a]
            # Z is a matrix of shape (r, mp*pn)
            
            Z = np.zeros((r, mp * pn))
            for r_idx in range(r):
                Z[r_idx, :] = (V[:, r_idx] * W[:, r_idx]).flatten() # V[:, r] is (mp,1), W[:, r] is (pn,1) -> outer product
                # Actually V[:, r_idx] * W[:, r_idx] is not outer product.
                # V[b, r] * W[c, r] is the element at (b, c) of the outer product of V[:, r] and W[:, r]
                Z[r_idx, :] = np.outer(V[:, r_idx], W[:, r_idx]).flatten()
                
            # Solve U[a] Z = T_target[a]
            # U[a] = T_target[a] Z^T (Z Z^T)^-1
            # Use least squares
            try:
                U_new, _, _, _ = np.linalg.lstsq(Z, T_target.reshape(nm, -1).T, rcond=None)
                U = U_new.T
            except:
                pass
                
            # Update V
            Y = np.zeros((r, nm * pn))
            for r_idx in range(r):
                Y[r_idx, :] = np.outer(U[:, r_idx], W[:, r_idx]).flatten()
            try:
                V_new, _, _, _ = np.linalg.lstsq(Y, T_target.reshape(mp, -1).T, rcond=None)
                # Wait, T_target is (nm, mp, pn).
                # For V[b, r], we fix U, W.
                # T[a, b, c] = sum_r (U[a, r] W[c, r]) V[b, r]
                # Let X_r[a, c] = U[a, r] W[c, r]
                # T[:, b, :] = sum_r V[b, r] X_r
                # T[:, b, :] is a vector of length nm*pn.
                # X is (r, nm*pn)
                V_new, _, _, _ = np.linalg.lstsq(Y, T_target.transpose(0, 2, 1).reshape(pn, -1).T, rcond=None)
                # Wait, T_target.transpose(0, 2, 1) is (nm, pn, mp).
                # Reshape to (pn, nm*mp)?
                # We want to solve for V[b] for each b.
                # T[:, b, :] flattened is length nm*pn.
                # So we need T_target reshaped to (mp, nm*pn).
                V_new, _, _, _ = np.linalg.lstsq(Y, T_target.transpose(1, 0, 2).reshape(mp, -1).T, rcond=None)
                V = V_new.T
            except:
                pass
                
            # Update W
            X = np.zeros((r, nm * mp))
            for r_idx in range(r):
                X[r_idx, :] = np.outer(U[:, r_idx], V[:, r_idx]).flatten()
            try:
                W_new, _, _, _ = np.linalg.lstsq(X, T_target.reshape(pn, -1).T, rcond=None)
                # T[a, b, c] = sum_r (U[a, r] V[b, r]) W[c, r]
                # Let Z_r[a, b] = U[a, r] V[b, r]
                # T[:, :, c] = sum_r W[c, r] Z_r
                # T[:, :, c] flattened is length nm*mp.
                # So T_target reshaped to (pn, nm*mp).
                W_new, _, _, _ = np.linalg.lstsq(X, T_target.transpose(2, 0, 1).reshape(pn, -1).T, rcond=None)
                W = W_new.T
            except:
                pass
            
            # Snap and check
            result = check_and_snap(U, V, W, r)
            if result:
                return result
                
            # Calculate error for tracking
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.sum((T_recon - T_target) ** 2)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
        # If loop finishes, check best
        result = check_and_snap(best_U, best_V, best_W, r)
        if result:
            return result
        return None

    # Fallback schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find lower rank
    # Try ranks from 40 to 49
    for r in range(40, 50):
        if time.time() > deadline:
            break
        for seed in [42, 123, 999, 111, 222]:
            if time.time() > deadline:
                break
            result = als_decompose(r, seed)
            if result:
                best_result = result
                best_rank = r
                break
        if best_rank < rank_school:
            break

    return best_result
# EVOLVE-BLOCK-END
