import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 12, 8
    
    # Build the target tensor T
    # T[a, b, c] where a=i*m+j, b=j'*p+k, c=k'*n+i'
    # T[a, b, c] = 1 if i=i', j=j', k=k', else 0
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def snap_half_int(arr):
        # Snap to nearest 0.5
        return np.round(arr * 2) / 2.0

    def verify(U, V, W, tol=1e-8):
        # Check if sum_r U[a,r] V[b,r] W[c,r] == T[a,b,c]
        # U: (nm, R), V: (mp, R), W: (pn, R)
        R = U.shape[1]
        # Efficient check
        # T_comp = np.einsum('ar,br,cr->abc', U, V, W)
        # But for small R, loop is fine
        T_comp = np.zeros((nm, mp, pn))
        for r in range(R):
            T_comp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.max(np.abs(T_comp - T)) < tol

    def solve_als(R, seed, max_iters=200):
        np.random.seed(seed)
        # Initialize with small random values
        U = np.random.randn(nm, R) * 0.1
        V = np.random.randn(mp, R) * 0.1
        W = np.random.randn(pn, R) * 0.1
        
        prev_cost = np.inf
        for it in range(max_iters):
            # Solve for U
            # For each a, A_a u_a^T = T[a, :, :].flatten()
            # A_a[(b, c), r] = V[b, r] W[c, r]
            # We can precompute A_a for all a? No, V and W change.
            # But we can compute the matrix product for each a.
            # Actually, we can form a large matrix?
            # Let's just loop a.
            for a in range(nm):
                # Construct A_a: (mp * pn, R)
                # V[b, r] W[c, r]
                # We can do: A_a_flat = (V[:, None, :] * W[None, :, :]).reshape(-1, R)
                # V is (mp, R), W is (pn, R)
                # V[:, None, :] is (mp, 1, R)
                # W[None, :, :] is (1, pn, R)
                # Product is (mp, pn, R)
                # Reshape to (mp*pn, R)
                A_a = (V[:, None, :] * W[None, :, :]).reshape(-1, R)
                t_a = T[a, :, :].reshape(-1)
                # Solve A_a x = t_a
                U[a, :], residuals, rank, sv = np.linalg.lstsq(A_a, t_a, rcond=None)
            
            # Solve for V
            for b in range(mp):
                # B_b[(a, c), r] = U[a, r] W[c, r]
                B_b = (U[:, None, :] * W[None, :, :]).reshape(-1, R)
                t_b = T[:, b, :].reshape(-1)
                V[b, :], residuals, rank, sv = np.linalg.lstsq(B_b, t_b, rcond=None)
            
            # Solve for W
            for c in range(pn):
                # C_c[(a, b), r] = U[a, r] V[b, r]
                C_c = (U[:, None, :] * V[None, :, :]).reshape(-1, R)
                t_c = T[:, :, c].reshape(-1)
                W[c, :], residuals, rank, sv = np.linalg.lstsq(C_c, t_c, rcond=None)
            
            # Calculate cost
            T_comp = np.zeros((nm, mp, pn))
            for r in range(R):
                T_comp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            cost = np.sum((T_comp - T)**2)
            
            if cost < 1e-10:
                break
            if abs(prev_cost - cost) < 1e-12:
                break
            prev_cost = cost
            
        # Try snapping
        U_s = snap_half_int(U)
        V_s = snap_half_int(V)
        W_s = snap_half_int(W)
        
        if verify(U_s, V_s, W_s, tol=1e-8):
            return U_s, V_s, W_s, cost
            
        # If snapped version fails, check if un-snapped is good enough?
        # The prompt requires integer/half-integer entries.
        # So we MUST return snapped values.
        # If snapped is not valid, this rank/restart failed.
        return None, None, None, cost

    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try ranks from 20 to 23
    for R in [20, 21, 22, 23]:
        if time.time() > deadline:
            break
            
        found = False
        # Number of restarts depends on remaining time
        # R=20 is the most important. Give it more restarts.
        if R == 20:
            num_restarts = 50
        elif R == 21:
            num_restarts = 20
        else:
            num_restarts = 10
            
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            U, V, W, cost = solve_als(R, seed=42 + R * 100 + restart)
            if U is not None:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback: Schoolbook rank 24
    R = n * m * p
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
