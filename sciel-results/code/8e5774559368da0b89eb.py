import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Build the target tensor T
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try ranks from low to high
    # Lower bound: max(n*m, m*p, n*p) = max(15, 30, 18) = 30
    # But tensor rank is typically higher
    best_result = None
    
    for rank in range(30, 91):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # ALS iterations
            converged = False
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Update W: for each c, solve M_c = Σ_r W[c,r] * (U[:,r] ⊗ V[:,r])
                # M_c[a,b] = T[a,b,c]
                # Vectorize: vec(M_c) = Σ_r W[c,r] * vec(U[:,r] * V[:,r]^T)
                # Design matrix D: shape (nm*mp, rank), D[:,r] = vec(U[:,r] * V[:,r]^T)
                D = np.zeros((nm * mp, rank))
                for r in range(rank):
                    D[:, r] = (U[:, r][:, None] * V[:, r][None, :]).flatten()
                
                # Solve for W: T.reshape(nm*mp, pn)^T = D @ W
                # T_flat: (nm*mp, pn) where T_flat[a*mp+b, c] = T[a,b,c]
                T_flat = T.reshape(nm * mp, pn)
                # W = pinv(D) @ T_flat^T ... but this is (rank, nm*mp) @ (nm*mp, pn) = (rank, pn)
                # Then W^T = (pn, rank)
                try:
                    W_new = np.linalg.lstsq(D, T_flat.T, rcond=None)[0].T
                except:
                    break
                
                # Update U: for each a, solve N_a[b,c] = T[a,b,c] = Σ_r U[a,r] * (V[:,r] ⊗ W[:,r])
                # N_a: (mp, pn), N_a[b,c] = T[a,b,c]
                # vec(N_a) = Σ_r U[a,r] * vec(V[:,r] * W[:,r]^T)
                D_u = np.zeros((mp * pn, rank))
                for r in range(rank):
                    D_u[:, r] = (V[:, r][:, None] * W[:, r][None, :]).flatten()
                
                T_u = T.reshape(nm, mp * pn)  # T_u[a, b*pn+c] = T[a,b,c]
                try:
                    U_new = np.linalg.lstsq(D_u, T_u.T, rcond=None)[0].T
                except:
                    break
                
                # Update V: for each b, solve P_b[a,c] = T[a,b,c] = Σ_r V[b,r] * (U[:,r] ⊗ W[:,r])
                # P_b: (nm, pn), P_b[a,c] = T[a,b,c]
                # vec(P_b) = Σ_r V[b,r] * vec(U[:,r] * W[:,r]^T)
                D_v = np.zeros((nm * pn, rank))
                for r in range(rank):
                    D_v[:, r] = (U[:, r][:, None] * W[:, r][None, :]).flatten()
                
                T_v = T.reshape(mp, nm * pn)  # T_v[b, a*pn+c] = T[a,b,c]
                try:
                    V_new = np.linalg.lstsq(D_v, T_v.T, rcond=None)[0].T
                except:
                    break
                
                # Check convergence and error
                err = np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V)) + np.max(np.abs(W_new - W))
                U, V, W = U_new, V_new, W_new
                if err < 1e-10:
                    converged = True
                    break
            
            # Check if we have an exact solution
            # Recompute T_approx
            T_approx = np.zeros((nm, mp, pn), dtype=np.float64)
            for r in range(rank):
                T_approx += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            max_err = np.max(np.abs(T_approx - T))
            if max_err < 1e-8:
                # Round to nearest half-integers or integers
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                # Verify rounded solution
                T_approx_round = np.zeros((nm, mp, pn), dtype=np.float64)
                for r in range(rank):
                    T_approx_round += np.einsum('a,b,c->abc', U_round[:, r], V_round[:, r], W_round[:, r])
                
                max_err_round = np.max(np.abs(T_approx_round - T))
                if max_err_round < 1e-8:
                    # Convert to lists
                    return {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }
        
        # If we found a solution at this rank, return it
        # (the above already returns if found)
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
