import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0
    
    # Schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_sb.tolist(), V_sb.tolist(), W_sb.tolist()
    best_rank = rank_sb

    # Precompute equation structures
    # Equations are indexed by (i, j, jp, k, kp, ip)
    # rhs = 1 if i==ip, j==jp, k==kp else 0
    
    # For solving U[i,j,:], we need equations where i,j are fixed.
    # The variables are U[i,j,r] for r in 0..R-1.
    # The coefficient for U[i,j,r] in equation (i, j, jp, k, kp, ip) is V[jp,k,r] * W[kp,ip,r].
    # The target is rhs.
    
    def try_rank(R):
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # Normalize to avoid large numbers
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        for _ in range(200):
            if time.time() - start_time > deadline:
                return None
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # Build A (144 x R) and y (144)
                    A = np.zeros((m * p * p * n, R))
                    y = np.zeros(m * p * p * n)
                    idx = 0
                    for jp in range(m):
                        for k in range(p):
                            for kp in range(p):
                                for ip in range(n):
                                    rhs = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                                    y[idx] = rhs
                                    for r in range(R):
                                        A[idx, r] = V[jp * p + k, r] * W[kp * n + ip, r]
                                    idx += 1
                    # Solve A u = y
                    u, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                    U[i * m + j, :] = u
                    
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n * m * p * n, R))
                    y = np.zeros(n * m * p * n)
                    idx = 0
                    for i in range(n):
                        for jp in range(m):
                            for kp in range(p):
                                for ip in range(n):
                                    rhs = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                                    y[idx] = rhs
                                    for r in range(R):
                                        A[idx, r] = U[i * m + jp, r] * W[kp * n + ip, r]
                                    idx += 1
                    v, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                    V[j * p + k, :] = v
                    
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((n * m * m * p, R))
                    y = np.zeros(n * m * m * p)
                    idx = 0
                    for jp in range(m):
                        for kp in range(p):
                            for ip in range(n):
                                for ii in range(n): # i'
                                    # Wait, W index is (k, i). 
                                    # Equation indices: i, j, jp, k, kp, ip
                                    # W[kp, ip, r] is the variable? No, W[k, i, r] is the variable for this block.
                                    # The equation uses W[kp, ip, r].
                                    # If we are updating W[k, i, :], then the variable is W[k, i, r].
                                    # It appears in equations where kp=k, ip=i.
                                    # So we need to iterate over equations where kp=k, ip=i.
                                    pass
                    # This part is tricky. Let's re-verify.
                    # Variable W[k, i, r].
                    # Appears in equation (i_eq, j_eq, jp_eq, k_eq, kp_eq, ip_eq) if kp_eq == k and ip_eq == i.
                    # So for fixed k, i, we iterate over i_eq, j_eq, jp_eq, k_eq.
                    # Count: n * m * m * p = 3*3*3*4 = 108.
                    A = np.zeros((n * m * m * p, R))
                    y = np.zeros(n * m * m * p)
                    idx = 0
                    for i_eq in range(n):
                        for j_eq in range(m):
                            for jp_eq in range(m):
                                for k_eq in range(p):
                                    # kp_eq is k, ip_eq is i
                                    rhs = 1.0 if (i_eq==i and j_eq==jp_eq and k_eq==k) else 0.0
                                    y[idx] = rhs
                                    for r in range(R):
                                        A[idx, r] = U[i_eq * m + j_eq, r] * V[jp_eq * p + k_eq, r]
                                    idx += 1
                    w, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                    W[k * n + i, :] = w
                    
        # Check error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                rhs = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                                val = 0.0
                                for r in range(R):
                                    val += U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r]
                                err += (val - rhs)**2
        if err < 1e-6:
            return U, V, W
        return None

    # Try ranks
    for R in range(35, 11, -1):
        res = try_rank(R)
        if res:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
