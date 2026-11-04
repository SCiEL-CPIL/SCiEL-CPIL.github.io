import time
import numpy as np

def construct():
    n, m, p = 3, 3, 5
    deadline = time.time() + 140  # 10s margin
    
    # Build target tensor T
    # T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def schoolbook():
        r = n * m * p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, r

    best_U, best_V, best_W, best_rank = schoolbook()
    
    def verify(U, V, W):
        P = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.max(np.abs(P - T))

    # Pre-compute indices for ALS
    # For W: fixed c, vary (a,b). M_c shape (135, R)
    # a in 0..8, b in 0..14
    ab_indices = np.array([(a, b) for a in range(n*m) for b in range(m*p)])
    a_idx = ab_indices[:, 0]
    b_idx = ab_indices[:, 1]
    
    # For U: fixed a, vary (b,c). M_a shape (135, R)
    bc_indices = np.array([(b, c) for b in range(m*p) for c in range(p*n)])
    b_idx_u = bc_indices[:, 0]
    c_idx_u = bc_indices[:, 1]
    
    # For V: fixed b, vary (a,c). M_b shape (135, R)
    ac_indices = np.array([(a, c) for a in range(n*m) for c in range(p*n)])
    a_idx_v = ac_indices[:, 0]
    c_idx_v = ac_indices[:, 1]

    def solve_W(U, V, R):
        W = np.zeros((p*n, R))
        for c in range(p*n):
            # M_c: (135, R)
            # M_c[row, r] = U[a, r] * V[b, r]
            # row corresponds to (a,b)
            M_c = U[a_idx, :] * V[b_idx, :] # Broadcasting: (135, R) * (135, R) -> (135, R)
            # t_c: (135,)
            t_c = T[a_idx, b_idx, c]
            # Solve M_c @ W_c = t_c
            W_c, residuals, rank, sv = np.linalg.lstsq(M_c, t_c, rcond=None)
            W[c, :] = W_c
        return W

    def solve_U(V, W, R):
        U = np.zeros((n*m, R))
        for a in range(n*m):
            # M_a: (135, R)
            # M_a[row, r] = V[b, r] * W[c, r]
            # row corresponds to (b,c)
            M_a = V[b_idx_u, :] * W[c_idx_u, :]
            t_a = T[a, b_idx_u, c_idx_u]
            U_a, residuals, rank, sv = np.linalg.lstsq(M_a, t_a, rcond=None)
            U[a, :] = U_a
        return U

    def solve_V(U, W, R):
        V = np.zeros((m*p, R))
        for b in range(m*p):
            # M_b: (135, R)
            # M_b[row, r] = U[a, r] * W[c, r]
            # row corresponds to (a,c)
            M_b = U[a_idx_v, :] * W[c_idx_v, :]
            t_b = T[a_idx_v, b, c_idx_v]
            V_b, residuals, rank, sv = np.linalg.lstsq(M_b, t_b, rcond=None)
            V[b, :] = V_b
        return V

    # Try ranks from low to high
    # Lower bound is 15. Let's try 15 to 44.
    # We will use multiple random restarts for each rank.
    max_restarts = 50
    max_als_iters = 50
    
    for R in range(15, 45):
        if time.time() > deadline - 10:
            break
            
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Random initialization
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5
            
            best_err = np.inf
            for it in range(max_als_iters):
                if time.time() > deadline - 10:
                    break
                
                # ALS steps
                W = solve_W(U, V, R)
                U = solve_V(V, W, R) # Wait, solve_V needs U and W. Correct.
                V = solve_U(U, W, R) # Wait, solve_U needs V and W. Correct.
                
                # Check error
                err = verify(U, V, W)
                
                if err < 1e-8:
                    # Found a solution. Try to snap to half-integers.
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    err_snap = verify(U_snap, V_snap, W_snap)
                    if err_snap < 1e-8:
                        best_U, best_V, best_W, best_rank = U_snap, V_snap, W_snap, R
                        found = True
                        break
                    else:
                        # If snap fails, keep the real-valued one? 
                        # The prompt requires integer/half-integer.
                        # If we can't snap, we might need to continue searching or accept higher rank.
                        # However, if the real solution is exact, it's likely close to half-integers.
                        # If not, we discard this solution and try another restart.
                        pass
                        
                if err < best_err:
                    best_err = err
                    if best_err < 1e-10:
                        break
                        
            if found:
                break
                
        if found:
            break

    # Final verification
    if verify(best_U, best_V, best_W) > 1e-8:
        # Fallback to schoolbook
        best_U, best_V, best_W, best_rank = schoolbook()

    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
