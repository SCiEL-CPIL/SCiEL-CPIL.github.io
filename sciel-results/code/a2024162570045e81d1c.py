import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    # Build the target tensor
    # T[a,b,c,d,i,k] = delta_{a,i} * delta_{b,c} * delta_{d,k}
    # We'll work with the flattened system
    
    # Precompute indices
    # For the linear systems, we need to build the coefficient matrices
    
    # Number of equations: n*m*m*p*n*p
    n_eq = n * m * m * p * n * p
    
    def build_system_U(V, W, R):
        """Build linear system for U given V, W."""
        # Unknowns: U[i*m+j, r] for i in [0,n), j in [0,m), r in [0,R)
        n_unknowns = n * m * R
        
        # Equation index: (a,b,c,d,i,k) -> eq_idx
        # Coefficient of U[a*m+b, r] in equation (a,b,c,d,i,k):
        #   V[c*p+d, r] * W[k*n+i, r]
        
        A_mat = np.zeros((n_eq, n_unknowns))
        b_vec = np.zeros(n_eq)
        
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    for d in range(p):
                        for i in range(n):
                            for k in range(p):
                                eq_idx = ((a * m + b) * m + c) * (p * n * p) + (d * n + i) * p + k
                                # Wait, let me define a proper indexing
                                # eq_idx = a * (m*m*p*n*p) + b * (m*p*n*p) + c * (p*n*p) + d * (n*p) + i * p + k
                                eq_idx = a * (m*m*p*n*p) + b * (m*p*n*p) + c * (p*n*p) + d * (n*p) + i * p + k
                                target = 1.0 if (a == i and b == c and d == k) else 0.0
                                b_vec[eq_idx] = target
                                for r in range(R):
                                    unknown_idx = (a * m + b) * R + r
                                    A_mat[eq_idx, unknown_idx] = V[c*p+d, r] * W[k*n+i, r]
        
        return A_mat, b_vec
    
    def build_system_V(U, W, R):
        """Build linear system for V given U, W."""
        n_unknowns = m * p * R
        
        A_mat = np.zeros((n_eq, n_unknowns))
        b_vec = np.zeros(n_eq)
        
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    for d in range(p):
                        for i in range(n):
                            for k in range(p):
                                eq_idx = a * (m*m*p*n*p) + b * (m*p*n*p) + c * (p*n*p) + d * (n*p) + i * p + k
                                target = 1.0 if (a == i and b == c and d == k) else 0.0
                                b_vec[eq_idx] = target
                                for r in range(R):
                                    unknown_idx = (c * p + d) * R + r
                                    A_mat[eq_idx, unknown_idx] = U[a*m+b, r] * W[k*n+i, r]
        
        return A_mat, b_vec
    
    def build_system_W(U, V, R):
        """Build linear system for W given U, V."""
        n_unknowns = p * n * R
        
        A_mat = np.zeros((n_eq, n_unknowns))
        b_vec = np.zeros(n_eq)
        
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    for d in range(p):
                        for i in range(n):
                            for k in range(p):
                                eq_idx = a * (m*m*p*n*p) + b * (m*p*n*p) + c * (p*n*p) + d * (n*p) + i * p + k
                                target = 1.0 if (a == i and b == c and d == k) else 0.0
                                b_vec[eq_idx] = target
                                for r in range(R):
                                    unknown_idx = (k * n + i) * R + r
                                    A_mat[eq_idx, unknown_idx] = U[a*m+b, r] * V[c*p+d, r]
        
        return A_mat, b_vec
    
    # Precompute the b vector (it's the same for all systems)
    b_vec = np.zeros(n_eq)
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            eq_idx = a * (m*m*p*n*p) + b * (m*p*n*p) + c * (p*n*p) + d * (n*p) + i * p + k
                            b_vec[eq_idx] = 1.0 if (a == i and b == c and d == k) else 0.0
    
    # Schoolbook solution
    rank_school = n * m * p
    U = np.zeros((n*m, rank_school))
    V = np.zeros((m*p, rank_school))
    W = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_school
    
    # Try alternating least squares for lower ranks
    for target_rank in range(rank_school - 1, max(rank_school - 3, 10), -1):
        if time.time() > deadline:
            break
        
        for trial in range(20):
            if time.time() > deadline:
                break
            
            # Random initialization
            np.random.seed(1000 + trial)
            U_t = np.random.randn(n*m, target_rank) * 0.1
            V_t = np.random.randn(m*p, target_rank) * 0.1
            W_t = np.random.randn(p*n, target_rank) * 0.1
            
            # Alternating least squares
            converged = False
            for iteration in range(100):
                if time.time() > deadline:
                    break
                
                # Solve for U
                A_U, _ = build_system_U(V_t, W_t, target_rank)
                U_t, res_U, _, _ = np.linalg.lstsq(A_U, b_vec, rcond=None)
                U_t = U_t.reshape(n*m, target_rank)
                
                # Solve for V
                A_V, _ = build_system_V(U_t, W_t, target_rank)
                V_t, res_V, _, _ = np.linalg.lstsq(A_V, b_vec, rcond=None)
                V_t = V_t.reshape(m*p, target_rank)
                
                # Solve for W
                A_W, _ = build_system_W(U_t, V_t, target_rank)
                W_t, res_W, _, _ = np.linalg.lstsq(A_W, b_vec, rcond=None)
                W_t = W_t.reshape(p*n, target_rank)
                
                # Check convergence
                # Compute residual
                total_res = 0.0
                for a in range(n):
                    for b in range(m):
                        for c in range(m):
                            for d in range(p):
                                for i in range(n):
                                    for k in range(p):
                                        target = 1.0 if (a == i and b == c and d == k) else 0.0
                                        val = np.sum(U_t[a*m+b, :] * V_t[c*p+d, :] * W_t[k*n+i, :])
                                        total_res += (val - target)**2
                
                if total_res < 1e-10:
                    converged = True
                    break
                
                if time.time() > deadline:
                    break
            
            if converged:
                # Verify and check if entries are rational
                U_t = np.round(U_t, 6)
                V_t = np.round(V_t, 6)
                W_t = np.round(W_t, 6)
                
                # Verify
                total_res = 0.0
                for a in range(n):
                    for b in range(m):
                        for c in range(m):
                            for d in range(p):
                                for i in range(n):
                                    for k in range(p):
                                        target = 1.0 if (a == i and b == c and d == k) else 0.0
                                        val = np.sum(U_t[a*m+b, :] * V_t[c*p+d, :] * W_t[k*n+i, :])
                                        total_res += (val - target)**2
                
                if total_res < 1e-6:
                    best_U, best_V, best_W = U_t, V_t, W_t
                    best_rank = target_rank
                    break
        
        if best_rank < rank_school:
            break
    
    # Convert to list
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
