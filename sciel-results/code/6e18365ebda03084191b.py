import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Schoolbook solution
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
    
    deadline = time.time() + 140  # Leave 10s margin for safety
    
    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        """Check if the decomposition is correct."""
        n, m, p = DIMENSIONS
        rank = U_mat.shape[1]
        for i in range(n):
            for k in range(p):
                for i2 in range(n):
                    for j in range(m):
                        for j2 in range(m):
                            for k2 in range(p):
                                expected = 1.0 if (i2 == i and j == j2 and k2 == k) else 0.0
                                actual = np.dot(W_mat[k*n+i, :], U_mat[i2*m+j, :] * V_mat[j2*p+k2, :])
                                if abs(actual - expected) > tol:
                                    return False
        return True
    
    # Precompute constraint structure for efficiency
    # Constraints: for each (i,k,i2,j,j2,k2), target = 1 if (i2==i and j==j2 and k2==k) else 0
    # We'll build linear systems on the fly but cache where possible
    
    def solve_for_W(U_mat, V_mat, rank):
        """Solve for W given U and V."""
        num_unknowns = p * n * rank
        num_constraints = n * p * n * m * m * p
        
        A = np.zeros((num_constraints, num_unknowns))
        b = np.zeros(num_constraints)
        
        idx = 0
        for i in range(n):
            for k in range(p):
                for i2 in range(n):
                    for j in range(m):
                        for j2 in range(m):
                            for k2 in range(p):
                                target = 1.0 if (i2 == i and j == j2 and k2 == k) else 0.0
                                b[idx] = target
                                # Only add non-zero coefficients
                                uv_prod = U_mat[i2*m+j, :] * V_mat[j2*p+k2, :]
                                nonzero = np.where(np.abs(uv_prod) > 1e-15)[0]
                                for c in nonzero:
                                    w_idx = (k * n + i) * rank + c
                                    A[idx, w_idx] += uv_prod[c]
                                idx += 1
        
        try:
            W_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except:
            W_flat = np.zeros(num_unknowns)
        
        return W_flat.reshape(p * n, rank)
    
    def solve_for_U(V_mat, W_mat, rank):
        """Solve for U given V and W."""
        num_unknowns = n * m * rank
        num_constraints = n * p * n * m * m * p
        
        A = np.zeros((num_constraints, num_unknowns))
        b = np.zeros(num_constraints)
        
        idx = 0
        for i in range(n):
            for k in range(p):
                for i2 in range(n):
                    for j in range(m):
                        for j2 in range(m):
                            for k2 in range(p):
                                target = 1.0 if (i2 == i and j == j2 and k2 == k) else 0.0
                                b[idx] = target
                                wv_prod = W_mat[k*n+i, :] * V_mat[j2*p+k2, :]
                                nonzero = np.where(np.abs(wv_prod) > 1e-15)[0]
                                for c in nonzero:
                                    u_idx = (i2 * m + j) * rank + c
                                    A[idx, u_idx] += wv_prod[c]
                                idx += 1
        
        try:
            U_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except:
            U_flat = np.zeros(num_unknowns)
        
        return U_flat.reshape(n * m, rank)
    
    def solve_for_V(U_mat, W_mat, rank):
        """Solve for V given U and W."""
        num_unknowns = m * p * rank
        num_constraints = n * p * n * m * m * p
        
        A = np.zeros((num_constraints, num_unknowns))
        b = np.zeros(num_constraints)
        
        idx = 0
        for i in range(n):
            for k in range(p):
                for i2 in range(n):
                    for j in range(m):
                        for j2 in range(m):
                            for k2 in range(p):
                                target = 1.0 if (i2 == i and j == j2 and k2 == k) else 0.0
                                b[idx] = target
                                uw_prod = W_mat[k*n+i, :] * U_mat[i2*m+j, :]
                                nonzero = np.where(np.abs(uw_prod) > 1e-15)[0]
                                for c in nonzero:
                                    v_idx = (j2 * p + k2) * rank + c
                                    A[idx, v_idx] += uw_prod[c]
                                idx += 1
        
        try:
            V_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except:
            V_flat = np.zeros(num_unknowns)
        
        return V_flat.reshape(m * p, rank)
    
    def alternating_minimization(U_init, V_init, W_init, rank, max_iter=20, tol=1e-10):
        """Run alternating least squares."""
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            W = solve_for_W(U, V, rank)
            U = solve_for_U(V, W, rank)
            V = solve_for_V(U, W, rank)
            
            if verify(U, V, W, tol):
                return U, V, W, True
        
        return U, V, W, verify(U, V, W, tol)
    
    # Try ranks from low to high
    for rank in range(1, best_rank):
        if time.time() > deadline - 5:
            break
        
        found = False
        
        # Try a few random initializations
        n_init = min(3, max(1, int((deadline - time.time()) / 10)))
        for init in range(n_init):
            if time.time() > deadline - 5:
                break
            
            U_init = np.random.randn(n*m, rank)
            V_init = np.random.randn(m*p, rank)
            W_init = np.random.randn(p*n, rank)
            
            U_sol, V_sol, W_sol, success = alternating_minimization(
                U_init, V_init, W_init, rank, max_iter=15, tol=1e-10
            )
            
            if success and rank < best_rank:
                best_U, best_V, best_W = U_sol, V_sol, W_sol
                best_rank = rank
                found = True
                break
        
        if found:
            continue
        
        # Try structured initialization from schoolbook
        if time.time() > deadline - 5:
            break
        
        if rank <= rank_school:
            U_init = U[:, :rank].copy()
            V_init = V[:, :rank].copy()
            W_init = W[:, :rank].copy()
            
            U_sol, V_sol, W_sol, success = alternating_minimization(
                U_init, V_init, W_init, rank, max_iter=15, tol=1e-10
            )
            
            if success and rank < best_rank:
                best_U, best_V, best_W = U_sol, V_sol, W_sol
                best_rank = rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
