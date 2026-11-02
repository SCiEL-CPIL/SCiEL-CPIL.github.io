"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill
    
    def _elapsed():
        return time.time() - start_time
    
    def _verify(U, V, W, tol=1e-9):
        """Verify decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k"""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        r = U.shape[1]
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.allclose(T, 1.0, atol=tol)
    
    def _schoolbook():
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
    
    # Start with schoolbook
    U_best, V_best, W_best = _schoolbook()
    best_rank = n * m * p
    
    # Strategy: Try to find lower-rank decompositions using coordinate descent
    # on the tensor constraint. We work with the residual approach.
    # 
    # Key idea: For a given rank R, we want U (nm x R), V (mp x R), W (pn x R)
    # such that einsum('ijr,jkr,kir->ijk', U3, V3, W3) = ones(n,m,p)
    #
    # Coordinate descent: fix two factors, optimize the third.
    # When optimizing U with V,W fixed:
    #   For each r, U[:,r] should satisfy: sum_r U[:,r] * (V3[:,k,r]*W3[k,:,r] summed over k) = 1
    #   More precisely, for fixed V,W, the constraint on U is:
    #   sum_r U[i*m+j, r] * (sum_k V[j*p+k, r] * W[k*n+i, r]) = 1 for all i,j
    #   This is a linear system in U! For each (i,j), we have:
    #   sum_r U[i*m+j, r] * A_{(i,j),r} = 1 where A_{(i,j),r} = sum_k V[j*p+k,r]*W[k*n+i,r]
    #   So U[i*m+j, :] is a row vector x such that x @ A_row = 1
    #   If A_row is nonzero, we can set x = A_row / (A_row @ A_row) (minimum norm solution)
    
    def _optimize_U(V, W):
        """Given V,W, find U that satisfies the constraint (if possible)."""
        V3 = V.reshape(m, p, -1)
        W3 = W.reshape(p, n, -1)
        R = V.shape[1]
        U = np.zeros((n * m, R))
        for i in range(n):
            for j in range(m):
                # A_{(i,j),r} = sum_k V[j*p+k, r] * W[k*n+i, r]
                # = sum_k V3[j,k,r] * W3[k,i,r]
                A_row = np.sum(V3[j, :, :] * W3[:, i, :], axis=0)  # shape (R,)
                norm_sq = np.dot(A_row, A_row)
                if norm_sq > 1e-12:
                    U[i * m + j, :] = A_row / norm_sq
                else:
                    # Degenerate: can't satisfy this constraint with current V,W
                    U[i * m + j, :] = 0
        return U
    
    def _optimize_V(U, W):
        """Given U,W, find V that satisfies the constraint (if possible)."""
        U3 = U.reshape(n, m, -1)
        W3 = W.reshape(p, n, -1)
        R = U.shape[1]
        V = np.zeros((m * p, R))
        for j in range(m):
            for k in range(p):
                # A_{(j,k),r} = sum_i U[i*m+j, r] * W[k*n+i, r]
                # = sum_i U3[i,j,r] * W3[k,i,r]
                A_row = np.sum(U3[:, j, :] * W3[k, :, :], axis=0)  # shape (R,)
                norm_sq = np.dot(A_row, A_row)
                if norm_sq > 1e-12:
                    V[j * p + k, :] = A_row / norm_sq
                else:
                    V[j * p + k, :] = 0
        return V
    
    def _optimize_W(U, V):
        """Given U,V, find W that satisfies the constraint (if possible)."""
        U3 = U.reshape(n, m, -1)
        V3 = V.reshape(m, p, -1)
        R = U.shape[1]
        W = np.zeros((p * n, R))
        for k in range(p):
            for i in range(n):
                # A_{(k,i),r} = sum_j U[i*m+j, r] * V[j*p+k, r]
                # = sum_j U3[i,j,r] * V3[j,k,r]
                A_row = np.sum(U3[i, :, :] * V3[:, k, :], axis=0)  # shape (R,)
                norm_sq = np.dot(A_row, A_row)
                if norm_sq > 1e-12:
                    W[k * n + i, :] = A_row / norm_sq
                else:
                    W[k * n + i, :] = 0
        return W
    
    def _compute_error(U, V, W):
        U3 = U.reshape(n, m, -1)
        V3 = V.reshape(m, p, -1)
        W3 = W.reshape(p, n, -1)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.sum((T - 1.0) ** 2)
    
    def _coordinate_descent(U, V, W, max_sweeps=200, tol=1e-12):
        """Run coordinate descent to minimize the error."""
        err = _compute_error(U, V, W)
        for sweep in range(max_sweeps):
            if _elapsed() > deadline:
                break
            U = _optimize_U(V, W)
            V = _optimize_V(U, W)
            W = _optimize_W(U, V)
            new_err = _compute_error(U, V, W)
            if abs(new_err - err) < tol:
                break
            err = new_err
        return U, V, W, err
    
    # Try ranks from 119 down to some lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(20, 30, 24) = 30
    # But realistically, we won't get much below 119 quickly
    
    np.random.seed(42)
    
    for R in range(best_rank - 1, 100, -1):
        if _elapsed() > deadline:
            break
        
        # Try a few random initializations for this rank
        max_restarts = 3
        for restart in range(max_restarts):
            if _elapsed() > deadline:
                break
            
            # Random initialization with small values
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5
            
            # Run coordinate descent
            U, V, W, err = _coordinate_descent(U, V, W, max_sweeps=100)
            
            if err < 1e-10:
                # Found a valid decomposition!
                # Round to nearest half-integer
                def _round_half(X):
                    return np.round(X * 2) / 2.0
                
                U_r = _round_half(U)
                V_r = _round_half(V)
                W_r = _round_half(W)
                
                # Verify the rounded version
                if _verify(U_r, V_r, W_r, tol=1e-8):
                    best_rank = R
                    best_U, best_V, best_W = U_r, V_r, W_r
                    break
                else:
                    # Try without rounding
                    if _verify(U, V, W, tol=1e-9):
                        best_rank = R
                        best_U, best_V, best_W = U, V, W
                        break
        
        if best_rank < best_rank:  # This won't trigger, just structural
            break
    
    # Convert to lists
    def _to_list(X):
        return [[float(x) for x in row] for row in np.asarray(X)]
    
    U_out = _to_list(best_U)
    V_out = _to_list(best_V)
    W_out = _to_list(best_W)
    
    # Final verification
    if not _verify(U_out, V_out, W_out, tol=1e-9):
        # Fall back to schoolbook
        U_s, V_s, W_s = _schoolbook()
        U_out = _to_list(U_s)
        V_out = _to_list(V_s)
        W_out = _to_list(W_s)
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
