"""Matrix multiplication 2x3x6 - structured search over sparse W with linear solve for U,V."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    n, m, p = 2, 3, 6
    N = n * m  # 6
    M = m * p  # 18
    P = p * n  # 12

    def schoolbook():
        rank = n * m * p
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = schoolbook()
    best_rank = n * m * p

    def verify_exact(U, V, W, n_s, m_s, p_s):
        """Verify decomposition using exact integer arithmetic with half-integer support."""
        r = U.shape[1]
        U_int = np.round(U * 2)
        V_int = np.round(V * 2)
        W_int = np.round(W * 2)
        
        if not np.allclose(U_int, U * 2, atol=1e-9): return False
        if not np.allclose(V_int, V * 2, atol=1e-9): return False
        if not np.allclose(W_int, W * 2, atol=1e-9): return False
        
        U_int = U_int.astype(np.int64)
        V_int = V_int.astype(np.int64)
        W_int = W_int.astype(np.int64)
        
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    total = 0
                    for c in range(r):
                        total += U_int[i * m_s + j, c] * V_int[j * p_s + k, c] * W_int[k * n_s + i, c]
                    if total != 8:
                        return False
        return True

    def try_rank_structured(rank, max_patterns=200):
        """
        Structured search: fix W to have sparse patterns, solve for U,V via linear system.
        
        For each column c, the constraint is:
          sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
        
        If we fix W, this becomes a bilinear system. But we can reformulate:
        For fixed W, define for each (i,k): A_{i,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        Actually, let's use the matrix form. The tensor equation is:
          T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        For fixed W, this is linear in (U*V) but bilinear in U and V.
        
        Alternative: For fixed W, we can set up a system. Note that for each c,
        the contribution is rank-1 in the (i,j) vs k sense.
        
        Let's use a different approach: For fixed W, the equation becomes:
          For each (i,k): sum_j V[j*p+k, :] * (U[:, j*m+...]) -- this is getting complex.
        
        Simpler: Use the fact that if W is fixed, we can solve for U and V by:
          For each c, define u_c = U[:, c], v_c = V[:, c], w_c = W[:, c].
          Then sum_c (u_c ⊗ v_c ⊗ w_c) = T (the multiplication tensor).
        
        This is still nonlinear. But we can use the following trick:
        Fix W, then for each (i,k), the equation is:
          sum_j V[j*p+k, :]^T * (some matrix involving U) = 1
        
        Actually, let's use a direct linear algebra approach.
        
        For fixed W, the tensor equation can be written as:
          For each i: sum_k W[k*n+i, :] ⊙ (sum_j U[i*m+j, :] ⊗ V[j*p+k, :]) = e_{i}
        
        This is still nonlinear. Let's try a different strategy:
        
        Since n=2 is small, we can enumerate patterns for W more cleverly.
        """
        
        # Strategy: For small ranks, try random W with sparsity, then solve for U,V
        # using alternating least squares or direct linear solve.
        
        for seed in range(max_patterns):
            if time.time() > deadline - 5:
                return None
                
            np.random.seed(seed * 7919 + rank * 104729)
            
            # Generate sparse W: each column has few nonzeros
            W = np.zeros((P, rank))
            for c in range(rank):
                # Choose 1-3 non-zero positions in W[:, c]
                n_nonzero = np.random.randint(1, 4)
                positions = np.random.choice(P, n_nonzero, replace=False)
                for pos in positions:
                    W[pos, c] = np.random.choice([-1, 1])
            
            # Now solve for U and V using alternating minimization
            U = np.random.randn(N, rank) * 0.1
            V = np.random.randn(M, rank) * 0.1
            
            # Normalize initial columns
            for c in range(rank):
                nU = np.linalg.norm(U[:, c])
                nV = np.linalg.norm(V[:, c])
                nW = np.linalg.norm(W[:, c])
                if nU > 1e-10 and nV > 1e-10 and nW > 1e-10:
                    scale = (nU * nV * nW) ** (1.0/3.0)
                    U[:, c] /= scale
                    V[:, c] /= scale
                    W[:, c] /= scale
            
            # Alternating least squares: fix two, solve for third
            converged = False
            for iteration in range(50):
                if time.time() > deadline - 3:
                    break
                
                # Solve for U: fix V and W
                # For each i: sum_j,k U[i*m+j, :] * (V[j*p+k, :] * W[k*n+i, :]) = e_i
                # This is: for each i, sum_j (U[i*m+j, :] ⊙ (sum_k V[j*p+k, :] * W[k*n+i, :])) = e_i
                # Let's build the linear system for U.
                
                # Residual for U: R_U[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] - delta
                # For fixed V, W, this is linear in U.
                # The system is: for each (i,j), sum_k U[i*m+j, :] * (V[j*p+k, :] * W[k*n+i, :]) = 1 (for the right (i,j,k))
                
                # Actually, the tensor equation is: for all (i,j,k), sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
                # For fixed V, W, this is linear in U. The coefficient of U[i*m+j, c] is V[j*p+k, c] W[k*n+i, c]
                # So for each (i,j), we have: sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1 for each k
                # This gives p equations for each (i,j), but they must all equal 1.
                
                # Let's set up the system properly. For each (i,j), define vector b_{i,j} of length p:
                #   b_{i,j}[k] = 1
                # And matrix A_{i,j} of shape (p, rank):
                #   A_{i,j}[k, c] = V[j*p+k, c] * W[k*n+i, c]
                # Then we want: A_{i,j} @ U[i*m+j, :] = b_{i,j}
                
                # Solve least squares for each (i,j)
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, rank))
                        for k in range(p):
                            for c in range(rank):
                                A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        b = np.ones(p)
                        # Solve A @ x = b in least squares
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                            U_new[i*m+j, :] = sol
                        except:
                            U_new[i*m+j, :] = U[i*m+j, :]
                
                # Solve for V: fix U and W
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, rank))
                        for i in range(n):
                            for c in range(rank):
                                A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        b = np.ones(n)
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                            V_new[j*p+k, :] = sol
                        except:
                            V_new[j*p+k, :] = V[j*p+k, :]
                
                # Check convergence
                diff_U = np.linalg.norm(U_new - U)
                diff_V = np.linalg.norm(V_new - V)
                U = U_new
                V = V_new
                
                if diff_U < 1e-10 and diff_V < 1e-10:
                    converged = True
                    break
            
            # Verify the solution
            # Compute residual
            residual = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        residual += (val - 1)**2
            
            if residual < 1e-10:
                # Snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                if verify_exact(U_snap, V_snap, W_snap, n, m, p):
                    return U_snap, V_snap, W_snap
                
                # Try integers
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                if verify_exact(U_int, V_int, W_int, n, m, p):
                    return U_int, V_int, W_int
        
        return None

    def try_rank_original(rank, max_seeds=5):
        """Original Gauss-Newton approach as fallback."""
        size_U = N * rank
        size_V = M * rank
        size_W = P * rank
        total_size = size_U + size_V + size_W

        for seed in range(max_seeds):
            if time.time() > deadline - 5:
                return None

            np.random.seed(seed * 1000 + rank)
            
            x0 = np.random.randn(total_size)
            U0 = x0[:size_U].reshape(N, rank)
            V0 = x0[size_U:size_U + size_V].reshape(M, rank)
            W0 = x0[size_U + size_V:].reshape(P, rank)
            
            for c in range(rank):
                nU = np.linalg.norm(U0[:, c])
                nV = np.linalg.norm(V0[:, c])
                nW = np.linalg.norm(W0[:, c])
                if nU > 1e-10 and nV > 1e-10 and nW > 1e-10:
                    scale = (nU * nV * nW) ** (1.0/3.0)
                    U0[:, c] /= scale
                    V0[:, c] /= scale
                    W0[:, c] /= scale
            
            x = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])

            try:
                import scipy.optimize as opt
            except ImportError:
                return None

            def fun(x):
                U = x[:size_U].reshape(N, rank)
                V = x[size_U:size_U + size_V].reshape(M, rank)
                W = x[size_U + size_V:].reshape(P, rank)
                residuals = np.zeros(n * m * p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(rank):
                                val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            residuals[idx] = val - 1.0
                            idx += 1
                return residuals

            try:
                res = opt.least_squares(fun, x, method='lm', max_nfev=300)
                err = np.sum(res.fun**2)
                
                if err < 1e-12:
                    U_opt = res.x[:size_U].reshape(N, rank)
                    V_opt = res.x[size_U:size_U + size_V].reshape(M, rank)
                    W_opt = res.x[size_U + size_V:].reshape(P, rank)
                    
                    U_snap = np.round(U_opt * 2) / 2
                    V_snap = np.round(V_opt * 2) / 2
                    W_snap = np.round(W_opt * 2) / 2
                    
                    if verify_exact(U_snap, V_snap, W_snap, n, m, p):
                        return U_snap, V_snap, W_snap
                        
                    U_int = np.round(U_opt)
                    V_int = np.round(V_opt)
                    W_int = np.round(W_opt)
                    if verify_exact(U_int, V_int, W_int, n, m, p):
                        return U_int, V_int, W_int
                        
            except Exception:
                pass

        return None

    # Search for lower ranks using structured approach first, then fallback
    for rank in range(18, best_rank):
        if time.time() > deadline - 10:
            break
            
        # Try structured approach first (exploits rectangular asymmetry)
        result = try_rank_structured(rank, max_patterns=50)
        if result:
            U_final, V_final, W_final = result
            if rank < best_rank:
                best_rank = rank
                best_result = {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
                if rank <= 19:
                    break
            continue
        
        # Fallback to original approach
        seeds = 5 if rank < 25 else 3
        result = try_rank_original(rank, max_seeds=seeds)
        if result:
            U_final, V_final, W_final = result
            if rank < best_rank:
                best_rank = rank
                best_result = {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
                if rank <= 19:
                    break

    return best_result
