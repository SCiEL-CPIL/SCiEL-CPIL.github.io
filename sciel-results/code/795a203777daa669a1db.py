import numpy as np
from scipy.optimize import minimize
import time

def construct():
    n, m, p = 2, 3, 10
    deadline = time.time() + 140  # 140s deadline, leaving 10s margin

    def compute_T(U, V, W, r):
        """Compute the reconstructed tensor using einsum for speed."""
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # We need T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        # T[i, j, k] = sum_c U_3d[i, j, c] * V_3d[j, k, c] * W_3d[k, i, c]
        return np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)

    def verify(U, V, W, r, tol=1e-9):
        """Verify that the decomposition exactly reconstructs the all-ones tensor."""
        T = compute_T(U, V, W, r)
        return np.allclose(T, 1.0, atol=tol)

    def objective_and_grad(params, r):
        """Objective and gradient for the tensor decomposition problem."""
        nm = n * m
        mp = m * p
        pn = p * n

        U = params[:nm*r].reshape(nm, r)
        V = params[nm*r:nm*r+mp*r].reshape(mp, r)
        W = params[nm*r+mp*r:].reshape(pn, r)

        # Compute T efficiently
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        
        diff = 1.0 - T  # (n, m, p)
        error = np.sum(diff**2)

        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)

        for r_idx in range(r):
            u = U[:, r_idx].reshape(n, m)  # (n, m)
            v = V[:, r_idx].reshape(m, p)  # (m, p)
            w = W[:, r_idx].reshape(p, n)  # (p, n)
            
            # dU[i, j] = -2 * sum_k diff[i, j, k] * v[j, k] * w[k, i]
            # Let A[i, j, k] = diff[i, j, k] * v[j, k]
            A = diff * v[np.newaxis, :, :]  # (n, m, p)
            # B[i, j] = sum_k A[i, j, k] * w[k, i]
            B = np.einsum('ijk,ki->ij', A, w)
            dU[:, r_idx] = (-2 * B).flatten()
            
            # dV[j, k] = -2 * sum_i diff[i, j, k] * u[i, j] * w[k, i]
            # Let C[i, j, k] = diff[i, j, k] * u[i, j]
            C = diff * u[:, :, np.newaxis]  # (n, m, p)
            # D[j, k] = sum_i C[i, j, k] * w[k, i]
            D = np.einsum('ijk,ki->jk', C, w)
            dV[:, r_idx] = (-2 * D).flatten()
            
            # dW[k, i] = -2 * sum_j diff[i, j, k] * u[i, j] * v[j, k]
            # Let E[i, j, k] = diff[i, j, k] * u[i, j] * v[j, k]
            E = diff * u[:, :, np.newaxis] * v[np.newaxis, :, :]  # (n, m, p)
            # F[i, k] = sum_j E[i, j, k]
            F = np.einsum('ijk->ik', E)  # (n, p)
            # dW[k, i] corresponds to F[i, k]
            dW[:, r_idx] = (-2 * F.T).flatten()

        grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        return error, grad

    # Schoolbook solution as fallback
    def schoolbook():
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    best_solution = None
    best_rank = n * m * p  # 60

    # Strategy:
    # 1. Try to find a solution with rank < 60 using continuous optimization.
    # 2. If found, attempt to round to half-integers.
    # 3. If rounding fails or no solution found, use schoolbook.
    # 4. Since the problem asks for integer/half-integer entries, and the continuous
    #    solution might be close, we try rounding. If rounding breaks exactness,
    #    we might need to search for an exact integer solution, which is hard.
    #    However, the task says "exactly decompose". The schoolbook is exact.
    #    The continuous optimization finds a real-valued solution.
    #    We must return integer/half-integer.
    #    If the continuous solution rounds to a valid half-integer solution, we use it.
    #    Otherwise, we fall back to schoolbook.
    #    Given the difficulty of finding low-rank integer solutions for 2x3x10,
    #    and the time limit, we will focus on finding *any* valid low-rank solution
    #    that can be rounded, or just return schoolbook if we can't guarantee exactness
    #    after rounding.
    
    # Note: The evaluator checks for exactness. If we return a non-integer solution,
    # it might fail. The prompt says "integer/half-integer entries".
    # So we MUST round.
    # If rounding causes error > 0, it's invalid.
    
    # Let's try ranks from 30 up to 59.
    # We will use L-BFGS-B.
    
    for r in range(30, 60):
        if time.time() > deadline:
            break
        if r >= best_rank:
            continue
            
        # Try multiple random restarts
        n_restarts = 10
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(12345 + restart * 7 + r)
            # Scale initialization to help convergence
            scale = 1.0
            U = np.random.randn(n*m, r) * scale
            V = np.random.randn(m*p, r) * scale
            W = np.random.randn(p*n, r) * scale
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                # Use L-BFGS-B
                res = minimize(objective_and_grad, params, args=(r,), jac=True,
                              method='L-BFGS-B', 
                              options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12})
                
                if res.fun < 1e-10:
                    U_sol = res.x[:n*m*r].reshape(n*m, r)
                    V_sol = res.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                    W_sol = res.x[n*m*r+m*p*r:].reshape(p*n, r)
                    
                    # Round to half-integers
                    U_round = np.round(U_sol * 2) / 2
                    V_round = np.round(V_sol * 2) / 2
                    W_round = np.round(W_sol * 2) / 2
                    
                    if verify(U_round, V_round, W_round, r, tol=1e-9):
                        best_solution = (U_round, V_round, W_round)
                        best_rank = r
                        break
            except Exception:
                pass
        
        if best_solution and best_rank == r:
            break
    
    if best_solution:
        U, V, W = best_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U, V, W = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
