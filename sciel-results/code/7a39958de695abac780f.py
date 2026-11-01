"""Direct construction: attempt to find a rank-minimal tensor decomposition for 2x3 x 3x7 matrix mult."""
import time
import numpy as np
from scipy.optimize import minimize, fsolve

DIMENSIONS = [2, 3, 7]
DEADLINE = 145.0
START_TIME = time.time()

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the standard matrix multiplication tensor T[a,b,c]
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition (rank 42)
    U_school = np.zeros((nm, 42))
    V_school = np.zeros((mp, 42))
    W_school = np.zeros((pn, 42))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1.0
                V_school[b, col] = 1.0
                W_school[c, col] = 1.0
                col += 1
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = 42
    
    # Try to find lower-rank decompositions using optimization
    # For a given rank r, we need U (nm x r), V (mp x r), W (pn x r)
    # such that sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t] = T
    
    # Flattened parameter vector: U.ravel() + V.ravel() + W.ravel()
    def tensor_residual(params, r):
        """Compute the residual T - sum_t U_t ⊗ V_t ⊗ W_t, flattened."""
        U = params[:nm*r].reshape(nm, r)
        V = params[nm*r:nm*r+mp*r].reshape(mp, r)
        W = params[nm*r+mp*r:nm*r+mp*r+pn*r].reshape(pn, r)
        
        # Compute the tensor approximation
        T_approx = np.zeros((nm, mp, pn), dtype=float)
        for t in range(r):
            T_approx += np.einsum('a,b,c->abc', U[:, t], V[:, t], W[:, t])
        
        return (T - T_approx).ravel()
    
    def tensor_residual_and_jac(params, r):
        """Compute residual and Jacobian for Newton's method."""
        U = params[:nm*r].reshape(nm, r)
        V = params[nm*r:nm*r+mp*r].reshape(mp, r)
        W = params[nm*r+mp*r:nm*r+mp*r+pn*r].reshape(pn, r)
        
        T_approx = np.zeros((nm, mp, pn), dtype=float)
        for t in range(r):
            T_approx += np.einsum('a,b,c->abc', U[:, t], V[:, t], W[:, t])
        
        residual = (T - T_approx).ravel()
        
        # Numerical Jacobian (expensive but necessary for accuracy)
        jac = np.zeros((nm*mp*pn, 3*r*(nm+mp+pn)))
        eps = 1e-8
        for i in range(jac.shape[1]):
            params_pert = params.copy()
            params_pert[i] += eps
            res_pert = tensor_residual(params_pert, r)
            jac[:, i] = (res_pert - residual) / eps
        
        return residual, jac
    
    # Try different target ranks from high to low
    for target_rank in range(40, 20, -1):
        if time.time() - START_TIME > DEADLINE - 15:
            break
        
        # Multiple random initializations
        for attempt in range(5):
            if time.time() - START_TIME > DEADLINE - 10:
                break
            
            np.random.seed(42 + target_rank * 100 + attempt)
            U_rand = np.random.randn(nm, target_rank) * 0.1
            V_rand = np.random.randn(mp, target_rank) * 0.1
            W_rand = np.random.randn(pn, target_rank) * 0.1
            
            params = np.concatenate([U_rand.ravel(), V_rand.ravel(), W_rand.ravel()])
            
            # Use fsolve to find a solution
            try:
                solution, info, ier, msg = fsolve(
                    lambda x: tensor_residual(x, target_rank),
                    params,
                    full_output=True,
                    xtol=1e-12,
                    maxfev=5000
                )
                
                if ier == 1:
                    # Check the residual
                    res = tensor_residual(solution, target_rank)
                    error = np.linalg.norm(res)
                    
                    if error < 1e-10:
                        # Valid decomposition found!
                        U_sol = solution[:nm*target_rank].reshape(nm, target_rank)
                        V_sol = solution[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                        W_sol = solution[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                        
                        # Verify
                        T_check = np.zeros((nm, mp, pn), dtype=float)
                        for t in range(target_rank):
                            T_check += np.einsum('a,b,c->abc', U_sol[:, t], V_sol[:, t], W_sol[:, t])
                        
                        if np.allclose(T_check, T, atol=1e-10):
                            best_result = {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
                            best_rank = target_rank
                            print(f"Found rank {target_rank} decomposition!")
                            break
            except Exception as e:
                pass
        
        if best_rank < target_rank:
            break
    
    return best_result
