"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    
    def get_target_indices():
        indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    indices.append((u_idx, v_idx, w_idx))
        return indices

    target_indices = get_target_indices()
    
    def solve_rank(R):
        np.random.seed(42)
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        def loss_and_grad(params):
            U = params[:size_u*R].reshape(size_u, R)
            V = params[size_u*R:(size_u+size_v)*R].reshape(size_v, R)
            W = params[(size_u+size_v)*R:].reshape(size_w, R)
            
            loss = 0.0
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for (u_idx, v_idx, w_idx) in target_indices:
                prod_VW = V[v_idx] * W[w_idx]
                prod_UW = U[u_idx] * W[w_idx]
                prod_UV = U[u_idx] * V[v_idx]
                
                val = np.sum(prod_UV * W[w_idx])
                err = val - 1.0
                
                loss += err ** 2
                
                dU[u_idx] += 2 * err * prod_VW
                dV[v_idx] += 2 * err * prod_UW
                dW[w_idx] += 2 * err * prod_UV
            
            grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
            return loss, grad

        x0 = np.random.randn((size_u + size_v + size_w) * R) * 0.5
        res = minimize(loss_and_grad, x0, method='L-BFGS-B', jac=True, options={'maxiter': 2000, 'ftol': 1e-12})
        
        if res.success and res.fun < 1e-8:
            U = res.x[:size_u*R].reshape(size_u, R)
            V = res.x[size_u*R:(size_u+size_v)*R].reshape(size_v, R)
            W = res.x[(size_u+size_v)*R:].reshape(size_w, R)
            
            T_comp = np.zeros((n*m, m*p, p*n))
            for r in range(R):
                T_comp += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
            
            max_err = 0
            for (u_idx, v_idx, w_idx) in target_indices:
                max_err = max(max_err, abs(T_comp[u_idx, v_idx, w_idx] - 1.0))
            
            for _ in range(50):
                u = np.random.randint(n*m)
                v = np.random.randint(m*p)
                w = np.random.randint(p*n)
                if (u, v, w) not in target_indices:
                    max_err = max(max_err, abs(T_comp[u, v, w]))
            
            if max_err < 1e-6:
                best_scale = None
                best_err = float('inf')
                for scale in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                    sU = U * scale
                    sV = V * scale
                    sW = W * scale
                    rU = np.round(sU)
                    rV = np.round(sV)
                    rW = np.round(sW)
                    err = max(np.max(np.abs(rU - sU)), np.max(np.abs(rV - sV)), np.max(np.abs(rW - sW)))
                    if err < 1e-4 and err < best_err:
                        best_err = err
                        best_scale = scale
                        best_U = rU
                        best_V = rV
                        best_W = rW
                
                if best_scale:
                    T_int = np.zeros((n*m, m*p, p*n))
                    for r in range(R):
                        T_int += np.einsum('i,j,k->ijk', best_U[:,r], best_V[:,r], best_W[:,r])
                    
                    valid = True
                    for (u_idx, v_idx, w_idx) in target_indices:
                        if abs(T_int[u_idx, v_idx, w_idx] - 1.0) > 1e-9:
                            valid = False
                            break
                    
                    if valid:
                        return {
                            "U": [[int(x) for x in row] for row in best_U],
                            "V": [[int(x) for x in row] for row in best_V],
                            "W": [[int(x) for x in row] for row in best_W]
                        }
        return None

    for R in [8, 9, 10]:
        res = solve_rank(R)
        if res:
            return res
            
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
