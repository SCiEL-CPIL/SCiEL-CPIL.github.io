import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Vectorized error function
    def tensor_error(params, rank):
        U = params[:rank * n * m].reshape(rank, n, m)
        V = params[rank * n * m : rank * n * m + rank * m * p].reshape(rank, m, p)
        W = params[rank * n * m + rank * m * p :].reshape(rank, p, n)
        
        # T[i,j,k] = sum_c U[c,i,j] * V[c,j,k] * W[c,k,i]
        # Using einsum: 'cij,cjk,cki->ijk'
        T = np.einsum('cij,cjk,cki->ijk', U, V, W)
        error = np.sum((T - 1.0) ** 2)
        return error

    def tensor_error_grad(params, rank):
        U = params[:rank * n * m].reshape(rank, n, m)
        V = params[rank * n * m : rank * n * m + rank * m * p].reshape(rank, m, p)
        W = params[rank * n * m + rank * m * p :].reshape(rank, p, n)
        
        T = np.einsum('cij,cjk,cki->ijk', U, V, W)
        diff = 2.0 * (T - 1.0)
        
        # Gradient with respect to U[c,i,j] is sum_{k} diff[i,j,k] * V[c,j,k] * W[c,k,i]
        # grad_U[c,i,j] = einsum('ijk,jk,ki->ij', diff, V[c], W[c]) ... wait, V and W depend on c.
        # grad_U[c] = einsum('ijk,jk,ki->ij', diff, V[c], W[c])
        # We can do this for all c at once?
        # diff is (n,m,p). V is (rank,m,p). W is (rank,p,n).
        # grad_U[c,i,j] = sum_k diff[i,j,k] * V[c,j,k] * W[c,k,i]
        # This is a batched operation.
        
        # Let's compute grad_U, grad_V, grad_W
        # grad_U shape (rank, n, m)
        # grad_U[c,i,j] = np.einsum('ijk,jk,ki->ij', diff, V[c], W[c])
        
        # Using einsum for batched:
        # 'cij' is U. We want grad_U[c,i,j].
        # Term is diff[i,j,k] * V[c,j,k] * W[c,k,i].
        # Sum over k.
        # grad_U = np.einsum('ijk,cjk,cki->cij', diff, V, W)
        
        # grad_V[c,j,k] = sum_i diff[i,j,k] * U[c,i,j] * W[c,k,i]
        # grad_V = np.einsum('ijk,cij,cki->cjk', diff, U, W)
        
        # grad_W[c,k,i] = sum_j diff[i,j,k] * U[c,i,j] * V[c,j,k]
        # grad_W = np.einsum('ijk,cij,cjk->cki', diff, U, V)
        
        grad_U = np.einsum('ijk,cjk,cki->cij', diff, V, W)
        grad_V = np.einsum('ijk,cij,cki->cjk', diff, U, W)
        grad_W = np.einsum('ijk,cij,cjk->cki', diff, U, V)
        
        grad = np.concatenate([
            grad_U.flatten(),
            grad_V.flatten(),
            grad_W.flatten()
        ])
        return grad

    best_result = None
    best_rank = n * m * p  # fallback to schoolbook

    # The rank of 3x3x3 is 23. 3x3x6 is larger.
    # Let's try ranks from 20 to 40.
    # To save time, we can start from a higher rank and go down, or just try a few promising ones.
    # Random initialization is key.
    
    for rank in range(20, 36):
        found = False
        for attempt in range(10):
            np.random.seed(42 + attempt * 100 + rank)
            
            # Sparse initialization: mostly zeros, some ones
            # This helps avoid local minima in the interior
            U_init = np.zeros((rank, n, m))
            V_init = np.zeros((rank, m, p))
            W_init = np.zeros((rank, p, n))
            
            # Randomly set some entries to 1 or -1
            for c in range(rank):
                # Pick a few non-zeros
                num_nonzero = max(1, n*m // 3)
                idx = np.random.choice(n*m, num_nonzero, replace=False)
                U_init[c, idx // m, idx % m] = np.random.choice([-1, 1], num_nonzero)
                
                num_nonzero = max(1, m*p // 3)
                idx = np.random.choice(m*p, num_nonzero, replace=False)
                V_init[c, idx // p, idx % p] = np.random.choice([-1, 1], num_nonzero)
                
                num_nonzero = max(1, p*n // 3)
                idx = np.random.choice(p*n, num_nonzero, replace=False)
                W_init[c, idx // n, idx % n] = np.random.choice([-1, 1], num_nonzero)

            params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            result = minimize(
                tensor_error, params0, args=(rank,),
                jac=tensor_error_grad,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < 1e-8:
                U_sol = result.x[:rank * n * m].reshape(n * m, rank)
                V_sol = result.x[rank * n * m : rank * n * m + rank * m * p].reshape(m * p, rank)
                W_sol = result.x[rank * n * m + rank * m * p :].reshape(p * n, rank)
                
                # Verify
                max_err = 0
                # Vectorized verify
                U_v = U_sol.T # (rank, n, m)
                V_v = V_sol.T # (rank, m, p)
                W_v = W_sol.T # (rank, p, n)
                T_v = np.einsum('cij,cjk,cki->ijk', U_v, V_v, W_v)
                max_err = np.max(np.abs(T_v - 1.0))
                
                if max_err < 1e-6:
                    best_rank = rank
                    best_result = (U_sol, V_sol, W_sol)
                    found = True
                    break
        
        if found:
            break

    if best_result is not None:
        U_sol, V_sol, W_sol = best_result
        U_list = [[float(x) for x in row] for row in U_sol]
        V_list = [[float(x) for x in row] for row in V_sol]
        W_list = [[float(x) for x in row] for row in W_sol]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
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
