import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 4, 4, 5
    nm, mp, pn = n*m, m*p, p*n  # 16, 20, 20
    
    # Create the sparse tensor indices
    idx_a = []
    idx_b = []
    idx_c = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                idx_a.append(a)
                idx_b.append(b)
                idx_c.append(c)
    
    idx_a = np.array(idx_a)
    idx_b = np.array(idx_b)
    idx_c = np.array(idx_c)
    
    # Target tensor
    T_target = np.zeros((nm, mp, pn))
    T_target[idx_a, idx_b, idx_c] = 1.0
    
    # Try to find a low rank decomposition
    # Known rank for 4x4x4 is 49. For 4x4x5, it should be similar.
    # Let's try rank 50.
    
    best_rank = n * m * p
    best_factors = None
    
    # Fallback schoolbook
    def get_schoolbook():
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][column] = 1
                    V_sb[j * p + k][column] = 1
                    W_sb[k * n + i][column] = 1
                    column += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Optimization function
    def loss_and_grad(params, rank):
        U = params[:nm*rank].reshape(nm, rank)
        V = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
        W = params[nm*rank+mp*rank:].reshape(pn, rank)
        
        # Compute T_full
        T_full = np.einsum('ar,br,cr->abc', U, V, W)
        residual = T_full - T_target
        loss = np.sum(residual**2)
        
        # Gradients
        dU = 2 * np.einsum('abc,br,cr->ar', residual, V, W)
        dV = 2 * np.einsum('abc,ar,cr->br', residual, U, W)
        dW = 2 * np.einsum('abc,ar,br->cr', residual, U, V)
        
        grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        return loss, grad

    # Try a few ranks
    for rank in [50, 55, 60]:
        np.random.seed(rank)
        # Initialize with small random values
        init_U = np.random.randn(nm, rank) * 0.1
        init_V = np.random.randn(mp, rank) * 0.1
        init_W = np.random.randn(pn, rank) * 0.1
        
        x0 = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
        
        # Use L-BFGS-B or similar
        # We need to wrap loss_and_grad for scipy
        def func(x):
            l, g = loss_and_grad(x, rank)
            return l, g
            
        try:
            res = minimize(func, x0, jac=True, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
            if res.success or res.fun < 1e-6:
                # Extract factors
                U = res.x[:nm*rank].reshape(nm, rank)
                V = res.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                W = res.x[nm*rank+mp*rank:].reshape(pn, rank)
                
                # Snap to half-integers
                U_half = np.round(U * 2) / 2
                V_half = np.round(V * 2) / 2
                W_half = np.round(W * 2) / 2
                
                # Verify
                T_half = np.einsum('ar,br,cr->abc', U_half, V_half, W_half)
                error = np.sum((T_half - T_target)**2)
                
                if error < 1e-10:
                    return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
        except Exception as e:
            pass
            
    # Fallback
    return get_schoolbook()
