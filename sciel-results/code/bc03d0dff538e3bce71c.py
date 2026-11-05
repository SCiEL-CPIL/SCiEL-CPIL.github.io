import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin before 150s kill

    n, m, p = DIMENSIONS
    NM, MP, PN = n * m, m * p, p * n
    rank_schoolbook = n * m * p  # 72

    # Build the correct matrix multiplication tensor T
    # T[a, b, c] = 1 if a=(i,j), b=(j',k), c=(i',k') and i=i', j=j', k=k'
    # With W index convention: W[k*n+i, r]
    T = np.zeros((NM, MP, PN), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    T_flat = T.flatten()

    def check_exact(U, V, W, rank, tol=1e-8):
        """
        U: (NM, rank)
        V: (MP, rank)
        W: (PN, rank)
        Check that sum_r U[a,r] V[b,r] W[c,r] equals T[a,b,c] exactly.
        """
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_rec.flatten(), T_flat, atol=tol)

    def get_schoolbook_solution():
        rank = rank_schoolbook
        U = np.zeros((NM, rank))
        V = np.zeros((MP, rank))
        W = np.zeros((PN, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def als_decompose(U, V, W, rank, max_iter=100, tol=1e-12, time_limit=10.0):
        """
        Alternating Least Squares for tensor decomposition.
        Alternately fix two factors and solve for the third via linear least squares.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        start = time.time()
        
        for it in range(max_iter):
            if time.time() - start > time_limit:
                break
            
            # Solve for U: fix V and W
            # T_rec[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For each a: T[a,:,:] = U[a,:] @ (V * W) where V*W is outer product per column
            # Actually: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # = sum_r U[a,r] * (V[b,r] * W[c,r])
            # So for fixed a: T[a,:,:] flattened = U[a,:] @ M_a where M_a[b,c] = sum_r V[b,r]*W[c,r]
            # But that's not quite right. Let me think again.
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # Let's define for each a: the equation is
            # T[a,b,c] for all b,c = U[a,0]*(V[:,0]*W[:,0]) + U[a,1]*(V[:,1]*W[:,1]) + ...
            # So T[a] (shape MP*PN) = U[a] (shape rank) @ H_a (shape rank x MP*PN)
            # where H_a[r, b*PN+c] = V[b,r] * W[c,r]
            
            # Build H matrix: H[r, b*PN+c] = V[b,r] * W[c,r]
            # Shape: (rank, MP*PN)
            HW = np.einsum('br,cr->bcr', V, W).reshape(rank, MP * PN)
            # Now T_flat_a = T[a, :, :].flatten() for each a
            # T_all = T.reshape(NM, MP*PN)
            T_reshaped = T.reshape(NM, MP * PN)
            
            # Solve: T_reshaped[a] = U[a] @ HW
            # This is: T_reshaped = U @ HW
            # So U = T_reshaped @ HW^+ (pseudo-inverse)
            # Or solve U @ HW = T_reshaped for U
            try:
                U_new, _, _, _ = np.linalg.lstsq(HW.T, T_reshaped.T, rcond=None)
                U = U_new.T
            except:
                pass
            
            # Solve for V: fix U and W
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed b: T[:,:,b] = V[b,:] @ (U * W)
            # U*W shape: (rank, NM*PN) where (U*W)[r, a*PN+c] = U[a,r]*W[c,r]
            UW = np.einsum('ar,cr->acr', U, W).reshape(rank, NM * PN)
            T_reshaped_v = T.reshape(MP, NM * PN).T  # (NM*PN, MP)
            try:
                V_new, _, _, _ = np.linalg.lstsq(UW.T, T_reshaped_v, rcond=None)
                V = V_new.T
            except:
                pass
            
            # Solve for W: fix U and V
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed c: T[:,:,c] = W[c,:] @ (U * V)
            UV = np.einsum('ar,br->abr', U, V).reshape(rank, NM * MP)
            T_reshaped_w = T.reshape(PN, NM * MP).T  # (NM*MP, PN)
            try:
                W_new, _, _, _ = np.linalg.lstsq(UV.T, T_reshaped_w, rcond=None)
                W = W_new.T
            except:
                pass
            
            # Check convergence
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.sum((T_rec - T) ** 2)
            if err < tol:
                break
        
        return U, V, W, err

    def try_rank_als(rank, num_restarts=10, time_per_restart=8.0):
        """
        Try to find a decomposition of given rank using ALS.
        """
        for seed in range(num_restarts):
            if time.time() > deadline - 15:
                return None
            
            rng = np.random.RandomState(seed * 7919 + 42)
            
            # Initialize with random values
            U = rng.randn(NM, rank) * 0.5
            V = rng.randn(MP, rank) * 0.5
            W = rng.randn(PN, rank) * 0.5
            
            # Run ALS
            U_opt, V_opt, W_opt, err = als_decompose(U, V, W, rank, max_iter=50, time_limit=time_per_restart)
            
            if err < 1e-8:
                # Check exactness
                if check_exact(U_opt, V_opt, W_opt, rank):
                    return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
                
                # Try rounding to half-integers
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                if check_exact(U_round, V_round, W_round, rank):
                    return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                
                # Try rounding to integers
                U_int = np.round(U_opt)
                V_int = np.round(V_opt)
                W_int = np.round(W_opt)
                if check_exact(U_int, V_int, W_int, rank):
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
        
        return None

    def try_rank_optimize(rank, num_restarts=5, time_per_restart=8.0):
        """
        Try to find a decomposition of given rank using L-BFGS-B optimization.
        """
        for seed in range(num_restarts):
            if time.time() > deadline - 15:
                return None
            
            rng = np.random.RandomState(seed * 104729 + 13)
            
            # Initialize
            U = rng.randn(NM, rank) * 0.1
            V = rng.randn(MP, rank) * 0.1
            W = rng.randn(PN, rank) * 0.1
            
            try:
                from scipy.optimize import minimize
                
                def objective(x):
                    U_x = x[:NM*rank].reshape(NM, rank)
                    V_x = x[NM*rank:NM*rank+MP*rank].reshape(MP, rank)
                    W_x = x[NM*rank+MP*rank:].reshape(PN, rank)
                    T_rec = np.einsum('ar,br,cr->abc', U_x, V_x, W_x)
                    err = T_rec.flatten() - T_flat
                    return np.sum(err**2)
                
                x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                # Run optimization with time limit
                res = minimize(objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 30, 'ftol': 1e-15, 'gtol': 1e-15})
                
                if res.fun < 1e-8:
                    x_opt = res.x
                    U_opt = x_opt[:NM*rank].reshape(NM, rank)
                    V_opt = x_opt[NM*rank:NM*rank+MP*rank].reshape(MP, rank)
                    W_opt = x_opt[NM*rank+MP*rank:].reshape(PN, rank)
                    
                    # Check exactness
                    if check_exact(U_opt, V_opt, W_opt, rank):
                        return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
                    
                    # Try rounding to half-integers
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    if check_exact(U_round, V_round, W_round, rank):
                        return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                    
                    # Try rounding to integers
                    U_int = np.round(U_opt)
                    V_int = np.round(V_opt)
                    W_int = np.round(W_opt)
                    if check_exact(U_int, V_int, W_int, rank):
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                        
            except Exception:
                pass
        
        return None

    # Search for lower rank
    best_result = None
    
    # Lower bound: max(n*p, m*p, n*m) = max(18, 24, 12) = 24
    # Try ranks from reasonable lower bound up to schoolbook - 1
    # Be strategic: try a few key ranks
    ranks_to_try = []
    # Start from 60 down to 50, then some lower ones
    for rank in range(60, 49, -1):
        ranks_to_try.append(rank)
    for rank in range(45, 39, -1):
        ranks_to_try.append(rank)
    for rank in range(35, 29, -1):
        ranks_to_try.append(rank)
    
    for rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
        
        # Try ALS first (usually faster)
        result = try_rank_als(rank, num_restarts=3, time_per_restart=5.0)
        if result is not None:
            best_result = result
            break
        
        # Try optimization if time allows
        if time.time() > deadline - 25:
            continue
        result = try_rank_optimize(rank, num_restarts=3, time_per_restart=5.0)
        if result is not None:
            best_result = result
            break

    if best_result is None:
        best_result = get_schoolbook_solution()

    return best_result
