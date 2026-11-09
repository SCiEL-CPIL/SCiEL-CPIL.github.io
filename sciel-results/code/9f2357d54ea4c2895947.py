import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = 150.0 - 10.0  # 10-second safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T as a 3D array
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W form a valid rank-r decomposition."""
        r = U.shape[1]
        T_rec = np.zeros((nm, mp, pn))
        for r_idx in range(r):
            T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        return np.allclose(T_rec, T, atol=tol)
    
    def round_to_half_integer(x):
        """Round to nearest integer or half-integer."""
        return np.round(x * 2) / 2.0

    def optimize_rank_levenberg(r, max_restarts=50):
        """
        Try to find a rank-r decomposition using Levenberg-Marquardt optimization.
        """
        if time.time() > deadline:
            return None
        
        # Flatten U, V, W into a single vector for scipy
        # Size: r * (nm + mp + pn)
        dim = r * (nm + mp + pn)
        
        def unpack(x):
            U = x[:r*nm].reshape(nm, r)
            V = x[r*nm:r*(nm+mp)].reshape(mp, r)
            W = x[r*(nm+mp):].reshape(pn, r)
            return U, V, W
        
        def loss_and_jacobian(x):
            """
            Compute loss and Jacobian for Levenberg-Marquardt.
            Returns: (loss, jacobian_matrix)
            """
            U, V, W = unpack(x)
            
            # Compute tensor reconstruction
            T_rec = np.zeros((nm, mp, pn))
            for r_idx in range(r):
                T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
            
            error = T_rec - T
            loss = np.sum(error**2)
            
            # Add small penalty to encourage integer/half-integer values
            penalty = 0.0
            for mat in [U, V, W]:
                for val in mat:
                    rounded = np.round(val * 2)
                    penalty += (val * 2 - rounded)**2
            
            total_loss = loss + 0.01 * penalty
            
            # Compute Jacobian using automatic differentiation approach
            # For Levenberg-Marquardt, we need the Jacobian of the residual
            # The residual is the flattened error tensor
            residual = error.flatten()
            
            # Compute Jacobian numerically using finite differences
            # This is expensive but works for small problems
            jacobian = np.zeros((residual.size, dim))
            
            # Use central differences for better accuracy
            h = 1e-7
            for i in range(dim):
                x_plus = x.copy()
                x_plus[i] += h
                U_plus, V_plus, W_plus = unpack(x_plus)
                T_rec_plus = np.zeros((nm, mp, pn))
                for r_idx in range(r):
                    T_rec_plus += np.einsum('a,b,c->abc', U_plus[:, r_idx], V_plus[:, r_idx], W_plus[:, r_idx])
                error_plus = (T_rec_plus - T).flatten()
                jacobian[:, i] = (error_plus - residual) / h
            
            # Add penalty gradient to Jacobian (approximate)
            # For the penalty term, we add its gradient
            grad_U_pen = 0.02 * (2 * U - np.round(2 * U))
            grad_V_pen = 0.02 * (2 * V - np.round(2 * V))
            grad_W_pen = 0.02 * (2 * W - np.round(2 * W))
            
            # For simplicity, we'll just use the main Jacobian
            # The penalty is small enough that it shouldn't affect convergence much
            
            return total_loss, jacobian
        
        best_result = None
        best_loss = float('inf')
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + restart * 13)
            
            # Initialize with small random values
            x0 = np.random.randn(dim) * 0.1
            
            # Try to optimize using Levenberg-Marquardt
            try:
                res = minimize(loss_and_jacobian, x0, method='lm', 
                               options={'maxiter': 50, 'ftol': 1e-10})
            except Exception:
                continue
            
            if res.fun < best_loss:
                U, V, W = unpack(res.x)
                
                # Round to nearest half-integer
                U_rounded = round_to_half_integer(U)
                V_rounded = round_to_half_integer(V)
                W_rounded = round_to_half_integer(W)
                
                # Check if rounded solution is exact
                if check_decomposition(U_rounded, V_rounded, W_rounded):
                    return (U_rounded, V_rounded, W_rounded)
                
                best_loss = res.fun
                best_result = (U_rounded, V_rounded, W_rounded)
        
        # If we didn't find an exact solution, return None
        return None

    def optimize_rank_l_bfgs(r, max_restarts=50):
        """
        Try to find a rank-r decomposition using L-BFGS-B.
        """
        if time.time() > deadline:
            return None
        
        # Flatten U, V, W into a single vector for scipy
        # Size: r * (nm + mp + pn)
        dim = r * (nm + mp + pn)
        
        def unpack(x):
            U = x[:r*nm].reshape(nm, r)
            V = x[r*nm:r*(nm+mp)].reshape(mp, r)
            W = x[r*(nm+mp):].reshape(pn, r)
            return U, V, W
        
        def loss_and_grad(x):
            U, V, W = unpack(x)
            
            # Compute tensor reconstruction
            T_rec = np.zeros((nm, mp, pn))
            for r_idx in range(r):
                T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
            
            error = T_rec - T
            loss = np.sum(error**2)
            
            # Add penalty to encourage integer/half-integer values
            penalty = 0.0
            for mat in [U, V, W]:
                for val in mat:
                    rounded = np.round(val * 2)
                    penalty += (val * 2 - rounded)**2
            
            total_loss = loss + 0.1 * penalty
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r_idx in range(r):
                # Gradient of loss w.r.t U[:, r_idx]
                temp = error * np.einsum('b,c->bc', V[:, r_idx], W[:, r_idx])[None, :, :]
                grad_U[:, r_idx] = np.sum(temp, axis=(1, 2))
                
                temp = error * np.einsum('a,c->ac', U[:, r_idx], W[:, r_idx])[None, :, :]
                grad_V[:, r_idx] = np.sum(temp, axis=(0, 2))
                
                temp = error * np.einsum('a,b->ab', U[:, r_idx], V[:, r_idx])[None, :, :]
                grad_W[:, r_idx] = np.sum(temp, axis=(0, 1))
            
            # Gradient of penalty
            grad_U_pen = 0.4 * (2 * U - np.round(2 * U))
            grad_V_pen = 0.4 * (2 * V - np.round(2 * V))
            grad_W_pen = 0.4 * (2 * W - np.round(2 * W))
            
            grad_U += grad_U_pen
            grad_V += grad_V_pen
            grad_W += grad_W_pen
            
            # Flatten gradient
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            return total_loss, grad
        
        best_result = None
        best_loss = float('inf')
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + restart * 13)
            
            # Initialize with small random values
            x0 = np.random.randn(dim) * 0.1
            
            # Try to optimize
            try:
                res = minimize(loss_and_grad, x0, method='L-BFGS-B', jac=True, 
                               options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            except Exception:
                continue
            
            if res.success or res.fun < best_loss:
                U, V, W = unpack(res.x)
                
                # Round to nearest half-integer
                U_rounded = round_to_half_integer(U)
                V_rounded = round_to_half_integer(V)
                W_rounded = round_to_half_integer(W)
                
                # Check if rounded solution is exact
                if check_decomposition(U_rounded, V_rounded, W_rounded):
                    return (U_rounded, V_rounded, W_rounded)
                
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_result = (U_rounded, V_rounded, W_rounded)
        
        # If we didn't find an exact solution, return None
        return None

    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: max(nm, mp, pn) = 21
    # Known lower bounds for matrix multiplication are higher.
    # Let's try 30 to 63.
    
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # Try ranks in increasing order? Or decreasing?
    # We want the MINIMAL rank. So we should try low ranks first.
    # But low ranks are harder to find.
    # Let's try a range.
    
    ranks_to_try = list(range(30, 64))
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
            
        # First try Levenberg-Marquardt (more robust for non-convex problems)
        result = optimize_rank_levenberg(r, max_restarts=10)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = (U, V, W)
            break
        
        # If LM fails, try L-BFGS-B
        result = optimize_rank_l_bfgs(r, max_restarts=10)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = (U, V, W)
            break
    
    U, V, W = best_result
    
    # Convert to lists of lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
