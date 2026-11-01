"""Matrix multiplication 2x3x3 rank-minimal exact tensor decomposition search."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]
_DEADLINE = 130.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _DEADLINE - _elapsed()

def construct():
    n, m, p = DIMENSIONS
    # Tensor T has shape (n*m, m*p, p*n)
    # Indices: U: i*m+j, V: j*p+k, W: k*n+i
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    T_flat = T.reshape(-1)
    
    # Schoolbook rank
    rank_sb = n * m * p
    U0 = np.zeros((n*m, rank_sb))
    V0 = np.zeros((m*p, rank_sb))
    W0 = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank_sb
    
    def _flatten_params(U, V, W, r):
        return np.concatenate([U.ravel(), V.ravel(), W.ravel()])
    
    def _unflatten(params, r):
        sz_u = n*m*r
        sz_v = m*p*r
        sz_w = p*n*r
        U = params[:sz_u].reshape(n*m, r)
        V = params[sz_u:sz_u+sz_v].reshape(m*p, r)
        W = params[sz_u+sz_v:sz_u+sz_v+sz_w].reshape(p*n, r)
        return U, V, W
    
    def _objective(params, r):
        U, V, W = _unflatten(params, r)
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_approx.reshape(-1) - T_flat
        return np.sum(diff*diff)
    
    def _objective_grad(params, r):
        U, V, W = _unflatten(params, r)
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_approx - T
        gU = np.einsum('abc,br,cr->ar', diff, V, W)
        gV = np.einsum('abc,ar,cr->br', diff, U, W)
        gW = np.einsum('abc,ar,br->cr', diff, U, V)
        return np.concatenate([gU.ravel(), gV.ravel(), gW.ravel()])
    
    def _verify_exact(U, V, W, tol=1e-7):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        if np.max(np.abs(T_approx - T)) > tol:
            return False
        # Check entries are close to integer or half-integer
        for mat in [U, V, W]:
            for val in mat:
                nearest = round(val * 2) / 2
                if abs(val - nearest) > 1e-5:
                    return False
        return True

    def _discrete_polish(U, V, W, r, max_steps=1000):
        """
        Local search on the half-integer lattice.
        """
        current_error = _objective(_flatten_params(U, V, W, r), r)
        if current_error < 1e-10:
            return U, V, W
        
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        mats = [U, V, W]
        
        # Pre-compute neighbors for half-integers
        # Values are k/2. Neighbors are (k-1)/2 and (k+1)/2
        
        for step in range(max_steps):
            if _remaining() < 5:
                break
            
            # Pick a random matrix and element
            mat_idx = np.random.choice(3)
            mat = mats[mat_idx]
            rows, cols = mat.shape
            r_idx = np.random.choice(rows)
            c_idx = np.random.choice(cols)
            
            current_val = mat[r_idx, c_idx]
            # Try neighboring half-integers
            # Current val is approx v. Neighbors are v-0.5, v+0.5
            # Also try v itself? No, just move.
            
            # Determine current half-integer index
            h_idx = int(round(current_val * 2))
            
            # Candidates: h_idx - 1, h_idx + 1
            # Also maybe jump to 0?
            candidates = [h_idx - 1, h_idx + 1]
            # Add a small probability to jump further
            if np.random.rand() < 0.1:
                candidates.append(h_idx + np.random.choice([-2, 2, 3, -3]))
            
            best_candidate_val = current_val
            best_candidate_err = current_error
            
            for c_val in candidates:
                new_val = c_val / 2.0
                # Skip if same
                if abs(new_val - current_val) < 1e-9:
                    continue
                # Apply change
                old_val = mat[r_idx, c_idx]
                mat[r_idx, c_idx] = new_val
                
                err = _objective(_flatten_params(mats[0], mats[1], mats[2], r), r)
                
                if err < best_candidate_err:
                    best_candidate_err = err
                    best_candidate_val = new_val
                else:
                    # Revert
                    mat[r_idx, c_idx] = old_val
            
            # If we found a better value, keep it (we already set it in the loop that won? No, we reverted if not best)
            # Wait, the logic above reverts if not best. So if best_candidate_val != current_val, we need to set it.
            if abs(best_candidate_val - current_val) > 1e-9:
                mat[r_idx, c_idx] = best_candidate_val
                current_error = best_candidate_err
            
            if current_error < 1e-10:
                return U, V, W
                
        return U, V, W

    def _round_to_half(U, V, W):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        return U_r, V_r, W_r
    
    # Search ranks
    # We prioritize finding the minimal rank.
    # Try 13, then 14.
    ranks_to_try = [13, 14]
    
    rng = np.random.default_rng(42)
    
    for rank in ranks_to_try:
        if _remaining() < 10:
            break
            
        # Number of restarts depends on remaining time
        # Estimate time per restart: 0.5s - 1s
        max_restarts = 30 if rank == 13 else 50
        
        found = False
        for restart in range(max_restarts):
            if _remaining() < 5:
                break
                
            # Initialization
            # Use half-integers directly to help discrete search
            vals = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
            
            U = rng.choice(vals, size=(n*m, rank))
            V = rng.choice(vals, size=(m*p, rank))
            W = rng.choice(vals, size=(p*n, rank))
            
            params = _flatten_params(U, V, W, rank)
            
            try:
                # 1. Continuous Optimization
                # Use L-BFGS-B
                res = minimize(
                    _objective, params, args=(rank,),
                    jac=_objective_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'gtol': 1e-6, 'ftol': 1e-8}
                )
                
                U, V, W = _unflatten(res.x, rank)
                
                # 2. Snap to half-integers
                U, V, W = _round_to_half(U, V, W)
                
                # Check if exact after snap
                if _verify_exact(U, V, W, tol=1e-7):
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    found = True
                    break
                
                # 3. Discrete Polish
                # Only do this if continuous error was reasonably low
                if res.fun < 10.0:
                    U, V, W = _discrete_polish(U, V, W, rank, max_steps=500)
                    
                    if _verify_exact(U, V, W, tol=1e-7):
                        best_U, best_V, best_W = U, V, W
                        best_rank = rank
                        found = True
                        break
                        
            except Exception:
                pass
        
        if found:
            # If we found rank 13, we can stop if we want minimal, but let's see if we can find 12?
            # Lower bound for 2x3x3 is likely >= 11 or 12.
            # If we found 13, it's a huge improvement over 18.
            # We can try to verify if 12 is possible?
            # For now, if we find 13, we break and return 13.
            # If we want to be safe, we can try 12 in the remaining time.
            if rank == 13 and _remaining() > 15:
                # Try rank 12?
                # It's unlikely to be 12, but let's try a few restarts.
                pass 
            else:
                break
                
    # Finalize
    if best_rank < rank_sb:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        
        # Final verification
        U_v = np.array(U_list)
        V_v = np.array(V_list)
        W_v = np.array(W_list)
        T_approx = np.einsum('ar,br,cr->abc', U_v, V_v, W_v)
        if np.max(np.abs(T_approx - T)) < 1e-7:
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    U_list = [[int(x) for x in row] for row in U0]
    V_list = [[int(x) for x in row] for row in V0]
    W_list = [[int(x) for x in row] for row in W0]
    return {"U": U_list, "V": V_list, "W": W_list}
