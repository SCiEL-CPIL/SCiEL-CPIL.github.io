import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p

    # Target tensor indices
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Pre-compute the non-zero entries of the target tensor
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                target_indices.append((idx_u, idx_v, idx_w))
                
    # Construct the full target tensor (1600 entries, tiny)
    T_target = np.zeros((n*m, m*p, p*n))
    for idx_u, idx_v, idx_w in target_indices:
        T_target[idx_u, idx_v, idx_w] = 1.0
        
    # Verify function: checks if U, V, W exactly reconstruct T_target
    # T_approx[idx_u, idx_v, idx_w] = sum_t U[idx_u, t] V[idx_v, t] W[idx_w, t]
    def verify(U, V, W):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # Efficient check: 
        # For each t, we have rank-1 tensor U_t \otimes V_t \otimes W_t
        # Summing them gives T_approx.
        # We can check specific entries.
        # To be safe, check all target entries are 1 and all others are 0.
        # However, checking all 1600 entries for every candidate is fast.
        
        # Let's use a vectorized approach if possible, or just loop.
        # r is small (<40).
        # T_approx = np.einsum('ut,vt,wt->uvw', U, V, W)
        # This creates a (8, 20, 10) array. 1600 elements. Very fast.
        T_approx = np.einsum('ut,vt,wt->uvw', U, V, W)
        
        # Check exactness with tolerance (integers/half-integers should be exact)
        # But floating point might introduce tiny errors.
        # The problem asks for integer/half-integer entries.
        # So we should snap to nearest half-integer and check?
        # Or just check if close to 0 or 1.
        
        # Since the inputs are integers/half-integers, the outputs are integers.
        # Let's check if T_approx is close to T_target.
        diff = np.abs(T_approx - T_target)
        return np.all(diff < 1e-9)

    # Schoolbook solution (Rank 40)
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_solution = {'U': U_school, 'V': V_school, 'W': W_school}
    best_rank = rank_schoolbook

    # Strategy: Search for lower ranks using structured Kronecker-like approaches
    # or random integer/half-integer search with ALS refinement.
    
    # Given the difficulty of finding a generic rank reduction for 2x4x5,
    # let's try a specific structured decomposition.
    # Note: R(2,4,5) <= R(2,2,2)*R(1,2,2.5)? No.
    # R(2,4,5) <= R(2,2,5)*R(1,2,1)?
    # R(2,4,5) <= R(2,4,2)*R(1,1,2.5)?
    
    # Let's try to find a rank 39 or lower solution by perturbing the schoolbook solution
    # and using ALS to converge to an integer/half-integer solution.
    # This is unlikely to jump the gap to a known lower rank if one exists structurally,
    # but it might find a "sparse" lower rank if the tensor allows.
    
    # Actually, a more promising search is to look for decompositions of rank < 40
    # by solving for U, V, W with integer constraints.
    # We can use a random restart strategy with ALS on the continuous relaxation,
    # then snap to nearest half-integer and verify.
    
    # ALS Update for CP Decomposition:
    # Minimize || T - sum_t U_t \otimes V_t \otimes W_t ||_F^2
    # Update U: 
    #   For each row u of U, we want to minimize || T_u - U_u (V \odot W)^T ||?
    #   Actually, T_u is a matrix of size (mp, pn).
    #   T_u[v, w] = sum_t U[u, t] V[v, t] W[w, t]
    #   Let M_t = V[:, t] \otimes W[:, t] (vector of length mp*pn)
    #   Then T_u_flat = sum_t U[u, t] M_t
    #   So U[u, :] is the least squares solution of T_u_flat = U[u, :] M^T
    #   where M is (r, mp*pn).
    
    # This is standard ALS.
    
    def als_decompose(T, r, max_iter=50, tol=1e-10):
        # T: (nm, mp, pn)
        nm, mp, pn = T.shape
        U = np.random.randn(nm, r)
        V = np.random.randn(mp, r)
        W = np.random.randn(pn, r)
        
        # Precompute M = V \odot W (element-wise product? No, Kronecker product of columns)
        # M[:, t] = V[:, t] \otimes W[:, t]
        
        for it in range(max_iter):
            # Update U
            # T_flat: (nm, mp*pn)
            T_flat = T.reshape(nm, mp * pn)
            # M: (r, mp*pn)
            M = np.einsum('vt,wt->tvp', V, W).reshape(r, mp * pn)
            # Solve U @ M.T = T_flat
            # U = T_flat @ M.T @ (M @ M.T)^-1
            try:
                U, residuals, rank, s = np.linalg.lstsq(M.T, T_flat.T, rcond=None)
                U = U.T
            except:
                break
                
            # Update V
            # T_mat_v: (mp, nm*pn)
            T_mat_v = T.transpose(1, 0, 2).reshape(mp, nm * pn)
            # N: (r, nm*pn)
            N = np.einsum('ut,wt->utw', U, W).reshape(r, nm * pn)
            try:
                V, residuals, rank, s = np.linalg.lstsq(N.T, T_mat_v.T, rcond=None)
                V = V.T
            except:
                break
                
            # Update W
            # T_mat_w: (pn, nm*mp)
            T_mat_w = T.transpose(2, 0, 1).reshape(pn, nm * mp)
            # P: (r, nm*mp)
            P = np.einsum('ut,vt->utv', U, V).reshape(r, nm * mp)
            try:
                W, residuals, rank, s = np.linalg.lstsq(P.T, T_mat_w.T, rcond=None)
                W = W.T
            except:
                break
                
            # Check convergence
            T_approx = np.einsum('ut,vt,wt->uvw', U, V, W)
            diff = np.linalg.norm(T_approx - T)
            if diff < tol:
                break
                
        return U, V, W

    # Search for lower ranks
    # We will try ranks from 39 down to 30.
    # For each rank, we will try a few random initializations and ALS.
    # After ALS, we snap to nearest half-integer and verify.
    
    # Note: ALS on continuous relaxation might converge to a non-integer solution.
    # Snapping might break the exactness.
    # However, if a low-rank integer solution exists, ALS might find a nearby continuous solution.
    
    # Alternative: Direct random search for small integer entries?
    # Too large space.
    
    # Let's try ALS and snapping.
    
    for r in range(rank_schoolbook - 1, 20, -1):
        if time.time() > deadline - 5:
            break
            
        max_restarts = 10
        # Reduce restarts if time is running out
        if time.time() > deadline - 20:
            max_restarts = 2
            
        for restart in range(max_restarts):
            if time.time() > deadline - 2:
                break
                
            U, V, W = als_decompose(T_target, r, max_iter=30)
            
            # Snap to nearest half-integer
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            # Verify
            if verify(U_snap, V_snap, W_snap):
                best_solution = {'U': U_snap, 'V': V_snap, 'W': W_snap}
                best_rank = r
                # Found a better solution, continue searching for even lower
                break
                
            # Also try snapping to integers
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if verify(U_int, V_int, W_int):
                best_solution = {'U': U_int, 'V': V_int, 'W': W_int}
                best_rank = r
                break

    # Return the best found
    # Convert numpy arrays to lists of lists
    return {
        "U": best_solution['U'].tolist(),
        "V": best_solution['V'].tolist(),
        "W": best_solution['W'].tolist()
    }
