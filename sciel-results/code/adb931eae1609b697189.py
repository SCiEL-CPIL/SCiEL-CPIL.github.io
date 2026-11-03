"""Find rank-minimal exact tensor decomposition for 2x5 x 5x5 matrix multiplication."""
import numpy as np
import time
import sys

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 10
    mp = m * p  # 25
    pn = p * n  # 10
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill
    
    # Build the target tensor T with shape (nm, mp, pn)
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k; else 0
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    def verify(U, V, W, tol=1e-6):
        """Check if decomposition is exact using efficient einsum."""
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        max_err = np.max(np.abs(T_recon - T_target))
        return max_err
    
    def snap_to_half_integers(U, V, W):
        """Snap values to nearest half-integer."""
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        return U_snap, V_snap, W_snap
    
    def check_and_return(U, V, W, r):
        """Check if the decomposition is valid and return formatted result."""
        # First try without snapping
        err = verify(U, V, W)
        if err < 1e-8:
            U_out = [[float(x) for x in row] for row in U]
            V_out = [[float(x) for x in row] for row in V]
            W_out = [[float(x) for x in row] for row in W]
            return {"U": U_out, "V": V_out, "W": W_out}
        
        # Try snapping to half-integers
        U_snap, V_snap, W_snap = snap_to_half_integers(U, V, W)
        err_snap = verify(U_snap, V_snap, W_snap)
        if err_snap < 1e-8:
            U_out = [[float(x) for x in row] for row in U_snap]
            V_out = [[float(x) for x in row] for row in V_snap]
            W_out = [[float(x) for x in row] for row in W_snap]
            return {"U": U_out, "V": V_out, "W": W_out}
        
        return None
    
    # Schoolbook decomposition as fallback (rank 50)
    rank_school = n * m * p  # 50
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    best_result = None
    best_rank = rank_school
    
    # Convert schoolbook to list format for fallback
    def to_list(U, V, W):
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }
    
    fallback_result = to_list(U_school, V_school, W_school)
    
    # Try to find lower rank decompositions using alternating least squares
    # Strategy: for each target rank, initialize randomly and run ALS
    
    def als_solve(U, V, W, max_iter=100, tol=1e-10):
        """Alternating least squares to minimize ||T_recon - T_target||_F."""
        for it in range(max_iter):
            if time.time() > deadline - 20:
                break
            
            # Update U: fix V, W
            # For each a, U[a,:] should satisfy: U[a,:] @ (V[b,r]*W[c,r]) = T_target[a,b,c] for all b,c
            # This is a linear system: U[a,:] @ M_a = T_a where M_a[b,c] = sum_r V[b,r]*W[c,r]
            # But we can vectorize: reshape T_target[a,:,:] and compute the Gram matrix
            
            # More efficiently: 
            # T_target[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # Fix V,W: for each a, we want to find U[a,:] that best satisfies
            # sum_r U[a,r] * (V[:,r] * W[:,r]) = T_target[a,:,:].reshape(-1)
            
            # Let's build the design matrix for U
            # For a given a, the equation is: U[a,:] @ A_a = T_a
            # where A_a[r, (b,c)] = V[b,r] * W[c,r]
            # T_a = T_target[a,:,:].reshape(-1)
            
            for a in range(nm):
                # Build A_a: shape (r, mp*pn)
                # A_a[r, b*pn+c] = V[b,r] * W[c,r]
                A_a = np.einsum('br,cr->rbc', V, W).reshape(-1, mp*pn).T  # (r, mp*pn)
                T_a = T_target[a].reshape(-1)
                
                # Solve least squares: U[a,:] @ A_a = T_a
                # i.e., A_a.T @ U[a,:] = T_a
                try:
                    U[a, :] = np.linalg.lstsq(A_a.T, T_a, rcond=None)[0]
                except:
                    pass
            
            # Update V: fix U, W
            for b in range(mp):
                A_b = np.einsum('ar,cr->rac', U, W).reshape(nm*pn, -1).T  # (r, nm*pn)
                T_b = T_target[:, b, :].reshape(-1)
                try:
                    V[b, :] = np.linalg.lstsq(A_b.T, T_b, rcond=None)[0]
                except:
                    pass
            
            # Update W: fix U, V
            for c in range(pn):
                A_c = np.einsum('ar,br->rab', U, V).reshape(nm*mp, -1).T  # (r, nm*mp)
                T_c = T_target[:, :, c].reshape(-1)
                try:
                    W[c, :] = np.linalg.lstsq(A_c.T, T_c, rcond=None)[0]
                except:
                    pass
            
            # Check convergence
            err = verify(U, V, W)
            if err < tol:
                break
        
        return U, V, W
    
    # Try ranks from 49 down to 25
    np.random.seed(42)
    
    for target_rank in range(49, 24, -1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple initializations
        for attempt in range(3):
            if time.time() > deadline - 30:
                break
            
            # Random initialization
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            # Run ALS
            U, V, W = als_solve(U, V, W, max_iter=50, tol=1e-10)
            
            # Check result
            result = check_and_return(U, V, W, target_rank)
            if result is not None:
                best_result = result
                best_rank = target_rank
                break
        
        # If we found a solution at this rank, try lower ranks
        if best_rank == target_rank:
            continue
    
    # Return best found or fallback
    if best_result is not None:
        return best_result
    else:
        return fallback_result
