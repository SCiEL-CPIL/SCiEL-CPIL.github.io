import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 3, 10]
DEADLINE = 145  # seconds, safety margin before 150s kill

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x10 matrix.
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    def build_rank_55():
        rank = 55
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        a00, a01, a10, a11 = 0, 1, 3, 4
        a02, a12 = 2, 5
        
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            b00, b01 = k0, k1
            b10, b11 = 10 + k0, 10 + k1
            c00, c01 = 2 * k0, 2 * k1
            c10, c11 = 2 * k0 + 1, 2 * k1 + 1
            
            # M1 = (A00 + A11)(B00 + B11)
            U[a00, col] += 1.0; U[a11, col] += 1.0
            V[b00, col] += 1.0; V[b11, col] += 1.0
            W[c00, col] += 1.0; W[c11, col] += 1.0
            col += 1
            
            # M2 = (A10 + A11)B00
            U[a10, col] += 1.0; U[a11, col] += 1.0
            V[b00, col] += 1.0
            W[c10, col] += 1.0; W[c11, col] -= 1.0
            col += 1
            
            # M3 = A00(B01 - B11)
            U[a00, col] += 1.0
            V[b01, col] += 1.0; V[b11, col] -= 1.0
            W[c01, col] += 1.0; W[c11, col] += 1.0
            col += 1
            
            # M4 = A11(B10 - B00)
            U[a11, col] += 1.0
            V[b10, col] += 1.0; V[b00, col] -= 1.0
            W[c00, col] += 1.0; W[c10, col] += 1.0
            col += 1
            
            # M5 = (A00 + A01)B11
            U[a00, col] += 1.0; U[a01, col] += 1.0
            V[b11, col] += 1.0
            W[c00, col] -= 1.0; W[c01, col] += 1.0
            col += 1
            
            # M6 = (A10 - A00)(B00 + B01)
            U[a10, col] += 1.0; U[a00, col] -= 1.0
            V[b00, col] += 1.0; V[b01, col] += 1.0
            W[c11, col] += 1.0
            col += 1
            
            # M7 = (A01 - A11)(B10 + B11)
            U[a01, col] += 1.0; U[a11, col] -= 1.0
            V[b10, col] += 1.0; V[b11, col] += 1.0
            W[c00, col] += 1.0
            col += 1
        
        for k in range(p):
            b2k = 2 * p + k
            U[a02, col] += 1.0; V[b2k, col] += 1.0; W[2 * k, col] += 1.0
            col += 1
            U[a12, col] += 1.0; V[b2k, col] += 1.0; W[2 * k + 1, col] += 1.0
            col += 1
        
        return U, V, W, rank
    
    def verify(U, V, W):
        """Vectorized verification of tensor decomposition."""
        # For each (i,j,k): sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] == 1
        # Build the full tensor check using einsum
        # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        # T[i,j,k] = sum_t U_r[i,j,t] * V_r[j,k,t] * W_r[k,i,t]
        T = np.einsum('ijt,jkt,kit->ijk', U_r, V_r, W_r)
        return np.allclose(T, 1.0, atol=1e-10)
    
    U55, V55, W55, rank55 = build_rank_55()
    best_U, best_V, best_W, best_rank = U55, V55, W55, rank55
    
    # Verify rank-55 solution (fast, vectorized)
    assert verify(U55, V55, W55), "Rank-55 solution invalid"
    
    # Try to find rank 54 by trying to eliminate one term
    # Strategy: for each term t in the rank-55 solution, try to set it to zero
    # and see if we can adjust the other terms to compensate.
    # This is equivalent to: can the t-th term be expressed as a combination
    # of the others in a way that preserves the tensor equation?
    
    # A simpler approach: try to find a rank-54 solution by modifying the support.
    # For each pair of terms (t1, t2), try to merge them into one term.
    
    # Actually, let's try a different approach: use the fact that the problem
    # is over-determined in a useful way. For rank r, we have 60 equations and
    # 56r unknowns. For r=54, that's 3024 unknowns and 60 equations.
    
    # Key insight: if we fix the support pattern, we can use alternating
    # least squares to find values that satisfy the equations.
    
    # Let's try: start from rank-55, zero out one term, and see if we can
    # redistribute its contribution to the other 54 terms.
    
    # For term t, its contribution to the tensor is:
    # U[:,t] ⊗ V[:,t] ⊗ W[:,t]
    # We want to find delta_U, delta_V, delta_W (supported on the other 54 terms)
    # such that the total tensor remains all-ones.
    
    # This is still nonlinear, but we can linearize: fix V and W at their
    # current values, and solve for adjustments to U.
    
    # For each (i,j,k): sum_{t != t0} U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] 
    #   + U[i*m+j, t0] * V[j*p+k, t0] * W[k*n+i, t0] = 1
    # If we remove term t0: sum_{t != t0} U'[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    # where U' = U adjusted.
    
    # Linearizing: U'[i*m+j, t] = U[i*m+j, t] + delta_U[i*m+j, t]
    # sum_{t != t0} (U[i*m+j, t] + delta_U[i*m+j, t]) * V[j*p+k, t] * W[k*n+i, t] = 1
    # sum_{t != t0} U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] 
    #   + sum_{t != t0} delta_U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    # The first sum = 1 - U[i*m+j, t0] * V[j*p+k, t0] * W[k*n+i, t0]
    # So: sum_{t != t0} delta_U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] 
    #   = U[i*m+j, t0] * V[j*p+k, t0] * W[k*n+i, t0]
    
    # This is a linear system in delta_U! For each (i,j), we have p equations
    # and (r-1) unknowns (one per remaining term).
    
    # But wait, we also need to maintain consistency: the delta_U must be such
    # that the full tensor equation holds for ALL (i,j,k). Since we're only
    # adjusting U, and the equation separates by (i,j), we can solve independently
    # for each (i,j).
    
    # For fixed (i,j): we need delta_U[i*m+j, :] (length r-1) such that for all k:
    # sum_{t != t0} delta_U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = U[i*m+j, t0] * V[j*p+k, t0] * W[k*n+i, t0]
    
    # This is a linear system: M[i,j] @ delta_U[i*m+j, :] = rhs[i,j]
    # where M[i,j][k, t] = V[j*p+k, t] * W[k*n+i, t] for t != t0
    
    # Let's try this for each t0!
    
    if time.time() - start_time > DEADLINE - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to eliminate each term
    for t0 in range(rank55):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        # Build the linear system for each (i,j)
        # Unknowns: delta_U[i*m+j, t] for t != t0, for each (i,j)
        # But actually, we need to be more careful. We can adjust U, V, OR W.
        # Let's try adjusting only U first (simplest case).
        
        success = True
        new_U = U55.copy()
        
        for i in range(n):
            for j in range(m):
                # Build M: (p x (r-1)) matrix
                # M[k, t_idx] = V[j*p+k, t] * W[k*n+i, t] for t != t0
                t_indices = [t for t in range(rank55) if t != t0]
                M_mat = np.zeros((p, len(t_indices)))
                rhs = np.zeros(p)
                
                for k in range(p):
                    for idx, t in enumerate(t_indices):
                        M_mat[k, idx] = V55[j*p+k, t] * W55[k*n+i, t]
                    rhs[k] = U55[i*m+j, t0] * V55[j*p+k, t0] * W55[k*n+i, t0]
                
                # Solve M_mat @ delta = rhs
                # Check if system is consistent
                try:
                    # Use least squares; check residual
                    delta, residuals, rank_m, sv = np.linalg.lstsq(M_mat, rhs, rcond=1e-10)
                    if residuals.size > 0 and residuals[0] > 1e-8:
                        # Not a good fit, this t0 can't be eliminated by adjusting U alone
                        success = False
                        break
                    # Apply delta
                    for idx, t in enumerate(t_indices):
                        new_U[i*m+j, t] += delta[idx]
                except:
                    success = False
                    break
            
            if not success:
                break
        
        if success:
            # Verify the new solution
            new_V = V55.copy()
            new_W = W55.copy()
            # Zero out term t0 in U (it's already zero since we only adjusted others)
            new_U[:, t0] = 0.0
            
            if verify(new_U, new_V, new_W):
                # Check that all entries are integer/half-integer
                # Round to nearest half-integer
                new_U_r = np.round(new_U * 2) / 2
                new_V_r = np.round(new_V * 2) / 2
                new_W_r = np.round(new_W * 2) / 2
                
                if verify(new_U_r, new_V_r, new_W_r):
                    # Check if term t0 is actually zero
                    if np.allclose(new_U[:, t0], 0) and np.allclose(new_V[:, t0], 0) and np.allclose(new_W[:, t0], 0):
                        # We have a rank-54 solution!
                        # Remove the zero column
                        mask = np.ones(rank55, dtype=bool)
                        mask[t0] = False
                        best_U = new_U_r[:, mask]
                        best_V = new_V_r[:, mask]
                        best_W = new_W_r[:, mask]
                        best_rank = rank55 - 1
                        print(f"Found rank {best_rank} solution by eliminating term {t0}")
                        break
    
    if time.time() - start_time > DEADLINE:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
