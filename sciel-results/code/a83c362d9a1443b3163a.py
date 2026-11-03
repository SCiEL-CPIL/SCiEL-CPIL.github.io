import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 9]


def construct():
    """
    Find a low-rank exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Strategy: 
    1. Build schoolbook decomposition (rank 144)
    2. Use vectorized verification for speed
    3. Try structured algebraic search: fix U,V patterns with small integers,
       then solve for W via least squares / linear system
    4. Try block decomposition using known low-rank 4x4*4x4 constructions
    """
    n, m, p = 4, 4, 9
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    # Build schoolbook decomposition
    R_sb = n * m * p  # 144
    U = np.zeros((n * m, R_sb), dtype=float)
    V = np.zeros((m * p, R_sb), dtype=float)
    W = np.zeros((p * n, R_sb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1

    def verify_fast(U, V, W, tol=1e-9):
        """Vectorized verification using tensor contraction."""
        n, m, p = 4, 4, 9
        R = U.shape[1]
        
        # Build the tensor T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Efficient: for each r, outer product of U[:,r], V[:,r], W[:,r]
        # T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        
        # Use einsum for efficiency
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        T = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check consistent indices: T[i*m+j, j*p+k, k*n+i] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    if abs(T[a, b, c] - 1.0) > tol:
                        return False
        
        # Check inconsistent indices: T[a,b,c] == 0
        # a = i1*m + j1, b = j2*p + k2, c = k3*n + i3
        # Inconsistent if j1 != j2 or i1 != i3 or k2 != k3
        for i1 in range(n):
            for j1 in range(m):
                a = i1 * m + j1
                for j2 in range(m):
                    for k2 in range(p):
                        b = j2 * p + k2
                        if j1 != j2:
                            # All c are inconsistent
                            if np.any(np.abs(T[a, b, :]) > tol):
                                return False
                        else:
                            # j1 == j2, check i and k
                            for k3 in range(p):
                                for i3 in range(n):
                                    c = k3 * n + i3
                                    if i1 != i3 or k2 != k3:
                                        if abs(T[a, b, c]) > tol:
                                            return False
        return True

    # Quick verify schoolbook (should pass)
    # Skip full verification to save time; schoolbook is known correct
    
    # Strategy: Try to find lower rank by structured search
    # Key insight: fix U and V with small integer/half-integer patterns,
    # then the constraints become linear in W (for each consistent triple)
    # and linear in W (for each inconsistent triple, but with U,V fixed it's still trilinear)
    # 
    # Better: fix U and V, then for each (a,b) pair, the constraint is:
    # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c] for all c
    # This is linear in W[:,r] for fixed (a,b) if we treat W[:,r] as unknowns.
    # Actually: for fixed (a,b), we need: sum_r (U[a,r]*V[b,r]) * W[c,r] = T[a,b,c]
    # Let M_ab = U[a,:] * V[b,:] (element-wise product, shape R)
    # Then: M_ab @ W[c,:] = T[a,b,c] for all c
    # This is: W @ M_ab^T = T[a,b,:] (where W is (p*n, R) and T[a,b,:] is (p*n,))
    # So: for each (a,b), we have a linear system in W.
    # But W must be the SAME for all (a,b). So we need W that satisfies all these.
    
    # Approach: Fix U and V. Then W must satisfy:
    # For each (a,b): (U[a,:] * V[b,:]) @ W[c,:] = T[a,b,c] for all c
    # Equivalently: for each c: sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
    # This is a linear system in W. We can set up a large linear system.
    
    # Number of equations: n*m * m*p * p*n = 16*36*36 = 20736
    # Number of unknowns: R * p*n = R * 36
    # For R < 144, this is overdetermined. We need exact solution.
    
    # Let's try: fix U and V with structured patterns, solve for W.
    
    def try_solve_for_W(U_fixed, V_fixed, R):
        """Given fixed U and V of shape (*, R), solve for W."""
        n, m, p = 4, 4, 9
        # Build linear system: for each (a,b,c): sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
        # This is: sum_r (U[a,r]*V[b,r]) * W[c,r] = T[a,b,c]
        # For fixed (a,b), this is a linear equation in W[:,r] for each c.
        # Let's collect all equations.
        
        # Number of equations: n*m * m*p * p*n
        # Number of unknowns: R * p*n (entries of W)
        
        # Build A matrix and b vector
        # Each equation: sum_{r} coeff_r * W[c, r] = T[a,b,c]
        # where coeff_r = U[a,r] * V[b,r]
        
        # We can vectorize: for each (a,b), the coefficients for W are U[a,:]*V[b,:]
        # and the RHS is T[a,b,:] (vector over c)
        
        # Let's build the system as:
        # For each (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        
        # Total equations: 16 * 36 * 36 = 20736
        # Total unknowns: R * 36
        
        # Build A (20736, R*36) and b (20736,)
        # This might be large but let's try for small R
        
        num_eq = n * m * m * p * p * n  # 4*4*4*9*9*4 = 20736
        num_unk = R * p * n  # R * 36
        
        if num_unk > 5000:  # Too large for direct solve
            return None
        
        A = np.zeros((num_eq, num_unk))
        b = np.zeros(num_eq)
        
        idx = 0
        for i1 in range(n):
            for j1 in range(m):
                a = i1 * m + j1
                for j2 in range(m):
                    for k2 in range(p):
                        b_idx = j2 * p + k2
                        for k3 in range(p):
                            for i3 in range(n):
                                c = k3 * n + i3
                                # Coefficients: U[a,r] * V[b_idx, r] for each r
                                coeffs = U_fixed[a, :] * V_fixed[b_idx, :]
                                # Unknowns: W[c, r] for each r
                                for r in range(R):
                                    A[idx, c * R + r] = coeffs[r]
                                # RHS
                                if j1 == j2 and i1 == i3 and k2 == k3:
                                    b[idx] = 1.0
                                else:
                                    b[idx] = 0.0
                                idx += 1
        
        # Solve least squares
        try:
            W_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            W_sol = W_flat.reshape(p * n, R)
        except np.linalg.LinAlgError:
            return None
        
        # Verify the solution
        if verify_fast(U_fixed, V_fixed, W_sol, tol=1e-7):
            return W_sol
        return None

    # Try structured U, V patterns
    # Pattern 1: U and V from schoolbook but with some columns removed
    # Pattern 2: Random small integer patterns
    
    best_result = None
    best_rank = R_sb
    
    # Try reducing rank by removing columns from schoolbook
    # This won't work since schoolbook columns are independent
    
    # Try: use Strassen-like structure for the 4x4 blocks
    # For 4x4*4x4, the known minimum rank is 49 (but we don't have explicit factors)
    # Let's try a simpler approach: block decomposition
    
    # Split 4x4*4x9 into 4x4*4x4 + 4x4*4x4 + 4x4*4x1
    # Rank: R(4,4,4) + R(4,4,4) + R(4,4,1)
    # R(4,4,1) = 4 (each output is a dot product of length 4)
    # R(4,4,4) <= 49 (Strassen), but let's try to find a good construction
    
    # For now, try random search with the solve-for-W approach
    np.random.seed(123)
    
    for R in [120, 100, 90, 80, 72, 64]:
        if time.time() > deadline - 30:
            break
        
        for trial in range(20):
            if time.time() > deadline - 30:
                break
            
            # Generate random U and V with small integer entries
            U_t = np.random.randint(-1, 2, (n * m, R)).astype(float)
            V_t = np.random.randint(-1, 2, (m * p, R)).astype(float)
            
            # Try to solve for W
            W_sol = try_solve_for_W(U_t, V_t, R)
            if W_sol is not None:
                best_rank = R
                best_result = (U_t, V_t, W_sol)
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        U_best, V_best, W_best = best_result
        return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
    
    # If no improvement found, return schoolbook
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
