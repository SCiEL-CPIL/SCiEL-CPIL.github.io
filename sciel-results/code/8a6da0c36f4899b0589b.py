import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def verify(U, V, W, tol=1e-9):
        """Check exact decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > tol:
                        return False
        # Check zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if (i2 == i and j2 == j and k2 == k):
                                    continue
                                # The tensor T[i2,j2,k2] should be 0 unless i2==i,j2==j,k2==k
                                # Actually, the multiplication tensor has T[i,j,k] = delta_{i,k}... 
                                # No, for matrix multiplication C = A*B where A is n×m, B is m×p
                                # C[i,k] = sum_j A[i,j]*B[j,k]
                                # The tensor is T[i,j,k] = 1 (it's the identity in the middle index)
                                # So all entries (i,j,k) should satisfy: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                                # There are no "zero" entries to check - all (i,j,k) in the valid range should give 1
                                pass
        return True
    
    def verify_full(U, V, W, tol=1e-9):
        """Full verification: all (i,j,k) must give exactly 1."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > tol:
                        return False
        return True
    
    # Build the linear system for a given rank
    # Variables: U (n*m*r), V (m*p*r), W (p*n*r)
    # Equations: for each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is trilinear, not linear. But if we fix two factors, it's linear in the third.
    
    def solve_linear_in_W(U, V, r):
        """Given U and V, solve for W linearly."""
        # For each (k,i): sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # For fixed (k,i): sum_c (sum_j U[i*m+j,c]*V[j*p+k,c]) * W[k*n+i,c] = 1
        # This is: M[k*n+i, c] * W[k*n+i, c] = 1 where M[k*n+i,c] = sum_j U[i*m+j,c]*V[j*p+k,c]
        # But this is not quite right because W is shared across different (j) for the same (k,i)
        
        # Actually, the equation is:
        # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # 
        # If we fix U and V, for each (k,i), the equation involves W[k*n+i,c] for all c,
        # but it's coupled through j. Let me reformulate.
        #
        # For fixed (k,i): sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = sum_j 1 = m
        # Wait, no. The equation is per (i,j,k), not summed over j.
        #
        # So for each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        #
        # If U and V are fixed, then for each (i,j,k), this is linear in W[k*n+i,c].
        # But W[k*n+i,c] appears in multiple equations (one for each j).
        #
        # Let's set up: for each (i,j,k), we have equation:
        # sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
        #
        # Group by (k,i): for fixed (k,i), and for each j, we have:
        # sum_c A_{j,c} * W_{k,i,c} = 1 where A_{j,c} = U[i*m+j,c] * V[j*p+k,c]
        #
        # So for fixed (k,i), we have m equations (one per j) in p*n*r unknowns... no, 
        # W[k*n+i,c] for c=0..r-1, so r unknowns.
        #
        # For fixed (k,i): m equations, r unknowns. If m < r, underdetermined.
        
        # Let's just use least squares on the full system.
        # Total equations: n*m*p = 72
        # Unknowns: p*n*r = 24*r
        
        num_eqs = n * m * p
        num_unk = p * n * r
        
        A = np.zeros((num_eqs, num_unk))
        b = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        # W[k*n+i, c] is at index (k*n+i)*r + c
                        A[eq_idx, (k*n+i)*r + c] = U[i*m+j, c] * V[j*p+k, c]
                    eq_idx += 1
        
        if num_unk <= num_eqs:
            sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        else:
            # Underdetermined: use minimum norm solution
            sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        
        W = sol.reshape(p*n, r)
        return W
    
    def solve_linear_in_U(V, W, r):
        """Given V and W, solve for U linearly."""
        num_eqs = n * m * p
        num_unk = n * m * r
        
        A = np.zeros((num_eqs, num_unk))
        b = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        A[eq_idx, (i*m+j)*r + c] = V[j*p+k, c] * W[k*n+i, c]
                    eq_idx += 1
        
        sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        U = sol.reshape(n*m, r)
        return U
    
    def solve_linear_in_V(U, W, r):
        """Given U and W, solve for V linearly."""
        num_eqs = n * m * p
        num_unk = m * p * r
        
        A = np.zeros((num_eqs, num_unk))
        b = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        A[eq_idx, (j*p+k)*r + c] = U[i*m+j, c] * W[k*n+i, c]
                    eq_idx += 1
        
        sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        V = sol.reshape(m*p, r)
        return V
    
    def try_rank(r, num_restarts=10, max_iters=30):
        """Try to find a rank-r decomposition using alternating minimization."""
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                return None
            
            np.random.seed(restart * 1000 + r)
            
            # Initialize with random values
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            for iteration in range(max_iters):
                if time.time() > deadline - 10:
                    return None
                
                # Alternate: solve for W, then U, then V
                W = solve_linear_in_W(U, V, r)
                U = solve_linear_in_V(W, V, r) if False else solve_linear_in_U(V, W, r)
                V = solve_linear_in_V(U, W, r)
                
                # Check convergence
                if verify_full(U, V, W, tol=1e-8):
                    # Try rounding to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if verify_full(U_r, V_r, W_r, tol=1e-9):
                        return U_r, V_r, W_r
                    else:
                        return U, V, W
            
            # Final check
            if verify_full(U, V, W, tol=1e-8):
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if verify_full(U_r, V_r, W_r, tol=1e-9):
                    return U_r, V_r, W_r
                else:
                    return U, V, W
        
        return None
    
    # Try ranks from 72 downward
    for target_rank in range(71, 40, -1):
        if time.time() > deadline - 15:
            break
        
        result = try_rank(target_rank, num_restarts=5, max_iters=20)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break
    
    # If still schoolbook, try a few more structured approaches
    if best_rank == rank_schoolbook and time.time() < deadline - 20:
        # Try with different initializations
        for target_rank in [64, 60, 56, 52, 48]:
            if time.time() > deadline - 15:
                break
            
            for restart in range(8):
                if time.time() > deadline - 15:
                    break
                
                np.random.seed(restart * 777 + target_rank * 13)
                
                # Structured initialization: block-diagonal pattern
                U = np.zeros((n*m, target_rank))
                V = np.zeros((m*p, target_rank))
                W = np.zeros((p*n, target_rank))
                
                # Initialize with random sparse patterns
                for c in range(target_rank):
                    # Randomly choose which entries are non-zero
                    mask_U = np.random.random(n*m) < 0.3
                    mask_V = np.random.random(m*p) < 0.3
                    mask_W = np.random.random(p*n) < 0.3
                    
                    U[mask_U, c] = np.random.choice([-1, 1], size=mask_U.sum())
                    V[mask_V, c] = np.random.choice([-1, 1], size=mask_V.sum())
                    W[mask_W, c] = np.random.choice([-1, 1], size=mask_W.sum())
                
                # Run alternating minimization
                for iteration in range(30):
                    if time.time() > deadline - 15:
                        break
                    
                    W = solve_linear_in_W(U, V, target_rank)
                    U = solve_linear_in_U(V, W, target_rank)
                    V = solve_linear_in_V(U, W, target_rank)
                
                if verify_full(U, V, W, tol=1e-8):
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if verify_full(U_r, V_r, W_r, tol=1e-9):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_r, V_r, W_r
                    else:
                        best_rank = target_rank
                        best_U, best_V, best_W = U, V, W
                    break
            
            if best_rank < rank_schoolbook:
                break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
