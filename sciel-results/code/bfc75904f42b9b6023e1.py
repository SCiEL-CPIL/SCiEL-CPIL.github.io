import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Try to find better ranks
    # We know 45 is the baseline. 
    # Theoretical lower bound is roughly (nm * mp * pn)^(1/3) ? No.
    # Let's try ranks 30 to 44.
    
    # Pre-compute indices for faster error calculation
    # Eq (i,j,k) -> sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    # We will use a randomized approach:
    # 1. Fix U, V (random small integers)
    # 2. Solve linear system for W
    # 3. Round W to half-integers
    # 4. Verify
    
    # To make it faster, we can restrict U, V to {-1, 0, 1} initially.
    
    def verify(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Check if sum_c U[i,j,c] V[j,k,c] W[k,i,c] == 1 for all i,j,k
        # And if entries are half-integers
        for r in range(U.shape[1]):
            if not np.all(np.isclose(U[:, r] * 2, np.round(U[:, r] * 2))):
                return False
            if not np.all(np.isclose(V[:, r] * 2, np.round(V[:, r] * 2))):
                return False
            if not np.all(np.isclose(W[:, r] * 2, np.round(W[:, r] * 2))):
                return False
        
        # Exact check
        # Construct the tensor sum
        # We can use einsum or loops
        # For speed, we can compute the product for each c and sum
        # T = sum_c outer(U[:,c], V[:,c]) ... no, indices are mixed.
        
        # Let's just loop over i, j, k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def solve_for_W(U, V, R):
        # Build linear system A x = b
        # Variables: W flattened (pn * R)
        # Equations: for each (i,j,k), sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
        
        num_eqs = n * m * p
        num_vars = pn * R
        A = np.zeros((num_eqs, num_vars))
        b = np.ones(num_eqs)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(R):
                        coeff = U[i*m+j, c] * V[j*p+k, c]
                        if coeff != 0:
                            # W index is k*n + i, column c
                            var_idx = (k*n + i) * R + c
                            A[idx, var_idx] += coeff
                    idx += 1
        
        # Solve least squares
        # Since it might be underdetermined, we want a sparse/simple solution?
        # lstsq gives minimum norm solution.
        try:
            W_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
        
        W = W_flat.reshape((pn, R))
        return W

    # Search loop
    # We'll try ranks from 35 to 44
    for R in range(35, 45):
        if time.time() > deadline:
            break
            
        # Limit iterations per rank
        max_iters = 500
        iters_done = 0
        while iters_done < max_iters and time.time() < deadline:
            iters_done += 1
            
            # Random U, V with entries in {-1, 0, 1}
            # We can bias towards sparsity
            U = np.random.choice([-1, 0, 1], size=(nm, R))
            V = np.random.choice([-1, 0, 1], size=(mp, R))
            
            W = solve_for_W(U, V, R)
            if W is None:
                continue
            
            # Round W to nearest half-integer
            W_rounded = np.round(W * 2) / 2
            
            # Check if rounded W is a valid solution
            if verify(U, V, W_rounded):
                best_rank = R
                best_U, best_V, best_W = U, V, W_rounded
                # If we found a solution, we can try lower ranks? 
                # The loop goes 35..44, so we will try lower R next.
                # But we break the inner loop to save time for lower R?
                # Yes, if we found a valid R, we should try R-1.
                break 
        
    # If we found a better rank, use it. Else schoolbook.
    if best_rank < rank_schoolbook:
        # Convert to lists of lists as required by interface
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        U_list = U_sb.tolist()
        V_list = V_sb.tolist()
        W_list = W_sb.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
