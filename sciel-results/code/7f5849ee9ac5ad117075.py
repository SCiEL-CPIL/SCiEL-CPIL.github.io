"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    tensor = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute lower bounds from matrix flattening
    T0 = tensor.reshape(n*m, -1)
    rank0 = np.linalg.matrix_rank(T0)
    T1 = tensor.transpose(1,0,2).reshape(m*p, -1)
    rank1 = np.linalg.matrix_rank(T1)
    T2 = tensor.transpose(2,0,1).reshape(p*n, -1)
    rank2 = np.linalg.matrix_rank(T2)
    lower_bound = max(rank0, rank1, rank2)
    
    # Trivial solution with rank = n*m*p
    rank_trivial = n * m * p
    U_trivial = np.zeros((n*m, rank_trivial))
    V_trivial = np.zeros((m*p, rank_trivial))
    W_trivial = np.zeros((p*n, rank_trivial))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i*m+j, col] = 1
                V_trivial[j*p+k, col] = 1
                W_trivial[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_trivial.copy(), V_trivial.copy(), W_trivial.copy()
    best_rank = rank_trivial
    
    # If lower bound equals trivial rank, we're done
    if lower_bound >= rank_trivial:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Strategy: Exploit rectangular structure by fixing U to identity-like pattern
    # This decouples the problem and allows structured search for V and W
    
    found_better = False
    
    # Try structured U patterns: identity-like with some randomization
    for trial_rank in range(max(lower_bound, n*m), rank_trivial):
        if time.time() > deadline - 20:
            break
        
        # Generate structured U: each row has exactly one non-zero (1 or -1)
        # This makes the system linear in V and W entries
        num_structured_attempts = 20
        for attempt in range(num_structured_attempts):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(42 + trial_rank * 1000 + attempt)
            
            # Create U with exactly one non-zero per row
            U_fixed = np.zeros((n*m, trial_rank))
            for a in range(n*m):
                c = np.random.randint(0, trial_rank)
                U_fixed[a, c] = np.random.choice([-1, 1])
            
            # With U having one non-zero per row, we can solve for V and W
            # more efficiently using the structure
            
            # Initialize V and W
            V_curr = np.random.randn(m*p, trial_rank)
            W_curr = np.random.randn(p*n, trial_rank)
            
            # Run ALS with more iterations and better convergence check
            converged = False
            prev_error = np.inf
            for iteration in range(200):
                if time.time() > deadline - 20:
                    break
                
                # Update W
                UtU = U_fixed.T @ U_fixed
                VtV = V_curr.T @ V_curr
                G_W = UtU * VtV
                
                rhs_W = np.einsum('abc,ar,br->cr', tensor, U_fixed, V_curr)
                
                reg = 1e-10 * np.eye(trial_rank)
                try:
                    W_curr = np.linalg.solve(G_W + reg, rhs_W.T).T
                except np.linalg.LinAlgError:
                    W_curr = np.linalg.lstsq(G_W + reg, rhs_W.T, rcond=None)[0].T
                
                # Update V
                VtV = V_curr.T @ V_curr
                WtW = W_curr.T @ W_curr
                G_V = UtU * WtW
                
                rhs_V = np.einsum('abc,ar,cr->br', tensor, U_fixed, W_curr)
                
                try:
                    V_curr = np.linalg.solve(G_V + reg, rhs_V.T).T
                except np.linalg.LinAlgError:
                    V_curr = np.linalg.lstsq(G_V + reg, rhs_V.T, rcond=None)[0].T
                
                # Check convergence
                recon = np.einsum('ar,br,cr->abc', U_fixed, V_curr, W_curr)
                error = np.linalg.norm(recon - tensor)
                
                if error < 1e-10:
                    converged = True
                    break
                
                # Early stopping if no improvement
                if error > prev_error * 0.99:
                    break
                prev_error = error
            
            if converged:
                # Try to round to rational values
                best_rounded = None
                best_round_error = np.inf
                
                for factor in [1, 2, 4, 8]:
                    V_r = np.round(V_curr * factor) / factor
                    W_r = np.round(W_curr * factor) / factor
                    
                    recon_r = np.einsum('ar,br,cr->abc', U_fixed, V_r, W_r)
                    error_r = np.linalg.norm(recon_r - tensor)
                    
                    if error_r < 1e-8 and error_r < best_round_error:
                        best_rounded = (U_fixed.copy(), V_r.copy(), W_r.copy())
                        best_round_error = error_r
                
                if best_rounded is not None:
                    best_U, best_V, best_W = best_rounded
                    best_rank = trial_rank
                    found_better = True
                    break
        
        if found_better:
            break
    
    # If structured search didn't work, try random sparse U
    if not found_better:
        for trial_rank in range(max(lower_bound, n*m), rank_trivial):
            if time.time() > deadline - 10:
                break
            
            num_random_attempts = 30
            for attempt in range(num_random_attempts):
                if time.time() > deadline - 10:
                    break
                
                np.random.seed(999 + trial_rank * 100 + attempt)
                U_fixed = np.zeros((n*m, trial_rank))
                for a in range(n*m):
                    num_nz = np.random.randint(1, 3)
                    cols = np.random.choice(trial_rank, size=num_nz, replace=False)
                    for c in cols:
                        U_fixed[a, c] = np.random.choice([-1, 1])
                
                V_curr = np.random.randn(m*p, trial_rank)
                W_curr = np.random.randn(p*n, trial_rank)
                
                converged = False
                for iteration in range(100):
                    if time.time() > deadline - 10:
                        break
                    
                    UtU = U_fixed.T @ U_fixed
                    VtV = V_curr.T @ V_curr
                    G_W = UtU * VtV
                    rhs_W = np.einsum('abc,ar,br->cr', tensor, U_fixed, V_curr)
                    reg = 1e-8 * np.eye(trial_rank)
                    try:
                        W_curr = np.linalg.solve(G_W + reg, rhs_W.T).T
                    except np.linalg.LinAlgError:
                        W_curr = np.linalg.lstsq(G_W + reg, rhs_W.T, rcond=None)[0].T
                    
                    VtV = V_curr.T @ V_curr
                    WtW = W_curr.T @ W_curr
                    G_V = UtU * WtW
                    rhs_V = np.einsum('abc,ar,cr->br', tensor, U_fixed, W_curr)
                    try:
                        V_curr = np.linalg.solve(G_V + reg, rhs_V.T).T
                    except np.linalg.LinAlgError:
                        V_curr = np.linalg.lstsq(G_V + reg, rhs_V.T, rcond=None)[0].T
                    
                    recon = np.einsum('ar,br,cr->abc', U_fixed, V_curr, W_curr)
                    error = np.linalg.norm(recon - tensor)
                    if error < 1e-8:
                        converged = True
                        break
                
                if converged:
                    for factor in [1, 2, 4]:
                        V_r = np.round(V_curr * factor) / factor
                        W_r = np.round(W_curr * factor) / factor
                        recon_r = np.einsum('ar,br,cr->abc', U_fixed, V_r, W_r)
                        error_r = np.linalg.norm(recon_r - tensor)
                        if error_r < 1e-6:
                            best_U, best_V, best_W = U_fixed.copy(), V_r.copy(), W_r.copy()
                            best_rank = trial_rank
                            found_better = True
                            break
                    if found_better:
                        break
            
            if found_better:
                break
    
    # Return best solution found
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
