"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 20s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 15
    pn = p * n  # 10
    
    # Build the target tensor T[i,j,k] = 1 for all valid (i,j,k)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Start with schoolbook rank 30
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def to_lists(U, V, W):
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }
    
    # Verify a decomposition
    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Best found so far
    best_rank = rank_school
    best_result = to_lists(U_school, V_school, W_school)
    
    # Try to reduce rank by removing columns and adjusting
    # For each candidate rank r from 29 down to 25, try to find a decomposition
    for target_rank in range(29, 24, -1):
        if time.time() > deadline:
            break
        
        # Strategy: start from schoolbook, remove one column, and try to fix
        # by adjusting the remaining columns
        
        # Try all ways to remove one of the 30 columns
        for remove_idx in range(rank_school):
            if time.time() > deadline:
                break
            
            # Keep all columns except remove_idx
            keep = [c for c in range(rank_school) if c != remove_idx]
            
            U_sub = U_school[:, keep].copy()
            V_sub = V_school[:, keep].copy()
            W_sub = W_school[:, keep].copy()
            
            # Now we need to adjust U_sub, V_sub, W_sub so that the tensor is still represented
            # The current representation with these 29 terms will have some entries wrong
            # Specifically, the entry (i0,j0,k0) corresponding to remove_idx will be 0 instead of 1
            # And other entries might be affected if we change the values
            
            # Let's find which (i,j,k) triple was removed
            i0 = remove_idx // (m * p)
            j0 = (remove_idx // p) % m
            k0 = remove_idx % p
            
            # With the removed column, T[i0,j0,k0] = 0, but we need it to be 1
            # Other entries are still 1 (since each (i,j,k) only had one contributing term in schoolbook)
            
            # We need to modify the remaining 29 terms to add 1 to T[i0,j0,k0]
            # while keeping all other T[i,j,k] = 1
            
            # This is a hard constraint satisfaction problem. Let's use a randomized approach:
            # Pick a random column c' to modify, and try to add contributions to (i0,j0,k0)
            
            # Alternative: use the fact that we can scale and add. 
            # Let's try: for the removed triple (i0,j0,k0), we need some combination of the 
            # remaining terms to contribute 1 to this position.
            
            # The remaining terms that have non-zero in all three positions for (i0,j0,k0) are those
            # where U[i0*m+j0, c] != 0, V[j0*p+k0, c] != 0, W[k0*n+i0, c] != 0
            # In schoolbook, only the removed term had all three non-zero for (i0,j0,k0).
            # So we need to introduce new non-zero values.
            
            # Let's try a different approach: use a parameterized search
            # Add a small perturbation to some entries and solve
            
            # Actually, let's try a more systematic approach using linear algebra
            # Fix U and V, solve for W, then check
            
            # For a given U (6 x r) and V (15 x r), the constraint on W is:
            # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # This is linear in W!
            
            # Let's fix U and V from the schoolbook (with one column removed), 
            # and solve for W.
            
            r = target_rank
            # Use U_sub and V_sub as fixed
            # Build the linear system for W
            # Unknowns: W[k*n+i, c] for k in 0..p-1, i in 0..n-1, c in 0..r-1
            # That's p*n*r = 10*r unknowns
            # Equations: n*m*p = 30 equations
            
            # For each (i,j,k), the equation is:
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            
            # This couples different W entries through the sum over c, but for fixed (i,k),
            # only W[k*n+i, c] for c in 0..r-1 appear.
            
            # Actually, for fixed (i,k), the equation involves W[k*n+i, c] for all c,
            # weighted by U[i*m+j,c] * V[j*p+k,c] which depends on j.
            
            # So for fixed (i,k), we have m equations (one per j):
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for j=0..m-1
            
            # This is a system of m equations in r unknowns (W[k*n+i, 0..r-1])
            # If m <= r, this is underdetermined and we can solve it.
            
            # For m=3, r=29, we have 3 equations in 29 unknowns - very underdetermined.
            # We can set most W entries to 0 and solve for 3 of them.
            
            W_sol = np.zeros((pn, r))
            solvable = True
            
            for i in range(n):
                for k in range(p):
                    # Build A matrix: m x r, where A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    A = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            A[j, c] = U_sub[i*m+j, c] * V_sub[j*p+k, c]
                    
                    # Solve A @ W_col = ones(m)
                    # W_col = W_sol[k*n+i, :]
                    # Use least squares or find a particular solution
                    try:
                        # Find a solution with minimum norm
                        W_col, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        # Check if it's an exact solution
                        if np.max(np.abs(A @ W_col - np.ones(m))) > 1e-9:
                            solvable = False
                            break
                        W_sol[k*n+i, :] = W_col
                    except:
                        solvable = False
                        break
                if not solvable:
                    break
            
            if solvable:
                # Verify the full decomposition
                if verify(U_sub, V_sub, W_sol):
                    best_rank = target_rank
                    best_result = to_lists(U_sub, V_sub, W_sol)
                    break  # Found a valid decomposition at this rank
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
        
        # If we found a solution at this rank, we can stop (we're going from high to low rank)
        if best_rank == target_rank:
            break
    
    # Try even lower ranks with a different strategy: random initialization
    for target_rank in range(best_rank - 1, 20, -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 1000 + target_rank)
            
            U_rand = np.random.randn(nm, target_rank) * 0.5
            V_rand = np.random.randn(mp, target_rank) * 0.5
            W_rand = np.zeros((pn, target_rank))
            
            # Try to solve for W given U_rand and V_rand
            solvable = True
            for i in range(n):
                for k in range(p):
                    A = np.zeros((m, target_rank))
                    for j in range(m):
                        for c in range(target_rank):
                            A[j, c] = U_rand[i*m+j, c] * V_rand[j*p+k, c]
                    
                    try:
                        W_col, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        if np.max(np.abs(A @ W_col - np.ones(m))) > 1e-9:
                            solvable = False
                            break
                        W_rand[k*n+i, :] = W_col
                    except:
                        solvable = False
                        break
                if not solvable:
                    break
            
            if solvable and verify(U_rand, V_rand, W_rand):
                best_rank = target_rank
                best_result = to_lists(U_rand, V_rand, W_rand)
                break
        
        if best_rank == target_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END

