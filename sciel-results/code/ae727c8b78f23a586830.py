import numpy as np
import time
import itertools
from collections import defaultdict

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    U_fb, V_fb, W_fb = schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # The tensor equation: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for the "wrong" combinations, the sum should be 0.
    # Actually, the full condition is:
    # sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k'*n+i', r] = 1 if (i=i', j=j', k'=k) else 0
    # Wait, let me re-read the convention.
    # W[k*n+i, r] - so W is indexed by (k, i)
    # The reconstruction is: C[i, k] = sum_j sum_r A[i,j] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For this to equal sum_j A[i,j]*B[j,k], we need:
    # sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k'*n+i', r] = delta(i,i')*delta(j,j')*delta(k,k')... 
    # No wait. Let me think more carefully.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Using decomposition: C[i,k] = sum_j sum_r (A[i,j] U[i*m+j, r]) * (V[j*p+k, r]) * W[k*n+i, r]
    # Hmm, this doesn't look right either. Let me reconsider.
    
    # The standard tensor decomposition for matrix multiplication:
    # M_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} where M is the matmul tensor
    # M_{i,j,k} represents the coefficient such that C[i,k] = sum_j M_{i,j,k} A[i,j] B[j,k]
    # For standard matmul, M_{i,j,k} = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]B[j,k])
    # Wait no. The tensor T such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # T[i,j,k] = 1 for all valid i,j,k.
    
    # So the condition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # That's it! Just 1 for all valid indices.
    
    # But wait, that can't be right for rank minimization. Let me re-read the task.
    # "exactly decompose the matrix multiplication tensor"
    # The matmul tensor for (n,m,p) has entries T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # And we need T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    
    # So yes, we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    # This is a system of n*m*p = 54 equations in n*m*r + m*p*r + p*n*r = 6r + 27r + 18r = 51r unknowns.
    # For r >= 2, we have more unknowns than equations, so solutions should exist.
    # The question is whether there are solutions with half-integer entries.
    
    # Strategy: For small r, set up the system and try to find half-integer solutions.
    # Key insight: if we can find ANY real solution, we can try to round to half-integers.
    # But better: use a structured approach.
    
    # For n=2, we can exploit the structure. Let's try ranks from a lower bound upward.
    # Lower bound: the rank must be at least max(n*p, m*p, n*m) but for matmul it's typically higher.
    # Actually, for 2×3×9, let's just try from r=27 upward (the max of the three products).
    
    # Approach: For each rank r, try to solve the system.
    # The system is: for each (i,j,k): sum_r' U[i,j,r'] * V[j,k,r'] * W[k,i,r'] = 1
    # This is a system of cubic equations. Hard to solve directly.
    
    # Alternative: fix two of the three matrices and solve linearly for the third.
    # If we fix U and V, then for each (k,i): sum_r' W[k,i,r'] * (sum over j of U[i,j,r']*V[j,k,r'])... 
    # No, that's not right either.
    
    # Let me think differently. Fix W. Then for each (i,j,k):
    # sum_r' (U[i,j,r'] * V[j,k,r']) * W[k,i,r'] = 1
    # Let X[i,j,k,r'] = U[i,j,r'] * V[j,k,r']
    # Then sum_r' X[i,j,k,r'] * W[k,i,r'] = 1
    # This is linear in X for fixed W, but X is a product of U and V.
    
    # Better approach: use the fact that for n=2, we can write the problem in a structured way.
    
    # Let me try a direct approach: for small ranks, use random initialization and gradient descent
    # on the loss function L = sum_{i,j,k} (sum_r U[i,j,r]V[j,k,r]W[k,i,r] - 1)^2
    
    # But we need EXACT half-integer solutions. So we need to find solutions that are exactly 1.
    
    # New strategy: enumerate possible half-integer values for a subset of variables,
    # then solve linearly for the rest.
    
    # For rank r, we have 51r variables. If we fix enough variables to make the system linear,
    # we can solve it.
    
    # Key insight: if we fix all entries of W to be in {0, 0.5, 1}, then the equations become:
    # sum_r' U[i,j,r'] * V[j,k,r'] * w_{k,i,r'} = 1
    # where w_{k,i,r'} is known. This is still nonlinear (product of U and V).
    
    # But if we also fix U, then it's linear in V:
    # sum_r' (U[i,j,r'] * w_{k,i,r'}) * V[j,k,r'] = 1
    # This is a linear system in V!
    
    # So: fix W and U, solve linearly for V. Then check if V has half-integer entries.
    
    # Even better: fix W to have a specific sparse pattern, fix U to have a specific pattern,
    # solve for V, and check.
    
    # Let's try this approach systematically.
    
    # For each rank r from lower bound to best_rank-1:
    #   Try various sparse patterns for W (each entry in {0, 0.5, 1})
    #   For each W pattern, try various sparse patterns for U
    #   Solve linear system for V
    #   Check if V is half-integer and the solution is exact
    
    # This is combinatorial, but for small r and with smart patterns, it might work.
    
    # Actually, let's try a simpler approach first: for each rank r, set up the full system
    # and use a numerical solver, then try to round to half-integers.
    
    # The system: F_{i,j,k} = sum_r' U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] - 1 = 0
    # for all i in range(n), j in range(m), k in range(p)
    
    # This is 54 equations. Unknowns: 6r + 27r + 18r = 51r.
    
    # For r=27: 1377 unknowns, 54 equations. Very underdetermined.
    # We can set most variables to 0 and solve for the rest.
    
    # Strategy: for each rank r, try to find a solution where most variables are 0,
    # and the nonzero ones are half-integers.
    
    # Let's try: for rank r, assume that each of the 54 equations involves exactly one
    # "active" term (like schoolbook but with sharing).
    
    # Actually, the most promising approach for a structural change:
    # Use the known fact that for 2×n×n matrix multiplication, the rank is at most 2n^2 - n
    # (or similar bounds). For 2×3×9, this doesn't directly apply.
    
    # Let me just try the numerical approach with better initialization and rounding.
    
    def solve_for_rank(r, max_attempts=20):
        """Try to find a half-integer decomposition of rank r."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                return None
            
            # Random initialization with half-integers
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            # Initialize with random half-integers in {-1, -0.5, 0, 0.5, 1}
            vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            
            # Make it sparse: only a fraction of entries are nonzero
            density = 0.3
            for idx in np.random.choice(n * m * r, size=int(n * m * r * density), replace=False):
                row = idx // r
                col = idx % r
                U[row, col] = np.random.choice(vals)
            
            for idx in np.random.choice(m * p * r, size=int(m * p * r * density), replace=False):
                row = idx // r
                col = idx % r
                V[row, col] = np.random.choice(vals)
            
            for idx in np.random.choice(p * n * r, size=int(p * n * r * density), replace=False):
                row = idx // r
                col = idx % r
                W[row, col] = np.random.choice(vals)
            
            # Now try to improve via local search: flip individual entries to half-integers
            # to reduce the error
            best_err = np.inf
            for it in range(50):
                if time.time() > deadline - 20:
                    break
                
                # Compute current error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            err += (val - 1.0) ** 2
                
                if err < best_err:
                    best_err = err
                
                if err < 1e-12:
                    # Check if it's exactly half-integer (it should be by construction)
                    return U, V, W
                
                # Try flipping one entry in U, V, or W
                improved = False
                
                # Try U
                for _ in range(100):
                    row = np.random.randint(0, n * m)
                    col = np.random.randint(0, r)
                    old = U[row, col]
                    for new_val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                        if new_val == old:
                            continue
                        U[row, col] = new_val
                        # Quick check: recompute error for affected equations
                        # Only equations involving (row, col) are affected
                        # row = i*m + j, so i = row//m, j = row%m
                        i = row // m
                        j = row % m
                        local_err = 0.0
                        for k in range(p):
                            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            local_err += (val - 1.0) ** 2
                        if local_err < err:
                            err = local_err
                            improved = True
                            break
                        U[row, col] = old
                    if improved:
                        break
                
                if improved:
                    continue
                
                # Try V
                for _ in range(100):
                    row = np.random.randint(0, m * p)
                    col = np.random.randint(0, r)
                    old = V[row, col]
                    for new_val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                        if new_val == old:
                            continue
                        V[row, col] = new_val
                        j = row // p
                        k = row % p
                        local_err = 0.0
                        for i in range(n):
                            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            local_err += (val - 1.0) ** 2
                        if local_err < err:
                            err = local_err
                            improved = True
                            break
                        V[row, col] = old
                    if improved:
                        break
                
                if improved:
                    continue
                
                # Try W
                for _ in range(100):
                    row = np.random.randint(0, p * n)
                    col = np.random.randint(0, r)
                    old = W[row, col]
                    for new_val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                        if new_val == old:
                            continue
                        W[row, col] = new_val
                        k = row // n
                        i = row % n
                        local_err = 0.0
                        for j in range(m):
                            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            local_err += (val - 1.0) ** 2
                        if local_err < err:
                            err = local_err
                            improved = True
                            break
                        W[row, col] = old
                    if improved:
                        break
                
                if not improved:
                    break
            
            # Final check
            if best_err < 1e-12:
                return U, V, W
        
        return None
    
    # Try ranks from lower bound upward
    lower_bound = 27  # max(n*p, m*p, n*m)
    
    for r in range(lower_bound, best_rank):
        if time.time() > deadline - 30:
            break
        
        result = solve_for_rank(r, max_attempts=10)
        if result is not None:
            U_sol, V_sol, W_sol = result
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_sol[i * m + j, :], V_sol[j * p + k, :] * W_sol[k * n + i, :])
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to lists
                best_rank = r
                best_U = U_sol.tolist()
                best_V = V_sol.tolist()
                best_W = W_sol.tolist()
                break  # Found the minimal rank, stop
    
    # Return the best found
    if best_rank < n * m * p:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U_fb, "V": V_fb, "W": W_fb}
