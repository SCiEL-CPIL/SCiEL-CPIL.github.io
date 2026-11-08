"""Matrix multiplication 3,3,6 - rank-minimal tensor decomposition search."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Build the multiplication tensor T[i, j, k] = 1 iff output (i,k) = sum_l A[i,l]*B[l,k]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # For each term c: U[:,c] * V[:,c] * W[:,c] should sum to T
    # T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k)
    
    # Create tensor as 3D array
    T = np.zeros((n*m, m*p, p*n), dtype=np.int32)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Start with schoolbook
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school), dtype=np.float64)
    V_school = np.zeros((m*p, rank_school), dtype=np.float64)
    W_school = np.zeros((p*n, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank decompositions
    # Use randomized approach: try various ranks and optimize
    rng = np.random.RandomState(42)
    
    # Try a structured approach: exploit the rectangular shape
    # A is 3x3, B is 3x6, so we can think of B as [B1, B2] where each is 3x3
    # AB = A*B1, A*B2 (two 3x3 multiplications)
    # If rank of 3x3 mult is r, then rank of 3x3 * 3x6 is at most 2r
    # Known: rank of 3x3 mult is between 19 and 23 (Landsberg bound)
    # Strassen gives 7 for 2x2, so for 3x3 we can use block methods
    
    # Let's try to construct a decomposition using known low-rank methods
    # For 3x3: use the fact that we can embed in 4x4 and use known results
    # Or use explicit constructions
    
    # Try random optimization for various target ranks
    target_ranks = [36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]
    
    for target_r in target_ranks:
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        max_attempts = 3 if target_r > 30 else 5
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with small random values
            U = rng.randn(n*m, target_r) * 0.5
            V = rng.randn(m*p, target_r) * 0.5
            W = rng.randn(p*n, target_r) * 0.5
            
            # Simple iterative optimization: for each term, try to minimize residual
            # This is a non-convex problem, so use simple heuristics
            
            # Alternative: use the structure - each term should capture some "rank-1" part
            # Let's try a greedy approach
            
            # Actually, let's try a different strategy:
            # Use the fact that we can decompose using Kronecker products or block structure
            
            # For 3x3 * 3x6:
            # Write A as 3x3, B as 3x6
            # We can use the identity: AB where B = [B1 B2]
            # Compute A*B1 and A*B2 separately
            # If we have a rank-r decomposition for 3x3*3x3, we get rank 2r for 3x3*3x6
            
            # Let me try to construct a rank-19 or rank-20 decomposition for 3x3*3x3
            # using known explicit constructions
            
            # Actually, let's just try random search more aggressively
            # Use gradient-free approach: perturb and keep improvements
            
            residual = np.zeros((n*m, m*p, p*n), dtype=np.float64)
            for c in range(target_r):
                residual += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
            
            current_error = np.sum((residual - T.astype(np.float64))**2)
            
            # Try to improve by adjusting one term at a time
            improved = False
            for c in range(min(target_r, 10)):  # Only adjust first few terms
                if time.time() > deadline - 15:
                    break
                for coord in range(3):  # U, V, or W
                    if time.time() > deadline - 15:
                        break
                    if coord == 0:
                        for idx in range(n*m):
                            old_val = U[idx, c]
                            for delta in [-0.1, -0.05, 0.05, 0.1]:
                                U[idx, c] = old_val + delta
                                res = np.zeros((n*m, m*p, p*n), dtype=np.float64)
                                for cc in range(target_r):
                                    res += np.einsum('a,b,c->abc', U[:,cc], V[:,cc], W[:,cc])
                                err = np.sum((res - T.astype(np.float64))**2)
                                if err < current_error:
                                    current_error = err
                                    improved = True
                                    break
                                else:
                                    U[idx, c] = old_val
                            else:
                                continue
                            break
                    elif coord == 1:
                        for idx in range(m*p):
                            old_val = V[idx, c]
                            for delta in [-0.1, -0.05, 0.05, 0.1]:
                                V[idx, c] = old_val + delta
                                res = np.zeros((n*m, m*p, p*n), dtype=np.float64)
                                for cc in range(target_r):
                                    res += np.einsum('a,b,c->abc', U[:,cc], V[:,cc], W[:,cc])
                                err = np.sum((res - T.astype(np.float64))**2)
                                if err < current_error:
                                    current_error = err
                                    improved = True
                                    break
                                else:
                                    V[idx, c] = old_val
                            else:
                                continue
                            break
                    else:
                        for idx in range(p*n):
                            old_val = W[idx, c]
                            for delta in [-0.1, -0.05, 0.05, 0.1]:
                                W[idx, c] = old_val + delta
                                res = np.zeros((n*m, m*p, p*n), dtype=np.float64)
                                for cc in range(target_r):
                                    res += np.einsum('a,b,c->abc', U[:,cc], V[:,cc], W[:,cc])
                                err = np.sum((res - T.astype(np.float64))**2)
                                if err < current_error:
                                    current_error = err
                                    improved = True
                                    break
                                else:
                                    W[idx, c] = old_val
                            else:
                                continue
                            break
            
            if current_error < 1e-6:
                # Found a valid decomposition
                best_rank = target_r
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                found = True
                break
        if found:
            break
    
    # Convert to lists and return
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
