import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit
    n, m, p = 2, 4, 5
    
    # Target: find minimal rank r
    # U: (n*m, r) -> (8, r)
    # V: (m*p, r) -> (20, r)
    # W: (p*n, r) -> (10, r)
    
    # Indices:
    # U[i*m + j, s]
    # V[j*p + k, s]
    # W[k*n + i, s]
    
    # Equation for (i, j, k):
    # sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1
    
    # Linear in W for fixed U, V.
    # For fixed (k, i), let x_s = W[k*n+i, s].
    # sum_s (U[i*m+j, s] * V[j*p+k, s]) * x_s = 1 for j=0..m-1
    
    # Let A_{k,i} be a matrix of size (m, r) where A_{k,i}[j, s] = U[i*m+j, s] * V[j*p+k, s]
    # We need to solve A_{k,i} x = 1 (vector of ones of length m)
    
    def try_rank(r, rng):
        # Initialize U and V randomly
        U = rng.standard_normal((n*m, r))
        V = rng.standard_normal((m*p, r))
        
        # Check solvability for all (k, i)
        W = np.zeros((p*n, r))
        
        for k in range(p):
            for i in range(n):
                # Construct A_{k,i} of shape (m, r)
                # A[j, s] = U[i*m + j, s] * V[j*p + k, s]
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i*m + j, :] * V[j*p + k, :]
                
                # Solve A x = 1
                b = np.ones(m)
                try:
                    # Use least squares to find x
                    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    # Check if the solution is exact (or close enough)
                    if np.linalg.norm(A @ x - b) < 1e-8:
                        W[k*n + i, :] = x
                    else:
                        return None
                except np.linalg.LinAlgError:
                    return None
        
        return U, V, W, r

    # Try ranks from lower bound up
    # Lower bound for rank of 2x4x5 tensor?
    # Rank >= max(n*m, m*p, p*n) / something?
    # Schoolbook is 40.
    # Let's start from 10 and go up.
    # Actually, let's start from a reasonable estimate like 20 or 30.
    # But random search might fail for small r if the manifold of solutions is small.
    # Let's try r = 16, 20, 24, 28, 32, 36.
    
    rng = np.random.default_rng(42)
    
    best_result = None
    
    # We can also try to use the Strassen structure as a starting point for r=36
    # But random search is simpler to code.
    
    # To be safe, let's try a range of r values.
    # If we find a solution for r, we can try smaller r.
    
    # Let's try r from 12 to 40 in steps of 2.
    # If we find a solution, we record it.
    
    found = False
    current_rank = 40
    
    # Optimization: If we find a solution for r, we can try r-2, etc.
    # But random search might be slow.
    
    # Let's try a few random restarts for each r.
    
    for r in range(12, 41, 2):
        if time.time() > deadline - 5:
            break
            
        # Try a few random initializations
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
                
            res = try_rank(r, rng)
            if res is not None:
                U, V, W, rank = res
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            if abs(val - 1.0) > 1e-6:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if valid:
                    current_rank = rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    found = True
                    # Since we want minimal rank, and we are iterating up, 
                    # we should actually iterate DOWN if we know an upper bound.
                    # But we don't know if lower r works.
                    # So we should iterate from low to high and stop at first success?
                    # No, random search might fail for low r even if a solution exists.
                    # So we should try low r first.
                    # If we find a solution for r=12, that's great.
                    # If not, try 14, etc.
                    # So the loop `for r in range(12, 41, 2)` is correct.
                    # We can break out of the attempt loop, but should we break the r loop?
                    # Yes, if we found a solution for the current smallest r, it's likely optimal or near optimal.
                    # However, it's possible r=12 has no solution, but r=14 does.
                    # So we should continue to the next r only if we didn't find a solution for current r.
                    # But we found one for r. So we can stop?
                    # Wait, what if r=12 failed, r=14 succeeded? We return 14.
                    # What if r=12 succeeded? We return 12.
                    # So yes, break the r loop as well?
                    # Actually, we should try to find the *minimal* rank.
                    # If we find a solution for r, it's a valid rank.
                    # Is it minimal? We don't know.
                    # But since we started from low, it's the lowest we found.
                    # We should probably not break immediately if we want to be sure, 
                    # but given time constraints, the first success at low r is a good bet.
                    # Let's break both loops to save time, as finding a lower rank is unlikely if we just found one at low r?
                    # No, if r=12 works, it's better than r=14.
                    # So we should break.
                    break
            # End if res is not None
        # End for attempt
        if found:
            break
    # End for r
    
    if not found:
        # Fallback to known valid decomposition (rank 36)
        # Re-construct the Strassen-based rank 36 solution
        def build_rank36():
            rank = 36
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            def add_strassen(col_offset, j_offset, k_offset):
                U[0*m + j_offset + 0, col_offset + 0] += 1
                U[1*m + j_offset + 1, col_offset + 0] += 1
                V[(j_offset + 0)*p + k_offset + 0, col_offset + 0] += 1
                V[(j_offset + 1)*p + k_offset + 1, col_offset + 0] += 1
                W[(k_offset + 0)*n + 0, col_offset + 0] += 1
                W[(k_offset + 1)*n + 1, col_offset + 0] += 1
                
                U[1*m + j_offset + 0, col_offset + 1] += 1
                U[1*m + j_offset + 1, col_offset + 1] += 1
                V[(j_offset + 0)*p + k_offset + 0, col_offset + 1] += 1
                W[(k_offset + 0)*n + 1, col_offset + 1] += 1
                W[(k_offset + 1)*n + 1, col_offset + 1] += -1
                
                U[0*m + j_offset + 0, col_offset + 2] += 1
                V[(j_offset + 0)*p + k_offset + 1, col_offset + 2] += 1
                V[(j_offset + 1)*p + k_offset + 1, col_offset + 2] += -1
                W[(k_offset + 1)*n + 0, col_offset + 2] += 1
                W[(k_offset + 1)*n + 1, col_offset + 2] += 1
                
                U[1*m + j_offset + 1, col_offset + 3] += 1
                V[(j_offset + 0)*p + k_offset + 0, col_offset + 3] += -1
                V[(j_offset + 1)*p + k_offset + 0, col_offset + 3] += 1
                W[(k_offset + 0)*n + 0, col_offset + 3] += 1
                W[(k_offset + 0)*n + 1, col_offset + 3] += 1
                
                U[0*m + j_offset + 0, col_offset + 4] += 1
                U[0*m + j_offset + 1, col_offset + 4] += 1
                V[(j_offset + 1)*p + k_offset + 1, col_offset + 4] += 1
                W[(k_offset + 0)*n + 0, col_offset + 4] += -1
                W[(k_offset + 1)*n + 0, col_offset + 4] += 1
                
                U[0*m + j_offset + 0, col_offset + 5] += -1
                U[1*m + j_offset + 0, col_offset + 5] += 1
                V[(j_offset + 0)*p + k_offset + 0, col_offset + 5] += 1
                V[(j_offset + 0)*p + k_offset + 1, col_offset + 5] += 1
                W[(k_offset + 1)*n + 1, col_offset + 5] += 1
                
                U[0*m + j_offset + 1, col_offset + 6] += 1
                U[1*m + j_offset + 1, col_offset + 6] += -1
                V[(j_offset + 1)*p + k_offset + 0, col_offset + 6] += 1
                V[(j_offset + 1)*p + k_offset + 1, col_offset + 6] += 1
                W[(k_offset + 0)*n + 0, col_offset + 6] += 1

            def add_schoolbook_2x2x1(col_offset, j_offset, k_val):
                for i in range(2):
                    for j_local in range(2):
                        idx = i*2 + j_local
                        U[i*m + j_offset + j_local, col_offset + idx] += 1
                        V[(j_offset + j_local)*p + k_val, col_offset + idx] += 1
                        W[k_val*n + i, col_offset + idx] += 1
            
            col = 0
            add_strassen(col, 0, 0); col += 7
            add_strassen(col, 0, 2); col += 7
            add_schoolbook_2x2x1(col, 0, 4); col += 4
            add_strassen(col, 2, 0); col += 7
            add_strassen(col, 2, 2); col += 7
            add_schoolbook_2x2x1(col, 2, 4); col += 4
            
            return U, V, W, rank

        U, V, W, rank = build_rank36()
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return best_result
