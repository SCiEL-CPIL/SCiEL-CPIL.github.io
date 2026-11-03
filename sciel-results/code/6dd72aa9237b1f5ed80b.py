"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Base solution: Rank 55
    # Structure:
    # 1. j=2 (last column of A, last row of B): 2x1x10 trivial -> rank 20
    # 2. j=0,1 (first 2 cols of A, first 2 rows of B): 2x2x10
    #    Split into 5 blocks of 2x2x2. Use Strassen (rank 7) for each.
    #    Total: 5 * 7 = 35
    # Total: 20 + 35 = 55
    
    rank_base = 55
    U = np.zeros((n*m, rank_base))
    V = np.zeros((m*p, rank_base))
    W = np.zeros((p*n, rank_base))
    
    col = 0
    
    # Part 1: j=2 (trivial)
    # A[i, 2] * B[2, k] -> C[i, k]
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1.0
            V[v_idx, col] = 1.0
            W[w_idx, col] = 1.0
            col += 1
            
    # Part 2: j=0,1 (Strassen on 2x2x2 blocks)
    # Indices for U (A matrix): i in {0,1}, j in {0,1}
    # u_00 = 0, u_01 = 1, u_10 = 3, u_11 = 4
    # Indices for V (B matrix): j in {0,1}, k in {0,1} (local)
    # v_00 = 0, v_01 = 1, v_10 = 10, v_11 = 11
    # Indices for W (C matrix): k in {0,1}, i in {0,1}
    # w_00 = 0, w_10 = 1, w_01 = 2, w_11 = 3
    
    u_00 = 0
    u_01 = 1
    u_10 = 3
    u_11 = 4
    
    # Strassen coefficients for 2x2x2
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    
    for b in range(5):
        # Local k indices for V and W
        # V: row j, col k. Global col k = 2*b + k_local
        v_00_idx = 0 * p + (2*b + 0)
        v_01_idx = 0 * p + (2*b + 1)
        v_10_idx = 1 * p + (2*b + 0)
        v_11_idx = 1 * p + (2*b + 1)
        
        # W: row k, col i. Global row k = 2*b + k_local
        w_00_idx = (2*b + 0) * n + 0
        w_10_idx = (2*b + 0) * n + 1
        w_01_idx = (2*b + 1) * n + 0
        w_11_idx = (2*b + 1) * n + 1
        
        # M1
        U[u_00, col] += 1.0; U[u_11, col] += 1.0
        V[v_00_idx, col] += 1.0; V[v_11_idx, col] += 1.0
        W[w_00_idx, col] += 1.0; W[w_11_idx, col] += 1.0
        col += 1
        
        # M2
        U[u_10, col] += 1.0; U[u_11, col] += 1.0
        V[v_00_idx, col] += 1.0
        W[w_10_idx, col] += 1.0; W[w_11_idx, col] += -1.0
        col += 1
        
        # M3
        U[u_00, col] += 1.0
        V[v_01_idx, col] += 1.0; V[v_11_idx, col] += -1.0
        W[w_01_idx, col] += 1.0; W[w_11_idx, col] += 1.0
        col += 1
        
        # M4
        U[u_11, col] += 1.0
        V[v_10_idx, col] += 1.0; V[v_00_idx, col] += -1.0
        W[w_00_idx, col] += 1.0; W[w_10_idx, col] += 1.0
        col += 1
        
        # M5
        U[u_00, col] += 1.0; U[u_01, col] += 1.0
        V[v_11_idx, col] += 1.0
        W[w_00_idx, col] += -1.0; W[w_01_idx, col] += 1.0
        col += 1
        
        # M6
        U[u_10, col] += 1.0; U[u_00, col] += -1.0
        V[v_00_idx, col] += 1.0; V[v_01_idx, col] += 1.0
        W[w_11_idx, col] += 1.0
        col += 1
        
        # M7
        U[u_01, col] += 1.0; U[u_11, col] += -1.0
        V[v_10_idx, col] += 1.0; V[v_11_idx, col] += 1.0
        W[w_00_idx, col] += 1.0
        col += 1
        
    # Verify base solution
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback to trivial if construction is wrong
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt Rank Reduction Search
    # Strategy: Fix U and V, solve for W.
    # We try to merge columns.
    # Let r be the target rank.
    # We select r columns from the base solution (or random combinations) to form U_r, V_r.
    # Then we solve for W_r such that the decomposition holds.
    # The condition is: for all i,j,k, sum_{c=1}^r U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    # This is a linear system for W.
    # Let x be the vector of W entries.
    # For each (i,j,k), we have an equation.
    # The coefficient for W[k*n+i, c] in the equation for (i,j,k) is U[i*m+j, c] * V[j*p+k, c].
    # So we have a system A * w = b, where b is all 1s.
    # A has dimensions (n*m*p) x (p*n*r).
    # We want to find a sparse W (or just any W) that satisfies this.
    # If the system is consistent, we have a valid decomposition of rank r.
    
    # We try ranks from 54 downwards.
    # To select columns, we can use a greedy approach or random sampling.
    # Given time constraints, we try a few random subsets of columns of size r.
    
    best_rank = rank_base
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # We will try to reduce rank by 1, 2, etc.
    # For each target rank r, we pick r indices from range(rank_base).
    # To be efficient, we might prioritize indices that cover the "trivial" part vs "Strassen" part.
    # However, random sampling is a good start.
    
    # Optimization: Only try if time permits.
    if time.time() < deadline - 30:
        rng = np.random.default_rng(42)
        
        for target_rank in range(rank_base - 1, max(rank_base - 10, 40), -1):
            if time.time() > deadline - 10:
                break
                
            found = False
            # Try multiple random subsets
            for _ in range(50):
                if time.time() > deadline - 10:
                    break
                
                indices = rng.choice(rank_base, size=target_rank, replace=False)
                U_r = U[:, indices]
                V_r = V[:, indices]
                
                # Build linear system for W
                # Variables: W[k*n+i, c] for k in 0..p-1, i in 0..n-1, c in 0..target_rank-1
                # Total variables: p*n*target_rank
                # Equations: n*m*p
                # Coefficient for equation (i,j,k) and variable (k', i', c) is 1 if k==k', i==i' and coeff is U[i*m+j, c]*V[j*p+k, c]
                
                # We can construct the matrix A.
                # A[e, v] where e = i*m*p + j*p + k, v = (k*n+i)*target_rank + c
                # A[e, v] = U[i*m+j, c] * V[j*p+k, c] if k'==k and i'==i else 0
                
                # This is sparse. We can use scipy.sparse if available, or just numpy for small sizes.
                # n*m*p = 60 equations.
                # p*n*target_rank = 20 * 50 = 1000 variables (approx).
                # Small enough for dense numpy.
                
                A = np.zeros((n*m*p, p*n*target_rank))
                b = np.ones(n*m*p)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            e = i*m*p + j*p + k
                            for c in range(target_rank):
                                v_idx = (k*n+i)*target_rank + c
                                A[e, v_idx] = U_r[i*m+j, c] * V_r[j*p+k, c]
                
                # Solve A * w = b
                # We want a solution. If rank(A) < number of variables, there are infinite solutions.
                # We just need one. np.linalg.lstsq gives the minimum norm solution.
                try:
                    w_sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                except Exception:
                    continue
                    
                # Check if solution is valid
                # Residuals should be close to 0
                if len(residuals) > 0 and residuals[0] > 1e-9:
                    continue
                
                # Reconstruct W
                W_r = np.zeros((p*n, target_rank))
                for k in range(p):
                    for i in range(n):
                        for c in range(target_rank):
                            W_r[k*n+i, c] = w_sol[(k*n+i)*target_rank + c]
                
                # Final verification
                if verify(U_r, V_r, W_r):
                    best_rank = target_rank
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    found = True
                    break
                    
            if found:
                # If we found a lower rank, we could try to go lower, 
                # but for now we just break out of the target_rank loop if we found a significant improvement?
                # Actually, we loop downwards, so we keep going to find the minimum.
                # But we break the inner loop. The outer loop continues to target_rank - 1.
                pass
        
    # Convert to lists for output
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
