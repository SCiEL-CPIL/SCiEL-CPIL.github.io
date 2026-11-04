import numpy as np
import random

def construct():
    n, m, p = 2, 3, 8
    
    # Target tensor T[i, j, k] = 1
    # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # Precompute indices
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def check_solution(U, V, W, rank):
        for i, j, k in indices:
            val = 0.0
            for r in range(rank):
                val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
            if abs(val - 1.0) > 1e-9:
                return False
        return True

    def local_search(U, V, W, rank, max_iters=500):
        # Convert to flat arrays for easier manipulation? 
        # Or just mutate lists.
        # Values should be half-integers.
        # Start with random initialization? Or schoolbook subset?
        # Better: Start with schoolbook, then try to merge columns?
        # Or start with random and minimize error.
        
        # Let's try a different init: 
        # Pick 'rank' random (i,j,k) triples and set U,V,W to 1 at those positions.
        # This is a sparse init.
        
        best_error = float('inf')
        best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
        
        for _ in range(max_iters):
            # Calculate current error
            error = 0.0
            for i, j, k in indices:
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                error += (val - 1.0)**2
            
            if error < 1e-9:
                return U, V, W, True
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
                
            # Mutate one random entry in U, V, or W
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                row = random.randint(0, n*m-1)
                col = random.randint(0, rank-1)
                old_val = U[row][col]
                U[row][col] = old_val + random.choice([-0.5, 0.5, -1.0, 1.0])
                # Keep values in a reasonable range, e.g., [-2, 2]
                if U[row][col] < -2 or U[row][col] > 2:
                    U[row][col] = old_val
            elif mat == 'V':
                row = random.randint(0, m*p-1)
                col = random.randint(0, rank-1)
                old_val = V[row][col]
                V[row][col] = old_val + random.choice([-0.5, 0.5, -1.0, 1.0])
                if V[row][col] < -2 or V[row][col] > 2:
                    V[row][col] = old_val
            else:
                row = random.randint(0, p*n-1)
                col = random.randint(0, rank-1)
                old_val = W[row][col]
                W[row][col] = old_val + random.choice([-0.5, 0.5, -1.0, 1.0])
                if W[row][col] < -2 or W[row][col] > 2:
                    W[row][col] = old_val
                    
        # Return best found so far (might not be exact)
        return best_U, best_V, best_W, False

    # Try ranks lower than schoolbook
    schoolbook_rank = n * m * p
    for rank in range(schoolbook_rank - 4, schoolbook_rank): # Try 44, 45, 46, 47
        # Initialize with schoolbook-like structure but compressed?
        # Simple init: Zero matrix, then set random non-zeros.
        U = [[0.0]*rank for _ in range(n*m)]
        V = [[0.0]*rank for _ in range(m*p)]
        W = [[0.0]*rank for _ in range(p*n)]
        
        # Seed with some 1s to help convergence
        for r in range(rank):
            i, j, k = random.randint(0, n-1), random.randint(0, m-1), random.randint(0, p-1)
            U[i*m+j][r] = 1.0
            V[j*p+k][r] = 1.0
            W[k*n+i][r] = 1.0
            
        success = False
        # Try a few random restarts
        for restart in range(20):
            U_res, V_res, W_res, is_exact = local_search(U, V, W, rank, max_iters=100)
            if is_exact:
                return {"U": U_res, "V": V_res, "W": W_res}
                
    # Fallback to Schoolbook
    rank = schoolbook_rank
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
                
    return {"U": U, "V": V, "W": W}
