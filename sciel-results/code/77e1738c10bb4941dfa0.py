"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    # Schoolbook decomposition (rank 84) as guaranteed fallback
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
        return {"U": U, "V": V, "W": W}

    # Verify a decomposition
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    fallback = schoolbook()

    # For the all-ones tensor of shape (n, m, p), try structured decompositions.
    # Key idea: use the fact that 1 = sum of products where we can use 
    # intermediate values. Try ranks from a low bound upward.
    
    # Lower bound: at least max(n, m, p) for non-trivial, but let's start lower.
    # The rank of all-ones tensor (a,b,c) is at least max(a,b,c) in some cases.
    # Let's try from rank 10 upward.
    
    best_result = None
    best_rank = n * m * p + 1
    
    # Strategy: For each rank r, try random U, V with entries in small set,
    # then solve for W exactly (since W entries are determined by U, V).
    # Then verify.
    
    for rank in range(10, n * m * p + 1):
        if time.time() > deadline:
            break
        
        found = False
        # Number of attempts: more for smaller ranks (harder), fewer for larger
        num_attempts = 500 if rank < 30 else 200
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            # Generate random U, V with entries from a small set
            # Use {0, 1, -1} for sparsity
            entries = [0, 1, -1]
            U = [[random.choice(entries) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice(entries) for _ in range(rank)] for _ in range(m * p)]
            
            # Compute M[k][i][r] = sum_j U[i*m+j][r] * V[j*p+k][r]
            M = [[[0] * rank for _ in range(n)] for _ in range(p)]
            for k in range(p):
                for i in range(n):
                    for r in range(rank):
                        s = 0
                        for j in range(m):
                            s += U[i * m + j][r] * V[j * p + k][r]
                        M[k][i][r] = s
            
            # Build W: for each (k,i), need sum_r W[k*n+i][r] * M[k][i][r] = 1
            # This is a linear equation in W[k*n+i][:]. 
            # We need to find W[k*n+i] such that dot(W_row, M[k][i]) = 1
            # For each (k,i), this is one linear equation with rank unknowns.
            # We have p*n equations total, each with rank unknowns (W has p*n rows of length rank).
            # Actually W is (p*n) x rank, and for each (k,i), the constraint is:
            #   sum_r W[k*n+i][r] * M[k][i][r] = 1
            # So for each (k,i), we need to pick W[k*n+i] from {values} such that the dot product = 1.
            
            W = [[0] * rank for _ in range(p * n)]
            valid = True
            for k in range(p):
                for i in range(n):
                    # Find W row such that sum_r W[r] * M[k][i][r] = 1
                    # Try simple values: for each r, if M[k][i][r] is nonzero, 
                    # we can set W[r] = 1/M[k][i][r] and others 0.
                    # But we need integer/half-integer entries.
                    found_w = False
                    for r in range(rank):
                        mv = M[k][i][r]
                        if mv == 1:
                            W[k * n + i][r] = 1
                            found_w = True
                            break
                        elif mv == -1:
                            W[k * n + i][r] = -1
                            found_w = True
                            break
                        elif mv == 2:
                            W[k * n + i][r] = 0.5
                            found_w = True
                            break
                        elif mv == -2:
                            W[k * n + i][r] = -0.5
                            found_w = True
                            break
                        elif mv == 3:
                            W[k * n + i][r] = 1.0 / 3
                            found_w = True
                            break
                        elif mv == -3:
                            W[k * n + i][r] = -1.0 / 3
                            found_w = True
                            break
                    if not found_w:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                if verify(U, V, W, rank):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    return fallback
# EVOLVE-BLOCK-END
