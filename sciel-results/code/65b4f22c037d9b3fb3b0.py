import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s limit
    n, m, p = DIMENSIONS
    
    # Start with the trivial decomposition (rank = n*m*p = 54)
    rank0 = n * m * p
    U = [[0] * rank0 for _ in range(n * m)]
    V = [[0] * rank0 for _ in range(m * p)]
    W = [[0] * rank0 for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank0
    
    # Try to find a lower rank decomposition using random search + verification
    # The tensor T is defined by T[i,j,k] = 1 if we're multiplying A[i,j] * B[j,k]
    # We need U, V, W such that sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    
    # Actually, the condition is: for each (i,j,k), the sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    
    # Let's try to find a rank-r decomposition where r < 54
    # We'll use a greedy approach: start with random vectors and try to satisfy the constraints
    
    # For small ranks, we can try to find integer/half-integer solutions
    # The problem is that for 2x3x9, the rank is likely 54 (since it's a direct sum structure)
    
    # Let's verify: the tensor M_{2,3,9} can be seen as 9 copies of M_{2,3,1}
    # M_{2,3,1} has rank 6 (since 2*3*1 = 6 and it's a multiplication tensor)
    # But the 9 copies share the A part (the 2x3 matrix)
    
    # Actually, for M_{n,m,p}, the rank is at most n*m*p
    # For M_{2,3,9}, we can potentially do better by exploiting structure
    
    # Let's try a structured approach:
    # The output C[i,k] = sum_j A[i,j] * B[j,k]
    # For each k, we have a 2x3 to 2x1 multiplication
    # These 9 multiplications share the A matrix
    
    # A potential optimization: use the same U vectors for all k, and vary V and W
    # U has shape (6, r), V has shape (27, r), W has shape (18, r)
    
    # For rank 54, we have one term per (i,j,k)
    # Can we do better?
    
    # Let's try to find if rank < 54 is possible
    # Given the time constraint, let's try a few random searches for lower rank
    
    # Try ranks from 53 down to maybe 40
    for r in range(rank0 - 1, max(20, rank0 - 20), -1):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random integer vectors
            U_r = np.zeros((n * m, r), dtype=float)
            V_r = np.zeros((m * p, r), dtype=float)
            W_r = np.zeros((p * n, r), dtype=float)
            
            # Set some random non-zero entries
            for idx in range(r):
                # Choose random positions
                ui = np.random.randint(0, n * m)
                vj = np.random.randint(0, m * p)
                wk = np.random.randint(0, p * n)
                
                U_r[ui, idx] = 1
                V_r[vj, idx] = 1
                W_r[wk, idx] = 1
            
            # Verify if this decomposition works
            # For each (i,j,k), check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r_idx in range(r):
                            val += U_r[i*m+j, r_idx] * V_r[j*p+k, r_idx] * W_r[k*n+i, r_idx]
                        if abs(val - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Convert to list of lists with integer/half-integer entries
            U_list = [[int(U_r[i, r_idx]) if abs(U_r[i, r_idx] - round(U_r[i, r_idx])) < 1e-9 else U_r[i, r_idx] for r_idx in range(r)] for i in range(n*m)]
            V_list = [[int(V_r[i, r_idx]) if abs(V_r[i, r_idx] - round(V_r[i, r_idx])) < 1e-9 else V_r[i, r_idx] for r_idx in range(r)] for i in range(m*p)]
            W_list = [[int(W_r[i, r_idx]) if abs(W_r[i, r_idx] - round(W_r[i, r_idx])) < 1e-9 else W_r[i, r_idx] for r_idx in range(r)] for i in range(p*n)]
            
            best = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = r
            break
    
    return best
