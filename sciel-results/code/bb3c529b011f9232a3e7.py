import numpy as np
import itertools
import time

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Tensor: T[i,j,k] = 1 for all valid (i,j,k)
    # We need U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # such that for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Schoolbook rank = n*m*p = 36
    # Try to find lower rank
    
    # The matrix multiplication tensor M_{2,3,6} has a known structure.
    # Let's try systematic search for small integer entries.
    
    best_rank = n * m * p
    best_result = None
    
    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try rank 28, 24, 21, etc. with small integer coefficients
    # Use a structured approach: fix some patterns and search
    
    # Approach: Use the fact that for 2x3*3x6, we can try to find
    # decompositions with entries in {-1, 0, 1}
    
    # Let's try a more clever approach: use bilinear complexity bounds
    # Known: 2x3 matrix multiplication has rank 5 (Strassen-like)
    # 3x6 can be seen as 3x(2*3), potentially reusing structure
    
    # For now, let's do a randomized/structured search for small integers
    import random
    
    def verify(U, V, W):
        """Verify exact decomposition."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    def verify_numpy(U_arr, V_arr, W_arr):
        """Verify using numpy for speed."""
        n, m, p = DIMENSIONS
        U = U_arr  # (n*m, rank)
        V = V_arr  # (m*p, rank)
        W = W_arr  # (p*n, rank)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if s != 1:
                        return False
        return True
    
    # Try structured decompositions with entries in {-1, 0, 1}
    # Strategy: random search with verification
    
    max_attempts = 500000
    attempt = 0
    
    # Try ranks from low to high
    for target_rank in [20, 21, 22, 24, 25, 26, 27, 28]:
        if time.time() > deadline - 30:
            break
        
        # For each rank, try random small-integer matrices
        for trial in range(max_attempts // 8):
            attempt += 1
            if time.time() > deadline - 30:
                break
            
            # Generate random U with small entries
            U_list = []
            for row in range(n * m):
                row_data = []
                for c in range(target_rank):
                    val = random.choice([-1, 0, 0, 0, 1])
                    row_data.append(val)
                U_list.append(row_data)
            
            V_list = []
            for row in range(m * p):
                row_data = []
                for c in range(target_rank):
                    val = random.choice([-1, 0, 0, 0, 1])
                    row_data.append(val)
                V_list.append(row_data)
            
            # Now compute W from constraints:
            # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # This is a system of linear equations in W entries
            # W[k*n+i, c] for each c is a variable, grouped by (k,n+i)
            
            # Variables: W[k*n+i, c] for k in [0,p), i in [0,n), c in [0,rank)
            # Total variables: p*n*rank
            # Equations: n*m*p = 36
            # Each equation: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            
            # This is underdetermined if p*n*rank > n*m*p
            # p*n*rank = 12*rank, n*m*p = 36
            # For rank=3: 36=36 (just determined)
            # For rank>3: underdetermined
            
            # Let's set up the linear system
            num_eqs = n * m * p
            num_vars = p * n * target_rank
            
            # Build A matrix where A[eq, var] = U[i*m+j,c] * V[j*p+k,c]
            # for the appropriate (i,j,k) -> eq and (k,i,c) -> var
            
            A = np.zeros((num_eqs, num_vars), dtype=np.float64)
            b = np.ones(num_eqs, dtype=np.float64)
            
            for idx, (i, j, k) in enumerate(itertools.product(range(n), range(m), range(p))):
                for c in range(target_rank):
                    var_idx = (k * n + i) * target_rank + c
                    A[idx, var_idx] = U_list[i*m+j][c] * V_list[j*p+k][c]
            
            # Solve: find W such that A @ W_flat = b
            # Use least squares or check consistency
            try:
                # Check if system is consistent
                # Use SVD
                U_svd, s, Vt = np.linalg.svd(A, full_matrices=False)
                
                # Check rank of A vs rank of [A|b]
                rank_A = np.sum(s > 1e-10)
                
                # Project b onto column space of A
                b_proj = U_svd @ (U_svd.T @ b)
                
                # Check consistency
                residual = b - b_proj
                if np.max(np.abs(residual)) > 1e-8:
                    continue  # Inconsistent
                
                # Find a solution (minimum norm)
                W_flat = Vt.T @ (s[(s > 1e-10)] ** -1) @ (U_svd[:len(s[s>1e-10]).T] @ b)
                
                # Round to nearest integer/half-integer
                W_rounded = np.round(W_flat * 2) / 2.0
                
                # Check if rounded solution still satisfies equations
                if not np.allclose(A @ W_rounded, b, atol=1e-8):
                    # Try rounding to integers
                    W_rounded_int = np.round(W_flat)
                    if not np.allclose(A @ W_rounded_int, b, atol=1e-8):
                        continue
                
                # Format W
                W_list = []
                for k in range(p):
                    for i in range(n):
                        row = []
                        for c in range(target_rank):
                            row.append(W_rounded[(k*n+i)*target_rank + c])
                        W_list.append(row)
                
                # Final verification
                if verify_numpy(np.array(U_list, dtype=np.float64), 
                                np.array(V_list, dtype=np.float64),
                                np.array(W_list, dtype=np.float64)):
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
            except:
                continue
    
    # If search fails, return schoolbook
    if time.time() > deadline - 5:
        return schoolbook()
    
    return schoolbook()
