"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U = np.zeros((nm, rank_school))
    V = np.zeros((mp, rank_school))
    W = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_school
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Build list of support indices: (i_idx, j_idx, k_idx) where tensor value is 1
    support = []
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i_idx = a*m + b
                j_idx = b*p + c
                k_idx = c*n + a
                support.append((i_idx, j_idx, k_idx))
    
    # Structural approach: exploit block structure of matrix multiplication
    # The tensor T[i,j,k] = 1 if i=a*m+b, j=b*p+c, k=c*n+a else 0
    # This has a natural block structure: for each (a,b,c), we have one 1
    
    # Try structured low-rank decompositions using the block structure
    # Key insight: we can group terms by their (a,c) pair and vary b
    
    # Strategy 1: Try rank = n*p + m*p + n*m - something (like Strassen-like)
    # Strategy 2: Use the fact that for fixed (a,c), the b-index creates a "line"
    
    # Let's try a different structural approach:
    # Represent the multiplication as sum over "intermediate" indices
    # For 3x5 * 5x6, we can think of it as:
    # C[a,c] = sum_b A[a,b] * B[b,c]
    
    # The tensor is T[a,b,c] = A[a,b] * B[b,c] for the (a,b,c) triple
    
    # Try to find decompositions of the form:
    # sum_r U[a,b,r] * V[b,c,r] * W[c,a,r] = 1 if (a,b,c) is valid
    
    # Use random search with structured initialization
    # Try various rank values below schoolbook
    
    np.random.seed(42)
    
    # Try structured random search for lower ranks
    for target_rank in range(80, 55, -1):  # Try ranks from 80 down to 56
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            # Initialize with structured random values
            U_try = np.random.randn(nm, target_rank) * 0.5
            V_try = np.random.randn(mp, target_rank) * 0.5
            W_try = np.random.randn(pn, target_rank) * 0.5
            
            # Use gradient descent to minimize error
            lr = 0.001
            for step in range(200):
                if time.time() > deadline:
                    break
                
                # Compute errors on support
                errors = np.zeros(len(support))
                for idx, (i_idx, j_idx, k_idx) in enumerate(support):
                    recon = np.dot(U_try[i_idx, :] * V_try[j_idx, :], W_try[k_idx, :])
                    errors[idx] = recon - 1.0
                
                err_norm = np.max(np.abs(errors))
                if err_norm < 1e-9:
                    # Check off-support validity
                    valid = True
                    for _ in range(100):
                        ii = np.random.randint(0, nm)
                        jj = np.random.randint(0, mp)
                        kk = np.random.randint(0, pn)
                        a, b = divmod(ii, m)
                        b2, c = divmod(jj, p)
                        c2, a2 = divmod(kk, n)
                        if b != b2 or c != c2 or a != a2:
                            recon = np.dot(U_try[ii, :] * V_try[jj, :], W_try[kk, :])
                            if abs(recon) > 1e-7:
                                valid = False
                                break
                    
                    if valid:
                        best_U, best_V, best_W = U_try, V_try, W_try
                        best_rank = target_rank
                        found = True
                        break
                
                # Compute gradients (simplified)
                dU = np.zeros_like(U_try)
                dV = np.zeros_like(V_try)
                dW = np.zeros_like(W_try)
                
                for idx, (i_idx, j_idx, k_idx) in enumerate(support):
                    e = errors[idx]
                    # dU[i_idx, r] += e * V[j_idx, r] * W[k_idx, r]
                    dU[i_idx, :] += e * V_try[j_idx, :] * W_try[k_idx, :]
                    dV[j_idx, :] += e * U_try[i_idx, :] * W_try[k_idx, :]
                    dW[k_idx, :] += e * U_try[i_idx, :] * V_try[j_idx, :]
                
                # Update
                U_try -= lr * dU
                V_try -= lr * dV
                W_try -= lr * dW
                
                # Adaptive learning rate
                if step % 20 == 0 and step > 0:
                    lr *= 0.9
            
            if found:
                break
        
        if found:
            break
    
    # Also try the removal-based approach from inspirations as a fallback
    if best_rank == rank_school and time.time() < deadline - 30:
        # Try removing terms from schoolbook
        U_curr = U.copy()
        V_curr = V.copy()
        W_curr = W.copy()
        curr_rank = rank_school
        
        for remove_idx in range(rank_school - 1, 60, -1):
            if time.time() > deadline - 20:
                break
            
            # Try removing this column
            test_U = np.delete(U_curr, remove_idx, axis=1)
            test_V = np.delete(V_curr, remove_idx, axis=1)
            test_W = np.delete(W_curr, remove_idx, axis=1)
            test_rank = curr_rank - 1
            
            # Check if still valid (fast check on support)
            valid = True
            for (i_idx, j_idx, k_idx) in support:
                recon = np.dot(test_U[i_idx, :] * test_V[j_idx, :], test_W[k_idx, :])
                if abs(recon - 1.0) > 1e-9:
                    valid = False
                    break
            
            if valid:
                # Quick off-support check
                off_valid = True
                for _ in range(50):
                    ii = np.random.randint(0, nm)
                    jj = np.random.randint(0, mp)
                    kk = np.random.randint(0, pn)
                    a, b = divmod(ii, m)
                    b2, c = divmod(jj, p)
                    c2, a2 = divmod(kk, n)
                    if b != b2 or c != c2 or a != a2:
                        recon = np.dot(test_U[ii, :] * test_V[jj, :], test_W[kk, :])
                        if abs(recon) > 1e-8:
                            off_valid = False
                            break
                
                if off_valid:
                    U_curr, V_curr, W_curr = test_U, test_V, test_W
                    curr_rank = test_rank
                    best_U, best_V, best_W = U_curr, V_curr, W_curr
                    best_rank = curr_rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
