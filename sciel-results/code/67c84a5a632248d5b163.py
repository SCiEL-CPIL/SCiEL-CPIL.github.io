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
    
    # Precompute off-support indices for full validation
    all_indices = set()
    for a in range(n):
        for b in range(m):
            for c in range(p):
                all_indices.add((a*m+b, b*p+c, c*n+a))
    
    off_support = []
    for ii in range(nm):
        for jj in range(mp):
            for kk in range(pn):
                a, b = divmod(ii, m)
                b2, c = divmod(jj, p)
                c2, a2 = divmod(kk, n)
                if not (b == b2 and c == c2 and a == a2):
                    off_support.append((ii, jj, kk))
    
    def validate_full(U_mat, V_mat, W_mat, tol=1e-8):
        """Full validation: check all support and off-support entries"""
        # Check support
        for (i_idx, j_idx, k_idx) in support:
            recon = np.dot(U_mat[i_idx, :] * V_mat[j_idx, :], W_mat[k_idx, :])
            if abs(recon - 1.0) > tol:
                return False
        # Check off-support
        for (ii, jj, kk) in off_support:
            recon = np.dot(U_mat[ii, :] * V_mat[jj, :], W_mat[kk, :])
            if abs(recon) > tol:
                return False
        return True
    
    def validate_quick(U_mat, V_mat, W_mat, tol=1e-7, n_samples=200):
        """Quick validation with sampling"""
        np.random.seed(12345)
        # Check all support
        for (i_idx, j_idx, k_idx) in support:
            recon = np.dot(U_mat[i_idx, :] * V_mat[j_idx, :], W_mat[k_idx, :])
            if abs(recon - 1.0) > tol:
                return False
        # Sample off-support
        for _ in range(n_samples):
            idx = np.random.randint(0, len(off_support))
            ii, jj, kk = off_support[idx]
            recon = np.dot(U_mat[ii, :] * V_mat[jj, :], W_mat[kk, :])
            if abs(recon) > tol:
                return False
        return True
    
    # Strategy: Try structured decompositions based on block structure
    # For matrix multiplication C = A * B where A is n×m, B is m×p
    # We can try various structured approaches
    
    np.random.seed(42)
    
    # Approach 1: Try to find lower rank via gradient descent with better init
    # Use structured initialization: group by (a,c) pairs
    for target_rank in range(rank_school - 1, max(50, rank_school - 40), -1):
        if time.time() > deadline - 5:
            break
        
        found = False
        n_attempts = max(1, min(5, int((deadline - time.time()) / 10)))
        
        for attempt in range(n_attempts):
            if time.time() > deadline - 5:
                break
            
            # Structured initialization
            U_try = np.zeros((nm, target_rank))
            V_try = np.zeros((mp, target_rank))
            W_try = np.zeros((pn, target_rank))
            
            # Initialize with random but structured values
            for r in range(target_rank):
                # Random assignment of (a,b) to U, (b,c) to V, (c,a) to W
                for a in range(n):
                    for b in range(m):
                        U_try[a*m+b, r] = np.random.randn() * 0.3
                for b in range(m):
                    for c in range(p):
                        V_try[b*p+c, r] = np.random.randn() * 0.3
                for c in range(p):
                    for a in range(n):
                        W_try[c*n+a, r] = np.random.randn() * 0.3
            
            # Gradient descent
            lr = 0.005
            best_err = float('inf')
            for step in range(300):
                if time.time() > deadline - 5:
                    break
                
                # Compute errors on support
                errors = np.zeros(len(support))
                for idx, (i_idx, j_idx, k_idx) in enumerate(support):
                    recon = np.dot(U_try[i_idx, :] * V_try[j_idx, :], W_try[k_idx, :])
                    errors[idx] = recon - 1.0
                
                err_norm = np.max(np.abs(errors))
                if err_norm < best_err:
                    best_err = err_norm
                
                if err_norm < 1e-10:
                    if validate_quick(U_try, V_try, W_try, tol=1e-8):
                        # Full validation
                        if validate_full(U_try, V_try, W_try, tol=1e-8):
                            best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                            best_rank = target_rank
                            found = True
                            break
                
                # Compute gradients
                dU = np.zeros_like(U_try)
                dV = np.zeros_like(V_try)
                dW = np.zeros_like(W_try)
                
                for idx, (i_idx, j_idx, k_idx) in enumerate(support):
                    e = errors[idx]
                    dU[i_idx, :] += e * V_try[j_idx, :] * W_try[k_idx, :]
                    dV[j_idx, :] += e * U_try[i_idx, :] * W_try[k_idx, :]
                    dW[k_idx, :] += e * U_try[i_idx, :] * V_try[j_idx, :]
                
                # Update with adaptive learning rate
                U_try -= lr * dU
                V_try -= lr * dV
                W_try -= lr * dW
                
                if step % 30 == 0 and step > 0:
                    lr *= 0.95
            
            if found:
                break
        
        if found:
            break
    
    # Approach 2: Try removing columns from schoolbook (greedy)
    if best_rank == rank_school and time.time() < deadline - 20:
        U_curr = U.copy()
        V_curr = V.copy()
        W_curr = W.copy()
        curr_rank = rank_school
        
        # Try removing each column, keep if still valid
        removed = 0
        for _ in range(min(30, int((deadline - time.time()) / 5))):
            if time.time() > deadline - 15:
                break
            
            improved = False
            for remove_idx in range(curr_rank - 1, -1, -1):
                if time.time() > deadline - 15:
                    break
                
                test_U = np.delete(U_curr, remove_idx, axis=1)
                test_V = np.delete(V_curr, remove_idx, axis=1)
                test_W = np.delete(W_curr, remove_idx, axis=1)
                
                # Quick check on support
                valid = True
                for (i_idx, j_idx, k_idx) in support:
                    recon = np.dot(test_U[i_idx, :] * test_V[j_idx, :], test_W[k_idx, :])
                    if abs(recon - 1.0) > 1e-9:
                        valid = False
                        break
                
                if valid:
                    # Quick off-support check
                    if validate_quick(test_U, test_V, test_W, tol=1e-8, n_samples=100):
                        U_curr, V_curr, W_curr = test_U, test_V, test_W
                        curr_rank -= 1
                        best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
                        best_rank = curr_rank
                        improved = True
                        break
            
            if not improved:
                break
    
    # Final polish: if we found a solution, try to verify with full validation
    if best_rank < rank_school:
        if not validate_full(best_U, best_V, best_W, tol=1e-8):
            # Fall back to schoolbook
            best_U, best_V, best_W = U, V, W
            best_rank = rank_school
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
