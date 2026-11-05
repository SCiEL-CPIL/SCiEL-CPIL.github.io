import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Precompute index mappings for verification
    ijks = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    U_idx = np.array([i*m + j for i, j, k in ijks])
    V_idx = np.array([j*p + k for i, j, k in ijks])
    W_idx = np.array([k*n + i for i, j, k in ijks])
    
    def verify_decomposition(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        R = U.shape[1]
        # Check shape
        if U.shape != (n*m, R) or V.shape != (m*p, R) or W.shape != (p*n, R):
            return False
        # Check each component
        for idx, (i, j, k) in enumerate(ijks):
            expected = 1.0
            actual = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
            if abs(actual - expected) > 1e-9:
                return False
        return True
    
    # Start with schoolbook decomposition (rank 100)
    schoolbook_rank = n * m * p
    best_U = np.zeros((n*m, schoolbook_rank))
    best_V = np.zeros((m*p, schoolbook_rank))
    best_W = np.zeros((p*n, schoolbook_rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j, idx] = 1.0
                best_V[j*p+k, idx] = 1.0
                best_W[k*n+i, idx] = 1.0
                idx += 1
    
    best_rank = schoolbook_rank
    
    # Try to find better decompositions using structured approaches
    # Strategy: Use Strassen-like block decomposition for submatrices
    
    # First, try to decompose using known fast matrix multiplication for smaller blocks
    # For 4x5x5, we can try to exploit structure
    
    # Approach: Try rank 80, 75, 70, etc. with random sparse initialization
    target_ranks = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50]
    
    np.random.seed(42)  # For reproducibility
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
            
        found = False
        num_attempts = 10
        attempts_per_rank = max(1, int((deadline - time.time()) / 5))  # ~5s per attempt
        
        for attempt in range(min(num_attempts, attempts_per_rank)):
            if time.time() > deadline:
                break
            
            # Create sparse random matrices
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Initialize with structured sparsity
            # Each column of U, V, W has a few non-zero entries
            for r in range(target_rank):
                # Choose which rows will have non-zero entries
                # For U: rows correspond to (i,j) pairs
                u_nonzero_rows = np.random.choice(n*m, size=min(3, target_rank//10+2), replace=False)
                v_nonzero_rows = np.random.choice(m*p, size=min(3, target_rank//10+2), replace=False)
                w_nonzero_rows = np.random.choice(p*n, size=min(3, target_rank//10+2), replace=False)
                
                U[u_nonzero_rows, r] = np.random.choice([-1.0, 1.0], size=len(u_nonzero_rows))
                V[v_nonzero_rows, r] = np.random.choice([-1.0, 1.0], size=len(v_nonzero_rows))
                W[w_nonzero_rows, r] = np.random.choice([-1.0, 1.0], size=len(w_nonzero_rows))
            
            # Try local optimization
            max_iter = 200
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                    
                # Compute current error
                errors = np.array([
                    abs(sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(target_rank)) - 1.0)
                    for i, j, k in ijks
                ])
                max_err = errors.max()
                
                if max_err < 1e-9:
                    break
                
                # Find worst error component
                worst_idx = np.argmax(errors)
                i, j, k = ijks[worst_idx]
                
                # Try to fix this component by adjusting one factor
                # Pick a random rank index
                r = np.random.randint(target_rank)
                
                # Try adjusting U
                old_val = U[i*m+j, r]
                for delta in [-0.5, -0.25, 0.25, 0.5, 1.0]:
                    U[i*m+j, r] = old_val + delta
                    new_err = abs(sum(U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr] for rr in range(target_rank)) - 1.0)
                    if new_err < errors[worst_idx]:
                        break
                else:
                    U[i*m+j, r] = old_val
                
                # Try adjusting V
                old_val = V[j*p+k, r]
                for delta in [-0.5, -0.25, 0.25, 0.5, 1.0]:
                    V[j*p+k, r] = old_val + delta
                    new_err = abs(sum(U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr] for rr in range(target_rank)) - 1.0)
                    if new_err < errors[worst_idx]:
                        break
                else:
                    V[j*p+k, r] = old_val
                
                # Try adjusting W
                old_val = W[k*n+i, r]
                for delta in [-0.5, -0.25, 0.25, 0.5, 1.0]:
                    W[k*n+i, r] = old_val + delta
                    new_err = abs(sum(U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr] for rr in range(target_rank)) - 1.0)
                    if new_err < errors[worst_idx]:
                        break
                else:
                    W[k*n+i, r] = old_val
            
            # Final check
            if verify_decomposition(U, V, W):
                best_rank = target_rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                found = True
                break
        
        if found:
            continue
        else:
            # If we can't find a decomposition at this rank, stop
            if target_rank < best_rank:
                break
    
    # Return the best decomposition found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
