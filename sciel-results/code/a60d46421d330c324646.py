import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # 10s margin before 150s limit
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_schoolbook = np.zeros((n * m, rank_schoolbook))
    V_schoolbook = np.zeros((m * p, rank_schoolbook))
    W_schoolbook = np.zeros((p * n, rank_schoolbook))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, column] = 1
                V_schoolbook[j * p + k, column] = 1
                W_schoolbook[k * n + i, column] = 1
                column += 1
    
    best_U = U_schoolbook
    best_V = V_schoolbook
    best_W = W_schoolbook
    best_rank = rank_schoolbook
    
    # Try to find rank reductions
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r to equal 1
    
    # Try ranks from schoolbook-1 down, but limit the search
    # For efficiency, try a few key ranks
    ranks_to_try = []
    # Try a range of ranks below schoolbook
    for r in range(rank_schoolbook - 1, max(rank_schoolbook - 50, 0), -1):
        ranks_to_try.append(r)
    
    # Also try some specific ranks that might be theoretically interesting
    for r in [rank_schoolbook - 5, rank_schoolbook - 10, rank_schoolbook - 20, rank_schoolbook - 30]:
        if r > 0 and r not in ranks_to_try:
            ranks_to_try.append(r)
    
    # Sort by rank (lower is better)
    ranks_to_try = sorted(set(ranks_to_try), key=lambda x: x)
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that the decomposition exactly reconstructs the multiplication tensor."""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def check_integer_half_integer(arr, tol=1e-10):
        """Check if all entries are integer or half-integer."""
        for val in arr.flatten():
            # Check if val is close to k/2 for some integer k
            closest_half = round(val * 2) / 2
            if abs(val - closest_half) > tol:
                return False
        return True
    
    def try_rank(r, max_attempts=100):
        """Try to find a valid decomposition of rank r."""
        if r >= best_rank:
            return None
        
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # Initialize with random values
        np.random.seed(42)
        U[:] = np.random.randn(n * m, r)
        V[:] = np.random.randn(m * p, r)
        W[:] = np.random.randn(p * n, r)
        
        # Simple iterative refinement
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Compute current error
            errors = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        errors.append(s - 1.0)
            
            max_err = max(abs(e) for e in errors)
            if max_err < 1e-8:
                # Check if entries are integer/half-integer
                if check_integer_half_integer(U) and check_integer_half_integer(V) and check_integer_half_integer(W):
                    return (U, V, W)
                # Try to round to nearest half-integers and verify
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                if verify_decomposition(U_rounded, V_rounded, W_rounded, tol=1e-8):
                    return (U_rounded, V_rounded, W_rounded)
            
            # Update: for each term r, adjust to reduce error
            # Simple coordinate descent
            for r_idx in range(r):
                # Compute gradient-like update for U[:, r_idx]
                # d/dU[i*m+j, r_idx] = V[j*p+k, r_idx] * W[k*n+i, r_idx] * (sum - 1)
                # This is complex, so use a simple heuristic: scale down terms with large contribution
                
                # Compute contribution of term r_idx
                contrib = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            contrib[i, j, k] = U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                
                # If this term contributes too much, scale it down
                max_contrib = np.max(np.abs(contrib))
                if max_contrib > 0.1:
                    scale = 0.9
                    U[:, r_idx] *= scale
                    V[:, r_idx] *= scale
                    W[:, r_idx] *= scale
                
                # Add small random perturbation
                U[:, r_idx] += np.random.randn(n * m) * 0.01
                V[:, r_idx] += np.random.randn(m * p) * 0.01
                W[:, r_idx] += np.random.randn(p * n) * 0.01
        
        return None
    
    # Try each rank
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, max_attempts=50)
        if result is not None:
            U, V, W = result
            if verify_decomposition(U, V, W, tol=1e-8):
                if check_integer_half_integer(U) and check_integer_half_integer(V) and check_integer_half_integer(W):
                    best_U = U
                    best_V = V
                    best_W = W
                    best_rank = r
                    # Continue trying to find even lower ranks
                    continue
                else:
                    # Entries aren't clean, try to clean them up
                    U_clean = np.round(U * 2) / 2
                    V_clean = np.round(V * 2) / 2
                    W_clean = np.round(W * 2) / 2
                    if verify_decomposition(U_clean, V_clean, W_clean, tol=1e-8):
                        best_U = U_clean
                        best_V = V_clean
                        best_W = W_clean
                        best_rank = r
                        continue
    
    # Convert to lists for the required format
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
