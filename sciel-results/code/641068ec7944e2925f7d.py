import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        r = U.shape[1]
        # Compute the reconstructed tensor
        X = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Build target tensor
        T = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    T[a, b, c] = 1.0
        
        return np.allclose(X, T, atol=tol)
    
    def try_rank_reduction(U, V, W, target_rank, max_iter=10000):
        """
        Try to reduce rank by removing terms that are linearly dependent
        or can be represented by combinations of other terms.
        """
        current_rank = U.shape[1]
        if current_rank <= target_rank:
            return U, V, W
        
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Try to identify redundant terms
        # A term r is redundant if its contribution can be expressed as a linear combination
        # of other terms
        
        for attempt in range(max_iter):
            if time.time() - start_time > time_limit - 10:
                break
            
            if U.shape[1] <= target_rank:
                break
            
            # Try removing a random term
            r_to_try = random.randint(0, U.shape[1] - 1)
            
            # Check if removing this term still gives a valid decomposition
            # by trying to adjust other terms
            U_test = U.copy()
            V_test = V.copy()
            W_test = W.copy()
            
            # Zero out the term
            U_test[:, r_to_try] = 0
            V_test[:, r_to_try] = 0
            W_test[:, r_to_try] = 0
            
            # Check if the remaining terms can still represent the tensor
            # This is a hard problem, so we'll use a simpler heuristic:
            # Check if the removed term's pattern is a linear combination of others
            
            # For now, just try random perturbations to see if we can maintain validity
            improved = False
            for _ in range(100):
                # Randomly perturb some entries
                mat_choice = random.random()
                if mat_choice < 0.333:
                    row = random.randint(0, n*m - 1)
                    col = random.randint(0, U_test.shape[1] - 1)
                    if col != r_to_try:
                        U_test[row, col] += random.choice([-0.5, 0.5])
                elif mat_choice < 0.667:
                    row = random.randint(0, m*p - 1)
                    col = random.randint(0, V_test.shape[1] - 1)
                    if col != r_to_try:
                        V_test[row, col] += random.choice([-0.5, 0.5])
                else:
                    row = random.randint(0, p*n - 1)
                    col = random.randint(0, W_test.shape[1] - 1)
                    if col != r_to_try:
                        W_test[row, col] += random.choice([-0.5, 0.5])
                
                if verify_decomposition(U_test, V_test, W_test):
                    improved = True
                    break
            
            if improved:
                # Remove the column
                mask = [i for i in range(U_test.shape[1]) if i != r_to_try]
                U = U_test[:, mask]
                V = V_test[:, mask]
                W = W_test[:, mask]
        
        return U, V, W
    
    # Strategy: Use known constructions for small matrix multiplication
    # For 3x3x3, Strassen gives rank 7 (vs schoolbook 27)
    # For 3x3x4, we can try to extend Strassen-like ideas
    
    # Known result: rank 33 is achievable for 3x3x4
    # Let's try a structured approach based on block decomposition
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
                nz_indices.append((a, b, c))
    
    # Approach: Start with schoolbook and try to find rank reduction
    # using a more systematic local search
    
    values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    def fast_error(U, V, W):
        """Fast error computation."""
        X = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((X - T)**2)
    
    def local_search_from_schoolbook(max_iter=5000, time_budget=60.0):
        """Local search starting from schoolbook decomposition."""
        U, V, W = get_schoolbook()
        current_err = fast_error(U, V, W)
        
        ns = time.time()
        for it in range(max_iter):
            if time.time() - ns > time_budget:
                break
            if current_err < 1e-9:
                break
            
            # Try to modify one entry
            mat_choice = random.random()
            if mat_choice < 0.333:
                arr = U
                rows, cols = U.shape
            elif mat_choice < 0.667:
                arr = V
                rows, cols = V.shape
            else:
                arr = W
                rows, cols = W.shape
            
            row = random.randint(0, rows - 1)
            col = random.randint(0, cols - 1)
            
            old_val = arr[row, col]
            best_val = old_val
            best_err = current_err
            
            for val in values:
                if val == old_val:
                    continue
                arr[row, col] = val
                new_err = fast_error(U, V, W)
                if new_err < best_err - 1e-12:
                    best_err = new_err
                    best_val = val
                arr[row, col] = old_val
            
            if best_val != old_val:
                arr[row, col] = best_val
                current_err = best_err
        
        return U, V, W, current_err
    
    # Try local search from schoolbook
    U, V, W, err = local_search_from_schoolbook(max_iter=10000, time_budget=80.0)
    
    if err < 1e-9 and verify_decomposition(U, V, W):
        # Check if we can reduce rank further
        rank = U.shape[1]
        if rank < n * m * p:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If local search didn't find improvement, try a different strategy
    # Use a more aggressive search with random restarts
    
    best_rank = n * m * p
    best_result = None
    
    for restart in range(20):
        if time.time() - start_time > time_limit - 20:
            break
        
        # Random initialization with half-integer values
        r_target = 33  # Known achievable rank
        U = np.random.choice(values, size=(n*m, r_target))
        V = np.random.choice(values, size=(m*p, r_target))
        W = np.random.choice(values, size=(p*n, r_target))
        
        # Local search
        current_err = fast_error(U, V, W)
        ns = time.time()
        for it in range(3000):
            if time.time() - ns > 5.0:
                break
            if current_err < 1e-9:
                break
            
            mat_choice = random.random()
            if mat_choice < 0.333:
                arr = U
            elif mat_choice < 0.667:
                arr = V
            else:
                arr = W
            
            rows, cols = arr.shape
            row = random.randint(0, rows - 1)
            col = random.randint(0, cols - 1)
            
            old_val = arr[row, col]
            best_val = old_val
            best_err = current_err
            
            for val in values:
                if val == old_val:
                    continue
                arr[row, col] = val
                new_err = fast_error(U, V, W)
                if new_err < best_err - 1e-12:
                    best_err = new_err
                    best_val = val
                arr[row, col] = old_val
            
            if best_val != old_val:
                arr[row, col] = best_val
                current_err = best_err
        
        if current_err < 1e-9 and verify_decomposition(U, V, W):
            best_rank = r_target
            best_result = (U, V, W)
            break
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
