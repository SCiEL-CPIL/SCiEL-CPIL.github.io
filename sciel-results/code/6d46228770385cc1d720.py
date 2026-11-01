"""
Refined program for Matrix Multiplication 2x3x4 Tensor Decomposition.
Uses vectorized local search with structured initialization to find rank-minimal
exact decompositions.
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    Returns a dictionary with 'U', 'V', 'W' matrices of shape (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 12
    dim_w = p * n  # 8
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Pre-compute the target tensor indices
    # T[i*m+j, j*p+k, k*n+i] = 1
    valid_indices = []
    zero_indices = []
    
    # For local search, we need to evaluate error.
    # Error = sum over all (a,b,c) of (sum_r U[a,r]V[b,r]W[c,r] - T[a,b,c])^2
    # This is O(rank * dim_u * dim_v * dim_w).
    # To speed this up, we can compute the "contribution" of each rank component.
    
    # Strategy:
    # 1. Try ranks from low to high (e.g., 12 to 24).
    # 2. For each rank, perform random restarts of local search.
    # 3. Local search minimizes the L2 error of the tensor decomposition.
    # 4. If error becomes 0 (within tolerance), we have a valid solution.
    
    # Known lower bound for 2x3x4 is likely > 12. Schoolbook is 24.
    # Literature suggests rank 24 is optimal for 2x3x4? Or maybe lower?
    # Actually, for 2x3x4, the rank is known to be 24? No, wait.
    # Strassen-like methods usually help for larger matrices.
    # For small matrices, 24 might be optimal, or close to it.
    # Let's search for ranks 20, 21, 22, 23 first to see if we can beat 24.
    
    def compute_error(U, V, W):
        """
        Compute the sum of squared errors for the tensor decomposition.
        U: (dim_u, rank), V: (dim_v, rank), W: (dim_w, rank)
        """
        rank = U.shape[1]
        # Initialize tensor with zeros
        # We don't want to build the full tensor if it's too big, but 6x12x8 is small (576 entries).
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        
        # Efficient way: 
        # T = np.einsum('ar,br,cr->abc', U, V, W)
        T = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Target tensor
        # Target T_target[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i else 0
        # We can precompute the target tensor once.
        T_target = np.zeros((dim_u, dim_v, dim_w))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_target[a, b, c] = 1.0
        
        diff = T - T_target
        return np.sum(diff * diff)

    # Pre-compute T_target outside the loop for speed? 
    # No, it's small, but let's do it once.
    T_target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def fast_error(U, V, W):
        T = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T - T_target
        return np.sum(diff * diff)

    def local_search(U, V, W, max_iters=2000):
        """
        Perform local search to minimize error.
        U, V, W are numpy arrays.
        Returns (error, U, V, W)
        """
        current_error = fast_error(U, V, W)
        if current_error < 1e-9:
            return current_error, U, V
        
        rank = U.shape[1]
        best_error = current_error
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Candidate values for updates
        candidates = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            if current_error < 1e-9:
                break
                
            # Pick a random matrix to update
            mat_type = random.choice(['U', 'V', 'W'])
            # Pick a random row and column
            if mat_type == 'U':
                row = random.randint(0, dim_u - 1)
            elif mat_type == 'V':
                row = random.randint(0, dim_v - 1)
            else:
                row = random.randint(0, dim_w - 1)
            col = random.randint(0, rank - 1)
            
            old_val = None
            new_val = None
            
            if mat_type == 'U':
                old_val = U[row, col]
                # Try setting to 0 or other candidates
                # If old_val is 0, try non-zero. Else try 0 or other.
                if old_val == 0:
                    new_val = random.choice([c for c in candidates if c != 0])
                else:
                    new_val = random.choice(candidates)
                    if new_val == old_val:
                        new_val = random.choice([c for c in candidates if c != old_val])
                
                # Compute error change efficiently?
                # U[row, col] affects T[row, b, c] for all b, c.
                # Delta T[b, c] += (new_val - old_val) * V[b, col] * W[c, col]
                delta_val = new_val - old_val
                delta_T = np.outer(V[:, col], W[:, col]) * delta_val
                
                # Current T for this row
                # T[row, :, :] = sum_r U[row, r] V[:, r] W[:, r]
                # We can compute the current T row and update it.
                # But recomputing full error is O(rank * dim_v * dim_w).
                # Since dims are small, let's just update the value and recompute full error.
                # Optimization: Only if error decreases, accept.
                
                U[row, col] = new_val
                new_error = fast_error(U, V, W)
                
                if new_error < current_error:
                    current_error = new_error
                    if new_error < best_error:
                        best_error = new_error
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                else:
                    U[row, col] = old_val
            elif mat_type == 'V':
                old_val = V[row, col]
                if old_val == 0:
                    new_val = random.choice([c for c in candidates if c != 0])
                else:
                    new_val = random.choice(candidates)
                    if new_val == old_val:
                        new_val = random.choice([c for c in candidates if c != old_val])
                    
                V[row, col] = new_val
                new_error = fast_error(U, V, W)
                
                if new_error < current_error:
                    current_error = new_error
                    if new_error < best_error:
                        best_error = new_error
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                else:
                    V[row, col] = old_val
            else:
                old_val = W[row, col]
                if old_val == 0:
                    new_val = random.choice([c for c in candidates if c != 0])
                else:
                    new_val = random.choice(candidates)
                    if new_val == old_val:
                        new_val = random.choice([c for c in candidates if c != old_val])
                    
                W[row, col] = new_val
                new_error = fast_error(U, V, W)
                
                if new_error < current_error:
                    current_error = new_error
                    if new_error < best_error:
                        best_error = new_error
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                else:
                    W[row, col] = old_val
                        
        return best_error, best_U, best_V, best_W

    def generate_initial_solution(rank):
        """
        Generate an initial solution with structured sparsity.
        """
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        
        # Strategy: Assign each rank component to cover a "slice" of the multiplication.
        # There are n*m*p = 24 scalar multiplications.
        # If rank >= 24, we can map 1-to-1.
        # If rank < 24, we must share components.
        
        # Simple random initialization with density
        density = 0.3
        for r in range(rank):
            # Randomly select indices
            u_idx = np.random.choice(dim_u, size=max(1, int(dim_u * density)), replace=False)
            v_idx = np.random.choice(dim_v, size=max(1, int(dim_v * density)), replace=False)
            w_idx = np.random.choice(dim_w, size=max(1, int(dim_w * density)), replace=False)
            
            for i in u_idx:
                U[i, r] = random.choice([-1, 1])
            for i in v_idx:
                V[i, r] = random.choice([-1, 1])
            for i in w_idx:
                W[i, r] = random.choice([-1, 1])
                
        return U, V, W

    # Schoolbook solution
    rank_schoolbook = n * m * p
    U_sb = np.zeros((dim_u, rank_schoolbook))
    V_sb = np.zeros((dim_v, rank_schoolbook))
    W_sb = np.zeros((dim_w, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_schoolbook
    best_sol = (U_sb, V_sb, W_sb)
    
    # Search for lower ranks
    # Try ranks from 12 up to 23
    # Use random restarts for each rank
    max_restarts_per_rank = 5
    
    for target_rank in range(12, 24):
        if time.time() > deadline - 10:
            break
            
        found_solution = False
        for restart in range(max_restarts_per_rank):
            if time.time() > deadline - 5:
                break
                
            U, V, W = generate_initial_solution(target_rank)
            
            # Local search
            err, U, V, W = local_search(U, V, W, max_iters=500)
            
            if err < 1e-9:
                # Verify exactness
                # Convert to list of lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Check if entries are close to half-integers
                # The problem allows integer/half-integer entries.
                # Our search uses -1, -0.5, 0, 0.5, 1.
                
                best_rank = target_rank
                best_sol = (U, V, W)
                found_solution = True
                break
        
        if found_solution:
            break
            
    # Return the best solution found
    U_final, V_final, W_final = best_sol
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
# EVOLVE-BLOCK-END
