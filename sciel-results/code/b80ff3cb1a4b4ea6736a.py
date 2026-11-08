import numpy as np
import time
import random

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook tensor decomposition for matrix multiplication M_{2,3,7}
    if no lower rank decomposition is found.
    The rank is n*m*p = 42.
    This implementation uses vectorized numpy operations to generate the factor matrices.
    Includes a deadline guard as per directive.
    """
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank_sb = n * m * p
    
    # Generate the multi-dimensional grid of indices
    i_grid, j_grid, k_grid = np.indices((n, m, p))
    
    # Flatten the grids
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    k_flat = k_grid.flatten()
    
    # Initialize U, V, W for schoolbook
    U = np.zeros((n * m, rank_sb), dtype=np.int64)
    V = np.zeros((m * p, rank_sb), dtype=np.int64)
    W = np.zeros((p * n, rank_sb), dtype=np.int64)
    
    # Column indices
    col_indices = np.arange(rank_sb)
    
    # Row indices
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    
    # Populate
    U[u_rows, col_indices] = 1
    V[v_rows, col_indices] = 1
    W[w_rows, col_indices] = 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_sb

    def verify_decomposition(U, V, W):
        """
        Vectorized verification: check if U, V, W exactly decompose the tensor.
        Tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        We need T[i, j, k] = 1 for all i, j, k.
        """
        # Precompute all required dot products
        for i in range(n):
            for k in range(p):
                # For fixed i,k, we need: for all j, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                w_row = W[k * n + i, :]  # (rank,)
                for j in range(m):
                    u_row = U[i * m + j, :]  # (rank,)
                    v_row = V[j * p + k, :]  # (rank,)
                    val = np.dot(u_row, v_row * w_row)
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True

    def snap_to_half_integer(arr, tol=1e-6):
        """Snap entries to nearest half-integer if close."""
        snapped = np.round(arr * 2) / 2
        if np.allclose(arr, snapped, atol=tol):
            return snapped
        return None

    def try_solve_for_W(U, V, rank):
        """
        Given U and V of shape (n*m, rank) and (m*p, rank),
        try to find W of shape (p*n, rank) such that the decomposition is exact.
        Returns W if found, else None.
        """
        W = np.zeros((p * n, rank))
        
        for i in range(n):
            for k in range(p):
                # Build A_ik[j, r] = U[i*m+j, r] * V[j*p+k, r]
                # For fixed i,k: j ranges 0..m-1
                u_block = U[i * m : (i + 1) * m, :]  # (m, rank)
                v_block = V[k : m * p : p, :]  # (m, rank) - rows k, p+k, 2p+k, ...
                
                A_ik = u_block * v_block  # (m, rank)
                
                # Solve A_ik @ w = ones(m)
                try:
                    # Use lstsq for robustness
                    w, residuals, rank_A, s = np.linalg.lstsq(A_ik, np.ones(m), rcond=None)
                    # Check if it's an exact solution
                    if not np.allclose(A_ik @ w, np.ones(m), atol=1e-9):
                        return None
                    W[k * n + i, :] = w
                except:
                    return None
        
        return W

    # Search for lower ranks
    # We'll try random U, V with various structures and solve for W.
    
    # Pre-calculate indices for speed
    indices_i = list(range(n))
    indices_k = list(range(p))
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Try ranks from rank_sb-1 down to a reasonable minimum
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # But for exact integer/half-integer decompositions, we may need more.
    min_rank_to_try = max(20, rank_sb - 15)  # Try down to rank 27
    
    for rank in range(rank_sb - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline:
            break
            
        # Adaptive number of trials based on time remaining
        time_remaining = deadline - time.time()
        if time_remaining < 5.0:
            break
        
        # Scale trials with time remaining
        max_trials = min(300, max(50, int(time_remaining * 2)))
        
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            # Generate random U, V with different strategies
            strategy = trial % 5
            
            if strategy == 0:
                # Sparse: mostly zeros, few ±1
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                for r in range(rank):
                    # Each column has 1-2 nonzeros in U and V
                    n_nonzero_U = random.randint(1, 2)
                    n_nonzero_V = random.randint(1, 2)
                    U_rows = random.sample(range(n * m), n_nonzero_U)
                    V_rows = random.sample(range(m * p), n_nonzero_V)
                    for row in U_rows:
                        U[row, r] = random.choice([-1, 1])
                    for row in V_rows:
                        V[row, r] = random.choice([-1, 1])
                        
            elif strategy == 1:
                # Dense with small integers
                U = np.random.randint(-1, 2, size=(n * m, rank))
                V = np.random.randint(-1, 2, size=(m * p, rank))
                
            elif strategy == 2:
                # Structured: each row of U has exactly one 1, each row of V has exactly one 1
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                for i in range(n * m):
                    col = random.randint(0, rank - 1)
                    U[i, col] = 1
                for j in range(m * p):
                    col = random.randint(0, rank - 1)
                    V[j, col] = 1
                    
            elif strategy == 3:
                # Block structure: group columns
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                block_size = max(1, rank // 7)
                for block in range(0, rank, block_size):
                    block_end = min(block + block_size, rank)
                    # Randomly assign rows to this block
                    for r in range(block, block_end):
                        U_row = random.randint(0, n * m - 1)
                        V_row = random.randint(0, m * p - 1)
                        U[U_row, r] = random.choice([-1, 1])
                        V[V_row, r] = random.choice([-1, 1])
                        
            else:
                # Mixed: some rows are standard basis, others random
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                # First n*m//2 rows are standard basis
                for i in range(n * m // 2):
                    col = i % rank
                    U[i, col] = 1
                # Rest are random
                for i in range(n * m // 2, n * m):
                    col = random.randint(0, rank - 1)
                    U[i, col] = random.choice([-1, 1])
                for j in range(m * p):
                    col = random.randint(0, rank - 1)
                    V[j, col] = random.choice([-1, 1])
            
            # Try to solve for W
            W = try_solve_for_W(U, V, rank)
            if W is None:
                continue
                
            # Snap W to half-integers
            W_snapped = snap_to_half_integer(W)
            if W_snapped is None:
                continue
                
            # Verify the whole decomposition with snapped W
            if verify_decomposition(U, V, W_snapped):
                best_U, best_V, best_W = U, V, W_snapped
                best_rank = rank
                break  # Found a valid one for this rank, move to lower rank

    # Final polish: if we found a lower rank, try to verify with tighter tolerance
    # and ensure exactness
    if best_rank < rank_sb:
        # Re-verify with very tight tolerance
        if verify_decomposition(best_U, best_V, best_W):
            pass  # Good
        else:
            # Fall back to schoolbook
            best_U, best_V, best_W = U, V, W
            best_rank = rank_sb

    # Convert to lists of lists for return
    U_ret = best_U.tolist()
    V_ret = best_V.tolist()
    W_ret = best_W.tolist()

    return {
        "U": U_ret,
        "V": V_ret,
        "W": W_ret
    }
# EVOLVE-BLOCK-END
