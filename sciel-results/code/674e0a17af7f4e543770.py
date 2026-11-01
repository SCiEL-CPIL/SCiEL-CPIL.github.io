"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 42
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def to_list(mat):
        return mat.tolist()

    def verify(U, V, W):
        """Verify that U, V, W correctly decompose the tensor."""
        rank = U.shape[1]
        if rank != V.shape[1] or rank != W.shape[1]:
            return False
        
        # Compute the tensor decomposition using einsum for speed
        # T_comp[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        T_comp = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        
        # Check if T_comp matches T
        # We need exact match for non-zeros and zeros for others.
        # Since we are dealing with floats, we check closeness.
        # However, the problem requires integer/half-integer entries.
        # So we should check if the values are close to integers/half-integers first?
        # No, verify should just check if the decomposition is correct.
        
        if not np.allclose(T_comp, T, atol=1e-9):
            return False
            
        # Additionally, check if U, V, W are close to half-integers
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-9):
            return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-9):
            return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-9):
            return False
            
        return True

    # Schoolbook decomposition as baseline
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=np.float64)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=np.float64)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    # Try to find a lower rank decomposition using local search
    # We will try to reduce the rank from 42 to 41, then 40, etc.
    
    def try_rank_search(target_rank, max_iters=500):
        """
        Try to find a decomposition of target_rank.
        Uses a simple local search: start with random/schoolbook mix, perturb, check.
        """
        if time.time() > deadline:
            return None
        
        # Initialize with schoolbook terms, but randomly drop (42 - target_rank) terms
        # and try to fix the remaining ones?
        # Or start with random sparse vectors.
        
        best_error = np.inf
        best_U, best_V, best_W = None, None, None
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = np.zeros((n * m, target_rank), dtype=np.float64)
            V = np.zeros((m * p, target_rank), dtype=np.float64)
            W = np.zeros((p * n, target_rank), dtype=np.float64)
            
            # Each rank-1 term should be "sparse" to mimic the structure
            for r_idx in range(target_rank):
                # Pick random supports
                # Support size 2-3 is typical for these decompositions
                u_size = np.random.randint(2, 4)
                v_size = np.random.randint(2, 4)
                w_size = np.random.randint(2, 4)
                
                u_pos = np.random.choice(n * m, size=u_size, replace=False)
                v_pos = np.random.choice(m * p, size=v_size, replace=False)
                w_pos = np.random.choice(p * n, size=w_size, replace=False)
                
                # Assign random half-integer values
                U[u_pos, r_idx] = np.random.choice([-2, -1, -0.5, 0.5, 1, 2], size=u_size)
                V[v_pos, r_idx] = np.random.choice([-2, -1, -0.5, 0.5, 1, 2], size=v_size)
                W[w_pos, r_idx] = np.random.choice([-2, -1, -0.5, 0.5, 1, 2], size=w_size)
            
            # Compute error
            T_comp = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            error = np.sum((T_comp - T) ** 2)
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if error < 1e-9:
                    # Found a solution!
                    # Round to half-integers to be safe
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    if verify(U_int, V_int, W_int):
                        return U_int, V_int, W_int
            
            # If we found a very good solution but not exact, try to polish it?
            # For now, just keep searching.
            
        return None

    # Try rank 41
    result = try_rank_search(41, max_iters=200)
    if result is not None:
        U, V, W = result
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
        
    # Try rank 40
    result = try_rank_search(40, max_iters=100)
    if result is not None:
        U, V, W = result
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

    # If rank reduction fails, return schoolbook
    return {
        "U": to_list(U_sb),
        "V": to_list(V_sb),
        "W": to_list(W_sb)
    }
# EVOLVE-BLOCK-END
