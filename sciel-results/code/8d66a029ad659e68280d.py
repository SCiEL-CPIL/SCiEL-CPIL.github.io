"""Program for matrix multiplication tensor decomposition task."""
import time
import random
import numpy as np
import scipy.optimize

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    Strategy: The 3x3 by 3x6 multiplication splits into two independent 3x3 by 3x3 multiplications.
    We search for a low-rank decomposition of the 3x3x3 multiplication tensor.
    If a rank-R decomposition is found for 3x3x3, the total rank is 2R.
    We use a discrete search on half-integers to find a valid decomposition.
    """
    n, m, p = DIMENSIONS  # 3, 3, 6
    deadline = time.time() + 110  # Safety margin

    def snap_to_half(x):
        """Snap value to nearest 0.5 multiple."""
        return round(x * 2) / 2.0

    def snap_matrix(M):
        """Snap all entries in matrix to nearest 0.5 multiple."""
        return np.array([[snap_to_half(x) for x in row] for row in M])

    def verify_3x3x3(U, V, W, rank, tol=1e-9):
        """
        Verify if U, V, W (snapped) form an exact decomposition of 3x3x3 multiplication.
        U: (9, rank), V: (9, rank), W: (9, rank)
        """
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def discrete_search(rank, max_time=10):
        """
        Attempt to find a rank-R decomposition for 3x3x3 multiplication using discrete search.
        """
        start_time = time.time()
        rng = np.random.RandomState(rank * 100 + 42)
        
        # Initialize with random half-integers
        U = (rng.randint(-2, 3, size=(9, rank)) / 2.0)
        V = (rng.randint(-2, 3, size=(9, rank)) / 2.0)
        W = (rng.randint(-2, 3, size=(9, rank)) / 2.0)
        
        T_target = np.zeros((9, 9, 9))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    T_target[i*3+j, j*3+k, k*3+i] = 1.0
        
        def compute_error(U, V, W):
            T_current = np.einsum('ur,vr,wr->uvw', U, V, W)
            Residual = T_current - T_target
            return np.sum(Residual**2)
        
        current_error = compute_error(U, V, W)
        best_error = current_error
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        T = 1.0
        T_min = 1e-4
        
        while time.time() - start_time < max_time:
            if current_error < 1e-9:
                break
            
            # Choose a random matrix and entry to change
            mat_idx = rng.randint(0, 3)
            if mat_idx == 0:
                mat = U
            elif mat_idx == 1:
                mat = V
            else:
                mat = W
            
            row, col = rng.randint(0, 8), rng.randint(0, rank - 1)
            old_val = mat[row, col]
            new_val = old_val + (0.5 if rng.rand() > 0.5 else -0.5)
            
            # Apply change
            mat[row, col] = new_val
            
            # Compute new error
            if mat_idx == 0:
                new_error = compute_error(U, V, W)
            elif mat_idx == 1:
                new_error = compute_error(U, V, W)
            else:
                new_error = compute_error(U, V, W)
            
            delta = new_error - current_error
            if delta < 0 or rng.rand() < np.exp(-delta / T):
                current_error = new_error
                if current_error < best_error:
                    best_error = current_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                # Revert
                mat[row, col] = old_val
            
            T *= 0.999
            
            if verify_3x3x3(best_U, best_V, best_W, rank):
                return best_U.tolist(), best_V.tolist(), best_W.tolist()
                
        if verify_3x3x3(best_U, best_V, best_W, rank):
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
            
        return None

    # Try ranks from low to high for 3x3x3
    # Lower bound is 19. Try 19, 20, 21, 22, 23.
    best_decomp = None
    best_rank = None
    
    for rank in range(19, 24):
        if time.time() > deadline - 20:
            break
        decomp = discrete_search(rank, max_time=5)
        if decomp is not None:
            best_decomp = decomp
            best_rank = rank
            break # Found the lowest rank, stop
            
    if best_decomp is not None:
        U3, V3, W3 = best_decomp
        rank_3x3 = best_rank
        total_rank = 2 * rank_3x3
        
        # Construct full 3x3x6 decomposition
        U_full = [[0] * total_rank for _ in range(9)]
        V_full = [[0] * total_rank for _ in range(18)]
        W_full = [[0] * total_rank for _ in range(18)]
        
        # First 3x3 block (k=0,1,2)
        for r in range(rank_3x3):
            for i in range(3):
                for j in range(3):
                    U_full[i*3+j][r] = U3[i*3+j][r]
                for j in range(3):
                    for k in range(3):
                        V_full[j*3+k][r] = V3[j*3+k][r]
                    for i in range(3):
                        W_full[k*3+i][r] = W3[k*3+i][r]
                        
        # Second 3x3 block (k=3,4,5)
        for r in range(rank_3x3):
            for i in range(3):
                for j in range(3):
                    U_full[i*3+j][r+rank_3x3] = U3[i*3+j][r]
                for j in range(3):
                    for k in range(3):
                        V_full[j*3+(k+3)][r+rank_3x3] = V3[j*3+k][r]
                    for i in range(3):
                        W_full[(k+3)*3+i][r+rank_3x3] = W3[k*3+i][r]
                        
        return {"U": U_full, "V": V_full, "W": W_full}
    else:
        # Fallback: Schoolbook
        rank = 54
        U = [[0] * rank for _ in range(9)]
        V = [[0] * rank for _ in range(18)]
        W = [[0] * rank for _ in range(18)]
        column = 0
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    U[i * 3 + j][column] = 1
                    V[j * 6 + k][column] = 1
                    W[k * 3 + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
