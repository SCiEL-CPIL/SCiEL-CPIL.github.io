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
    We try ranks from 20 upwards (lower bound for 3x3x3 is 19).
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

    def find_3x3x3_decomposition(rank, max_time=10):
        """
        Attempt to find a rank-R decomposition for 3x3x3 multiplication.
        Returns (U, V, W) if found, else None.
        Uses scipy.optimize for better convergence.
        """
        start_time = time.time()
        rng = np.random.RandomState(rank * 100 + 42)
        
        T_target = np.zeros((9, 9, 9))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    T_target[i*3+j, j*3+k, k*3+i] = 1.0
        
        # Precompute indices for faster residual calculation if needed, but einsum is fine
        # We will flatten U, V, W into a single vector for scipy.optimize
        
        def objective(x):
            U = x[:9*rank].reshape(9, rank)
            V = x[9*rank:18*rank].reshape(9, rank)
            W = x[18*rank:].reshape(9, rank)
            T_current = np.einsum('ur,vr,wr->uvw', U, V, W)
            Residual = T_current - T_target
            return np.sum(Residual**2)
            
        def gradient(x):
            U = x[:9*rank].reshape(9, rank)
            V = x[9*rank:18*rank].reshape(9, rank)
            W = x[18*rank:].reshape(9, rank)
            T_current = np.einsum('ur,vr,wr->uvw', U, V, W)
            Residual = T_current - T_target
            
            grad_U = np.einsum('uvw,vr,wr->ur', Residual, V, W)
            grad_V = np.einsum('uvw,ur,wr->vr', Residual, U, W)
            grad_W = np.einsum('uvw,ur,vr->wr', Residual, U, V)
            
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

        for attempt in range(20):
            if time.time() - start_time > max_time:
                break
            
            # Initialize
            x0 = rng.randn(27 * rank) * 0.5
            
            try:
                res = scipy.optimize.minimize(objective, x0, method='L-BFGS-B', jac=gradient, options={'maxiter': 100, 'ftol': 1e-12})
            except Exception:
                continue
                
            if res.success or objective(res.x) < 1e-6:
                x_opt = res.x
                U = x_opt[:9*rank].reshape(9, rank)
                V = x_opt[9*rank:18*rank].reshape(9, rank)
                W = x_opt[18*rank:].reshape(9, rank)
                
                U_snapped = snap_matrix(U)
                V_snapped = snap_matrix(V)
                W_snapped = snap_matrix(W)
                
                if verify_3x3x3(U_snapped, V_snapped, W_snapped, rank):
                    return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
                    
        return None

    # Try ranks from low to high for 3x3x3
    # Lower bound is 19. Try 19, 20, 21, 22, 23.
    best_decomp = None
    best_rank = None
    
    for rank in range(19, 24):
        if time.time() > deadline - 20:
            break
        decomp = find_3x3x3_decomposition(rank, max_time=5)
        if decomp is not None:
            best_decomp = decomp
            best_rank = rank
            break # Found the lowest rank, stop
            
    if best_decomp is not None:
        U3, V3, W3 = best_decomp
        rank_3x3 = best_rank
        total_rank = 2 * rank_3x3
        
        # Construct full 3x3x6 decomposition
        # U_full: (9, total_rank)
        # V_full: (18, total_rank)
        # W_full: (18, total_rank)
        
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
