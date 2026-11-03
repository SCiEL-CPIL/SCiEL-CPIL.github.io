"""Program for matrix multiplication tensor decomposition task."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    Strategy: Split into two independent 3x3x3 multiplications.
    Search for a low-rank decomposition of the 3x3x3 multiplication tensor using ALS.
    If a rank-R decomposition is found for 3x3x3, the total rank is 2R.
    """
    n, m, p = DIMENSIONS  # 3, 3, 6
    deadline = time.time() + 115  # Safety margin
    
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
        
        # Check all 729 entries of the tensor
        T_current = np.einsum('ur,vr,wr->uvw', U, V, W)
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    u = i*3 + j
                    v = j*3 + k
                    w = k*3 + i
                    if abs(T_current[u, v, w] - 1.0) > tol:
                        return False
        
        # Check zeros
        for u in range(9):
            for v in range(9):
                for w in range(9):
                    i1 = u // 3
                    j1 = u % 3
                    j2 = v // 3
                    k1 = v % 3
                    k2 = w // 3
                    i2 = w % 3
                    
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        continue
                    else:
                        if abs(T_current[u, v, w]) > tol:
                            return False
        return True

    def find_3x3x3_decomposition(rank, max_time=10):
        """
        Attempt to find a rank-R decomposition for 3x3x3 multiplication using ALS.
        Returns (U, V, W) if found, else None.
        """
        start_time = time.time()
        
        # Target tensor
        T_target = np.zeros((9, 9, 9))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    T_target[i*3+j, j*3+k, k*3+i] = 1.0
        
        rng = np.random.RandomState(rank * 100 + 42)
        
        max_restarts = 50
        for attempt in range(max_restarts):
            if time.time() - start_time > max_time:
                break
            
            # Initialize with small random values
            U = rng.randn(9, rank)
            V = rng.randn(9, rank)
            W = rng.randn(9, rank)
            
            # ALS Refinement
            for step in range(100):
                if time.time() - start_time > max_time:
                    break
                
                # Solve for U: T = U (V^T * W) ? No.
                # T_{uvw} = sum_r U_{ur} V_{vr} W_{wr}
                # For fixed V, W, we want to minimize sum_{v,w} (T_{uvw} - sum_r U_{ur} V_{vr} W_{wr})^2
                # This is a linear least squares problem for each u.
                # Let M_{v,w,r} = V_{vr} W_{wr}. Then T_{u,v,w} approx sum_r U_{ur} M_{v,w,r}.
                # We can vectorize.
                
                # Reshape V and W to (9, 9, rank)
                V_w = V[:, :, None] * W[:, None, :] # (9, 9, rank)
                
                # For each u, solve U[u, :] such that U[u, :] dot V_w[v, w, :] approx T_target[u, v, w]
                # This is equivalent to solving U[u, :] @ V_w_flat.T = T_target[u, :, :].flatten()
                
                V_w_flat = V_w.transpose(2, 0, 1).reshape(rank, 81) # (rank, 81)
                T_flat = T_target.reshape(9, 81) # (9, 81)
                
                # Solve U @ V_w_flat.T = T_flat
                # U is (9, rank). V_w_flat is (rank, 81).
                # U @ V_w_flat.T is (9, 81).
                # We want to minimize || U @ V_w_flat.T - T_flat ||_F
                # Solution: U = T_flat @ V_w_flat.T @ (V_w_flat @ V_w_flat.T)^{-1}
                
                try:
                    # Regularize
                    reg = 1e-6 * np.eye(rank)
                    V_w_flat_T_V_w_flat = V_w_flat.T @ V_w_flat + reg
                    U_new = T_flat @ V_w_flat.T @ np.linalg.inv(V_w_flat_T_V_w_flat)
                except np.linalg.LinAlgError:
                    break
                
                # Solve for V
                U_w = U[:, :, None] * W[:, None, :] # (9, 9, rank)
                U_w_flat = U_w.transpose(2, 0, 1).reshape(rank, 81)
                try:
                    reg = 1e-6 * np.eye(rank)
                    U_w_flat_T_U_w_flat = U_w_flat.T @ U_w_flat + reg
                    V_new = T_flat @ U_w_flat.T @ np.linalg.inv(U_w_flat_T_U_w_flat)
                except np.linalg.LinAlgError:
                    break
                
                # Solve for W
                U_v = U[:, :, None] * V[:, None, :] # (9, 9, rank)
                U_v_flat = U_v.transpose(2, 0, 1).reshape(rank, 81)
                try:
                    reg = 1e-6 * np.eye(rank)
                    U_v_flat_T_U_v_flat = U_v_flat.T @ U_v_flat + reg
                    W_new = T_flat @ U_v_flat.T @ np.linalg.inv(U_v_flat_T_U_v_flat)
                except np.linalg.LinAlgError:
                    break
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                T_current = np.einsum('ur,vr,wr->uvw', U, V, W)
                if np.max(np.abs(T_current - T_target)) < 1e-12:
                    break
            
            # Final check with snapping
            U_snapped = snap_matrix(U)
            V_snapped = snap_matrix(V)
            W_snapped = snap_matrix(W)
            
            if verify_3x3x3(U_snapped, V_snapped, W_snapped, rank):
                return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
            
            # Try a polish step: fix U, V and solve for W exactly, then snap W
            # This might recover from slight numerical errors
            # But W must be half-integer.
            # We can try to solve for W in real space, then snap, then verify.
            # We already did that.
            
            # Try different initialization
            U = rng.randint(-1, 2, size=(9, rank)) * 0.5
            V = rng.randint(-1, 2, size=(9, rank)) * 0.5
            W = rng.randint(-1, 2, size=(9, rank)) * 0.5
            
        return None

    def construct_full(U3, V3, W3, rank_3x3):
        """
        Construct full 3x3x6 decomposition from two 3x3x3 decompositions.
        """
        total_rank = 2 * rank_3x3
        
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
            for k in range(3):
                for i in range(3):
                    W_full[k*3+i][r] = W3[k*3+i][r]
                    
        # Second 3x3 block (k=3,4,5)
        for r in range(rank_3x3):
            for i in range(3):
                for j in range(3):
                    U_full[i*3+j][r+rank_3x3] = U3[i*3+j][r]
            for j in range(3):
                for k in range(3, 6):
                    V_full[j*6+k][r+rank_3x3] = V3[j*3+(k-3)][r]
            for k in range(3, 6):
                for i in range(3):
                    W_full[k*3+i][r+rank_3x3] = W3[(k-3)*3+i][r]
                    
        return {"U": U_full, "V": V_full, "W": W_full}

    best_decomp = None
    best_rank = None
    
    # Try ranks from low to high for 3x3x3
    # Best known is 23. Try 23, 24, 25, 26.
    for rank in [23, 24, 25, 26]:
        if time.time() > deadline - 10:
            break
        decomp = find_3x3x3_decomposition(rank, max_time=20)
        if decomp is not None:
            best_decomp = decomp
            best_rank = rank
            break # Found the lowest rank, stop
            
    if best_decomp is not None:
        U3, V3, W3 = best_decomp
        return construct_full(U3, V3, W3, best_rank)
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
