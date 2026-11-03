import time
import numpy as np
import scipy.optimize

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    def snap_to_half(x):
        return round(x * 2) / 2.0

    def snap_matrix(M):
        return np.array([[snap_to_half(x) for x in row] for row in M])

    def verify_3x3x3(U, V, W, rank, tol=1e-9):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Target T[i,j,k] = 1 if i,j,k in 0..2 else 0? 
        # No, T is the multiplication tensor.
        # T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
        # So T[i*3+j, j*3+k, k*3+i] = 1.
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def find_3x3x3_decomposition(rank, max_time=20):
        start_time = time.time()
        rng = np.random.RandomState(rank * 100 + 42)
        
        T_target = np.zeros((9, 9, 9))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    T_target[i*3+j, j*3+k, k*3+i] = 1.0

        def objective(x):
            U = x[:9*rank].reshape(9, rank)
            V = x[9*rank:18*rank].reshape(9, rank)
            W = x[18*rank:].reshape(9, rank)
            T_current = np.einsum('ur,vr,wr->uvw', U, V, W)
            Residual = T_current - T_target
            
            # Penalty for not being half-integers
            # We want x to be close to 0.5 * integers
            # dist = x - 0.5 * round(2*x)
            # This is non-differentiable, so we use a smooth approximation or just rely on snapping later.
            # Better: Add a quadratic penalty that is periodic? No.
            # Just minimize residual. The snapping step is the "projection".
            
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

        best_res = None
        best_obj = float('inf')
        
        for attempt in range(100):
            if time.time() - start_time > max_time:
                break
            
            # Initialize with small noise
            x0 = rng.randn(27 * rank) * 0.5
            
            try:
                res = scipy.optimize.minimize(objective, x0, method='L-BFGS-B', jac=gradient, options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-8})
            except Exception:
                continue
                
            obj_val = objective(res.x)
            if obj_val < best_obj:
                best_obj = obj_val
                best_res = res.x
                
                # Check if snapping works
                U = best_res[:9*rank].reshape(9, rank)
                V = best_res[9*rank:18*rank].reshape(9, rank)
                W = best_res[18*rank:].reshape(9, rank)
                
                U_snapped = snap_matrix(U)
                V_snapped = snap_matrix(V)
                W_snapped = snap_matrix(W)
                
                if verify_3x3x3(U_snapped, V_snapped, W_snapped, rank):
                    return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
                    
        return None

    best_decomp = None
    best_rank = None
    
    # Try ranks from 23 upwards
    # 23 is the theoretical best for 3x3x3
    for rank in range(23, 28):
        if time.time() > deadline - 10:
            break
        decomp = find_3x3x3_decomposition(rank, max_time=10)
        if decomp is not None:
            best_decomp = decomp
            best_rank = rank
            break
            
    if best_decomp is not None:
        U3, V3, W3 = best_decomp
        rank_3x3 = best_rank
        total_rank = 2 * rank_3x3
        
        U_full = [[0] * total_rank for _ in range(9)]
        V_full = [[0] * total_rank for _ in range(18)]
        W_full = [[0] * total_rank for _ in range(18)]
        
        # First block
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
                    
        # Second block
        for r in range(rank_3x3):
            for i in range(3):
                for j in range(3):
                    U_full[i*3+j][r+rank_3x3] = U3[i*3+j][r]
            for j in range(3):
                for k in range(3):
                    V_full[j*3+(k+3)][r+rank_3x3] = V3[j*3+k][r]
            for k in range(3):
                for i in range(3):
                    W_full[(k+3)*3+i][r+rank_3x3] = W3[k*3+i][r]
                    
        return {"U": U_full, "V": V_full, "W": W_full}
    else:
        # Fallback
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
