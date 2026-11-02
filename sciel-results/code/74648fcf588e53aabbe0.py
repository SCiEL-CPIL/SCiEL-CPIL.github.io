import time
import sys
import numpy as np

DIMENSIONS = [3, 3, 6]

def _schoolbook():
    n, m, p = 3, 3, 6
    rank = n * m * p  # 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = 3, 3, 6
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    
    # Check if entries are half-integers
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _loss_and_grad(U, V, W, n=3, m=3, p=6):
    """Compute loss and gradients for the tensor decomposition."""
    rank = U.shape[1]
    residual = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                residual[i, j, k] = s - 1.0
    
    loss = np.sum(residual ** 2)
    
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r_ijk = residual[i, j, k]
                dU[i*m+j, :] += 2 * r_ijk * (V[j*p+k, :] * W[k*n+i, :])
                dV[j*p+k, :] += 2 * r_ijk * (U[i*m+j, :] * W[k*n+i, :])
                dW[k*n+i, :] += 2 * r_ijk * (U[i*m+j, :] * V[j*p+k, :])
    
    return loss, dU, dV, dW

def _try_rank_vectorized(rank, time_limit=5.0, max_restarts=20):
    """
    Try to find a valid decomposition of the given rank using vectorized local search.
    """
    n, m, p = 3, 3, 6
    start_time = time.time()
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with small random integers or half-integers
        rng = np.random.RandomState(restart + 12345)
        U = rng.choice([0, 0.5, -0.5, 1, -1], size=(n*m, rank))
        V = rng.choice([0, 0.5, -0.5, 1, -1], size=(m*p, rank))
        W = rng.choice([0, 0.5, -0.5, 1, -1], size=(p*n, rank))
        
        lr = 0.01
        best_loss = float('inf')
        
        for it in range(150):
            if time.time() - start_time > time_limit:
                break
                
            loss, dU, dV, dW = _loss_and_grad(U, V, W)
            
            if loss < best_loss:
                best_loss = loss
                if loss < 1e-12:
                    break
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            lr *= 0.97
            
            # Snap to nearest half-integer every few steps
            if it % 5 == 4:
                U = np.round(U * 2) / 2.0
                V = np.round(V * 2) / 2.0
                W = np.round(W * 2) / 2.0
        
        if best_loss < 1e-9:
            U_rounded = np.round(U * 2) / 2.0
            V_rounded = np.round(V * 2) / 2.0
            W_rounded = np.round(W * 2) / 2.0
            
            U_list = U_rounded.tolist()
            V_list = V_rounded.tolist()
            W_list = W_rounded.tolist()
            
            if _verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
                
    return None

def _try_block_decomposition(time_limit=30.0):
    """
    Try to use block decomposition: split 3x6 into two 3x3 blocks.
    Find a 3x3x3 decomposition and use it twice.
    """
    n, m, p = 3, 3, 6
    start_time = time.time()
    
    # For 3x3x3, try to find a low-rank decomposition
    # Schoolbook for 3x3x3 is rank 27
    # Try ranks from 26 down to 20
    
    for rank3 in range(26, 19, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Try to find a 3x3x3 decomposition of rank rank3
        for restart in range(10):
            if time.time() - start_time > time_limit:
                break
            
            rng = np.random.RandomState(restart + 99999)
            U3 = rng.choice([0, 0.5, -0.5, 1, -1], size=(9, rank3))
            V3 = rng.choice([0, 0.5, -0.5, 1, -1], size=(9, rank3))
            W3 = rng.choice([0, 0.5, -0.5, 1, -1], size=(9, rank3))
            
            lr = 0.01
            best_loss = float('inf')
            
            for it in range(100):
                if time.time() - start_time > time_limit:
                    break
                
                # Compute loss for 3x3x3
                residual = np.zeros((3, 3, 3))
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            s = np.dot(U3[i*3+j, :], V3[j*3+k, :] * W3[k*3+i, :])
                            residual[i, j, k] = s - 1.0
                
                loss = np.sum(residual ** 2)
                if loss < best_loss:
                    best_loss = loss
                    if loss < 1e-12:
                        break
                
                # Gradients
                dU3 = np.zeros_like(U3)
                dV3 = np.zeros_like(V3)
                dW3 = np.zeros_like(W3)
                
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            r_ijk = residual[i, j, k]
                            dU3[i*3+j, :] += 2 * r_ijk * (V3[j*3+k, :] * W3[k*3+i, :])
                            dV3[j*3+k, :] += 2 * r_ijk * (U3[i*3+j, :] * W3[k*3+i, :])
                            dW3[k*3+i, :] += 2 * r_ijk * (U3[i*3+j, :] * V3[j*3+k, :])
                
                U3 -= lr * dU3
                V3 -= lr * dV3
                W3 -= lr * dW3
                lr *= 0.97
                
                if it % 5 == 4:
                    U3 = np.round(U3 * 2) / 2.0
                    V3 = np.round(V3 * 2) / 2.0
                    W3 = np.round(W3 * 2) / 2.0
            
            if best_loss < 1e-9:
                U3_r = np.round(U3 * 2) / 2.0
                V3_r = np.round(V3 * 2) / 2.0
                W3_r = np.round(W3 * 2) / 2.0
                
                # Verify 3x3x3
                valid = True
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            s = np.dot(U3_r[i*3+j, :], V3_r[j*3+k, :] * W3_r[k*3+i, :])
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Build 3x3x6 decomposition using two copies
                    total_rank = 2 * rank3
                    U = np.zeros((9, total_rank))
                    V = np.zeros((18, total_rank))
                    W = np.zeros((18, total_rank))
                    
                    # First block: k=0,1,2
                    U[:, :rank3] = U3_r
                    V[:9, :rank3] = V3_r
                    W[:9, :rank3] = W3_r
                    
                    # Second block: k=3,4,5
                    U[:, rank3:] = U3_r
                    V[9:, rank3:] = V3_r
                    W[9:, rank3:] = W3_r
                    
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    
                    if _verify(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # Leave 5s margin
    
    # Try block decomposition first (may find rank 40-46)
    if time.time() < deadline - 30:
        result = _try_block_decomposition(time_limit=min(30.0, deadline - time.time() - 5))
        if result is not None:
            return result
    
    # Try direct search for lower ranks
    for rank in [53, 52, 51, 50, 49, 48]:
        if time.time() > deadline - 5:
            break
        remaining_time = deadline - time.time()
        if remaining_time < 3:
            break
            
        time_limit = min(8.0, remaining_time - 2.0)
        result = _try_rank_vectorized(rank, time_limit=time_limit, max_restarts=15)
        
        if result is not None:
            return result
            
    # Fallback to schoolbook
    return _schoolbook()
