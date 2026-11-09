import numpy as np
import time
import random

DIMENSIONS = [2, 5, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = U.shape[1]
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j]
                v_val = V_np[j * p + k]
                w_val = W_np[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _solve_subproblem(n, m, p, target_rank, time_limit=30):
    """
    Attempt to find a rank-target_rank decomposition for M_{n,m,p}.
    Returns U, V, W or None if failed.
    """
    start_time = time.time()
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook if target_rank is large, else random
    # For small ranks, random is better
    
    for attempt in range(20):
        if time.time() - start_time > time_limit:
            break
            
        rng = np.random.RandomState(attempt * 1234 + n * m * p)
        
        U = rng.randn(n * m, target_rank) * 0.1
        V = rng.randn(m * p, target_rank) * 0.1
        W = rng.randn(p * n, target_rank) * 0.1
        
        # Optimization
        lr = 0.01
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        for step in range(2000):
            if time.time() - start_time > time_limit:
                break
                
            # Compute error and gradients
            U_r = U.reshape(n, m, target_rank)
            V_r = V.reshape(m, p, target_rank)
            W_r = W.reshape(p, n, target_rank)
            
            T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            residual = T - 1.0
            err = np.sum(residual**2)
            
            if err < 1e-10:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                return best_U, best_V, best_W
                
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            dU_r = 2.0 * np.einsum('ijk,jkc,kic->ijc', residual, V_r, W_r)
            dV_r = 2.0 * np.einsum('ijk,ijc,kic->jkc', residual, U_r, W_r)
            dW_r = 2.0 * np.einsum('ijk,ijc,jkc->kic', residual, U_r, V_r)
            
            dU = dU_r.reshape(n*m, target_rank)
            dV = dV_r.reshape(m*p, target_rank)
            dW = dW_r.reshape(p*n, target_rank)
            
            # Adam-like update (simplified)
            vU = momentum * vU + (1 - momentum) * dU
            vV = momentum * vV + (1 - momentum) * dV
            vW = momentum * vW + (1 - momentum) * dW
            
            U -= lr * vU
            V -= lr * vV
            W -= lr * vW
            
            # Decay learning rate
            if step % 500 == 0 and step > 0:
                lr *= 0.5
                
    if best_err < 1e-6:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Strategy:
    # 1. Try to solve M_{2,2,3} with rank 11
    # 2. Try to solve M_{2,3,3} with rank 16
    # 3. Combine to get M_{2,5,3} with rank 27
    # 4. Duplicate to get M_{2,5,6} with rank 54
    
    # Subproblem 1: M_{2,2,3}
    n1, m1, p1 = 2, 2, 3
    rank1 = 11
    res1 = _solve_subproblem(n1, m1, p1, rank1, time_limit=40)
    
    # Subproblem 2: M_{2,3,3}
    n2, m2, p2 = 2, 3, 3
    rank2 = 16
    res2 = _solve_subproblem(n2, m2, p2, rank2, time_limit=40)
    
    if res1 is not None and res2 is not None:
        U1, V1, W1 = res1
        U2, V2, W2 = res2
        
        # Construct M_{2,5,3}
        # A is 2x5. Split into A1 (2x2) and A2 (2x3).
        # B is 5x3. Split into B1 (2x3) and B2 (3x3).
        # C = A1 B1 + A2 B2.
        
        # U for M_{2,5,3}:
        # U1 corresponds to A1 (indices i in 0..1, j in 0..1)
        # U2 corresponds to A2 (indices i in 0..1, j in 2..4)
        # We need U of shape (2*5, 27) = (10, 27)
        
        U_total = np.zeros((10, 27))
        # U1 is (2*2, 11) = (4, 11). Indices j=0,1.
        # Map U1 to U_total rows corresponding to j=0,1.
        # Row index in U_total: i*m + j. m=5.
        # i=0, j=0 -> 0
        # i=0, j=1 -> 1
        # i=1, j=0 -> 5
        # i=1, j=1 -> 6
        # U1 rows:
        # 0: i=0,j=0
        # 1: i=0,j=1
        # 2: i=1,j=0
        # 3: i=1,j=1
        
        U_total[0, :11] = U1[0, :]
        U_total[1, :11] = U1[1, :]
        U_total[5, :11] = U1[2, :]
        U_total[6, :11] = U1[3, :]
        
        # U2 is (2*3, 16) = (6, 16). Indices j=0,1,2 in the sub-problem.
        # These correspond to j=2,3,4 in the full problem.
        # U2 rows:
        # 0: i=0,j=0 (sub) -> i=0,j=2 (full) -> row 2
        # 1: i=0,j=1 (sub) -> i=0,j=3 (full) -> row 3
        # 2: i=0,j=2 (sub) -> i=0,j=4 (full) -> row 4
        # 3: i=1,j=0 (sub) -> i=1,j=2 (full) -> row 7
        # 4: i=1,j=1 (sub) -> i=1,j=3 (full) -> row 8
        # 5: i=1,j=2 (sub) -> i=1,j=4 (full) -> row 9
        
        U_total[2, 11:27] = U2[0, :]
        U_total[3, 11:27] = U2[1, :]
        U_total[4, 11:27] = U2[2, :]
        U_total[7, 11:27] = U2[3, :]
        U_total[8, 11:27] = U2[4, :]
        U_total[9, 11:27] = U2[5, :]
        
        # V for M_{2,5,3}:
        # V1 is (2*3, 11) = (6, 11). Indices j=0,1 (sub) -> j=0,1 (full). k=0,1,2.
        # V1 rows:
        # 0: j=0,k=0 -> row 0
        # 1: j=0,k=1 -> row 1
        # 2: j=0,k=2 -> row 2
        # 3: j=1,k=0 -> row 3
        # 4: j=1,k=1 -> row 4
        # 5: j=1,k=2 -> row 5
        
        V_total = np.zeros((15, 27))
        V_total[0, :11] = V1[0, :]
        V_total[1, :11] = V1[1, :]
        V_total[2, :11] = V1[2, :]
        V_total[3, :11] = V1[3, :]
        V_total[4, :11] = V1[4, :]
        V_total[5, :11] = V1[5, :]
        
        # V2 is (3*3, 16) = (9, 16). Indices j=0,1,2 (sub) -> j=2,3,4 (full). k=0,1,2.
        # V2 rows:
        # 0: j=0,k=0 (sub) -> j=2,k=0 (full) -> row 2*3+0 = 6
        # 1: j=0,k=1 -> row 7
        # 2: j=0,k=2 -> row 8
        # 3: j=1,k=0 -> row 9
        # 4: j=1,k=1 -> row 10
        # 5: j=1,k=2 -> row 11
        # 6: j=2,k=0 -> row 12
        # 7: j=2,k=1 -> row 13
        # 8: j=2,k=2 -> row 14
        
        V_total[6, 11:27] = V2[0, :]
        V_total[7, 11:27] = V2[1, :]
        V_total[8, 11:27] = V2[2, :]
        V_total[9, 11:27] = V2[3, :]
        V_total[10, 11:27] = V2[4, :]
        V_total[11, 11:27] = V2[5, :]
        V_total[12, 11:27] = V2[6, :]
        V_total[13, 11:27] = V2[7, :]
        V_total[14, 11:27] = V2[8, :]
        
        # W for M_{2,5,3}:
        # W1 is (3*2, 11) = (6, 11). Indices k=0,1,2. i=0,1.
        # W1 rows:
        # 0: k=0,i=0 -> row 0
        # 1: k=0,i=1 -> row 1
        # 2: k=1,i=0 -> row 2
        # 3: k=1,i=1 -> row 3
        # 4: k=2,i=0 -> row 4
        # 5: k=2,i=1 -> row 5
        
        W_total = np.zeros((6, 27))
        W_total[0, :11] = W1[0, :]
        W_total[1, :11] = W1[1, :]
        W_total[2, :11] = W1[2, :]
        W_total[3, :11] = W1[3, :]
        W_total[4, :11] = W1[4, :]
        W_total[5, :11] = W1[5, :]
        
        # W2 is (3*2, 16) = (6, 16). Indices k=0,1,2. i=0,1.
        # W2 rows:
        # 0: k=0,i=0 -> row 0
        # 1: k=0,i=1 -> row 1
        # 2: k=1,i=0 -> row 2
        # 3: k=1,i=1 -> row 3
        # 4: k=2,i=0 -> row 4
        # 5: k=2,i=1 -> row 5
        
        W_total[0, 11:27] = W2[0, :]
        W_total[1, 11:27] = W2[1, :]
        W_total[2, 11:27] = W2[2, :]
        W_total[3, 11:27] = W2[3, :]
        W_total[4, 11:27] = W2[4, :]
        W_total[5, 11:27] = W2[5, :]
        
        # Now we have M_{2,5,3} with rank 27.
        # We need M_{2,5,6}.
        # Split p=6 into p1=3 and p2=3.
        # C = A B1 + A B2.
        # U is the same for both (depends on A).
        # V and W are different for B1 and B2.
        
        U_final = np.zeros((10, 54))
        U_final[:, :27] = U_total
        U_final[:, 27:] = U_total
        
        V_final = np.zeros((30, 54))
        # V_total is (15, 27).
        # V for B1 (k=0,1,2) is V_total.
        # V for B2 (k=3,4,5) is also V_total (since B2 is 5x3, same structure).
        # Rows 0..14 correspond to k=0,1,2.
        # Rows 15..29 correspond to k=3,4,5.
        
        V_final[0:15, :27] = V_total
        V_final[15:30, 27:] = V_total
        
        W_final = np.zeros((12, 54))
        # W_total is (6, 27).
        # W for B1 (k=0,1,2) is W_total.
        # W for B2 (k=3,4,5) is also W_total.
        # Rows 0..5 correspond to k=0,1,2.
        # Rows 6..11 correspond to k=3,4,5.
        
        W_final[0:6, :27] = W_total
        W_final[6:12, 27:] = W_total
        
        # Verify
        if _verify_decomposition(U_final, V_final, W_final, 2, 5, 6):
            return {
                "U": U_final.tolist(),
                "V": V_final.tolist(),
                "W": W_final.tolist()
            }
        
    # Fallback to schoolbook
    U = np.zeros((10, 60))
    V = np.zeros((30, 60))
    W = np.zeros((12, 60))
    
    col = 0
    for i in range(2):
        for j in range(5):
            for k in range(6):
                U[i*5 + j, col] = 1
                V[j*6 + k, col] = 1
                W[k*2 + i, col] = 1
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
