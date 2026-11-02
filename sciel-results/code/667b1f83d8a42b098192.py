import time
import numpy as np
import random

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify_decomposition(U, V, W, rank):
        # U, V, W are lists of lists
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_np[i*m + j] * V_np[j*p + k] * W_np[k*n + i])
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    def get_residuals(U, V, W, rank):
        """
        Compute sum of squared errors for the tensor decomposition.
        T_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}
        Target T_{ijk} = 1 if i,j,k valid, else 0? No, matrix multiplication tensor is 1 for all i,j,k in the support.
        Actually, the tensor is T_{i,j,k} = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
        """
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        total_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_np[i*m + j] * V_np[j*p + k] * W_np[k*n + i])
                    total_err += (val - 1.0) ** 2
        return total_err

    def snap_to_half_int(x):
        return np.round(x * 2) / 2

    def refine_decomposition(U, V, W, rank, time_limit):
        """
        Local search to snap to half-integers and reduce residual to 0.
        """
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Initial snap
        U = snap_to_half_int(U)
        V = snap_to_half_int(V)
        W = snap_to_half_int(W)
        
        current_err = get_residuals(U.tolist(), V.tolist(), W.tolist(), rank)
        
        # If already perfect, return
        if current_err < 1e-9:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
        values = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        
        # Coordinate Descent / Local Search
        # We will iterate through random entries and try changing them to nearby values
        # to reduce the error.
        
        while time.time() < time_limit:
            # Pick a random matrix and entry
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                idx = random.randint(0, nm - 1)
                r = random.randint(0, rank - 1)
                old_val = U[idx, r]
                # Try nearby values
                neighbors = [old_val - 0.5, old_val + 0.5]
                # Also allow jumping to other small values if stuck?
                # For now, simple local move.
                best_new_val = old_val
                best_new_err = current_err
                for nv in neighbors:
                    if nv < -2 or nv > 2: continue
                    U[idx, r] = nv
                    new_err = get_residuals(U.tolist(), V.tolist(), W.tolist(), rank)
                    if new_err < best_new_err:
                        best_new_err = new_err
                        best_new_val = nv
                    U[idx, r] = old_val
                
                if best_new_val != old_val:
                    U[idx, r] = best_new_val
                    current_err = best_new_err
            elif mat_idx == 1:
                idx = random.randint(0, mp - 1)
                r = random.randint(0, rank - 1)
                old_val = V[idx, r]
                neighbors = [old_val - 0.5, old_val + 0.5]
                best_new_val = old_val
                best_new_err = current_err
                for nv in neighbors:
                    if nv < -2 or nv > 2: continue
                    V[idx, r] = nv
                    new_err = get_residuals(U.tolist(), V.tolist(), W.tolist(), rank)
                    if new_err < best_new_err:
                        best_new_err = new_err
                        best_new_val = nv
                    V[idx, r] = old_val
                
                if best_new_val != old_val:
                    V[idx, r] = best_new_val
                    current_err = best_new_err
            else:
                idx = random.randint(0, pn - 1)
                r = random.randint(0, rank - 1)
                old_val = W[idx, r]
                neighbors = [old_val - 0.5, old_val + 0.5]
                best_new_val = old_val
                best_new_err = current_err
                for nv in neighbors:
                    if nv < -2 or nv > 2: continue
                    W[idx, r] = nv
                    new_err = get_residuals(U.tolist(), V.tolist(), W.tolist(), rank)
                    if new_err < best_new_err:
                        best_new_err = new_err
                        best_new_val = nv
                    W[idx, r] = old_val
                
                if best_new_val != old_val:
                    W[idx, r] = best_new_val
                    current_err = best_new_err
            
            if current_err < 1e-9:
                break
                
        if current_err < 1e-9:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None

    def svd_compress(U, V, W, target_rank):
        """
        Compress the rank 56 solution to target_rank using SVD.
        """
        # Construct the full tensor T of shape (n, m, p)
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1.0
        
        # We want to approximate T with rank r.
        # T_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}
        # This is a 3-way tensor decomposition.
        # We can use ALS (Alternating Least Squares) or just truncate SVD of a specific unfolding.
        # Unfold along U: T_{(i,j), (k,r)}? No.
        # Let's use a simple ALS approach to find U, V, W of target_rank.
        
        r = target_rank
        # Initialize randomly
        U = np.random.randn(nm, r)
        V = np.random.randn(mp, r)
        W = np.random.randn(pn, r)
        
        # Normalize to prevent scale issues
        U /= np.linalg.norm(U, axis=0, keepdims=True)
        V /= np.linalg.norm(V, axis=0, keepdims=True)
        W /= np.linalg.norm(W, axis=0, keepdims=True)
        
        # ALS Iterations
        for _ in range(50):
            # Update U
            # For each (i,j), T_{i,j,:} = sum_r U_{ij,r} * (V_{j,:,r} * W_{:,i,r})
            # Let's define auxiliary matrices
            # M_U[i,j, r] = sum_k V[j*p+k, r] * W[k*n+i, r]
            M_U = np.zeros((nm, r))
            for i in range(n):
                for j in range(m):
                    for r_idx in range(r):
                        s = 0
                        for k in range(p):
                            s += V[j*p + k, r_idx] * W[k*n + i, r_idx]
                        M_U[i*m + j, r_idx] = s
            # Solve U @ M_U^T = T_unfolded
            # T_unfolded shape (nm, p) ? No.
            # T_{i,j,k} = sum_r U_{i,j,r} * (V_{j,k,r} W_{k,i,r})
            # This is not a simple matrix multiplication because V and W depend on different indices.
            # This is a general 3-tensor decomposition.
            # ALS for general tensor is complex. 
            # Given time, I will skip complex ALS and rely on the Local Search from a random start or SVD of a flattened version if possible.
            # Actually, for small sizes, we can just use the Local Search directly on a random Rank 32 initialization.
            pass
            
        return None # Placeholder

    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find a better rank using Local Search from random starts
    # We know rank 32 is possible.
    # Let's try to find rank 32.
    
    target_ranks = [32, 33, 34, 35, 36, 37, 38, 39, 40]
    
    for R in target_ranks:
        if time.time() > deadline - 20.0:
            break
            
        success = False
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 20.0:
                break
                
            # Random initialization with small integers
            U = np.random.choice([-1, 0, 1], size=(nm, R))
            V = np.random.choice([-1, 0, 1], size=(mp, R))
            W = np.random.choice([-1, 0, 1], size=(pn, R))
            
            # Run local search
            time_limit = time.time() + 5.0
            result = refine_decomposition(U.tolist(), V.tolist(), W.tolist(), R, time_limit)
            if result is not None:
                best_result = result
                best_rank = R
                success = True
                break
                
        if success:
            break
            
    return best_result
