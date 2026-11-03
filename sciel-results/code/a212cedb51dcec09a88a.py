import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def _snap_to_half_int(matrix):
    """Snap matrix values to nearest half-integer."""
    return np.round(matrix * 2) / 2.0

def _check_exact(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the tensor."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    # Reshape for efficient computation
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # We need sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 for all i,j,k
    # This is a batched matrix multiplication or tensor contraction.
    # Let's compute the full tensor T_calc
    # T_calc[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    
    # We can do this by iterating over r
    T_calc = np.zeros((n, m, p))
    for r in range(rank):
        u_r = U_reshaped[:, :, r] # (n, m)
        v_r = V_reshaped[:, :, r] # (m, p)
        w_r = W_reshaped[:, :, r] # (p, n)
        
        # Product u_r[i,j] * v_r[j,k] * w_r[k,i]
        # Sum over j? No, we want T[i,j,k].
        # T[i,j,k] += u_r[i,j] * v_r[j,k] * w_r[k,i]
        
        # Create a tensor for this r
        # u_r is (n, m), v_r is (m, p), w_r is (p, n)
        # We need outer product contracted?
        # T_r[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[k,i]
        
        # Use np.einsum
        # 'ij, jk, ki -> ijk'
        term = np.einsum('ij,jk,ki->ijk', u_r, v_r, w_r)
        T_calc += term
        
    # Check if T_calc is all ones
    # Note: Floating point precision.
    # Since we snapped to half-integers, values should be exact if the decomposition is exact.
    # But accumulation might have error.
    # We check if T_calc is close to 1.
    if not np.allclose(T_calc, 1.0, atol=1e-9):
        return False
    return True

def _try_rank_optimize(n, m, p, target_rank, time_limit):
    """Try to find a decomposition of target_rank."""
    start_time = time.time()
    num_attempts = 50
    
    for attempt in range(num_attempts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with random half-integers
        # Scale to avoid huge numbers?
        # Values in {-1, -0.5, 0, 0.5, 1}
        
        U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, target_rank))
        V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, target_rank))
        W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, target_rank))
        
        # ALS
        for step in range(20):
            if time.time() - start_time > time_limit:
                break
                
            # Solve for U
            # Fix V, W.
            # T[i,j,k] = sum_r U[i,j,r] (V[j,k,r] W[k,i,r])
            # For each i,j, we have a linear system for U[i,j,:].
            # Let A_{ij, r} = V[j,k,r] * W[k,i,r] ? No, k is summed?
            # No, the equation is for EACH i,j,k.
            # T[i,j,k] = 1.
            # This is a system of equations.
            # Variables: U[i,j,r] for all i,j,r.
            # Equations: for each i,j,k: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
            
            # This is linear in U.
            # Let's construct the matrix for the least squares problem.
            # Number of equations: n*m*p = 24.
            # Number of unknowns: n*m*target_rank.
            
            # We can solve for U given V, W.
            # For each (i,j), the unknowns are U[i,j, 0..R-1].
            # The equations involving U[i,j] are for all k.
            # For fixed i,j:
            # sum_r U[i,j,r] (V[j,k,r] * W[k,i,r]) = 1 for all k.
            # This is a linear system: M_{k,r} * U_{i,j,r} = 1_k
            # where M_{k,r} = V[j,k,r] * W[k,i,r].
            
            U_new = np.zeros((n*m, target_rank))
            for i in range(n):
                for j in range(m):
                    # Build M (p x target_rank)
                    M = np.zeros((p, target_rank))
                    for k in range(p):
                        for r in range(target_rank):
                            M[k, r] = V[j*p + k, r] * W[k*n + i, r]
                    
                    # Solve M * U_col = ones(p)
                    # Use least squares
                    try:
                        U_col, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U_new[i*m + j, :] = U_col
                    except:
                        pass
                        
            U = _snap_to_half_int(U_new)
            
            # Similarly for V and W
            V_new = np.zeros((m*p, target_rank))
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, target_rank))
                    for i in range(n):
                        for r in range(target_rank):
                            M[i, r] = U[i*m + j, r] * W[k*n + i, r]
                    try:
                        V_col, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V_new[j*p + k, :] = V_col
                    except:
                        pass
            V = _snap_to_half_int(V_new)
            
            W_new = np.zeros((p*n, target_rank))
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, target_rank))
                    for j in range(m):
                        for r in range(target_rank):
                            M[j, r] = U[i*m + j, r] * V[j*p + k, r]
                    try:
                        W_col, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W_new[k*n + i, :] = W_col
                    except:
                        pass
            W = _snap_to_half_int(W_new)
            
            # Check exactness
            if _check_exact(U, V, W, n, m, p):
                # Convert to list of lists
                return U.tolist(), V.tolist(), W.tolist()
                
    return None

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    # Try to optimize
    # Start with rank 23
    deadline = time.time() + 100  # 100s budget for search
    
    for r in range(23, 10, -1):
        if time.time() > deadline:
            break
        result = _try_rank_optimize(n, m, p, r, time_limit=5)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    return {"U": U, "V": V, "W": W}
