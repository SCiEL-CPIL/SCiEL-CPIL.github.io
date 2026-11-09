"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 10.0 # 10 seconds for search
    
    # Attempt to find a lower rank decomposition via random search
    # This is a heuristic and may not find a solution, but it's a structural change.
    # We search for ranks from schoolbook_rank - 1 down to schoolbook_rank - 5
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_school, V_school, W_school
    
    for target_rank in range(schoolbook_rank - 1, schoolbook_rank - 10, -1):
        if time.time() > deadline:
            break
        # Try a few random initializations
        for _ in range(100):
            if time.time() > deadline:
                break
            # Randomly initialize U, V, W with small integers
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Verify
            if verify_decomposition(U, V, W, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                return {"U": best_U, "V": best_V, "W": best_W}
                
    return {"U": best_U, "V": best_V, "W": best_W}

def verify_decomposition(U, V, W, n, m, p):
    # Convert to numpy for speed
    U_np = np.array(U, dtype=int)
    V_np = np.array(V, dtype=int)
    W_np = np.array(W, dtype=int)
    
    # Compute tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i, j, k] == 1 for all i, j, k
    # This is a high-dimensional check.
    # We can check by computing the full tensor and comparing to ones.
    
    # Unroll for small dimensions
    rank = U_np.shape[1]
    T = np.zeros((n, m, p), dtype=int)
    
    for r in range(rank):
        u_vec = U_np[:, r]
        v_vec = V_np[:, r]
        w_vec = W_np[:, r]
        
        # u_vec has length n*m, indexed by i*m+j
        # v_vec has length m*p, indexed by j*p+k
        # w_vec has length p*n, indexed by k*n+i
        
        # Reshape u to (n, m)
        u_mat = u_vec.reshape(n, m)
        # Reshape v to (m, p)
        v_mat = v_vec.reshape(m, p)
        # Reshape w to (p, n)
        w_mat = w_vec.reshape(p, n)
        
        # The contribution to T[i, j, k] is u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
        # We can compute this efficiently?
        # T += u_mat[:, :, None] * v_mat[None, :, :] * w_mat.T[:, None, :]?
        # w_mat is (p, n). w_mat[k, i].
        # w_mat.T is (n, p). w_mat.T[i, k].
        # So term is u_mat[i, j] * v_mat[j, k] * w_mat.T[i, k]
        
        # u_mat: (n, m)
        # v_mat: (m, p)
        # w_mat.T: (n, p)
        
        # We need to sum over j? No, the tensor is T[i, j, k].
        # The term is a product of 3 matrices.
        # We can compute this as:
        # temp = u_mat[:, :, None] * v_mat[None, :, :]  # Shape (n, m, p)
        # temp *= w_mat.T[:, None, :]  # w_mat.T[i, k] -> w_mat.T[:, None, :] has shape (n, 1, p)
        # T += temp
        
        temp = u_mat[:, :, None] * v_mat[None, :, :]
        temp *= w_mat.T[:, None, :]
        T += temp

    # Check if T is all ones
    if np.all(T == 1):
        return True
    return False
# EVOLVE-BLOCK-END
