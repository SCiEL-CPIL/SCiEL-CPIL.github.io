import time
import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Known rank 17 solution
    rank_17 = 17
    U17 = np.zeros((n * m, rank_17))
    V17 = np.zeros((m * p, rank_17))
    W17 = np.zeros((p * n, rank_17))
    
    # Term 1
    U17[0, 0] = 1; U17[4, 0] = 1
    V17[0, 0] = 1; V17[4, 0] = 1
    W17[0, 0] = 1; W17[3, 0] = 1
    
    # Term 2
    U17[3, 1] = 1; U17[4, 1] = 1
    V17[0, 1] = 1
    W17[1, 1] = 1; W17[3, 1] = -1
    
    # Term 3
    U17[0, 2] = 1
    V17[1, 2] = 1; V17[4, 2] = -1
    W17[2, 2] = 1; W17[3, 2] = 1
    
    # Term 4
    U17[4, 3] = 1
    V17[3, 3] = 1; V17[0, 3] = -1
    W17[0, 3] = 1; W17[1, 3] = 1
    
    # Term 5
    U17[0, 4] = 1; U17[1, 4] = 1
    V17[4, 4] = 1
    W17[0, 4] = -1; W17[2, 4] = 1
    
    # Term 6
    U17[3, 5] = 1; U17[0, 5] = -1
    V17[0, 5] = 1; V17[1, 5] = 1
    W17[3, 5] = 1
    
    # Term 7
    U17[1, 6] = 1; U17[4, 6] = -1
    V17[3, 6] = 1; V17[4, 6] = 1
    W17[0, 6] = 1
    
    # Term 8
    U17[0, 7] = 1
    V17[2, 7] = 1
    W17[4, 7] = 1
    
    # Term 9
    U17[1, 8] = 1
    V17[5, 8] = 1
    W17[4, 8] = 1
    
    # Term 10
    U17[3, 9] = 1
    V17[2, 9] = 1
    W17[5, 9] = 1
    
    # Term 11
    U17[4, 10] = 1
    V17[5, 10] = 1
    W17[5, 10] = 1
    
    # Term 12
    U17[2, 11] = 1
    V17[6, 11] = 1
    W17[0, 11] = 1
    
    # Term 13
    U17[2, 12] = 1
    V17[7, 12] = 1
    W17[2, 12] = 1
    
    # Term 14
    U17[2, 13] = 1
    V17[8, 13] = 1
    W17[4, 13] = 1
    
    # Term 15
    U17[5, 14] = 1
    V17[6, 14] = 1
    W17[1, 14] = 1
    
    # Term 16
    U17[5, 15] = 1
    V17[7, 15] = 1
    W17[3, 15] = 1
    
    # Term 17
    U17[5, 16] = 1
    V17[8, 16] = 1
    W17[5, 16] = 1

    def to_list(mat):
        return mat.tolist()
    
    # Pre-compute the target tensor T_target of shape (n*m, m*p, p*n)
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0

    # Attempt to find rank 16 solution
    # Note: Rank 16 is likely impossible for 2x3x3, so we limit search time to ensure robustness
    start_time = time.time()
    deadline = start_time + 5.0 # 5 seconds for search, ample buffer for safety
    
    R = 16
    
    # Pre-compute the linear system matrix A for the non-zero entries
    # We need to solve for W given U and V.
    # The equations are: sum_r U[u, r] V[v, r] W[w, r] = 1 for (u,v,w) in support.
    # And sum_r U[u, r] V[v, r] W[w, r] = 0 for others.
    # We only use the 18 non-zero equations for the linear solve, then verify all.
    
    tensor_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                tensor_indices.append((u, v, w))
                
    # Number of non-zero entries is 18
    num_nonzero = 18
    # A_mat will be (num_nonzero, R * p * n)
    # We build A_mat inside the loop because it depends on U and V?
    # No, A_mat depends on U and V.
    # A[idx, w*R + r] = U[u, r] * V[v, r]
    
    for _ in range(10000):
        if time.time() > deadline:
            break
            
        # Random sparse initialization
        # Use integers -1, 0, 1
        U = np.random.randint(-1, 2, size=(n*m, R))
        V = np.random.randint(-1, 2, size=(m*p, R))
        
        # Build linear system for W
        A_mat = np.zeros((num_nonzero, R * p * n)) 
        b_vec = np.ones(num_nonzero)
        
        for idx, (u, v, w) in enumerate(tensor_indices):
            for r in range(R):
                coeff = U[u, r] * V[v, r]
                A_mat[idx, w * R + r] = coeff
        
        # Solve least squares
        try:
            W_sol, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            W_sol = W_sol.reshape((p*n, R))
            
            # Check if W_sol is close to integer or half-integer
            # Try integers
            W_int = np.round(W_sol)
            if np.allclose(W_sol, W_int, atol=1e-6):
                W_candidate = W_int
            else:
                # Try half-integers
                W_half = np.round(W_sol * 2) / 2.0
                if np.allclose(W_sol, W_half, atol=1e-6):
                    W_candidate = W_half
                else:
                    continue
            
            # Full verification using vectorized einsum
            # T_recon[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
            T_recon = np.einsum('ur,vr,wr->uvw', U, V, W_candidate)
            
            if np.allclose(T_recon, T_target, atol=1e-9):
                # Found a valid rank 16 solution
                return {"U": to_list(U), "V": to_list(V), "W": to_list(W_candidate)}
                
        except Exception:
            pass

    # Return the known rank 17 solution
    return {"U": to_list(U17), "V": to_list(V17), "W": to_list(W17)}
