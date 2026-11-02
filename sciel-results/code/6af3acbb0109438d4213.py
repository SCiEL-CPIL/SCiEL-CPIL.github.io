import numpy as np
import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank to try
    target_ranks = [37, 38, 39, 40, 42, 44, 46, 48]
    
    best_result = None
    best_rank = n * m * p
    
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin
    
    for rank in target_ranks:
        if time.time() > deadline:
            break
            
        # Try to find a decomposition of this rank
        # Using random initialization and alternating minimization
        # The tensor T is (n*m, m*p, p*n) -> (4, 24, 24)
        # We want T = sum_r U_r V_r W_r
        
        # Initialize U, V, W
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Scale to reasonable values
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        # Alternating Least Squares
        for it in range(500):
            if time.time() > deadline:
                break
                
            # Update U
            # T = sum U_r V_r W_r
            # For each column r of U, we have U[:, r] = (T - sum_{r'!=r} U_r' V_r' W_r') / (V_r W_r)
            # This is a linear least squares problem.
            
            # Efficient update:
            # T_flat is (4, 24, 24)
            # We can reshape T to (4*24, 24) and use matrix ops.
            
            # Let's use a simple iterative update
            # T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
            # Error = T - T_approx
            
            # This is too slow for large rank in pure Python loops.
            # Use vectorized operations.
            
            # Reshape T to (n*m, m*p*p*n) ? No.
            # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
            
            # Update U[:, r]:
            # For fixed r, U[:, r] minimizes || T - sum_{r'} U_{r'} V_{r'} W_{r'} ||
            # This is equivalent to:
            # (V_r W_r^T) * U[:, r] = T - (T - U[:, r] V_r W_r^T)
            # Let M = V_r * W_r (outer product? No, V_r is (m*p, 1), W_r is (p*n, 1))
            # The term U[:, r] V_r W_r^T is a rank 1 tensor.
            # We can flatten the tensor.
            
            # Let's just do a few iterations of a simpler method or use a library if available.
            # Since SciPy is available, we can use least squares.
            
            # For each r:
            # Reshape T to (n*m, (m*p)*(p*n))
            # T_flat = T.reshape(n*m, m*p*p*n)
            # The column corresponding to V_r W_r is the Kronecker product V_r (kron) W_r
            # U[:, r] = argmin || T_flat - U[:, r] (V_r kron W_r)^T ||
            
            T_flat = np.zeros((n * m, m * p * p * n))
            # Construct T
            # T[i, j, k] = 1 if i=m*j? No.
            # Standard basis:
            # T[i, j, k] = 1 if i == m * (j % m) + (j // m) ? No.
            # Indices:
            # i in 0..n-1 (row of A, col of C)
            # j in 0..m-1 (col of A, row of B)
            # k in 0..p-1 (col of B, col of C)
            # C[i, k] = sum_j A[i, j] B[j, k]
            # Tensor index: (i, j, k)
            # U index: i*m + j ? No, U is (n*m, rank). Index i*m+j?
            # In the seed: U[i*m+j, c]. So U row index is i*m + j.
            # V index: j*p + k.
            # W index: k*n + i.
            
            # So T[(i*m+j), (j*p+k), (k*n+i)] = 1.
            
            T = np.zeros((n * m, m * p, p * n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i * m + j, j * p + k, k * n + i] = 1.0
            
            T_flat = T.reshape(n * m, m * p * p * n)
            
            # Update U
            for r in range(rank):
                v_w = np.kron(V[:, r], W[:, r])
                # v_w is (m*p*p*n, 1)
                # We want U[:, r] such that U[:, r] v_w^T approximates T_flat
                # This is a rank 1 approximation.
                # U[:, r] = (T_flat v_w) / (v_w^T v_w)
                numerator = T_flat @ v_w
                denominator = v_w @ v_w
                if denominator > 1e-10:
                    U[:, r] = numerator / denominator
                    
            # Update V
            # Similar logic
            # T_flat2 = T.reshape(m*p, n*m*p*n)
            T_flat2 = T.reshape(m * p, n * m * p * n)
            for r in range(rank):
                u_w = np.kron(U[:, r], W[:, r])
                # Note: order in kron must match reshape order.
                # T[i,j,k]. U is i*m+j. V is j*p+k. W is k*n+i.
                # Reshape to (m*p, ...) means first index is j*p+k.
                # The remaining indices are i*m+j and k*n+i.
                # So the vector is kron(U, W) but with appropriate ordering.
                # Let's stick to the previous logic which worked for U.
                # For V, we fix V[:, r] and update.
                # T = sum U V W
                # T_flat2[:, :] = sum_r V[:, r] (U[:, r] kron W[:, r])^T ?
                # Actually, T[j, i, k] = sum_r U[i, r] V[j, r] W[k, r]
                # Flatten j to row.
                # Col index corresponds to (i, k).
                # Vector is U[i] * W[k] = kron(U, W).
                u_w = np.kron(U[:, r], W[:, r])
                numerator = T_flat2 @ u_w
                denominator = u_w @ u_w
                if denominator > 1e-10:
                    V[:, r] = numerator / denominator

            # Update W
            T_flat3 = T.reshape(p * n, n * m * m * p)
            for r in range(rank):
                u_v = np.kron(U[:, r], V[:, r])
                numerator = T_flat3 @ u_v
                denominator = u_v @ u_v
                if denominator > 1e-10:
                    W[:, r] = numerator / denominator
            
        # After ALS, round to nearest half-integer
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        # Verify
        T_approx = np.einsum('ir,jr,kr->ijk', U_round, V_round, W_round)
        error = np.linalg.norm(T - T_approx)
        
        if error < 1e-9:
            # Convert to list of lists
            U_list = U_round.tolist()
            V_list = V_round.tolist()
            W_list = W_round.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
            
        # If not found, continue to next rank or return best effort
        # For now, if we don't find a low rank, we return schoolbook
        if rank < best_rank:
             # Store best attempt? No, must be exact.
             pass

    # Fallback to schoolbook
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
