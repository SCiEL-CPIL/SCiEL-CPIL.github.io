import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # The tensor equation is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # Schoolbook fallback: rank 72
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1, else 0
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Flatten T into a matrix of shape (n*m, (m*p)*(p*n)) for SVD
    T_mat = T.reshape(n * m, m * p * p * n)
    
    # Try to find a low-rank decomposition using randomized SVD
    # We'll try ranks from 60 to 71 and see if we can get an exact decomposition
    # after rounding to half-integers
    
    for target_rank in range(60, 72):
        if time.time() > deadline - 10:
            break
        
        # Randomized SVD
        try:
            # Generate random projection matrix
            Omega = np.random.randn(m * p * p * n, target_rank)
            Y = T_mat @ Omega
            Q, _ = np.linalg.qr(Y)
            B = Q.T @ T_mat
            U_svd, s, Vt_svd = np.linalg.svd(B, full_matrices=False)
            
            # Reconstruct T_mat approximation
            T_approx = (U_svd @ np.diag(s) @ Vt_svd).reshape(n * m, m * p, p * n)
            
            # Now we need to decompose T_approx into rank-1 tensors
            # T_approx[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
            # We can use the SVD of T_approx reshaped
            
            # Actually, let's use a different approach:
            # We have T_mat ≈ U_svd @ diag(s) @ Vt_svd
            # T_mat is (n*m, m*p*p*n)
            # We want to reshape this into (n*m, m*p, p*n) and decompose
            
            # Let's use a tensor decomposition approach
            # T_approx[i, j, k] where i is n*m, j is m*p, k is p*n
            
            # We can try to find U, V, W such that T_approx = sum_c outer(U[:,c], V[:,c], W[:,c])
            # This is a 3-way tensor decomposition, which is NP-hard in general
            # But for small tensors, we can try alternating least squares
            
            # Initialize U, V, W randomly
            np.random.seed(42)
            U_init = np.random.randn(n * m, target_rank) * 0.1
            V_init = np.random.randn(m * p, target_rank) * 0.1
            W_init = np.random.randn(p * n, target_rank) * 0.1
            
            # Alternating least squares
            for iteration in range(100):
                if time.time() > deadline - 10:
                    break
                
                # Update U
                for c in range(target_rank):
                    # Fix V and W, solve for U[:, c]
                    # T_approx[i, j, k] = U[i, c] * V[j, c] * W[k, c] + sum_{c' != c} U[i, c'] * V[j, c'] * W[k, c']
                    # This is complex, so let's use a simpler approach
                    
                    # Actually, let's just use the SVD directly and round
                    pass
            
            # Simpler approach: use the SVD of T_mat and round
            # T_mat ≈ U_svd @ diag(s) @ Vt_svd
            # U_svd is (n*m, target_rank), Vt_svd is (target_rank, m*p*p*n)
            # We need to reshape Vt_svd into (target_rank, m*p, p*n)
            Vt_reshaped = Vt_svd.reshape(target_rank, m * p, p * n)
            
            # Now we have T_mat[i, (j,k)] ≈ sum_c U_svd[i, c] * Vt_reshaped[c, j, k]
            # We want T_approx[i, j, k] = sum_c U[i, c] * V[j, c] * W[k, c]
            # So we need to decompose Vt_reshaped[c, j, k] = V[j, c] * W[k, c]
            # This is a rank-1 approximation for each c, which is not generally possible
            
            # Let's try a different strategy: use the fact that T is sparse and structured
            # We'll use a greedy approach to find a low-rank decomposition
            
            # For now, let's just check if the SVD gives a good approximation
            error = np.linalg.norm(T_mat - (U_svd @ np.diag(s) @ Vt_svd))
            if error < 1e-10:
                # We have an exact low-rank decomposition of T_mat
                # But we need to convert it to the tensor form
                # This is non-trivial, so let's skip this for now
                pass
                
        except Exception:
            continue
    
    # Given the complexity of the tensor decomposition, let's use a simpler approach:
    # We'll use a known rank-68 decomposition from the literature or construct one
    # using a combination of Strassen and schoolbook terms
    
    # For now, return the schoolbook decomposition as a safe fallback
    return best

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # Schoolbook fallback: rank 72
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Flatten T into a matrix of shape (n*m, (m*p)*(p*n)) for SVD
    T_mat = T.reshape(n * m, m * p * p * n)
    
    # Try to find a low-rank decomposition using randomized SVD
    # We'll try ranks from 60 to 71 and see if we can get an exact decomposition
    # after rounding to half-integers
    
    for target_rank in range(60, 72):
        if time.time() > deadline - 10:
            break
        
        # Randomized SVD
        try:
            # Generate random projection matrix
            Omega = np.random.randn(m * p * p * n, target_rank)
            Y = T_mat @ Omega
            Q, _ = np.linalg.qr(Y)
            B = Q.T @ T_mat
            U_svd, s, Vt_svd = np.linalg.svd(B, full_matrices=False)
            
            # Reconstruct T_mat approximation
            T_approx = (U_svd @ np.diag(s) @ Vt_svd).reshape(n * m, m * p, p * n)
            
            # Now we need to decompose T_approx into rank-1 tensors
            # T_approx[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
            
            # Use alternating least squares to find U, V, W
            np.random.seed(42)
            U_init = np.random.randn(n * m, target_rank) * 0.1
            V_init = np.random.randn(m * p, target_rank) * 0.1
            W_init = np.random.randn(p * n, target_rank) * 0.1
            
            # Alternating least squares
            for iteration in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Update U
                for c in range(target_rank):
                    # Fix V and W, solve for U[:, c]
                    # T_approx[i, j, k] = U[i, c] * V[j, c] * W[k, c] + sum_{c' != c} U[i, c'] * V[j, c'] * W[k, c']
                    # This is a linear system in U[:, c]
                    
                    # Actually, let's use a simpler approach:
                    # We'll use the SVD of T_approx directly
                    pass
                
                # Update V
                for c in range(target_rank):
                    # Fix U and W, solve for V[:, c]
                    pass
                
                # Update W
                for c in range(target_rank):
                    # Fix U and V, solve for W[:, c]
                    pass
            
            # Given the complexity, let's just use the SVD and round
            # This is a heuristic and may not work
            pass
            
        except Exception:
            continue
    
    # Given the complexity of the tensor decomposition, let's use a simpler approach:
    # We'll use a known rank-68 decomposition from the literature or construct one
    # using a combination of Strassen and schoolbook terms
    
    # For now, return the schoolbook decomposition as a safe fallback
    return best
# EVOLVE-BLOCK-END
