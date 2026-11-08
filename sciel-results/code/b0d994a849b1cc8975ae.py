import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # 10s margin before 150s kill
    
    # Schoolbook decomposition
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=float)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=float)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    def validate(U, V, W, num_tests=20):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        R = U.shape[1]
        
        for _ in range(num_tests):
            A = np.random.randint(-10, 10, size=(n, m))
            B = np.random.randint(-10, 10, size=(m, p))
            C_true = A @ B
            
            # Compute C_decomp = sum_r W_r * (U_r^T A_flat) * (V_r^T B_flat)
            # But this isn't right. The correct formula is:
            # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
            # This is only valid if U and V are rank-1 in their respective indices.
            
            # Actually, the correct tensor decomposition condition is:
            # For the multiplication tensor, we need:
            # C[i,k] = sum_{j,r} A[i,j] * B[j,k] * T[i,j,k]
            # where T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # And we need T[i,j,k] = 1 for all i,j,k.
            
            # So let's check if T_decomp[i,j,k] = 1 for all i,j,k
            # T_decomp[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # Efficient check: compute the full tensor and compare to ones
            # But that's expensive. Instead, check via random A, B.
            
            # C_decomp[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
            # This is NOT correct in general. The issue is that U[i*m+j, r] and V[j*p+k, r] 
            # are coupled through j.
            
            # The correct approach: the tensor T is a 3-way array.
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # We need T[i,j,k] = 1 for all i,j,k.
            
            # Let's just build T_decomp and check.
            T_decomp = np.zeros((n, m, p))
            for r in range(R):
                u_r = U[:, r].reshape(n, m)
                v_r = V[:, r].reshape(m, p)
                w_r = W[:, r].reshape(p, n)
                # T[i,j,k] += u_r[i,j] * v_r[j,k] * w_r[k,i]
                # This is an outer product
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_decomp[i, j, k] += u_r[i, j] * v_r[j, k] * w_r[k, i]
            
            if not np.allclose(T_decomp, np.ones((n, m, p)), atol=1e-10):
                return False
        
        return True
    
    # Try random search for lower rank
    # Start from rank 40 down to 30
    for target_rank in range(40, 29, -1):
        if time.time() > deadline - 30:
            break
        
        # Try several random initializations
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with small values
            U_rand = np.random.randn(n * m, target_rank) * 0.1
            V_rand = np.random.randn(m * p, target_rank) * 0.1
            W_rand = np.random.randn(p * n, target_rank) * 0.1
            
            # Simple gradient-free search: perturb and check
            # This is very slow, so let's just try a few
            if validate(U_rand, V_rand, W_rand, num_tests=3):
                best_U, best_V, best_W = U_rand, V_rand, W_rand
                best_rank = target_rank
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
