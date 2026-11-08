"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as baseline
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook), dtype=np.float64)
    V = np.zeros((m * p, rank_schoolbook), dtype=np.float64)
    W = np.zeros((p * n, rank_schoolbook), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_schoolbook
    
    # Try to find a lower-rank decomposition
    # The tensor T[i,j,k] = 1 for all i,j,k in the multiplication sense
    # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Build the target tensor
    T_target = np.ones((n, m, p), dtype=np.float64)
    
    def verify(U_mat, V_mat, W_mat, rank):
        """Verify decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        T_recon = np.zeros((n, m, p), dtype=np.float64)
        for r in range(rank):
            u_col = U_mat[:, r].reshape(n, m)
            v_col = V_mat[:, r].reshape(m, p)
            w_col = W_mat[:, r].reshape(p, n)
            # T[i,j,k] += u_col[i,j] * v_col[j,k] * w_col[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
        return np.allclose(T_recon, T_target, atol=1e-10)
    
    # Try Strassen-based construction for rank reduction
    # For 2x2, Strassen uses rank 7 with entries in {-1,0,1,2}
    # We can try to build a 5x5 decomposition from smaller blocks
    
    # Attempt: use a rank-25 decomposition based on outer product structure
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is inherently rank 25 if we fix j and vary (i,k)... no, that's not right.
    
    # Let's try random search with structured initialization
    np.random.seed(42)
    
    for target_rank in [25, 30, 35, 40, 49, 50, 60, 75, 100]:
        if time.time() > deadline - 15:
            break
        
        # Try multiple random initializations
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            
            U_try = np.random.randn(n * m, target_rank)
            V_try = np.random.randn(m * p, target_rank)
            W_try = np.random.randn(p * n, target_rank)
            
            # Quick check using tensor contraction
            # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # Use einsum for speed
            U_reshaped = U_try.reshape(n, m, target_rank)
            V_reshaped = V_try.reshape(m, p, target_rank)
            W_reshaped = W_try.reshape(p, n, target_rank)
            
            # T[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
            # This is a 4-index contraction, let's do it efficiently
            # First contract U and V over j and r: temp[i,k,r] = sum_j U[i,j,r]*V[j,k,r]
            temp = np.einsum('ijr,jkr->ikr', U_reshaped, V_reshaped)
            # Then T[i,k] = sum_r temp[i,k,r] * W[k,i,r]
            T_recon = np.einsum('ikr,kir->ik', temp, W_reshaped)
            
            # Check if T_recon[i,k] matches what we need
            # For matrix multiplication, the tensor is T[i,j,k]=1, which means
            # sum_j T[i,j,k] should give... actually no.
            # The condition is: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            # This is NOT the same as the matrix product being correct.
            # Let me re-examine.
            
            # Actually, the tensor T[i,j,k] = 1 for ALL i,j,k is what we need.
            # So T_recon[i,j,k] must equal 1 for all i,j,k.
            # Let me redo the verification properly.
            
            T_full = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            
            if np.allclose(T_full, 1.0, atol=1e-10):
                best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                best_rank = target_rank
                break
        if best_rank < rank_schoolbook:
            break
    
    # Return the best decomposition found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
