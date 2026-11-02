"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin within 150s limit
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Known lower bounds: for (4,4,6), the Strassen-style or other methods may help
    # Try ranks from a lower bound up to schoolbook
    
    # The tensor T_{i,j,k} = e_{ij} ⊗ e_{jk} ⊗ e_{ki} (with appropriate indexing)
    # We need: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # Use random search with verification
    def verify_decomposition(U_mat, V_mat, W_mat, rank):
        """Check if the decomposition is exact."""
        # U_mat: (n*m, rank), V_mat: (m*p, rank), W_mat: (p*n, rank)
        # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        # For non-(i,j,k) combinations, the tensor is 0, so we only need the (i,j,k) entries
        # Actually the tensor T[i,j,k] = 1 for all valid (i,j,k), 0 otherwise
        # But in matrix multiplication, the tensor is T[i,j,k] = delta_{i,j,k} in the sense that
        # (A*B)_{i,k} = sum_j A_{i,j} * B_{j,k}
        # So the tensor is T[i,j,k] = 1 for all (i,j,k) in range, and the decomposition must satisfy:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U_mat[i*m+j][c] * V_mat[j*p+k][c] * W_mat[k*n+i][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_decomposition_numpy(U_mat, V_mat, W_mat, rank):
        """Vectorized verification using numpy."""
        U_arr = np.array(U_mat, dtype=float)  # (n*m, rank)
        V_arr = np.array(V_mat, dtype=float)  # (m*p, rank)
        W_arr = np.array(W_mat, dtype=float)  # (p*n, rank)
        
        # Reshape to 3D
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        # For each (i,j,k), compute sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # This is a batched operation
        # U_3d[i,j,c], V_3d[j,k,c], W_3d[k,i,c]
        # Need to align indices: for fixed (i,j,k), sum over c
        
        # Create full tensor
        result = np.zeros((n, m, p))
        for c in range(rank):
            # U_3d[:, :, c] has shape (n, m)
            # V_3d[:, :, c] has shape (m, p)  
            # W_3d[:, :, c] has shape (p, n)
            # For each (i,j,k): U[i,j,c] * V[j,k,c] * W[k,i,c]
            # This is element-wise product after proper broadcasting
            # U_3d[:,:,c] is (n,m), V_3d[:,:,c] is (m,p), W_3d[:,:,c] is (p,n)
            # We need to compute outer product structure
            # For fixed c: contribution[i,j,k] = U[i,j,c] * V[j,k,c] * W[k,i,c]
            # This can be computed as:
            # temp[i,j,k] = U[i,j,c] * V[j,k,c] * W[k,i,c]
            # = (U[:,:,c])_{i,j} * (V[:,:,c])_{j,k} * (W[:,:,c])_{k,i}
            
            # Use einsum or manual
            # U_c = U_3d[:,:,c] shape (n,m)
            # V_c = V_3d[:,:,c] shape (m,p)
            # W_c = W_3d[:,:,c] shape (p,n)
            # result_c[i,j,k] = U_c[i,j] * V_c[j,k] * W_c[k,i]
            # = np.einsum('ij,jk,ki->ijk', U_c, V_c, W_c)
            result += np.einsum('ij,jk,ki->ijk', U_3d[:,:,c], V_3d[:,:,c], W_3d[:,:,c])
        
        # Check if all entries are 1
        return np.allclose(result, np.ones((n,m,p)), atol=1e-9)
    
    # Random search for lower rank
    def random_search(rank, num_tries=100, seed=42):
        """Try random initializations and check if they form a valid decomposition."""
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)
        
        for _ in range(num_tries):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small values
            U_rand = np_rng.randn(n*m, rank) * 0.5
            V_rand = np_rng.randn(m*p, rank) * 0.5
            W_rand = np_rng.randn(p*n, rank) * 0.5
            
            # Convert to lists for verification
            U_list = U_rand.tolist()
            V_list = V_rand.tolist()
            W_list = W_rand.tolist()
            
            if verify_decomposition_numpy(U_rand, V_rand, W_rand, rank):
                return U_list, V_list, W_list, True
        
        return None, None, None, False
    
    # Try decreasing ranks
    # Lower bound for matrix multiplication rank is at least max(n, m, p) * min(n, m, p) / something
    # For (4,4,6), let's try ranks from 16 up to schoolbook
    # Actually, the trivial lower bound is max(n*p, m*p, n*m) = max(24, 24, 16) = 24
    # But we can do better with structured decompositions
    
    # Let's try a structured approach based on known results
    # For 4x4 * 4x6, we can potentially use rank 24 or lower
    
    # Try ranks from 16 to schoolbook
    for rank in range(16, rank_schoolbook + 1):
        if time.time() > deadline - 10:
            break
        
        if rank >= best_rank:
            continue
        
        # Try random search
        U_try, V_try, W_try, found = random_search(rank, num_tries=50, seed=rank*1000+42)
        
        if found:
            best_rank = rank
            best_result = {"U": U_try, "V": V_try, "W": W_try}
            break  # Found minimal rank, stop
    
    return best_result
# EVOLVE-BLOCK-END
