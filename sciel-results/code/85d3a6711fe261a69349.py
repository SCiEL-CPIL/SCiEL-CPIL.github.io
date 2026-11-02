"""Combined approach: structured search for low-rank exact tensor decomposition."""
import numpy as np
import time
import random
import itertools

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin within 150s limit

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

    # Precompute the target tensor T as a flat array for fast comparison
    # T has shape (n*m, m*p, p*n)
    # T[idx_u, idx_v, idx_w] = 1 if idx_u=(i,j), idx_v=(j,k), idx_w=(k,i)
    # else 0
    T_flat = np.zeros((n * m * m * p * p * n), dtype=np.float32)
    # Actually, we can just check the specific non-zero entries
    # There are n*m*p non-zero entries.
    # Let's store the indices of the non-zero entries
    nz_indices_u = []
    nz_indices_v = []
    nz_indices_w = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                nz_indices_u.append(idx_u)
                nz_indices_v.append(idx_v)
                nz_indices_w.append(idx_w)
                
    nz_indices_u = np.array(nz_indices_u, dtype=np.int32)
    nz_indices_v = np.array(nz_indices_v, dtype=np.int32)
    nz_indices_w = np.array(nz_indices_w, dtype=np.int32)

    def verify_decomposition_numpy(U_mat, V_mat, W_mat, rank):
        """Vectorized verification using numpy."""
        U_arr = np.array(U_mat, dtype=np.float32)
        V_arr = np.array(V_mat, dtype=np.float32)
        W_arr = np.array(W_mat, dtype=np.float32)

        # We need to check if sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] == 1 for all (i,j,k)
        # and 0 for others?
        # The problem statement says "exactly decompose the matrix multiplication tensor".
        # The tensor has 1s at (i,j,k) and 0s elsewhere.
        # So we must check that the result is 1 at nz_indices and 0 elsewhere.
        
        # Efficient check:
        # Compute the tensor product sum
        # result[idx_u, idx_v, idx_w] = sum_c U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
        
        # This is a large array (16 * 24 * 24 = 9216 elements).
        # We can compute the full tensor and check.
        
        # Reshape to (n*m, rank), (m*p, rank), (p*n, rank)
        # Use einsum to compute the outer product sum
        # result = np.einsum('ic,jc,kc->ijk', U_arr, V_arr, W_arr)
        
        result = np.einsum('ic,jc,kc->ijk', U_arr, V_arr, W_arr, optimize=True)
        
        # Check non-zero entries
        # result[idx_u, idx_v, idx_w] should be 1
        # We can't directly index a 3D array with 3 separate 1D arrays of the same length 
        # in a vectorized way that returns a 1D array easily without advanced indexing.
        # result[nz_indices_u, nz_indices_v, nz_indices_w] works if they are arrays.
        
        vals = result[nz_indices_u, nz_indices_v, nz_indices_w]
        if not np.allclose(vals, 1.0, atol=1e-5):
            return False
            
        # Check that other entries are 0?
        # The tensor T is sparse. If we assume the decomposition is exact, 
        # and we matched the 1s, do we need to check the 0s?
        # Yes, because the rank-1 terms might have "leakage" into other indices.
        # However, checking all 9216 entries is fast.
        # Let's flatten and check.
        # T_flat is 0 everywhere except the nz positions.
        # We can create a mask.
        
        # For speed, we might skip the zero-check if we trust the structure, 
        # but for exactness, we should check.
        # 9216 floats is small.
        target_flat = np.zeros(result.shape, dtype=np.float32)
        target_flat[nz_indices_u, nz_indices_v, nz_indices_w] = 1.0
        
        if not np.allclose(result, target_flat, atol=1e-5):
            return False
            
        return True

    def try_rank(rank, num_attempts=50):
        """Try to find a valid decomposition at the given rank."""
        if time.time() > deadline - 5:
            return None

        # Strategy: Random search with Gaussian initialization
        for attempt in range(num_attempts):
            if time.time() > deadline - 5:
                return None

            seed = hash((rank, attempt)) % (2**32)
            np_rng = np.random.RandomState(seed)

            # Initialize with small random values
            U = np_rng.randn(n*m, rank) * 0.5
            V = np_rng.randn(m*p, rank) * 0.5
            W = np_rng.randn(p*n, rank) * 0.5

            if verify_decomposition_numpy(U, V, W, rank):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
        return None

    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound is max(n*m, m*p, p*n) = 24
    # We start from 24 and go up.
    # Given the difficulty, we will likely only find 96.
    # But we try.
    
    # To save time, we can skip many ranks if we don't find anything quickly.
    # Let's try a few specific ranks: 24, 25, ..., 30, then jump to 90..96.
    
    ranks_to_try = list(range(24, 40)) + list(range(90, 97))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 5:
            break
        if rank >= best_rank:
            continue
            
        # Increase attempts for lower ranks? No, lower ranks are harder.
        # Maybe try more attempts for ranks close to 96?
        # Let's just do a fixed number of attempts.
        result = try_rank(rank, num_attempts=20)

        if result is not None:
            best_rank = rank
            best_result = result
            # If we found a very low rank, we can stop early
            if rank <= 30:
                break

    return best_result
# EVOLVE-BLOCK-END
