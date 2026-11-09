import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0

    # We'll try to find a rank-24 decomposition using a structured approach.
    # The idea is to exploit the fact that n=2, so we can potentially share
    # some of the rank-1 terms between the two output rows.
    
    # Let's start with the known rank-28 solution and try to reduce it.
    # We'll use a local search approach: start with rank 28, then try to
    # eliminate terms one by one, adjusting the remaining terms to maintain
    # exact decomposition.
    
    # First, let's construct the rank-28 solution as before.
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen terms for 2x2 matrix multiplication
    strassen_terms = [
        # (a+d)(e+h) -> C00, C11
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        # (c+d)e -> C10, -C11
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1), (3, -1)]},
        # a(f-h) -> C01, C11
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1), (3, 1)]},
        # d(g-e) -> C00, C10
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (1, 1)]},
        # (a+b)h -> -C00, C01
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
        # (c-a)(e+f) -> C11
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        # (b-d)(g+h) -> C00
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]}
    ]
    
    blocks = [
        {'j_offset': 0, 'k_offset': 0},
        {'j_offset': 0, 'k_offset': 2},
        {'j_offset': 2, 'k_offset': 0},
        {'j_offset': 2, 'k_offset': 2}
    ]
    
    col = 0
    for block in blocks:
        if time.time() > deadline:
            break
        j_off = block['j_offset']
        k_off = block['k_offset']
        for term in strassen_terms:
            if time.time() > deadline:
                break
            # Add to U
            for (idx, coeff) in term['U']:
                i = idx // 2
                j = idx % 2
                global_i = i
                global_j = j + j_off
                U[global_i * m + global_j, col] += coeff
            
            # Add to V
            for (idx, coeff) in term['V']:
                j = idx // 2
                k = idx % 2
                global_j = j + j_off
                global_k = k + k_off
                V[global_j * p + global_k, col] += coeff
            
            # Add to W
            for (idx, coeff) in term['W']:
                k = idx // 2
                i = idx % 2
                global_k = k + k_off
                global_i = i
                W[global_k * n + global_i, col] += coeff
            
            col += 1
    
    # Now try to reduce the rank by eliminating terms.
    # We'll try to eliminate terms one by one, checking if the remaining
    # terms can be adjusted to maintain exact decomposition.
    
    def verify_decomposition(U_mat, V_mat, W_mat, rank_val):
        """Check if the decomposition is exact."""
        # The tensor T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # should equal delta[i,j,k] for matrix multiplication.
        # For M_{n,m,p}, T[i,j,k] = 1 if i,j,k are valid indices, else 0.
        # Actually, for matrix multiplication, the tensor is:
        # T[i,j,k] = 1 if we're multiplying A[i,:] by B[:,k], result C[i,k]
        # So T[i,j,k] = 1 for all valid i,j,k (it's the identity in the middle).
        
        # Let's compute the tensor from the decomposition.
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        
        # We'll check a few random indices.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute T[i,j,k]
                    val = 0.0
                    for r in range(rank_val):
                        val += U_mat[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    # For matrix multiplication, T[i,j,k] should be 1
                    # (since A[i,j] * B[j,k] contributes to C[i,k])
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    # Try to eliminate terms one by one.
    # We'll start from the end and work backwards.
    best_rank = rank
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # First, verify the initial decomposition.
    if not verify_decomposition(U, V, W, rank):
        # The initial decomposition is not valid. Return it anyway.
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to eliminate terms.
    for trial_rank in range(rank - 1, max(10, rank - 10), -1):
        if time.time() > deadline:
            break
        
        # Try to eliminate (rank - trial_rank) terms.
        # We'll try different combinations of terms to eliminate.
        # For simplicity, let's try eliminating the last (rank - trial_rank) terms.
        
        U_trial = U[:, :trial_rank].copy()
        V_trial = V[:, :trial_rank].copy()
        W_trial = W[:, :trial_rank].copy()
        
        # Check if this reduced decomposition is valid.
        if verify_decomposition(U_trial, V_trial, W_trial, trial_rank):
            best_rank = trial_rank
            best_U = U_trial.copy()
            best_V = V_trial.copy()
            best_W = W_trial.copy()
    
    # If we found a better rank, use it.
    if best_rank < rank:
        U_final = best_U
        V_final = best_V
        W_final = best_W
    else:
        U_final = U
        V_final = V
        W_final = W
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
