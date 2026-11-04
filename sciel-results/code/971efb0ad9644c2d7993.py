"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U, 'V': V, 'W': W}
    
    best_result = get_schoolbook()
    best_rank = rank_schoolbook
    
    # Tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    def verify(U, V, W):
        # Check if entries are integer or half-integer
        for mat in [U, V, W]:
            # 2 * mat should be integer
            if not np.allclose(2 * mat, np.round(2 * mat), atol=1e-9):
                return False
        # Check exact decomposition
        # T_approx[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # We can do this efficiently with einsum
        T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.allclose(T_approx, T, atol=1e-9):
            return True
        return False

    def project_half_integer(A):
        return np.round(A * 2) / 2.0

    def try_rank(r, max_iters=50):
        if time.time() > deadline:
            return None
        
        # Initialize
        # Use random initialization with small values
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
                
            # Update U
            # U[:, r] = T[:, :, r] * W[:, r] / (V[:, r] * W[:, r]) ... wait, standard ALS
            # For each r:
            # U[:, r] = sum_k T[:, :, k] * W[:, r] * V[:, r] / sum_k (V[:, r] * W[:, r])? No.
            # T_{ijk} = U_{ir} V_{jr} W_{kr}
            # Fix V, W. Update U.
            # U_{ir} = sum_{j,k} T_{ijk} V_{jr} W_{kr} / sum_{j,k} (V_{jr} W_{kr})^2 ? No.
            # Least squares: minimize sum_{j,k} (T_{ijk} - U_{ir} V_{jr} W_{kr})^2
            # derivative wrt U_{ir}: -2 sum_{j,k} T_{ijk} V_{jr} W_{kr} + 2 U_{ir} sum_{j,k} (V_{jr} W_{kr})^2 = 0
            # U_{ir} = sum_{j,k} T_{ijk} V_{jr} W_{kr} / sum_{j,k} (V_{jr} W_{kr})^2
            
            # Efficient computation:
            # Let M_U = V * W^T (outer product per rank? No, V is (m*p, r), W is (p*n, r))
            # We need sum_k T_{ijk} W_{kr} for each r.
            # T is (n*m, m*p, p*n). W is (p*n, r).
            # Temp = np.einsum('ijk,kr->ijr', T, W)
            # Denom = np.einsum('jr,kr->jkr', V, W) # (m*p, p*n, r)
            # Denom_sq = np.einsum('jkr,jkr->jr', Denom, Denom)
            
            Temp_U = np.einsum('ijk,kr->ijr', T, W)
            Denom_U = np.einsum('jr,kr->jkr', V, W)
            Denom_U_sq = np.sum(Denom_U**2, axis=(0,1)) # shape (r,)
            
            # Avoid division by zero
            U_new = Temp_U / (Denom_U_sq[np.newaxis, np.newaxis, :] + 1e-12)
            U = project_half_integer(U_new)
            
            # Update V
            Temp_V = np.einsum('ijk,ir->jkr', T, U)
            Denom_V = np.einsum('ir,kr->ikr', U, W)
            Denom_V_sq = np.sum(Denom_V**2, axis=(0,2)) # shape (r,)
            
            V_new = Temp_V / (Denom_V_sq[np.newaxis, np.newaxis, :] + 1e-12)
            V = project_half_integer(V_new)
            
            # Update W
            Temp_W = np.einsum('ijk,jr->ikr', T, V)
            Denom_W = np.einsum('ir,jr->ijr', U, V)
            Denom_W_sq = np.sum(Denom_W**2, axis=(0,1)) # shape (r,)
            
            W_new = Temp_W / (Denom_W_sq[np.newaxis, np.newaxis, :] + 1e-12)
            W = project_half_integer(W_new)
            
            # Check convergence/exactness occasionally
            if it % 5 == 0:
                if verify(U, V, W):
                    return {'U': U, 'V': V, 'W': W}
                    
        if verify(U, V, W):
            return {'U': U, 'V': V, 'W': W}
        return None

    # Search for lower rank
    # Start from a reasonable lower bound. 
    # Rank is at least max(n*m, m*p, p*n) = max(12, 24, 18) = 24.
    # But usually much higher. Let's try a range.
    # We will try ranks from 60 down to 25? Or just a few specific values?
    # Given the time limit, we can't try all.
    # Let's try a few random restarts for a few ranks.
    
    # To be efficient, we'll try ranks that are likely candidates.
    # If no better rank is found, we return schoolbook.
    
    # Note: Finding a valid decomposition is hard. 
    # We will try ranks 71, 70, ... down to maybe 50 with a few restarts each.
    # If we find one, we return it.
    
    ranks_to_try = list(range(71, 50, -1))
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        for restart in range(3):
            if time.time() > deadline:
                break
            res = try_rank(r)
            if res is not None:
                best_result = res
                best_rank = r
                # If we found a lower rank, we can try to go lower
                # But we break the restart loop for this rank
                break
                
    return best_result
# EVOLVE-BLOCK-END
