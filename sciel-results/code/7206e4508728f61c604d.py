import numpy as np
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Initialize U, V, W for schoolbook as fallback
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

    # Attempt NMTF approach as per directive
    # This is a heuristic search. It is very unlikely to find an exact
    # low-rank decomposition for 4x4x5.
    # We will try a small rank to see if it converges to something valid.
    # If it fails, we return schoolbook.
    
    def try_nmtf(target_rank):
        # Random initialization with non-negative values
        # Use a fixed seed for reproducibility in evaluation, or random?
        # The prompt says "initializing a random rank-20 candidate".
        # Let's use rank 20 as suggested, or maybe lower?
        # 4x4x5 schoolbook is 80. 
        # Let's try rank 30, 40, etc? 
        # The directive says "initializing a random rank-20 candidate".
        
        r = target_rank
        # Initialize
        U = np.random.rand(n * m, r) * 0.1
        V = np.random.rand(m * p, r) * 0.1
        W = np.random.rand(p * n, r) * 0.1
        
        # Construct the target tensor T[i, j, k]
        # T is the multiplication tensor: C = A * B
        # C_{i, k} = sum_j A_{i, j} B_{j, k}
        # Tensor T_{i, j, k} = 1 if it contributes to C_{i, k} via A_{i, j} and B_{j, k}
        # In the decomposition: T = U \circ V \circ W
        # T_{i, j, k} = sum_c U_{i, c} V_{j, c} W_{k, c}
        # Note the indices in the task:
        # U shape (n*m, rank) -> index i*m+j
        # V shape (m*p, rank) -> index j*p+k
        # W shape (p*n, rank) -> index k*n+i
        
        # Target Tensor T_flat
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    T[idx_u, idx_v, idx_w] = 1.0
        
        # Multiplicative Update Rules for NMTF
        # Minimize || T - U \circ V \circ W ||_F^2
        # This is equivalent to minimizing sum_{i,j,k} (T_{i,j,k} - sum_c U_{i,c} V_{j,c} W_{k,c})^2
        
        # We can reshape T to a matrix for easier calculation?
        # No, NMTF is tri-factorization.
        # Updates:
        # U_{i,c} <- U_{i,c} * sum_{j,k} T_{i,j,k} V_{j,c} W_{k,c} / sum_{j,k} (sum_c' U_{i,c'} V_{j,c'} W_{k,c'}) V_{j,c} W_{k,c}
        
        # This is computationally heavy for 4x4x5 but feasible.
        # However, convergence to exact integer/half-integer is not guaranteed.
        
        # Given the low probability of success and the risk of invalid output,
        # and the strict requirement for "exact" decomposition,
        # we will perform a few iterations and check validity.
        
        max_iter = 100
        for _ in range(max_iter):
            # Calculate predictions
            # pred_{i,j,k} = sum_c U_{i,c} V_{j,c} W_{k,c}
            # We need to update U, V, W.
            
            # Helper to compute the tensor product
            def compute_tensor(U, V, W):
                tensor = np.zeros((n * m, m * p, p * n))
                for c in range(r):
                    # Outer product of U[:, c], V[:, c], W[:, c]
                    u_col = U[:, c]
                    v_col = V[:, c]
                    w_col = W[:, c]
                    # Tensor[i, j, k] += u_col[i] * v_col[j] * w_col[k]
                    # Using broadcasting
                    # u_col shape (n*m, 1, 1)
                    # v_col shape (1, m*p, 1)
                    # w_col shape (1, 1, p*n)
                    tensor += np.einsum('i,j,k->ijk', u_col, v_col, w_col)
                return tensor

            # This loop is O(r * n*m * m*p * p*n) per iteration.
            # 4*4 * 4*5 * 5*4 = 16 * 20 * 20 = 6400 ops per c.
            # r=20. 128,000 ops per iteration. 100 iterations = 12.8M ops.
            # This is fast enough.
            
            # However, the update rule requires sums over indices.
            # Let's implement the updates carefully.
            
            # Update U
            # Numerator_{i,c} = sum_{j,k} T_{i,j,k} V_{j,c} W_{k,c}
            # Denominator_{i,c} = sum_{j,k} Pred_{i,j,k} V_{j,c} W_{k,c}
            
            pred = compute_tensor(U, V, W)
            
            # Pre-compute terms for efficiency?
            # Let's just loop for clarity given small sizes.
            
            new_U = U.copy()
            for i in range(n * m):
                for c in range(r):
                    num = 0.0
                    den = 0.0
                    for j in range(m * p):
                        for k in range(p * n):
                            t_val = T[i, j, k]
                            p_val = pred[i, j, k]
                            v_val = V[j, c]
                            w_val = W[k, c]
                            num += t_val * v_val * w_val
                            den += p_val * v_val * w_val
                    if den > 1e-12:
                        new_U[i, c] = U[i, c] * (num / den)
                    # else keep U[i,c] or set to 0? 
                    # If den is 0, it means this component isn't used.
            U = new_U
            
            # Update V
            new_V = V.copy()
            for j in range(m * p):
                for c in range(r):
                    num = 0.0
                    den = 0.0
                    for i in range(n * m):
                        for k in range(p * n):
                            t_val = T[i, j, k]
                            p_val = pred[i, j, k]
                            u_val = U[i, c]
                            w_val = W[k, c]
                            num += t_val * u_val * w_val
                            den += p_val * u_val * w_val
                    if den > 1e-12:
                        new_V[j, c] = V[j, c] * (num / den)
            V = new_V
            
            # Update W
            new_W = W.copy()
            for k in range(p * n):
                for c in range(r):
                    num = 0.0
                    den = 0.0
                    for i in range(n * m):
                        for j in range(m * p):
                            t_val = T[i, j, k]
                            p_val = pred[i, j, k]
                            u_val = U[i, c]
                            v_val = V[j, c]
                            num += t_val * u_val * v_val
                            den += p_val * u_val * v_val
                    if den > 1e-12:
                        new_W[k, c] = W[k, c] * (num / den)
            W = new_W
            
            # Check convergence / validity
            pred_final = compute_tensor(U, V, W)
            error = np.linalg.norm(T - pred_final)
            
            if error < 1e-6:
                # Check if entries are close to half-integers
                is_valid = True
                for mat in [U, V, W]:
                    for val in mat.flatten():
                        # Check if val is close to k/2
                        nearest_half = round(val * 2) / 2
                        if abs(val - nearest_half) > 1e-5:
                            is_valid = False
                            break
                    if not is_valid:
                        break
                if is_valid:
                    # Convert to lists of floats (half-integers)
                    U_list = [[float(round(val * 2) / 2) for val in row] for row in U]
                    V_list = [[float(round(val * 2) / 2) for val in row] for row in V]
                    W_list = [[float(round(val * 2) / 2) for val in row] for row in W]
                    return r, U_list, V_list, W_list
        return None

    # Try a few ranks
    # The directive mentions rank 20.
    # Let's try 20, 30, 40.
    # If any succeeds, return it.
    for target_rank in [20, 30, 40, 50]:
        res = try_nmtf(target_rank)
        if res:
            r, U_l, V_l, W_l = res
            return {"U": U_l, "V": V_l, "W": W_l}
            
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
